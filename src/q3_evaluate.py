"""Q3 step 1: evaluate fixed posted prices on preference scenarios. Run: python -m src.q3_evaluate"""
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from src.data_loader import load_community
from src.dynamic_pricing import PostedPriceMemberModel
from src.model import MemberModel, ContractTerms
from src.dp_variants import run_dp_variants

ROOT = Path(__file__).resolve().parent.parent
SCEN_DIR = ROOT / "results" / "base" / "Q3_scenarios"
OUT = ROOT / "results" / "base" / "Q3_baseline"

def member_in_scenario(d, scen_member):
    """InputData of member d with the scenario's reference profile, daily minimum and hourly bounds."""
    load = next(iter(scen_member.values()))               # the member's load (FL_Lk)
    ref = d.load_max_kWh * np.asarray(load["reference_load_capacity_factor"])
    D = d.max_hourly_deviation_kWh
    return replace(d, reference_load=ref, min_daily_energy_kWh=float(ref.sum()),
                   load_lower_kWh=np.maximum(0.0, ref - D), load_upper_kWh=np.minimum(d.load_max_kWh, ref + D))

def evaluate(com, x, scen):
    """What happens when the members respond to prices x in one scenario."""
    ct = com.contract
    beta, alpha, cap = ct.tariff_discount_ratio, ct.penalty_DKK_per_kWh, np.asarray(ct.capacity_limit_kW)
    net, rows = np.zeros(24), []
    for d in com.members:
        cid = d.consumer_id
        ds = member_in_scenario(d, scen["members"][cid])
        resp = PostedPriceMemberModel(ds, x).build().solve()
        q = (resp.hourly["import"] - resp.hourly["export"]).to_numpy()
        net += q
        cost_I = MemberModel(ds, ContractTerms(beta=beta, cap=np.asarray(ct.individual_capacity_limit_kW[cid]),
                                               alpha=alpha)).build().solve().objective
        cost_R = MemberModel(ds).build().solve().objective
        rows.append({"consumer_id": cid, "theta": scen["theta"]["profile"][cid], "payment": float(x @ q),
                     "cost_DP": resp.objective, "cost_I": cost_I, "cost_R": cost_R})
    m = pd.DataFrame(rows).set_index("consumer_id")
    P_imp, P_exp = np.maximum(net, 0), np.maximum(-net, 0)
    exc = np.maximum(P_imp - cap, 0)
    p = np.asarray(com.energy_price)
    cp_cost = (p @ (P_imp - P_exp) + (1 - beta) * (com.import_tariff * P_imp.sum() + com.export_tariff * P_exp.sum())
               + alpha * exc.sum())
    budget = m["payment"].sum() - cp_cost
    return {"scenario": scen["id"], "exceedance_kWh": exc.sum(), "hours_above_cap": int((exc > 1e-6).sum()),
            "peak_kW": P_imp.max(), "penalty_DKK": alpha * exc.sum(), "cp_cost_DKK": cp_cost, "budget_DKK": budget,
            "community_cost_DKK": cp_cost + (m["cost_DP"] - m["payment"]).sum(),   # + PV + disutility
            "n_IR_viol_vs_I": int((m["cost_DP"] > m["cost_I"] + 1e-6).sum()),
            "max_IR_shortfall_vs_I_DKK": max(0.0, (m["cost_DP"] - m["cost_I"]).max()),
            "n_IR_viol_vs_R": int((m["cost_DP"] > m["cost_R"] + 1e-6).sum())}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    com = load_community("base", day="d1")
    dp = run_dp_variants(com)                           # the Q2 prices
    variant = "budget_balance_ir"                       # the deterministic baseline you choose from Q2
    x = np.asarray(dp[variant]["posted_price"])
    oos = json.load(open(SCEN_DIR / "OOS_N100.json"))["scenarios"]
    df = pd.DataFrame([evaluate(com, x, s) for s in oos]).set_index("scenario")
    df.to_csv(OUT / f"DP_{variant}_on_OOS.csv")
    print(df.describe(percentiles=[.05, .5, .95]).T.round(2).to_string())
    print("Deterministic DP (theta=1):", {k: dp[variant]["metrics"][k] for k in list(dp[variant]["metrics"])[:8]})
