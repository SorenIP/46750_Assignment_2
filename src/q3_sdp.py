"""Question 3, D4: stochastic dynamic pricing SDP^ind (one price vector per member), risk-neutral base.

The prices x_i,t are decided day-ahead and shared by all scenarios. In every scenario s each member's best response
to x_i is embedded through its KKT conditions (same construction as DP^ind in src/dp_variants.py).
Base model: minimise the expected community cost; budget balance and IR (vs C_ref = min(C_R, C_I)) in expectation.
Validation: N = 1 with theta = 1 gives DP^ind (372.162 DKK).

Run:  python -m src.q3_sdp --set expected          # validation
      python -m src.q3_sdp --set IS_N5 --time 1800
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path
import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB
from src.community_model import CommunityModel
from src.data_loader import CommunityData, load_community
from src.dp_variants import _build_member_kkt
from src.q3_evaluate import evaluate_prices, load_prices, load_scenarios, ref_costs, scenario_members, summarize

OUT = Path(__file__).resolve().parent.parent / "results" / "base" / "Q3_SDP"

def build_sdp(com: CommunityData, scenarios: list[dict] | None, set_name: str | None = None,
              risk: str = "neutral", requirements: str = "expected",
              dual_bound: float = 30.0, slack_bound: float = 500.0) -> dict:
    """Build SDP^ind. scenarios = None: the expected scenario only (theta = 1, the deterministic DP^ind).
    risk / requirements: 'neutral' / 'expected' is the base model; the D5 variants are added here later."""
    if risk != "neutral" or requirements != "expected":
        raise NotImplementedError("only the risk-neutral base model with requirements in expectation so far")
    ct = com.contract
    beta, alpha = ct.tariff_discount_ratio, ct.penalty_DKK_per_kWh
    cap = np.asarray(ct.capacity_limit_kW, dtype=float)
    p = np.asarray(com.energy_price, dtype=float)
    scen = scenarios if scenarios is not None else [None]
    ids = ["expected"] if scenarios is None else [s["id"] for s in scenarios]
    prob = [1.0] if scenarios is None else [float(s["probability"]) for s in scenarios]
    cref = np.minimum(ref_costs(com, scenarios, set_name, "R"), ref_costs(com, scenarios, set_name, "I"))

    m = gp.Model("SDP_ind")
    # ---- first stage: one price vector per member, the same in every scenario (non-anticipativity)
    x = {d.consumer_id: {t: m.addVar(lb=float(p[t] - (1 - beta) * com.export_tariff),
                                     ub=float(p[t] + (1 - beta) * com.import_tariff + alpha),
                                     name=f"price_{d.consumer_id}_{t}") for t in range(24)}
         for d in com.members}

    # ---- second stage: every scenario gets its own copy of the members' KKT and of the community meter
    exp_cost, exp_budget = gp.LinExpr(), gp.LinExpr()
    exp_member = {cid: gp.LinExpr() for cid in x}
    per_scen = []
    for k, (s, pk) in enumerate(zip(scen, prob)):
        kkt = {d.consumer_id: _build_member_kkt(m, d, x[d.consumer_id], f"{d.consumer_id}_s{k}",
                                                dual_bound=dual_bound, slack_bound=slack_bound, tight_slacks=True)
               for d in scenario_members(com, s)}
        P_imp = m.addVars(24, lb=0.0, name=f"P_imp_s{k}")
        P_exp = m.addVars(24, lb=0.0, name=f"P_exp_s{k}")
        E = m.addVars(24, lb=0.0, name=f"exceed_s{k}")
        for t in range(24):
            net = gp.quicksum(q.primal[q.indices["import"][t]] - q.primal[q.indices["export"][t]] for q in kkt.values())
            m.addConstr(net == P_imp[t] - P_exp[t], name=f"balance_s{k}_{t}")
            m.addConstr(P_imp[t] - float(cap[t]) <= E[t], name=f"cap_s{k}_{t}")
        meter = gp.quicksum(float(p[t]) * (P_imp[t] - P_exp[t])
                            + (1 - beta) * (com.import_tariff * P_imp[t] + com.export_tariff * P_exp[t])
                            + alpha * E[t] for t in range(24))
        private = gp.quicksum(q.private_cost for q in kkt.values())
        payments = gp.quicksum(q.payment_cost for q in kkt.values())   # = sum_t x_i,t * net_i,t (strong duality)
        exp_cost += pk * (meter + private)
        exp_budget += pk * (payments - meter)
        for cid, q in kkt.items():
            exp_member[cid] += pk * q.total_cost
        per_scen.append({"id": ids[k], "prob": pk, "kkt": kkt, "E": E, "P_imp": P_imp,
                         "meter": meter, "private": private, "payments": payments})

    # ---- requirements in expectation
    m.addConstr(exp_budget == 0.0, name="budget_balance_expected")
    for cid in x:
        rhs = float(sum(pk * cref.loc[i, cid] for i, pk in zip(ids, prob)))
        m.addConstr(exp_member[cid] <= rhs, name=f"IR_expected_{cid}")
    m.setObjective(exp_cost, GRB.MINIMIZE)
    return {"model": m, "x": x, "scen": per_scen, "exp_cost": exp_cost, "cref": cref}


def lower_bound(com: CommunityData, scenarios: list[dict] | None) -> float:
    """Expected cost of central dispatch C in the scenarios: no pricing model can do better (valid cut)."""
    scen = scenarios if scenarios is not None else [None]
    prob = [1.0] if scenarios is None else [float(s["probability"]) for s in scenarios]
    return sum(pk * CommunityModel(dataclasses.replace(com, members=scenario_members(com, s)), "C").solve().objective
               for s, pk in zip(scen, prob))


def run_sdp(set_name: str = "expected", time_limit: float = 1800.0, gap: float = 1e-4,
            oos_set: str | None = "OOS_N100", tag: str = "neutral") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    com = load_community("base", day="d1")
    scenarios = None if set_name == "expected" else load_scenarios(set_name)
    name = f"SDP_{tag}_{set_name}"

    t0 = time.time()
    sdp = build_sdp(com, scenarios, None if scenarios is None else set_name)
    m, x = sdp["model"], sdp["x"]
    lb = lower_bound(com, scenarios)
    m.addConstr(sdp["exp_cost"] >= lb - 1e-6, name="lower_bound_cut_from_C")       # speed-up
    start = load_prices("ind_ir_ref")                                              # speed-up: warm start from DP^ind
    for cid, xs in x.items():
        for t, v in xs.items():
            v.Start = float(np.clip(start[cid][t], v.LB, v.UB))
    build_s = time.time() - t0
    m.Params.MIPGap, m.Params.TimeLimit, m.Params.DualReductions = gap, time_limit, 0
    m.optimize()

    row = {"model": name, "set": set_name, "n_scenarios": len(sdp["scen"]), "status": m.Status,
           "build_s": build_s, "solve_s": m.Runtime, "n_vars": m.NumVars, "n_sos": m.NumSOS,
           "lower_bound_C_DKK": lb}
    if m.SolCount == 0:
        print(f"{name}: no solution (status {m.Status})")
        return row
    row.update({"in_sample_expected_cost_DKK": m.ObjVal, "mip_gap": m.MIPGap})
    prices = {cid: np.array([xs[t].X for t in range(24)]) for cid, xs in x.items()}
    pd.DataFrame(prices, index=pd.Index(range(24), name="hour")).to_csv(OUT / f"{name}_prices.csv")
    ins = pd.DataFrame([{"scenario": s["id"], "probability": s["prob"],
                         "community_cost_DKK": s["meter"].getValue() + s["private"].getValue(),
                         "budget_DKK": s["payments"].getValue() - s["meter"].getValue(),
                         "exceedance_kWh": sum(s["E"][t].X for t in range(24))} for s in sdp["scen"]])
    ins.to_csv(OUT / f"{name}_in_sample.csv", index=False)
    print(f"\n{name}: cost {m.ObjVal:.3f} (lower bound C {lb:.3f}), gap {m.MIPGap:.2%}, "
          f"build {build_s:.0f} s + solve {m.Runtime:.0f} s")

    if oos_set:                                                   # out-of-sample test with the same tester as M^D
        df, mem = evaluate_prices(com, prices, load_scenarios(oos_set), oos_set)
        df.to_csv(OUT / f"{name}_on_{oos_set}.csv")
        mem.to_csv(OUT / f"{name}_on_{oos_set}_members.csv", index=False)
        row.update({f"oos_{k}": v for k, v in summarize(df).items()})
        print(pd.Series(summarize(df)).round(3).to_string())
    path = OUT / "SDP_summary.csv"
    table = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        table = pd.concat([old[old["model"] != name], table], ignore_index=True)
    table.to_csv(path, index=False)
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="expected", help="expected (validation, N = 1) | IS_N5 | IS_N10 | IS_N10_cv")
    ap.add_argument("--time", type=float, default=1800.0, help="time limit in seconds")
    ap.add_argument("--no-oos", action="store_true", help="skip the out-of-sample test")
    a = ap.parse_args()
    run_sdp(a.set, a.time, oos_set=None if a.no_oos else "OOS_N100")