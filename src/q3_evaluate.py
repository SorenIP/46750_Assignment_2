"""Question 3, D6 + D9: the out-of-sample tester and the deterministic baseline M^D.

The tester takes a fixed hourly price vector x (posted day-ahead, before the preferences are realized) and a set of
preference scenarios, and computes what happens in every scenario:

  1. the members are rebuilt with the scenario's reference profile rho^theta, daily minimum and hourly bounds;
  2. every member solves its own LP at the posted prices x  ->  its optimal cost V_i;
  3. tie-breaking rule (OPTIMISTIC): among all schedules that keep every member at its optimal cost V_i, the one
     with the lowest community cost is selected (one joint LP per scenario). This is the selection the KKT models
     DP / SDP make, so in-sample and out-of-sample numbers use the same behavioural assumption. The schedule an LP
     solver returns when each member is solved on its own is also evaluated ("arbitrary"), to show how exposed the
     manager is to the tie-breaking;
  4. the reference of individual rationality is C_ref = min(C_R, C_I): the member's cost without a contract (R) or
     with its own individual contract (I, individual cap kappa_i * cap_t fixed day-ahead), in the same scenario
     (both cached per scenario set: they do not depend on x). The shortfall against R alone is also reported;
  5. the manager's bill at the connection point, penalty, exceedance, budget (payments - bill) and the IR shortfall
     of every member are computed.

The deterministic baseline M^D is the Q2 price vector computed on the expected parameters (theta = 1), read from
results/base/DP_variants/DP_<variant>_hourly.csv (no MILP is re-solved here).

Run:  python -m src.q3_evaluate                                   # both baselines, validation + OOS_N100
      python -m src.q3_evaluate --prices budget_balance --sets OOS_N100 IS_N10
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from src.data_loader import CommunityData, InputData, load_community
from src.dynamic_pricing import PostedPriceMemberModel
from src.model import ContractTerms, MemberModel

ROOT = Path(__file__).resolve().parent.parent
SCEN_DIR = ROOT / "results" / "base" / "Q3_scenarios"
DP_DIR = ROOT / "results" / "base" / "DP_variants"
QUEUE_DIR = ROOT / "results" / "base" / "DP_queue"
OUT = ROOT / "results" / "base" / "Q3_baseline"
TOL_IR = 1e-3           # DKK (0.1 oere): a member counts as worse off than its reference only above this; members
                        # whose IR binds sit at C_ref up to solver noise (~1e-5 DKK), which must not count
TOL_EXC = 1e-6          # kWh: an hour counts as above the cap only above this
BASELINES = ["ind_ir_ref", "budget_balance", "budget_balance_relaxed_ir"]   # main M^D first: DP^ind


# --------------------------------------------------------------------------- scenarios
def load_scenarios(name: str) -> list[dict]:
    """A saved scenario set (src/q3_scenarios.py), e.g. 'OOS_N100'."""
    return json.loads((SCEN_DIR / f"{name}.json").read_text())["scenarios"]


def member_in_scenario(d: InputData, scen_member: dict | None) -> InputData:
    """Member d with the scenario's reference profile, daily minimum (= reference energy) and hourly bounds.
    scen_member = None returns d unchanged (theta = 1, the expected scenario)."""
    if scen_member is None:
        return d
    load = next(iter(scen_member.values()))                       # the member's load (FL_Lk)
    ref = d.load_max_kWh * np.asarray(load["reference_load_capacity_factor"], dtype=float)
    D = d.max_hourly_deviation_kWh
    return replace(d, reference_load=ref, min_daily_energy_kWh=float(load["min_total_energy_per_day_kWh"]),
                   load_lower_kWh=np.maximum(d.load_min_kWh, ref - D),
                   load_upper_kWh=np.minimum(d.load_max_kWh, ref + D))


def scenario_members(com: CommunityData, scen: dict | None) -> list[InputData]:
    return [member_in_scenario(d, None if scen is None else scen["members"][d.consumer_id]) for d in com.members]


# --------------------------------------------------------------------------- references R and I per scenario
def _terms(com: CommunityData, d: InputData, model: str) -> ContractTerms:
    """R: no contract. I: the member's own individual contract (discount beta, cap kappa_i * cap_t, penalty alpha);
    the individual cap is agreed day-ahead, so it is the same in every scenario."""
    if model == "R":
        return ContractTerms()
    ct = com.contract
    return ContractTerms(beta=ct.tariff_discount_ratio, alpha=ct.penalty_DKK_per_kWh,
                         cap=np.asarray(ct.individual_capacity_limit_kW[d.consumer_id], dtype=float))


def ref_costs(com: CommunityData, scenarios: list[dict] | None, set_name: str | None = None,
              model: str = "R") -> pd.DataFrame:
    """Cost of every member in model R or I in every scenario (rows: scenario id, columns: member).
    scenarios = None gives the expected scenario (row 'expected'). Cached to csv per set."""
    path = SCEN_DIR / f"{set_name}_{model}_costs.csv" if set_name else None
    if path is not None and path.exists():
        cached = pd.read_csv(path, index_col="scenario")
        if len(cached) == len(scenarios):
            return cached
    rows = {}
    for s in (scenarios if scenarios is not None else [None]):
        sid = "expected" if s is None else s["id"]
        rows[sid] = {d.consumer_id: MemberModel(d, _terms(com, d, model)).build().solve().objective
                     for d in scenario_members(com, s)}
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "scenario"
    if path is not None:
        df.to_csv(path)
    return df


def r_costs(com: CommunityData, scenarios: list[dict], set_name: str | None = None) -> pd.DataFrame:
    return ref_costs(com, scenarios, set_name, "R")


# --------------------------------------------------------------------------- members' response to fixed prices
def _meter(com: CommunityData, net: np.ndarray) -> dict[str, float]:
    """Manager's bill at the connection point for an aggregated net import profile."""
    ct = com.contract
    beta, alpha, cap = ct.tariff_discount_ratio, ct.penalty_DKK_per_kWh, np.asarray(ct.capacity_limit_kW, dtype=float)
    p_imp, p_exp = np.maximum(net, 0.0), np.maximum(-net, 0.0)
    exc = np.maximum(p_imp - cap, 0.0)
    p = np.asarray(com.energy_price, dtype=float)
    out = {"energy_DKK": float(p @ (p_imp - p_exp)),
           "tariffs_DKK": float((1 - beta) * (com.import_tariff * p_imp.sum() + com.export_tariff * p_exp.sum())),
           "penalty_DKK": float(alpha * exc.sum()),
           "exceedance_kWh": float(exc.sum()), "hours_above_cap": int((exc > TOL_EXC).sum()),
           "peak_import_kW": float(p_imp.max()), "daily_import_kWh": float(p_imp.sum())}
    out["meter_bill_DKK"] = out["energy_DKK"] + out["tariffs_DKK"] + out["penalty_DKK"]
    return out


Prices = "np.ndarray | dict[str, np.ndarray]"


def price_of(x, cid: str) -> np.ndarray:
    """The price vector member cid faces: x itself (uniform price) or x[cid] (one price vector per member)."""
    return np.asarray(x[cid] if isinstance(x, dict) else x, dtype=float)


def respond(com: CommunityData, members: list[InputData], x: Prices) -> dict:
    """Members' response to prices x under the optimistic rule, plus the arbitrary (solver) response.

    x: one 24-h price vector for everybody, or a dict {member id: 24-h price vector} (price discrimination).
    Returns per-member net import (24 h) and cost parts for both selections."""
    ct = com.contract
    beta, alpha = ct.tariff_discount_ratio, ct.penalty_DKK_per_kWh
    cap = np.asarray(ct.capacity_limit_kW, dtype=float)

    # step 2: each member alone -> optimal value V_i (and the solver's arbitrary schedule)
    arbitrary, value = {}, {}
    for d in members:
        res = PostedPriceMemberModel(d, price_of(x, d.consumer_id)).build().solve()
        value[d.consumer_id] = res.objective
        arbitrary[d.consumer_id] = {"net": res.hourly["net_import"].to_numpy(),
                                    "pv_DKK": res.cost_breakdown["pv"],
                                    "disutility_DKK": res.cost_breakdown["disutility"]}

    # step 3: joint LP, every member held at its optimal value, min community cost
    m = gp.Model("optimistic_response")
    m.Params.OutputFlag = 0
    mm = {}
    for d in members:
        mm[d.consumer_id] = PostedPriceMemberModel(d, price_of(x, d.consumer_id), model=m).build()
        own = gp.quicksum(mm[d.consumer_id]._cost.values())
        v = value[d.consumer_id]
        m.addConstr(own <= v + 1e-7 * max(1.0, abs(v)), name=f"best_response_{d.consumer_id}")
    T = range(24)
    P_imp = m.addVars(T, lb=0.0, name="P_imp")
    P_exp = m.addVars(T, lb=0.0, name="P_exp")
    E = m.addVars(T, lb=0.0, name="exceed")
    for t in T:
        m.addConstr(gp.quicksum(k.var["import"][t] - k.var["export"][t] for k in mm.values()) == P_imp[t] - P_exp[t])
        m.addConstr(P_imp[t] - cap[t] <= E[t])
    p = np.asarray(com.energy_price, dtype=float)
    meter = gp.quicksum(p[t] * (P_imp[t] - P_exp[t]) + (1 - beta) * (com.import_tariff * P_imp[t]
                        + com.export_tariff * P_exp[t]) + alpha * E[t] for t in T)
    private = gp.quicksum(k._cost["pv"] + k._cost["disutility"] for k in mm.values())
    m.setObjective(meter + private, GRB.MINIMIZE)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"optimistic response LP: status {m.Status}")
    optimistic = {cid: {"net": np.array([k.var["import"][t].X - k.var["export"][t].X for t in T]),
                        "pv_DKK": k._cost["pv"].getValue(), "disutility_DKK": k._cost["disutility"].getValue()}
                  for cid, k in mm.items()}
    return {"optimistic": optimistic, "arbitrary": arbitrary, "value": value}


def _outcome(com, x, resp: dict, cost_R: pd.Series, cost_I: pd.Series) -> tuple[dict, list[dict]]:
    """Scenario metrics and per-member rows for one selection (optimistic or arbitrary).

    IR is measured against C_ref = min(C_R, C_I) (the report's definition: the member's best alternative);
    the shortfall against R alone is kept in the *_R columns."""
    net = sum(r["net"] for r in resp.values())
    out = _meter(com, net)
    rows = []
    for cid, r in resp.items():
        pay = float(price_of(x, cid) @ r["net"])
        cost = pay + r["pv_DKK"] + r["disutility_DKK"]
        cR, cI = float(cost_R[cid]), float(cost_I[cid])
        cref = min(cR, cI)
        rows.append({"consumer_id": cid, "payment_DKK": pay, "pv_DKK": r["pv_DKK"],
                     "disutility_DKK": r["disutility_DKK"], "cost_DP_DKK": cost,
                     "cost_R_DKK": cR, "cost_I_DKK": cI, "cost_ref_DKK": cref,
                     "IR_shortfall_DKK": max(0.0, cost - cref), "IR_shortfall_R_DKK": max(0.0, cost - cR)})
    mem = pd.DataFrame(rows)
    out["private_DKK"] = float((mem["pv_DKK"] + mem["disutility_DKK"]).sum())
    out["community_cost_DKK"] = out["meter_bill_DKK"] + out["private_DKK"]
    out["payments_DKK"] = float(mem["payment_DKK"].sum())
    out["budget_DKK"] = out["payments_DKK"] - out["meter_bill_DKK"]          # > 0 surplus, < 0 deficit
    for suffix, col in (("", "IR_shortfall_DKK"), ("_R", "IR_shortfall_R_DKK")):
        out[f"IR_violations{suffix}"] = int((mem[col] > TOL_IR).sum())
        out[f"IR_shortfall_total{suffix}_DKK"] = float(mem[col].sum())
        out[f"IR_shortfall_max{suffix}_DKK"] = float(mem[col].max())
    return out, rows


# --------------------------------------------------------------------------- the tester
def evaluate_prices(com: CommunityData, x: Prices, scenarios: list[dict] | None,
                    set_name: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate prices x (uniform vector or {member: vector}) on a scenario set (None = expected scenario theta = 1).

    Returns (one row per scenario, one row per scenario x member), optimistic selection; the scenario table also
    holds the community cost and budget of the arbitrary selection (suffix _arbitrary)."""
    x = {k: np.asarray(v, dtype=float) for k, v in x.items()} if isinstance(x, dict) else np.asarray(x, dtype=float)
    scen_list = scenarios if scenarios is not None else [None]
    R = ref_costs(com, scenarios, set_name, "R")
    I = ref_costs(com, scenarios, set_name, "I")
    rows, member_rows = [], []
    for s in scen_list:
        sid = "expected" if s is None else s["id"]
        resp = respond(com, scenario_members(com, s), x)
        out, mrows = _outcome(com, x, resp["optimistic"], R.loc[sid], I.loc[sid])
        arb, _ = _outcome(com, x, resp["arbitrary"], R.loc[sid], I.loc[sid])
        out = {"scenario": sid, **out,
               "community_cost_arbitrary_DKK": arb["community_cost_DKK"], "budget_arbitrary_DKK": arb["budget_DKK"],
               "IR_violations_arbitrary": arb["IR_violations"]}
        rows.append(out)
        for r in mrows:
            r["scenario"] = sid
            r["theta"] = 1.0 if s is None else float(s["theta"]["profile"][r["consumer_id"]])
        member_rows += mrows
    return pd.DataFrame(rows).set_index("scenario"), pd.DataFrame(member_rows)


def cvar(values, level: float = 0.95) -> float:
    """CVaR_level of a cost sample: mean of the worst (1 - level) share (upper tail), empirical."""
    v = np.sort(np.asarray(values, dtype=float))
    k = max(1, int(np.ceil((1 - level) * len(v))))
    return float(v[-k:].mean())


def summarize(df: pd.DataFrame) -> dict:
    """Distribution of the outcome over the scenarios: what the model comparison table needs."""
    c = df["community_cost_DKK"]
    return {"n_scenarios": len(df),
            "cost_mean_DKK": c.mean(), "cost_q05_DKK": c.quantile(0.05), "cost_q50_DKK": c.median(),
            "cost_q95_DKK": c.quantile(0.95), "cost_CVaR95_DKK": cvar(c, 0.95), "cost_max_DKK": c.max(),
            "penalty_mean_DKK": df["penalty_DKK"].mean(),
            "exceedance_mean_kWh": df["exceedance_kWh"].mean(),
            "P_cap_exceeded": float((df["exceedance_kWh"] > TOL_EXC).mean()),
            "budget_mean_DKK": df["budget_DKK"].mean(), "budget_q05_DKK": df["budget_DKK"].quantile(0.05),
            "P_budget_deficit": float((df["budget_DKK"] < -1e-6).mean()),
            "IR_violations_mean": df["IR_violations"].mean(),
            "P_any_IR_violation": float((df["IR_violations"] > 0).mean()),
            "IR_shortfall_total_mean_DKK": df["IR_shortfall_total_DKK"].mean(),
            "IR_violations_R_mean": df["IR_violations_R"].mean(),
            "P_any_IR_violation_R": float((df["IR_violations_R"] > 0).mean()),
            "IR_shortfall_total_R_mean_DKK": df["IR_shortfall_total_R_DKK"].mean(),
            "cost_mean_arbitrary_DKK": df["community_cost_arbitrary_DKK"].mean()}


def _variant_dir(variant: str) -> Path:
    """Where a solved Q2 variant lives: results/base/DP_variants (main.py --DP-variants) or
    results/base/DP_queue/<job> (run_dp_queue.py, e.g. ind_ir_ref = one price vector per member, IR vs C_ref)."""
    for d in (DP_DIR, QUEUE_DIR / variant):
        if (d / f"DP_{variant}_hourly.csv").exists():
            return d
    raise FileNotFoundError(f"DP_{variant}_hourly.csv not found in {DP_DIR} or {QUEUE_DIR / variant}: "
                            f"run  python main.py --DP-variants {variant}  or the run_dp_queue.py job first")


def load_prices(variant: str) -> Prices:
    """Posted prices of a solved Q2 variant (theta = 1): the deterministic baseline M^D.
    A uniform-price variant gives one 24-h vector; a price-discriminating one a dict {member: 24-h vector}."""
    h = pd.read_csv(_variant_dir(variant) / f"DP_{variant}_hourly.csv", index_col="hour")
    per_member = [c for c in h.columns if c.startswith("price_")]
    if per_member:
        return {c.removeprefix("price_"): h[c].to_numpy(dtype=float) for c in per_member}
    return h["posted_price"].to_numpy(dtype=float)


def _q2_reference(variant: str) -> dict:
    """The Q2 numbers of a variant at theta = 1 (for validating the tester), with its IR reference (R or C_ref)."""
    d = _variant_dir(variant)
    if d == DP_DIR:
        row = pd.read_csv(DP_DIR / "DP_variants_summary.csv", index_col="variant").loc[variant]
        ir_ref = "R"
    else:
        row = pd.read_csv(QUEUE_DIR / "queue_summary.csv", index_col="stage").loc[variant]
        ir_ref = "Cref" if row.get("ir_ref", "R") == "best" else "R"
    members = pd.read_csv(d / f"DP_{variant}_members.csv")        # benefit_vs_R_DKK is against the job's reference
    return {"status": row["status"], "community_cost_DKK": row["community_cost_DKK"],
            "exceedance_kWh": row["exceedance_kWh"], "budget_DKK": row["budget_balance_residual_DKK"],
            "IR_shortfall_DKK": float((-members["benefit_vs_R_DKK"]).clip(lower=0.0).sum()), "ir_ref": ir_ref}


# --------------------------------------------------------------------------- main
def main(variants: list[str], sets: list[str]) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    com = load_community("base", day="d1")
    summary = []
    for v in variants:
        x = load_prices(v)
        price_table = pd.DataFrame(x if isinstance(x, dict) else {"posted_price": x},
                                   index=pd.Index(range(24), name="hour"))
        price_table.to_csv(OUT / f"MD_{v}_prices.csv")

        # validation: theta = 1 must give back the Q2 numbers of the variant
        t0 = time.time()
        exp_df, exp_mem = evaluate_prices(com, x, None)
        q2 = _q2_reference(v)
        e = exp_df.iloc[0]
        val = {"variant": v, "set": "expected (validation)", "status_Q2": q2["status"],
               "Q2_community_cost_DKK": q2["community_cost_DKK"], "tester_community_cost_DKK": e["community_cost_DKK"],
               "Q2_exceedance_kWh": q2["exceedance_kWh"], "tester_exceedance_kWh": e["exceedance_kWh"],
               "Q2_budget_DKK": q2["budget_DKK"], "tester_budget_DKK": e["budget_DKK"],
               f"Q2_IR_shortfall_vs_{q2['ir_ref']}_DKK": q2["IR_shortfall_DKK"],
               "tester_IR_shortfall_vs_R_DKK": e["IR_shortfall_total_R_DKK"],
               "tester_IR_shortfall_vs_Cref_DKK": e["IR_shortfall_total_DKK"],
               "tester_IR_violations_vs_Cref": e["IR_violations"]}
        pd.Series(val).to_csv(OUT / f"MD_{v}_validation.csv", header=["value"])
        exp_mem.to_csv(OUT / f"MD_{v}_expected_members.csv", index=False)
        print(f"\n=== M^D = {v} ===\nvalidation at theta = 1 ({time.time() - t0:.1f} s):")
        print(pd.Series(val).to_string())
        summary.append({"model": f"MD_{v}", "set": "expected", **summarize(exp_df)})

        for s in sets:
            t0 = time.time()
            df, mem = evaluate_prices(com, x, load_scenarios(s), s)
            df.to_csv(OUT / f"MD_{v}_on_{s}.csv")
            mem.to_csv(OUT / f"MD_{v}_on_{s}_members.csv", index=False)
            row = {"model": f"MD_{v}", "set": s, **summarize(df), "eval_time_s": time.time() - t0}
            summary.append(row)
            print(f"\n{s}: {len(df)} scenarios in {row['eval_time_s']:.0f} s")
            print(pd.Series(row).drop(["model", "set"]).round(3).to_string())
    table = pd.DataFrame(summary)
    path = OUT / "MD_summary.csv"
    if path.exists():                     # keep rows of earlier runs, replace the ones re-run now
        old = pd.read_csv(path)
        keep = ~old.set_index(["model", "set"]).index.isin(table.set_index(["model", "set"]).index)
        table = pd.concat([old[keep], table], ignore_index=True)
    table.to_csv(path, index=False)
    print(f"\nsaved to {OUT}")
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prices", nargs="+", default=BASELINES, help="Q2 variants whose prices are the baseline M^D")
    ap.add_argument("--sets", nargs="+", default=["OOS_N100"], help="scenario sets to evaluate on")
    a = ap.parse_args()
    main(a.prices, a.sets)
