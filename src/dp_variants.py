"""Question 2 price optimization with budget-balance and IR variants."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from .community_model import CommunityModel
from .data_loader import CommunityData, InputData
from .model import ContractTerms, MemberModel


@dataclass
class MemberKKT:
    data: InputData
    primal: list[gp.Var]
    price: dict[int, gp.Var]
    indices: dict[str, dict[int, int]]
    payment_cost: gp.LinExpr
    private_cost: gp.LinExpr
    total_cost: gp.LinExpr
    daily_energy_value: gp.LinExpr


def _row_slack_ranges(lp_model: gp.Model) -> dict[int, float]:
    """Exact largest value of the slack of every inequality row over the member's primal feasible set.

    Each slack lies in [0, this value] at every feasible schedule, so it is a valid (and much tighter than a
    blanket 500 kWh) upper bound for the big-M of the complementarity pair of that row.
    """
    work = lp_model.copy()
    work.Params.OutputFlag = 0
    work.Params.Method = 1
    work.update()
    A = work.getA().tocsr()
    wvars, wcons = work.getVars(), work.getConstrs()
    out: dict[int, float] = {}
    for r, con in enumerate(wcons):
        if con.Sense == "=":
            continue
        row = A.getrow(r)
        expr = gp.quicksum(float(a) * wvars[int(j)] for j, a in zip(row.indices, row.data))
        # slack of "<" row: rhs - expr (maximise by minimising expr); of ">" row: expr - rhs
        work.setObjective(expr, GRB.MINIMIZE if con.Sense == "<" else GRB.MAXIMIZE)
        work.optimize()
        if work.Status != GRB.OPTIMAL:
            continue                                  # unbounded or numerical trouble: keep the blanket bound
        value = work.ObjVal
        out[r] = (float(con.RHS) - value) if con.Sense == "<" else (value - float(con.RHS))
    return out


def _build_member_kkt(model: gp.Model, data: InputData, posted_price: dict[int, gp.Var],
                      prefix: str, dual_bound: float = GRB.INFINITY, slack_bound: float = GRB.INFINITY,
                      tight_slacks: bool = False) -> MemberKKT:
    """Embed a member LP's primal feasibility, dual feasibility and SOS1 complementarity."""
    lp = MemberModel(data, ContractTerms(), verbose=False).build()
    lp.m.Params.OutputFlag = 0
    lp.m.update()
    slack_range = _row_slack_ranges(lp.m) if tight_slacks else {}
    matrix = lp.m.getA().tocsr()
    source_vars = lp.m.getVars()
    source_cons = lp.m.getConstrs()
    primal = [
        model.addVar(lb=v.LB, ub=v.UB, name=f"{prefix}_{v.VarName.replace('[', '_').replace(']', '')}")
        for v in source_vars
    ]

    row_duals: list[gp.Var] = []
    for r, con in enumerate(source_cons):
        coeff = matrix.getrow(r)
        row_expr = gp.quicksum(float(a) * primal[int(j)] for j, a in zip(coeff.indices, coeff.data))
        rhs = float(con.RHS)
        if con.Sense == "=":
            dual = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"{prefix}_dual_{r}")
            model.addConstr(row_expr == rhs, name=f"{prefix}_primal_eq_{r}")
        elif con.Sense == "<":
            dual = model.addVar(lb=-GRB.INFINITY, ub=0.0, name=f"{prefix}_dual_{r}")
            slack = model.addVar(lb=0.0, ub=min(slack_bound, slack_range.get(r, GRB.INFINITY) + 1e-6),
                                 name=f"{prefix}_slack_{r}")
            multiplier = model.addVar(lb=0.0, ub=dual_bound, name=f"{prefix}_multiplier_{r}")
            model.addConstr(row_expr + slack == rhs, name=f"{prefix}_primal_le_{r}")
            model.addConstr(dual + multiplier == 0.0, name=f"{prefix}_dual_sign_{r}")
            model.addSOS(GRB.SOS_TYPE1, [slack, multiplier])
        elif con.Sense == ">":
            dual = model.addVar(lb=0.0, ub=dual_bound, name=f"{prefix}_dual_{r}")
            slack = model.addVar(lb=0.0, ub=min(slack_bound, slack_range.get(r, GRB.INFINITY) + 1e-6),
                                 name=f"{prefix}_slack_{r}")
            model.addConstr(row_expr - slack == rhs, name=f"{prefix}_primal_ge_{r}")
            model.addSOS(GRB.SOS_TYPE1, [slack, dual])
        else:
            raise ValueError(f"Unsupported member constraint sense {con.Sense!r} in {con.ConstrName}")
        row_duals.append(dual)

    stationarity_terms: list[list[tuple[float, gp.Var]]] = [[] for _ in source_vars]
    for r in range(matrix.shape[0]):
        coeff = matrix.getrow(r)
        for j, value in zip(coeff.indices, coeff.data):
            stationarity_terms[int(j)].append((float(value), row_duals[r]))

    by_name: dict[str, dict[int, int]] = {}
    for j, var in enumerate(source_vars):
        stem = var.VarName.split("[", 1)[0]
        hour = int(var.VarName.rsplit("[", 1)[1][:-1])
        by_name.setdefault(stem, {})[hour] = j

    for j, var in enumerate(source_vars):
        stem = var.VarName.split("[", 1)[0]
        hour = int(var.VarName.rsplit("[", 1)[1][:-1])
        if stem == "import":
            objective_coefficient = posted_price[hour]
        elif stem == "export":
            objective_coefficient = -posted_price[hour]
        elif stem == "pv":
            objective_coefficient = float(data.pv_marginal_cost)
        elif stem in ("dev_plus", "dev_minus"):
            objective_coefficient = float(data.linear_disutility)
        else:
            objective_coefficient = 0.0
        stationarity = gp.LinExpr(objective_coefficient)
        if stationarity_terms[j]:
            stationarity.addTerms([-term[0] for term in stationarity_terms[j]],
                                  [term[1] for term in stationarity_terms[j]])
        model.addConstr(stationarity == 0.0, name=f"{prefix}_stationarity_{j}")

    def expression(stem: str, coefficient: float) -> gp.LinExpr:
        return gp.quicksum(coefficient * primal[j] for j in by_name.get(stem, {}).values())

    private_cost = (float(data.pv_marginal_cost) * expression("pv", 1.0)
                    + float(data.linear_disutility) * (expression("dev_plus", 1.0)
                                                       + expression("dev_minus", 1.0)))
    lower_level_cost = gp.quicksum(float(con.RHS) * row_duals[r] for r, con in enumerate(source_cons))
    min_energy_row = next(r for r, con in enumerate(source_cons) if con.ConstrName == "min_energy")
    daily_energy_value = -row_duals[min_energy_row]
    model.update()
    return MemberKKT(data, primal, by_name["import"], by_name, lower_level_cost - private_cost,
                     private_cost, lower_level_cost, daily_energy_value)


def _fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _progress_callback(label: str, prices: dict | None = None, best_path: Path | None = None,
                       stop_file: Path | None = None, log_path: Path | None = None):
    """Live progress line (elapsed time, best solution, bound, gap, nodes), refreshed about once per second.

    Every time the solver finds a better solution, its prices are saved to ``best_path`` (a small csv), so a partial
    result survives even if the run is killed.
    """
    state = {"last": 0.0, "last_log": 0.0}

    def callback(model, where):
        if where == GRB.Callback.MIPSOL and prices is not None and best_path is not None:
            keys = list(prices)
            frame = pd.DataFrame({"hour": [k[1] if isinstance(k, tuple) else k for k in keys],
                                  "posted_price": [model.cbGetSolution(prices[k]) for k in keys],
                                  "objective_DKK": model.cbGet(GRB.Callback.MIPSOL_OBJ)})
            if keys and isinstance(keys[0], tuple):
                frame.insert(0, "group", [k[0] for k in keys])
            frame.to_csv(best_path, index=False)
            return
        if where != GRB.Callback.MIP:
            return
        now = time.time()
        if now - state["last"] < 1.0:
            return
        state["last"] = now
        if stop_file is not None and stop_file.exists():      # remote stop: keep the best solution, move on
            try:
                stop_file.unlink()
            except OSError:
                pass
            model.terminate()
        best = model.cbGet(GRB.Callback.MIP_OBJBST)
        bound = model.cbGet(GRB.Callback.MIP_OBJBND)
        nodes = int(model.cbGet(GRB.Callback.MIP_NODCNT))
        if abs(best) >= 1e30:
            incumbent, gap = "none yet", "-"
        else:
            incumbent = f"{best:9.3f}"
            gap = "-" if abs(bound) >= 1e20 else f"{abs(best - bound) / max(abs(best), 1e-10) * 100:6.2f}%"
        bound_txt = "        -" if abs(bound) >= 1e20 else f"{bound:9.3f}"
        line = (f"{label} | elapsed {_fmt_time(now - state['t0'])} | best {incumbent} | bound {bound_txt} | "
                f"gap {gap} | nodes {nodes}")
        print(f"\r  {line}   ", end="", flush=True)
        if log_path is not None and now - state["last_log"] >= 10.0:     # a plain-text log line every 10 s
            state["last_log"] = now
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(time.strftime("%H:%M:%S") + "  " + line + "\n")

    callback_state = state
    return callback, callback_state


def _solve_variant(com: CommunityData, r_reference, name: str, budget_balance: bool,
                   individual_rationality: bool, relax_ir: bool, solver_log: bool = False,
                   label: str | None = None, mip_gap_target: float = 1e-6, speedups: bool = True,
                   start_prices=None, lower_bound: float | None = None, best_path: Path | None = None,
                   dual_bound: float = 30.0, slack_bound: float = 500.0,
                   stop_file: Path | None = None, log_path: Path | None = None,
                   shortfall_mode: str = "lexicographic", shortfall_cap: float | None = None,
                   ir_eps: float = 0.0, price_groups: str = "uniform", time_limit: float | None = None) -> dict:
    """Build and solve one KKT-based DP MILP.

    ir_eps: individual rationality is cost_i <= R_i + ir_eps (0 = strict IR; a small positive value is a relaxation).
    price_groups: "uniform" (one price per hour for everybody, the assignment's model), "asset" (one price vector per
    asset class: none / PV / battery / PV+battery) or "individual" (one per member): the extensions that remove the
    conflict between uniform prices and the requirements. time_limit: seconds (None = no limit).

    Elastic-IR modes (only with relax_ir): "lexicographic" (min shortfall, then cost), "shortfall_only"
    (min total shortfall; Gurobi focused on the bound, to prove how small it can get), "cost_given_shortfall"
    (min community cost with total shortfall <= shortfall_cap).

    speedups=True adds (all valid, none changes the optimal value unless a bound is too small, which is checked):
      * finite upper bounds on every multiplier (``dual_bound``, DKK/kWh) and slack (``slack_bound``, kWh) that
        appear in a complementarity pair, so Gurobi can turn the SOS1 sets into tight big-M constraints;
      * the cut  community cost >= lower_bound  (lower_bound = cost of model C, since every DP variant is C
        with extra restrictions), so the bound is tight from the start;
      * a MIP start with the prices ``start_prices`` (Gurobi completes the rest).
    """
    if relax_ir and not budget_balance:
        raise ValueError("The elastic-IR variant requires budget balance.")

    contract = com.contract
    beta = float(contract.tariff_discount_ratio)
    alpha = float(contract.penalty_DKK_per_kWh)
    cap = np.asarray(contract.capacity_limit_kW, dtype=float)
    model = gp.Model(f"DP_{name}")
    model.Params.OutputFlag = int(solver_log)
    model.Params.MIPGap = mip_gap_target
    model.Params.DualReductions = 0
    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)

    def group_of(d) -> str:
        if price_groups == "uniform":
            return "all"
        if price_groups == "individual":
            return d.consumer_id
        if price_groups == "asset":
            return "PV+BAT" if d.has_pv and d.has_battery else "PV" if d.has_pv else "BAT" if d.has_battery else "none"
        raise ValueError(f"unknown price_groups {price_groups!r}")

    group_prices: dict[str, dict[int, gp.Var]] = {}
    for data in com.members:
        g = group_of(data)
        if g in group_prices:
            continue
        group_prices[g] = {}
        for t in range(24):
            lower = float(com.energy_price[t] - (1.0 - beta) * com.export_tariff)
            upper = float(com.energy_price[t] + (1.0 - beta) * com.import_tariff + alpha)
            group_prices[g][t] = model.addVar(lb=lower, ub=upper,
                                              name=f"posted_price_{t}" if g == "all" else f"posted_price_{g}_{t}")
    prices = group_prices["all"] if price_groups == "uniform" else \
        {(g, t): v for g, pr in group_prices.items() for t, v in pr.items()}      # flat view for start / callback

    member_kkt: dict[str, MemberKKT] = {}
    for data in com.members:
        member_kkt[data.consumer_id] = _build_member_kkt(
            model, data, group_prices[group_of(data)], data.consumer_id,
            dual_bound=dual_bound if speedups else GRB.INFINITY, slack_bound=slack_bound if speedups else GRB.INFINITY,
            tight_slacks=speedups)

    p_imp = model.addVars(24, lb=0.0, name="community_import")
    p_exp = model.addVars(24, lb=0.0, name="community_export")
    exceed = model.addVars(24, lb=0.0, name="cap_exceedance")
    for t in range(24):
        net_consumption = gp.quicksum(
            member_kkt[d.consumer_id].primal[member_kkt[d.consumer_id].indices["import"][t]]
            - member_kkt[d.consumer_id].primal[member_kkt[d.consumer_id].indices["export"][t]]
            for d in com.members
        )
        model.addConstr(net_consumption == p_imp[t] - p_exp[t], name=f"community_balance_{t}")
        model.addConstr(p_imp[t] - float(cap[t]) <= exceed[t], name=f"cap_{t}")

    meter_cost = gp.quicksum(
        float(com.energy_price[t]) * (p_imp[t] - p_exp[t])
        + (1.0 - beta) * (com.import_tariff * p_imp[t] + com.export_tariff * p_exp[t])
        + alpha * exceed[t]
        for t in range(24)
    )
    private_cost = gp.quicksum(kkt.private_cost for kkt in member_kkt.values())
    community_cost = meter_cost + private_cost
    total_member_cost = gp.quicksum(kkt.total_cost for kkt in member_kkt.values())

    if budget_balance:
        model.addConstr(total_member_cost - private_cost == meter_cost, name="budget_balance")

    ir_shortfall: dict[str, gp.Var] = {}
    if individual_rationality or relax_ir:
        for data in com.members:
            kkt = member_kkt[data.consumer_id]
            baseline_cost = float(r_reference.member_cost.loc[data.consumer_id, "cost_DKK"])
            if individual_rationality:
                model.addConstr(kkt.total_cost <= baseline_cost + float(ir_eps), name=f"individual_rationality_{data.consumer_id}")
            if relax_ir:
                shortfall = model.addVar(lb=0.0, name=f"ir_shortfall_{data.consumer_id}")
                model.addConstr(kkt.total_cost - baseline_cost <= shortfall,
                                name=f"relaxed_ir_{data.consumer_id}")
                ir_shortfall[data.consumer_id] = shortfall

    if relax_ir and shortfall_mode == "shortfall_only":
        model.setObjective(gp.quicksum(ir_shortfall.values()), GRB.MINIMIZE)
        model.Params.MIPFocus = 3
    elif relax_ir and shortfall_mode == "cost_given_shortfall":
        model.addConstr(gp.quicksum(ir_shortfall.values()) <= float(shortfall_cap), name="shortfall_cap")
        model.setObjective(community_cost, GRB.MINIMIZE)
        model.addConstr(community_cost >= float(lower_bound if lower_bound is not None else 0.0) - 1e-6,
                        name="lower_bound_cut_from_C")
    elif relax_ir:
        model.setObjectiveN(gp.quicksum(ir_shortfall.values()), index=0, priority=2,
                            name="minimize_total_IR_shortfall")
        model.setObjectiveN(community_cost, index=1, priority=1, name="minimize_community_cost")
    else:
        model.setObjective(community_cost, GRB.MINIMIZE)
    if speedups and lower_bound is not None and not relax_ir:
        model.addConstr(community_cost >= float(lower_bound) - 1e-6, name="lower_bound_cut_from_C")
    if speedups and start_prices is not None:
        for key, var in prices.items():
            t = key[1] if isinstance(key, tuple) else key
            var.Start = float(np.clip(start_prices[t], var.LB, var.UB))
    callback, cb_state = _progress_callback(label or name, prices, best_path, stop_file, log_path)
    cb_state["t0"] = time.time()
    model.optimize(callback)
    runtime = time.time() - cb_state["t0"]
    print()   # end the live progress line

    # No time limit. Ctrl+C once stops this variant and keeps the best solution found so far (if any).
    has_solution = model.SolCount > 0
    if model.Status == GRB.INFEASIBLE:
        return {"name": name, "status": "infeasible", "runtime_s": runtime, "model": model}
    if model.Status == GRB.INF_OR_UNBD:
        return {"name": name, "status": "infeasible_or_unbounded", "runtime_s": runtime, "model": model}
    if not has_solution:
        return {"name": name, "status": f"no_solution_status_{model.Status}", "runtime_s": runtime, "model": model}
    if model.Status == GRB.OPTIMAL:
        status = "optimal"
    elif model.Status == GRB.INTERRUPTED:
        status = "interrupted_best_found"
    elif model.Status == GRB.TIME_LIMIT:
        status = "time_limit_best_found"
    else:
        status = "suboptimal"
    try:
        mip_gap = float(model.MIPGap)
    except (gp.GurobiError, AttributeError):   # multi-objective (relaxed IR): no single gap is reported
        mip_gap = 0.0 if status == "optimal" else float("nan")

    if speedups:
        at_bound = [v.VarName for v in model.getVars()
                    if (("_multiplier_" in v.VarName or "_dual_" in v.VarName)     # slacks may sit at their bound: harmless
                        and v.UB < 1e20 and v.UB > 0 and v.X >= 0.999 * v.UB)]
        if at_bound:
            print(f"  WARNING: {len(at_bound)} multiplier/slack variables sit at their upper bound (e.g. {at_bound[0]}); "
                  f"the bound may have cut off solutions. Re-run with larger bounds or with --DP-plain.", flush=True)

    rows = []
    response_members = {}
    for data in com.members:
        kkt = member_kkt[data.consumer_id]
        member_cost = float(kkt.total_cost.getValue())
        by_stem = kkt.indices
        pv_cost = float(data.pv_marginal_cost) * sum(
            kkt.primal[j].X for j in by_stem.get("pv", {}).values())
        disutility = float(data.linear_disutility) * sum(
            kkt.primal[j].X for stem in ("dev_plus", "dev_minus")
            for j in by_stem.get(stem, {}).values())
        payment = member_cost - pv_cost - disutility
        baseline_cost = float(r_reference.member_cost.loc[data.consumer_id, "cost_DKK"])
        shortfall = float(ir_shortfall[data.consumer_id].X) if data.consumer_id in ir_shortfall else 0.0
        rows.append({
            "consumer_id": data.consumer_id,
            "archetype": data.archetype,
            "price_group": group_of(data),
            "slope_DKK_per_kWh": float(data.linear_disutility),
            "R_cost_DKK": baseline_cost,
            "DP_cost_DKK": member_cost,
            "payment_DKK": payment,
            "benefit_vs_R_DKK": baseline_cost - member_cost,
            "IR_shortfall_DKK": shortfall,
            "daily_energy_value_DKK_per_kWh": float(kkt.daily_energy_value.getValue()),
        })
        frame = pd.DataFrame(index=pd.Index(range(24), name="hour"))
        for stem, indices in by_stem.items():
            if stem in ("load", "import", "export", "pv", "dev_plus", "dev_minus",
                        "charge", "discharge", "soc"):
                frame[stem] = [kkt.primal[indices[t]].X for t in range(24)]
        frame["net"] = frame["import"] - frame["export"]
        frame["reference_load"] = data.reference_load
        frame["deviation"] = frame["load"] - frame["reference_load"]
        response_members[data.consumer_id] = frame

    member_table = pd.DataFrame(rows).set_index("consumer_id")
    hourly = pd.DataFrame({
        "posted_price": [float(np.mean([pr[t].X for pr in group_prices.values()])) for t in range(24)],
        "day_ahead_price": com.energy_price,
        "community_import": [p_imp[t].X for t in range(24)],
        "community_export": [p_exp[t].X for t in range(24)],
        "cap": cap,
        "exceedance": [exceed[t].X for t in range(24)],
    }, index=pd.Index(range(24), name="hour"))
    if price_groups != "uniform":
        for g, pr in group_prices.items():
            hourly[f"price_{g}"] = [pr[t].X for t in range(24)]
    actual_manager_bill = float(meter_cost.getValue())
    collected = float(member_table["payment_DKK"].sum())
    agg_import = hourly["community_import"].to_numpy()
    shifted = sum(float(h["deviation"].abs().sum() / 2.0) for h in response_members.values())
    reference_objective = float(r_reference.objective)
    community_total = float(community_cost.getValue())
    cap_exceed = float(hourly["exceedance"].sum())
    metrics = {
        "community_cost_DKK": community_total,
        "community_benefit_vs_R_DKK": reference_objective - community_total,
        "community_cost_difference_vs_C_DKK": np.nan,
        "daily_import_kWh": float(agg_import.sum()),
        "peak_import_kW": float(agg_import.max()),
        "hours_above_cap": int((hourly["exceedance"] > 1e-3).sum()),
        "exceedance_kWh": cap_exceed,
        "shifted_kWh": shifted,
        "manager_meter_cost_DKK": actual_manager_bill,
        "member_payments_collected_DKK": collected,
        "budget_balance_residual_DKK": collected - actual_manager_bill,
        "total_IR_shortfall_DKK": float(member_table["IR_shortfall_DKK"].sum()),
        "mip_gap": mip_gap,
        "runtime_s": runtime,
        "members_better_off_vs_R": int((member_table["benefit_vs_R_DKK"] >= -1e-6).sum()),
        "members_worse_off_vs_R": int((member_table["benefit_vs_R_DKK"] < -1e-6).sum()),
    }
    return {"name": name, "status": status, "runtime_s": runtime, "mip_gap": mip_gap, "model": model, "metrics": metrics,
            "members": member_table, "hourly": hourly, "responses": response_members,
            "objective": community_total}


VARIANTS = [
    ("cap_only", False, False, False),
    ("budget_balance", True, False, False),
    ("budget_balance_ir", True, True, False),
    ("budget_balance_relaxed_ir", True, False, True),
    ("ir_shortfall_only", True, False, True),         # proof run: smallest possible total IR shortfall
    ("relaxed_ir_stage2", True, False, True),         # min cost with the shortfall fixed at the value found above
]
SHORTFALL_MODE = {"ir_shortfall_only": "shortfall_only", "relaxed_ir_stage2": "cost_given_shortfall"}
DEFAULT_VARIANTS = ["cap_only", "budget_balance", "budget_balance_ir", "budget_balance_relaxed_ir"]


def _save_variant(out: Path, name: str, result: dict) -> dict:
    """Write the files of one variant and return its row of the summary table."""
    row = {"variant": name, "status": result["status"], "runtime_s": result.get("runtime_s")}
    if "metrics" in result:
        row.update(result["metrics"])
        result["hourly"].to_csv(out / f"DP_{name}_hourly.csv", index_label="hour")
        result["members"].to_csv(out / f"DP_{name}_members.csv")
        for cid, frame in result["responses"].items():
            frame.to_csv(out / f"DP_{name}_{cid}_response.csv", index_label="hour")
    return row


def _read_best_shortfall(out) -> float:
    """Fallback for the stage-2 variant when the shortfall-only variant was not run in the same session."""
    members = Path(out) / "DP_budget_balance_relaxed_ir_members.csv"
    return float(pd.read_csv(members)["IR_shortfall_DKK"].sum())


def run_dp_variants(com: CommunityData, output_dir: str | Path | None = None,
                    solver_log: bool = False, only: list[str] | None = None,
                    mip_gap_target: float = 1e-6, speedups: bool = True) -> dict[str, dict]:
    """Run cap-only, BB, BB+IR, and elastic-IR variants of the uniform-price DP, one after the other.

    No time limit. A live progress line shows elapsed time, best solution, bound and gap. Each variant is saved
    as soon as it is finished (infeasible ones too, with their status), and the summary table is updated after
    every variant. ``only``: names of the variants to run (default: all four, in this order).
    """
    chosen = [v for v in VARIANTS if (v[0] in DEFAULT_VARIANTS if only is None else v[0] in only)]
    if only is not None:
        chosen.sort(key=lambda v: only.index(v[0]))   # run in the order given
    if only is not None:
        unknown = set(only) - {v[0] for v in VARIANTS}
        if unknown:
            raise ValueError(f"unknown variant(s) {sorted(unknown)}; available: {[v[0] for v in VARIANTS]}")
    out = Path(output_dir) if output_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    r_reference = CommunityModel(com, "R").solve()
    central = CommunityModel(com, "C").solve()
    results, summary_rows = {}, []
    start_prices = central.grid["pi"].to_numpy()          # first guess: the internal prices of model C
    t_all = time.time()
    for k, (name, bb, ir, relax) in enumerate(chosen, start=1):
        label = f"[{k}/{len(chosen)}] {name}"
        print(f"{label}: building and solving (Ctrl+C once = stop this variant, keep best found)", flush=True)
        best_path = (out / f"DP_{name}_best_so_far_prices.csv") if out is not None else None
        stop_file = (out / f"STOP_{name}.txt") if out is not None else None     # create this file to stop the variant
        log_path = (out / "progress_log.txt") if out is not None else None
        if stop_file is not None and stop_file.exists():
            stop_file.unlink()
        if log_path is not None:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(time.strftime("%H:%M:%S") + f"  START {label}\n")
        mode = SHORTFALL_MODE.get(name, "lexicographic")
        cap_value = None
        if mode == "cost_given_shortfall":
            prev = results.get("ir_shortfall_only", {}).get("metrics", {})
            cap_value = (prev["total_IR_shortfall_DKK"] if prev else _read_best_shortfall(out)) * 1.001 + 1e-4
            print(f"{label}: shortfall cap {cap_value:.4f} DKK", flush=True)
        if name in ("ir_shortfall_only", "relaxed_ir_stage2") and out is not None:
            seed = out / "DP_budget_balance_relaxed_ir_best_so_far_prices.csv"
            if seed.exists():
                start_prices = pd.read_csv(seed)["posted_price"].to_numpy()
        result = _solve_variant(com, r_reference, name, bb, ir, relax, solver_log, label=label,
                                mip_gap_target=mip_gap_target, speedups=speedups, start_prices=start_prices,
                                lower_bound=central.objective, best_path=best_path,
                                stop_file=stop_file, log_path=log_path,
                                shortfall_mode=mode, shortfall_cap=cap_value)
        if "metrics" in result:
            result["metrics"]["community_cost_difference_vs_C_DKK"] = (
                result["metrics"]["community_cost_DKK"] - central.objective)
            print(f"{label}: {result['status']} | cost {result['metrics']['community_cost_DKK']:.3f} DKK "
                  f"(C: {central.objective:.3f}) | gap {result['mip_gap'] * 100:.2f}% | "
                  f"time {_fmt_time(result['runtime_s'])}", flush=True)
        else:
            print(f"{label}: {result['status']} | time {_fmt_time(result['runtime_s'])}", flush=True)
        results[name] = result
        if log_path is not None:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(time.strftime("%H:%M:%S") + f"  END {label}: {result['status']} | time {_fmt_time(result['runtime_s'])}\n")
        if "hourly" in result:
            start_prices = result["hourly"]["posted_price"].to_numpy()    # next variant starts from these prices
        if out is not None:
            summary_rows.append(_save_variant(out, name, result))
            summary_path = out / "DP_variants_summary.csv"
            new = pd.DataFrame(summary_rows).set_index("variant")
            if summary_path.exists():                       # keep rows of earlier runs; replace the ones re-run now
                old = pd.read_csv(summary_path, index_col="variant")
                new = pd.concat([old.drop(index=[i for i in new.index if i in old.index]), new])
            new.to_csv(summary_path)
            print(f"    saved to {out}", flush=True)
    print(f"All done in {_fmt_time(time.time() - t_all)}.")
    return results
