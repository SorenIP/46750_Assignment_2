"""Question 2 price optimization with budget-balance and IR variants."""
from __future__ import annotations

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


def _build_member_kkt(model: gp.Model, data: InputData, posted_price: dict[int, gp.Var],
                      prefix: str) -> MemberKKT:
    """Embed a member LP's primal feasibility, dual feasibility and SOS1 complementarity."""
    lp = MemberModel(data, ContractTerms(), verbose=False).build()
    lp.m.Params.OutputFlag = 0
    lp.m.update()
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
            slack = model.addVar(lb=0.0, name=f"{prefix}_slack_{r}")
            multiplier = model.addVar(lb=0.0, name=f"{prefix}_multiplier_{r}")
            model.addConstr(row_expr + slack == rhs, name=f"{prefix}_primal_le_{r}")
            model.addConstr(dual + multiplier == 0.0, name=f"{prefix}_dual_sign_{r}")
            model.addSOS(GRB.SOS_TYPE1, [slack, multiplier])
        elif con.Sense == ">":
            dual = model.addVar(lb=0.0, name=f"{prefix}_dual_{r}")
            slack = model.addVar(lb=0.0, name=f"{prefix}_slack_{r}")
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


def _solve_variant(com: CommunityData, r_reference, name: str, budget_balance: bool,
                   individual_rationality: bool, relax_ir: bool, solver_log: bool = False) -> dict:
    """Build and solve one KKT-based DP MILP."""
    if relax_ir and not budget_balance:
        raise ValueError("The elastic-IR variant requires budget balance.")

    contract = com.contract
    beta = float(contract.tariff_discount_ratio)
    alpha = float(contract.penalty_DKK_per_kWh)
    cap = np.asarray(contract.capacity_limit_kW, dtype=float)
    prices: dict[int, gp.Var] = {}
    model = gp.Model(f"DP_{name}")
    model.Params.OutputFlag = int(solver_log)
    model.Params.MIPGap = 1e-6
    model.Params.DualReductions = 0
    for t in range(24):
        lower = float(com.energy_price[t] - (1.0 - beta) * com.export_tariff)
        upper = float(com.energy_price[t] + (1.0 - beta) * com.import_tariff + alpha)
        prices[t] = model.addVar(lb=lower, ub=upper, name=f"posted_price_{t}")

    member_kkt: dict[str, MemberKKT] = {}
    for data in com.members:
        member_kkt[data.consumer_id] = _build_member_kkt(model, data, prices, data.consumer_id)

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
                model.addConstr(kkt.total_cost <= baseline_cost, name=f"individual_rationality_{data.consumer_id}")
            if relax_ir:
                shortfall = model.addVar(lb=0.0, name=f"ir_shortfall_{data.consumer_id}")
                model.addConstr(kkt.total_cost - baseline_cost <= shortfall,
                                name=f"relaxed_ir_{data.consumer_id}")
                ir_shortfall[data.consumer_id] = shortfall

    if relax_ir:
        model.setObjectiveN(gp.quicksum(ir_shortfall.values()), index=0, priority=2,
                            name="minimize_total_IR_shortfall")
        model.setObjectiveN(community_cost, index=1, priority=1, name="minimize_community_cost")
    else:
        model.setObjective(community_cost, GRB.MINIMIZE)
    model.optimize()

    if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        return {"name": name, "status": "infeasible" if model.Status == GRB.INFEASIBLE else f"status_{model.Status}",
                "model": model}

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
        "posted_price": [prices[t].X for t in range(24)],
        "day_ahead_price": com.energy_price,
        "community_import": [p_imp[t].X for t in range(24)],
        "community_export": [p_exp[t].X for t in range(24)],
        "cap": cap,
        "exceedance": [exceed[t].X for t in range(24)],
    }, index=pd.Index(range(24), name="hour"))
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
        "members_better_off_vs_R": int((member_table["benefit_vs_R_DKK"] >= -1e-6).sum()),
        "members_worse_off_vs_R": int((member_table["benefit_vs_R_DKK"] < -1e-6).sum()),
    }
    return {"name": name, "status": "optimal", "model": model, "metrics": metrics,
            "members": member_table, "hourly": hourly, "responses": response_members,
            "objective": community_total}


def run_dp_variants(com: CommunityData, output_dir: str | Path | None = None,
                    solver_log: bool = False) -> dict[str, dict]:
    """Run cap-only, BB, BB+IR, and elastic-IR variants of the uniform-price DP."""
    r_reference = CommunityModel(com, "R").solve()
    central = CommunityModel(com, "C").solve()
    variants = [
        ("cap_only", False, False, False),
        ("budget_balance", True, False, False),
        ("budget_balance_ir", True, True, False),
        ("budget_balance_relaxed_ir", True, False, True),
    ]
    results = {}
    for name, bb, ir, relax in variants:
        result = _solve_variant(com, r_reference, name, bb, ir, relax, solver_log)
        if result["status"] == "optimal":
            result["metrics"]["community_cost_difference_vs_C_DKK"] = (
                result["metrics"]["community_cost_DKK"] - central.objective)
        results[name] = result

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        summary_rows = []
        for name, result in results.items():
            row = {"variant": name, "status": result["status"]}
            if result["status"] == "optimal":
                row.update(result["metrics"])
                result["hourly"].to_csv(out / f"DP_{name}_hourly.csv", index_label="hour")
                result["members"].to_csv(out / f"DP_{name}_members.csv")
                for cid, frame in result["responses"].items():
                    frame.to_csv(out / f"DP_{name}_{cid}_response.csv", index_label="hour")
            summary_rows.append(row)
        pd.DataFrame(summary_rows).set_index("variant").to_csv(out / "DP_variants_summary.csv")
    return results
