"""Model C: community contract with central dispatch (Question 1, contract type (b), direct control).

Self-contained on purpose, like model I: it *uses* the member LP of ``src/model.py`` (``MemberModel``, unchanged),
the solver wrapper ``src/lp_solve.py`` and the R and I solutions, and changes nothing in the R or I code.

What model C is
---------------
The community manager holds one contract for all members, behind one meter, and dispatches every member in one
optimization that minimizes the community's total cost:

    min  sum_t [ (p_t + (1-beta) tau_imp) P_imp_t - (p_t - (1-beta) tau_exp) P_exp_t + alpha e_t ]
         + sum_i ( PV cost_i + disutility_i )

    s.t. every member's constraints (as in R: load bounds, daily minimum energy, PV, battery),
         internal balance      sum_i q_i,t = P_imp_t - P_exp_t          (dual: internal price pi_t)
         cap                   e_t >= P_imp_t - cap_t,  e_t >= 0         (dual of the cap: congestion rent mu_t)

with q_i,t = q_imp_i,t - q_exp_i,t the net consumption of member i. Behind the meter members exchange energy freely
and without tariff: only the net q_i,t matters, the split into gross import and export of a member has no meaning
(it is free and not reported). The discounted tariffs and the penalty are paid at the connection point only.
It is one LP (16 members x 24 hours), so strong duality holds and the duals exist.

Settlement (dual pricing): every member pays the uniform internal price on its net consumption,

    payment_i = sum_t pi_t q_i,t,        cost^C_i = payment_i + PV cost_i + disutility_i,   pi_t = - dual of the balance.

The manager's budget is  sum_i payment_i - (cost at the connection point)  (the congestion rent when the cap binds).

Metric definitions (units: kWh, kW, DKK; "benefit" > 0 means better off than in the reference)
-------------------------------------------------------------------------------------------------
Same as model I (``src/experiments_I.py``), plus:
* benefit against R      B^R_i = C^R_i - C^C_i      (individual rationality: B^R_i >= 0)
* benefit against I      B^I_i = C^I_i - C^C_i      (is pooling better than signing alone?)
* budget of the manager  sum_i payment_i - cost at the connection point  (0 = balanced)
* DSO revenue            tariffs + penalty paid at the connection point(s); reference R (full tariffs, no penalty)
See ``report/metrics_hypotheses.tex`` for the table of all metrics (view, definition, unit, reference).
Community cost = cost at the connection point + PV costs + disutilities (payments between members and manager cancel).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gurobipy as gp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gurobipy import GRB

from .data_loader import CommunityData
from .experiments_I import community_metrics, solve_community_I
from .lp_solve import Solution, solve_lp
from .model import ContractTerms, MemberModel, aggregate, solve_community_R


@dataclass
class CommunityResult:
    """Solution of model C (or of the individual-meter variant used for validation)."""

    objective: float
    solver: str
    members: dict[str, pd.DataFrame]      # per member: load, pv, charge, ..., net (kWh/h), deviation
    grid: pd.DataFrame                    # per hour: price, P_imp, P_exp, exceed, cap, pi (internal price), mu (cap rent)
    cost: dict[str, float]                # community cost components: energy, tariffs, penalty, pv, disutility
    member_parts: pd.DataFrame            # per member: PV cost and disutility (DKK)
    checks: dict[str, float]              # validation residuals


def build_community_C(com: CommunityData, beta: float, alpha: float, cap_scale: float = 1.0,
                      meters: str = "community", own_flow_tariffs: bool = False):
    """Build model C as one Gurobi model. ``own_flow_tariffs=True`` charges the tariffs on every member's own gross
    import and export instead of on the community meter (validation only: with beta = 0, no cap and no penalty this
    must give R, see ``validation_C``). ``meters="individual"`` builds the limit case instead: every member behind
    its own meter at full tariffs, no cap (= model R in one model), used only for validation.
    Returns (model, member models, dict of community variables/constraints)."""
    T = range(24)
    m = gp.Model("community")
    members = [MemberModel(d, ContractTerms(), model=m).build() for d in com.members]   # unchanged member LPs
    parts = gp.quicksum(mm._cost["pv"] + mm._cost["disutility"] for mm in members)
    g = {}
    if meters == "individual":
        m.setObjective(gp.quicksum(gp.quicksum(mm._cost.values()) for mm in members), GRB.MINIMIZE)
        m.update()
        return m, members, g
    p = com.energy_price
    k = 1.0 - beta
    tau_i, tau_e = com.import_tariff, com.export_tariff
    cap = cap_scale * np.asarray(com.contract.capacity_limit_kW, dtype=float)
    free = dict(lb=-GRB.INFINITY)
    P_imp, P_exp, e = m.addVars(T, name="P_imp", **free), m.addVars(T, name="P_exp", **free), m.addVars(T, name="exceed", **free)
    g["P_imp"], g["P_exp"], g["exceed"], g["cap"] = P_imp, P_exp, e, cap
    # internal balance: sum of the members' net consumption = metered import - metered export (dual = - internal price)
    g["balance"] = m.addConstrs(
        (gp.quicksum(mm.var["import"][t] - mm.var["export"][t] for mm in members) - P_imp[t] + P_exp[t] == 0 for t in T),
        name="community_balance")
    g["cap_con"] = m.addConstrs((P_imp[t] - float(cap[t]) - e[t] <= 0 for t in T), name="cap")
    m.addConstrs((-P_imp[t] <= 0 for t in T), name="P_imp_nonneg")
    m.addConstrs((-P_exp[t] <= 0 for t in T), name="P_exp_nonneg")
    m.addConstrs((-e[t] <= 0 for t in T), name="exceed_nonneg")
    g["cost_expr"] = {
        "energy": gp.quicksum(float(p[t]) * (P_imp[t] - P_exp[t]) for t in T),
        "tariffs": (gp.quicksum(k * tau_i * mm.var["import"][t] + k * tau_e * mm.var["export"][t]
                                for mm in members for t in T) if own_flow_tariffs else
                    gp.quicksum(k * tau_i * P_imp[t] + k * tau_e * P_exp[t] for t in T)),
        "penalty": gp.quicksum(alpha * e[t] for t in T),
    }
    m.setObjective(gp.quicksum(g["cost_expr"].values()) + parts, GRB.MINIMIZE)
    m.update()
    return m, members, g


def solve_community_C(com: CommunityData, beta: float | None = None, alpha: float | None = None,
                      cap_scale: float = 1.0, meters: str = "community", solver: str = "auto",
                      own_flow_tariffs: bool = False) -> CommunityResult:
    """Solve model C (LP) and collect primal values, the internal price and the cap rent."""
    ct = com.contract
    beta = ct.tariff_discount_ratio if beta is None else beta
    alpha = ct.penalty_DKK_per_kWh if alpha is None else alpha
    m, members, g = build_community_C(com, beta, alpha, cap_scale, meters, own_flow_tariffs)
    sol = solve_lp(m, solver)
    T = list(range(24))

    mem, parts = {}, {}
    for mm in members:
        d = mm.d
        h = pd.DataFrame(index=pd.Index(T, name="hour"))
        for name, v in mm.var.items():
            h[name] = [sol.x(v[t]) for t in T]
        h["reference_load"] = d.reference_load
        h["net"] = h["import"] - h["export"]
        h["deviation"] = h["load"] - h["reference_load"]
        mem[d.consumer_id] = h
        parts[d.consumer_id] = {"pv_DKK": d.pv_marginal_cost * h["pv"].sum(),
                                "disutility_DKK": d.linear_disutility * (h["dev_plus"] + h["dev_minus"]).sum()}
    parts = pd.DataFrame(parts).T

    p = com.energy_price
    grid = pd.DataFrame({"price": p}, index=pd.Index(T, name="hour"))
    if meters == "individual":
        P = sum(h["import"] for h in mem.values()); X = sum(h["export"] for h in mem.values())
        grid["P_imp"], grid["P_exp"] = P, X
        cost = {"total": sol.objective}
        return CommunityResult(sol.objective, sol.solver, mem, grid, cost, parts, {})
    grid["P_imp"] = [sol.x(g["P_imp"][t]) for t in T]
    grid["P_exp"] = [sol.x(g["P_exp"][t]) for t in T]
    grid["exceed"] = [sol.x(g["exceed"][t]) for t in T]
    grid["cap"] = g["cap"]
    grid["pi"] = [-sol.pi(g["balance"][t]) for t in T]          # internal price, DKK/kWh
    grid["mu"] = [-sol.pi(g["cap_con"][t]) for t in T]          # congestion rent: value of one more kW of cap, DKK/kWh
    cost = {k_: _value(e_, sol) for k_, e_ in g["cost_expr"].items()}
    cost["pv"], cost["disutility"] = parts["pv_DKK"].sum(), parts["disutility_DKK"].sum()
    assert abs(sum(cost.values()) - sol.objective) < 1e-6 * max(1.0, abs(sol.objective))

    # ---- validation residuals ----
    k = 1.0 - beta
    dual_obj = sum(sol.pi(c) * c.RHS for c in m.getConstrs())
    importing = grid["P_imp"] > 1e-9
    checks = {
        "strong_duality_gap_DKK": abs(sol.objective - dual_obj),
        "internal_balance_kWh": float((sum(h["net"] for h in mem.values()) - (grid["P_imp"] - grid["P_exp"])).abs().max()),
        # while importing, the internal price is the discounted import price plus the cap rent
        "price_vs_import_price_plus_rent": float((grid["pi"] - (p + k * com.import_tariff + grid["mu"]))[importing].abs().max()),
        "cap_rent_zero_when_cap_slack": float(grid["mu"][grid["P_imp"] < grid["cap"] - 1e-6].abs().max()
                                              if (grid["P_imp"] < grid["cap"] - 1e-6).any() else 0.0),
        "exceed_def_error_kW": float((grid["exceed"] - np.maximum(grid["P_imp"] - grid["cap"], 0)).abs().max()),
    }
    return CommunityResult(sol.objective, sol.solver, mem, grid, cost, parts, checks)


def _value(expr: gp.LinExpr, sol: Solution) -> float:
    """Value of a linear expression at the solution (works for any solver)."""
    return sum(expr.getCoeff(i) * sol.x(expr.getVar(i)) for i in range(expr.size())) + expr.getConstant()


def settlement(com: CommunityData, res: CommunityResult, beta: float | None = None) -> pd.DataFrame:
    """Dual pricing: member i pays the internal price pi_t on its net consumption.
    Returns a per-member table with payment, cost and the manager's budget in ``.attrs['manager']``."""
    beta = com.contract.tariff_discount_ratio if beta is None else beta
    pi = res.grid["pi"]
    pay = pd.Series({cid: float((pi * h["net"]).sum()) for cid, h in res.members.items()}, name="payment_DKK")
    tab = res.member_parts.copy()
    tab["payment_DKK"] = pay
    tab["cost_C_DKK"] = tab["payment_DKK"] + tab["pv_DKK"] + tab["disutility_DKK"]
    collected = pay.sum()
    manager_cost = res.cost["energy"] + res.cost["tariffs"] + res.cost["penalty"]
    tab.attrs["manager"] = {"collected_DKK": collected, "cost_at_connection_point_DKK": manager_cost,
                            "budget_DKK": collected - manager_cost,
                            "cap_rent_DKK": float((res.grid["mu"] * res.grid["cap"]).sum())}
    return tab


def run_C(com: CommunityData, out: Path, solver: str = "auto") -> dict:
    """Solve R, I and C on the same community, compare them, settle model C, save tables and figures to ``out``.

    Files: ``C_summary.csv`` (R, I, C side by side), ``C_members.csv`` (per member: cost in R, I and C, benefits,
    payment, shifted energy), ``C_by_type.csv``, ``C_prices.csv`` (hourly grid, internal price, cap rent),
    ``C_manager.csv`` (budget), ``C_checks.csv``, ``C_profiles.png``, ``C_members.png``.
    """
    out.mkdir(parents=True, exist_ok=True)
    R = solve_community_R(com.members)
    I = solve_community_I(com)
    C = solve_community_C(com, solver=solver)
    cap = np.asarray(com.contract.capacity_limit_kW, dtype=float)

    # ---- community level ----
    mR, mI = community_metrics(com, R), community_metrics(com, I)
    P = C.grid["P_imp"].values
    exc = np.maximum(P - cap, 0.0)
    mC = {"daily_import_kWh": P.sum(), "peak_import_kW": P.max(), "hours_above_cap": int((exc > 1e-3).sum()),
          "exceedance_kWh": exc.sum(), "max_exceedance_kW": exc.max(), "total_cost_DKK": C.objective,
          **{f"cost_{k}_DKK": v for k, v in C.cost.items()}}
    summary = pd.DataFrame({"R": mR, "I": mI, "C": mC})
    # DSO view: what the DSO receives for the connection (tariffs + penalty; the energy cost goes to the market)
    summary.loc["dso_revenue_DKK"] = summary.loc["cost_tariffs_DKK"] + summary.loc["cost_penalty_DKK"]
    summary["C_minus_R"] = summary["C"] - summary["R"]
    summary["C_minus_I"] = summary["C"] - summary["I"]
    summary.to_csv(out / "C_summary.csv", index_label="metric")

    # ---- settlement and per member ----
    tab = settlement(com, C)
    rows = []
    for d in com.members:
        cid = d.consumer_id
        rows.append({
            "consumer_id": cid, "archetype": d.archetype, "slope_class": d.slope_class,
            "asset_class": ("PV+BAT" if d.has_pv and d.has_battery else "PV" if d.has_pv
                            else "BAT" if d.has_battery else "none"),
            "cost_R_DKK": R[cid].objective, "cost_I_DKK": I[cid].objective, "cost_C_DKK": tab.loc[cid, "cost_C_DKK"],
            "benefit_vs_R_DKK": R[cid].objective - tab.loc[cid, "cost_C_DKK"],
            "benefit_vs_I_DKK": I[cid].objective - tab.loc[cid, "cost_C_DKK"],
            "payment_DKK": tab.loc[cid, "payment_DKK"], "pv_DKK": tab.loc[cid, "pv_DKK"],
            "disutility_DKK": tab.loc[cid, "disutility_DKK"],
            "net_import_kWh": C.members[cid]["net"].sum(),
            "shifted_C_kWh": C.members[cid]["deviation"].abs().sum() / 2,
            "shifted_R_kWh": R[cid].hourly["deviation"].abs().sum() / 2,
        })
    members = pd.DataFrame(rows).set_index("consumer_id")
    # Rent redistribution (our proposal): the manager keeps no profit. Each member who lost against R is made whole
    # first, the rest of the rent is split equally. Possible when the rent covers all losses (checked in C_manager.csv).
    rent = tab.attrs["manager"]["budget_DKK"]
    loss = (-members["benefit_vs_R_DKK"]).clip(lower=0)
    rest = rent - loss.sum()
    members["rebate_DKK"] = loss + rest / len(members)
    members["benefit_after_rebate_DKK"] = members["benefit_vs_R_DKK"] + members["rebate_DKK"]
    tab.attrs["manager"]["losses_to_compensate_DKK"] = float(loss.sum())
    tab.attrs["manager"]["rent_left_after_make_whole_DKK"] = float(rest)
    members["benefit_vs_R_pct"] = 100 * members["benefit_vs_R_DKK"] / members["cost_R_DKK"]
    members.to_csv(out / "C_members.csv")
    cols = ["cost_R_DKK", "cost_I_DKK", "cost_C_DKK", "benefit_vs_R_DKK", "benefit_vs_I_DKK", "shifted_C_kWh"]
    pd.concat({"slope_class": members.groupby("slope_class")[cols].sum(),
               "asset_class": members.groupby("asset_class")[cols].sum()}).to_csv(out / "C_by_type.csv")
    C.grid.to_csv(out / "C_prices.csv", index_label="hour")
    pd.Series(tab.attrs["manager"]).to_csv(out / "C_manager.csv", header=["value"], index_label="item")
    pd.Series(C.checks).to_csv(out / "C_checks.csv", header=["value"], index_label="check")
    pd.Series(validation_C(com, solver)).to_csv(out / "C_validation.csv", header=["value_DKK"], index_label="check")

    # ---- figure 1: congestion and the internal price ----
    aggR, aggI = aggregate(R), aggregate(I)
    k = 1.0 - com.contract.tariff_discount_ratio
    colR, colI, colC = "#888888", "#1f4e79", "#c97b1a"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.0))
    a1.plot(aggR.index, aggR["import"], color=colR, lw=2, label="R: no contract")
    a1.plot(aggI.index, aggI["import"], color=colI, lw=2, label="I: individual contracts")
    a1.plot(C.grid.index, P, color=colC, lw=2.2, label="C: community contract")
    a1.axhline(cap.mean(), color="#c0392b", ls="--", lw=1.5, label=f"Cap {cap.mean():.1f} kW")
    a1.set_xlabel(f"Hour of day (day {com.day})"); a1.set_ylabel("Community import (kW)")
    a1.set_title(f"Exceedance: R {mR['exceedance_kWh']:.0f}, I {mI['exceedance_kWh']:.1f}, C {mC['exceedance_kWh']:.1f} kWh", fontsize=10)
    a1.legend(frameon=False, fontsize=8)
    a2.step(C.grid.index, C.grid["price"] + k * com.import_tariff, where="mid", color="#999999", ls=":", label="Day-ahead + discounted import tariff")
    a2.step(C.grid.index, C.grid["pi"], where="mid", color=colC, lw=2.2, label="Internal price $\\pi_t$")
    a2.step(C.grid.index, C.grid["price"], where="mid", color="#555555", lw=1.2, label="Day-ahead price")
    a2.set_xlabel(f"Hour of day (day {com.day})"); a2.set_ylabel("DKK/kWh")
    a2.set_title("Internal price = discounted import price + congestion rent when the cap binds", fontsize=9)
    a2.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "C_profiles.png", dpi=170); plt.close(fig)

    # ---- figure 2: who gains, against R and against I ----
    order = members.sort_values(["slope_class", "asset_class"]).index
    x = np.arange(len(order)); w = 0.27
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    ax.bar(x - w, members.loc[order, "benefit_vs_R_DKK"], w, color=colR, label="against R, internal price only")
    ax.bar(x, members.loc[order, "benefit_vs_I_DKK"], w, color=colI, label="against I (signing alone), internal price only")
    ax.bar(x + w, members.loc[order, "benefit_after_rebate_DKK"], w, color="#2a7f62", label="against R, after rent rebate")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=70, fontsize=8)
    ax.set_ylabel("Benefit of C (DKK/day)")
    ax.set_title(f"With the internal price alone {(members['benefit_vs_R_DKK'] > -1e-6).sum()} of {len(members)} members beat R; "
                 f"after returning the {tab.attrs['manager']['budget_DKK']:.0f} DKK congestion rent {(members['benefit_after_rebate_DKK'] > -1e-6).sum()} do", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "C_members.png", dpi=170); plt.close(fig)
    return {"summary": summary, "members": members, "manager": tab.attrs["manager"], "checks": C.checks,
            "solver": C.solver, "R": R, "I": I, "C": C}


def validation_C(com: CommunityData, solver: str = "auto") -> dict[str, float]:
    """Validation of model C against R and I (all numbers should be >= 0 or ~0 as stated).

    * own_flow_tariffs_minus_R_DKK ~ 0 (the PMO check): the community model (internal balance, one meter) with no cap, no
      discount, no penalty and the tariffs charged on every member's own gross flow is R. The energy price is linear, so
      netting it behind the meter changes nothing.
    * individual_meters_minus_R_DKK ~ 0: every member behind its own meter, full tariffs, no cap, in one model (bookkeeping check).
    * sum_I_minus_C_DKK >= 0: pooling is a relaxation of the individual contracts (the members' schedules in I are feasible
      in C with the same discounts, a pooled cap equal to the sum of the individual caps and no more penalty), so C never
      costs the community more than the sum of the individual contracts.
    * C_with_no_contract_terms_minus_R_DKK <= 0: with no discount, no cap and no penalty, only the netting of tariffs behind
      the meter is left, which can only save money.
    """
    R = solve_community_R(com.members)
    I = solve_community_I(com)
    sumR = sum(r.objective for r in R.values())
    sumI = sum(r.objective for r in I.values())
    lim = solve_community_C(com, meters="individual", solver=solver)
    C = solve_community_C(com, solver=solver)
    C0 = solve_community_C(com, beta=0.0, alpha=0.0, cap_scale=1e3, solver=solver)
    own = solve_community_C(com, beta=0.0, alpha=0.0, cap_scale=1e3, solver=solver, own_flow_tariffs=True)
    return {"own_flow_tariffs_minus_R_DKK": own.objective - sumR, "individual_meters_minus_R_DKK": lim.objective - sumR, "sum_I_minus_C_DKK": sumI - C.objective,
            "C_with_no_contract_terms_minus_R_DKK": C0.objective - sumR}
