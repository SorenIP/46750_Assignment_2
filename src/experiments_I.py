"""Model I: individual contracts (Question 1, contract type (a)).

Self-contained on purpose: it only *uses* the member LP of ``src/model.py`` (``MemberModel`` and ``ContractTerms``)
and the reference solution of model R; it changes nothing in the R code, so the R results stay reproducible.

What model I is
---------------
Every member signs a contract on its own meter and still schedules alone (no exchange between members):

    cap of member i in hour t :   cap_i,t = kappa_i * cap_t         (kappa_i from the contract file, sum = 1)
    member i pays             :   (1 - beta) * tau on its metered import and export,
                                  alpha DKK/kWh on every kWh imported above cap_i,t,
                                  plus the day-ahead energy price and its own PV and disutility costs.

Convention (to be confirmed, see README): the discounted tariff applies to *all* metered import, also above the cap,
and the penalty is charged on top of it on the exceedance.

Each member's problem is an LP, as in R, so duals exist and strong duality holds.

Metric definitions (units: kWh, kW, DKK; "benefit" > 0 means better off than in R)
-----------------------------------------------------------------------------------
* community import profile  P^imp_t = sum_i q_imp_i,t              (what the DSO sees)
* exceedance of the community cap   e_t = max(0, P^imp_t - cap_t)   (DSO point of view; same definition as in R)
* member exceedance  sum_t max(0, q_imp_i,t - cap_i,t)              (what members are penalized for)
* member cost C^I_i  (energy + tariffs + penalty + PV + disutility), reference cost C^R_i from model R
* benefit of member i       B_i  = C^R_i - C^I_i
* community benefit         B    = sum_i B_i                        (penalties and discounts are transfers with the DSO)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data_loader import CommunityData
from .model import ContractTerms, MemberModel, MemberResults, aggregate, solve_community_R


def solve_community_I(com: CommunityData, beta: float | None = None, alpha: float | None = None,
                      cap_scale: float = 1.0) -> dict[str, MemberResults]:
    """Model I: every member solves its own LP with its own contract.

    Defaults come from the contract file (``com.contract``). ``cap_scale`` multiplies every individual cap
    (cap tightness, 1.0 = base case). Returns {consumer_id: MemberResults}.
    """
    ct = com.contract
    beta = ct.tariff_discount_ratio if beta is None else beta
    alpha = ct.penalty_DKK_per_kWh if alpha is None else alpha
    out = {}
    for d in com.members:
        cap_i = cap_scale * np.asarray(ct.individual_capacity_limit_kW[d.consumer_id], dtype=float)
        out[d.consumer_id] = MemberModel(d, ContractTerms(beta=beta, cap=cap_i, alpha=alpha)).build().solve()
    return out


def community_metrics(com: CommunityData, res: dict[str, MemberResults], cap_scale: float = 1.0,
                      tol_kW: float = 1e-3) -> dict[str, float]:
    """Congestion seen by the DSO and total cost of a set of member solutions (R or I), see module docstring.

    ``tol_kW``: an hour counts as "above the cap" only if the import exceeds the cap by more than this. Needed
    because the shares kappa_i of the contract file are rounded (they sum to 1.000001), so the individual caps
    sum to the community cap + 1e-5 kW, and members that sit exactly at their caps would otherwise be counted."""
    cap = cap_scale * np.asarray(com.contract.capacity_limit_kW, dtype=float)
    P = aggregate(res)["import"].values
    exc = np.maximum(P - cap, 0.0)
    comp = pd.DataFrame({cid: r.cost_breakdown for cid, r in res.items()}).T
    return {
        "daily_import_kWh": P.sum(), "peak_import_kW": P.max(), "hours_above_cap": int((exc > tol_kW).sum()),
        "exceedance_kWh": exc.sum(), "max_exceedance_kW": exc.max(),
        "total_cost_DKK": comp.sum().sum(), **{f"cost_{k}_DKK": comp[k].sum() for k in comp.columns},
    }


def run_I(com: CommunityData, out: Path) -> dict:
    """Solve R and I on the same community, compare them, save tables and the figure to ``out``.

    Files: ``I_summary.csv`` (R and I side by side), ``I_members.csv`` (per-member cost, benefit against R,
    exceedance, shifted energy), ``I_by_type.csv`` (benefit by slope class and by asset class),
    ``I_aggregate_hourly.csv``, ``I_<member>_hourly.csv``, ``I_checks.csv``, ``I_vs_R.png``.
    """
    out.mkdir(parents=True, exist_ok=True)
    R = solve_community_R(com.members)          # reference (unchanged R code)
    I = solve_community_I(com)
    cap = np.asarray(com.contract.capacity_limit_kW, dtype=float)

    # ---- community level (DSO and community point of view) ----
    mR, mI = community_metrics(com, R), community_metrics(com, I)
    summary = pd.DataFrame({"R": mR, "I": mI})
    summary["I_minus_R"] = summary["I"] - summary["R"]
    summary.to_csv(out / "I_summary.csv", index_label="metric")

    # ---- per member ----
    ic = com.contract.individual_capacity_limit_kW
    rows = []
    for d in com.members:
        cid = d.consumer_id
        hR, hI = R[cid].hourly, I[cid].hourly
        rows.append({
            "consumer_id": cid, "archetype": d.archetype, "slope_class": d.slope_class,
            "asset_class": ("PV+BAT" if d.has_pv and d.has_battery else "PV" if d.has_pv
                            else "BAT" if d.has_battery else "none"),
            "cap_share_kappa": com.contract.share[cid],
            "cost_R_DKK": R[cid].objective, "cost_I_DKK": I[cid].objective,
            "benefit_DKK": R[cid].objective - I[cid].objective,
            "benefit_pct_of_R": 100 * (R[cid].objective - I[cid].objective) / R[cid].objective,
            "penalty_paid_DKK": I[cid].cost_breakdown["penalty"],
            "tariffs_R_DKK": R[cid].cost_breakdown["tariffs"], "tariffs_I_DKK": I[cid].cost_breakdown["tariffs"],
            "exceedance_R_kWh": float(np.maximum(hR["import"].values - ic[cid], 0).sum()),  # R import vs the cap it would get
            "exceedance_I_kWh": float(I[cid].hourly["exceed"].sum()),
            "import_R_kWh": hR["import"].sum(), "import_I_kWh": hI["import"].sum(),
            "shifted_I_kWh": hI["deviation"].abs().sum() / 2,
            "shifted_R_kWh": hR["deviation"].abs().sum() / 2,
        })
    members = pd.DataFrame(rows).set_index("consumer_id")
    members.to_csv(out / "I_members.csv")
    by_type = pd.concat({
        "slope_class": members.groupby("slope_class")[["cost_R_DKK", "cost_I_DKK", "benefit_DKK", "penalty_paid_DKK", "shifted_I_kWh"]].sum(),
        "asset_class": members.groupby("asset_class")[["cost_R_DKK", "cost_I_DKK", "benefit_DKK", "penalty_paid_DKK", "shifted_I_kWh"]].sum(),
    })
    by_type.to_csv(out / "I_by_type.csv")

    aggI = aggregate(I)
    aggI.assign(cap=cap, exceedance=np.maximum(aggI["import"].values - cap, 0)).to_csv(
        out / "I_aggregate_hourly.csv", index_label="hour")
    for cid, r in I.items():
        r.hourly.to_csv(out / f"I_{cid}_hourly.csv", index_label="hour")
    worst = {}
    for r in I.values():
        for k, v in r.checks.items():
            worst[k] = max(worst.get(k, 0.0), v)
    pd.Series(worst).to_csv(out / "I_checks.csv", header=["max_over_members"], index_label="check")

    # ---- figure: does signing alone reduce the congestion, and who gains? ----
    aggR = aggregate(R)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1.15, 1]})
    a1.plot(aggR.index, aggR["import"], color="#888888", lw=2, label="R: no contract")
    a1.plot(aggI.index, aggI["import"], color="#1f4e79", lw=2, label="I: individual contracts")
    a1.axhline(cap.mean(), color="#c0392b", ls="--", lw=1.5, label=f"Cap {cap.mean():.1f} kW")
    a1.set_xlabel(f"Hour of day (day {com.day})"); a1.set_ylabel("Community import (kW)")
    a1.set_title(f"Exceedance {mR['exceedance_kWh']:.0f} kWh (R) -> {mI['exceedance_kWh']:.0f} kWh (I)", fontsize=10)
    a1.legend(frameon=False, fontsize=8)
    order = members.sort_values(["slope_class", "asset_class"]).index
    colors = ["#2a7f62" if b >= 0 else "#c0392b" for b in members.loc[order, "benefit_DKK"]]
    a2.bar(range(len(order)), members.loc[order, "benefit_DKK"], color=colors)
    a2.axhline(0, color="black", lw=0.8)
    a2.set_xticks(range(len(order))); a2.set_xticklabels(order, rotation=70, fontsize=7)
    a2.set_ylabel("Benefit of I against R (DKK/day)")
    a2.set_title(f"Members better off: {(members['benefit_DKK'] > 1e-6).sum()} of {len(members)}; "
                 f"community {members['benefit_DKK'].sum():+.1f} DKK", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "I_vs_R.png", dpi=170); plt.close(fig)

    return {"summary": summary, "members": members, "checks": worst}


def limit_case_check(com: CommunityData, tol: float = 1e-6) -> dict[str, float]:
    """Validation: I with no discount, no penalty and no cap is R. Returns the largest per-member cost difference (DKK)
    and the largest hourly import difference (kWh/h); both must be ~0."""
    R = solve_community_R(com.members)
    I0 = {d.consumer_id: MemberModel(d, ContractTerms(beta=0.0, cap=None, alpha=0.0)).build().solve() for d in com.members}
    dc = max(abs(R[c].objective - I0[c].objective) for c in R)
    # with a cap far above any import and alpha = 0 the cap must not matter either
    big = {d.consumer_id: MemberModel(d, ContractTerms(beta=0.0, cap=np.full(24, 1e3), alpha=0.0)).build().solve()
           for d in com.members}
    dc2 = max(abs(R[c].objective - big[c].objective) for c in R)
    return {"max_cost_diff_no_cap_DKK": dc, "max_cost_diff_slack_cap_DKK": dc2}
