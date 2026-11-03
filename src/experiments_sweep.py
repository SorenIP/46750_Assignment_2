"""Sensitivity study of Question 1: discount beta, cap tightness and penalty alpha, for R, I and C.

One parameter at a time, the other two at their base value (contract file). R does not depend on the contract, so it is
solved once and is the reference in every figure. Hypotheses H1-H4 are in ``report/metrics_hypotheses.tex`` (written before
this study was run). Uses the solvers of models I and C as they are; it changes nothing in them.

Output (``results/<community>/sweep/``): ``sweep.csv`` (one row per model and parameter value), ``sweep_checks.csv``
(refined break-even discount, refined penalty threshold, hypothesis tests), ``sweep_members.csv`` (every member in every
run), ``sweep_by_type.csv`` (benefit by slope class and by asset class, before and after the rebate), ``sweep_cost.png``, ``sweep_settlement.png``.
Columns of ``sweep.csv``: exceedance, peak, hours above cap, DSO revenue, total cost, benefit against R, members better off
than in R, worst member benefit; for C also the congestion rent, the members' losses (to be covered by the rent), whether
the rent covers them, the members better off after the rebate and the members better off than in I.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data_loader import CommunityData
from .experiments_C import settlement, solve_community_C
from .experiments_I import community_metrics, solve_community_I
from .model import solve_community_R

GRID = {
    "discount": [0.0, 0.25, 0.5, 0.75, 1.0],
    "cap_scale": [0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 3.0],
    "penalty": [0.0, 1.0, 2.0, 4.0, 6.0, 10.0, 20.0],
}
TOL = 1e-6


def _kwargs(com: CommunityData, param: str, value: float) -> dict:
    ct = com.contract
    kw = {"beta": ct.tariff_discount_ratio, "alpha": ct.penalty_DKK_per_kWh, "cap_scale": 1.0}
    kw[{"discount": "beta", "cap_scale": "cap_scale", "penalty": "alpha"}[param]] = value
    return kw


def _asset_class(d) -> str:
    return "PV+BAT" if d.has_pv and d.has_battery else "PV" if d.has_pv else "BAT" if d.has_battery else "none"


def evaluate(com: CommunityData, R: dict, beta: float, alpha: float, cap_scale: float):
    """Solve I and C for one set of contract terms. Returns (rows, member_rows): one row of metrics for I and for C, and
    one row per member and model with its cost and benefits (C: internal price only, and with the rent rebate)."""
    sumR = sum(r.objective for r in R.values())
    I = solve_community_I(com, beta=beta, alpha=alpha, cap_scale=cap_scale)
    C = solve_community_C(com, beta=beta, alpha=alpha, cap_scale=cap_scale)
    mI = community_metrics(com, I, cap_scale)
    cap = cap_scale * np.asarray(com.contract.capacity_limit_kW, dtype=float)
    P = C.grid["P_imp"].values
    exc = np.maximum(P - cap, 0.0)
    bI = pd.Series({c: R[c].objective - I[c].objective for c in R})
    rowI = {"model": "I", **{k: mI[k] for k in ("exceedance_kWh", "peak_import_kW", "hours_above_cap", "total_cost_DKK")},
            "dso_revenue_DKK": mI["cost_tariffs_DKK"] + mI["cost_penalty_DKK"],
            "benefit_vs_R_DKK": sumR - mI["total_cost_DKK"], "members_better_off_vs_R": int((bI > -TOL).sum()),
            "worst_member_benefit_DKK": bI.min()}
    tab = settlement(com, C, beta)
    bC = pd.Series({c: R[c].objective - tab.loc[c, "cost_C_DKK"] for c in R})
    loss = (-bC).clip(lower=0).sum()
    rent = tab.attrs["manager"]["budget_DKK"]
    shifted = sum(h["deviation"].abs().sum() / 2 for h in C.members.values())
    rowC = {"model": "C", "exceedance_kWh": exc.sum(), "peak_import_kW": P.max(), "hours_above_cap": int((exc > 1e-3).sum()),
            "total_cost_DKK": C.objective, "dso_revenue_DKK": C.cost["tariffs"] + C.cost["penalty"],
            "benefit_vs_R_DKK": sumR - C.objective, "members_better_off_vs_R": int((bC > -TOL).sum()),
            "worst_member_benefit_DKK": bC.min(), "rent_DKK": rent, "member_losses_DKK": loss,
            "member_gains_DKK": bC.clip(lower=0).sum(), "rent_covers_losses": bool(rent >= loss - TOL), "shifted_kWh": shifted,
            "members_better_off_vs_I": int((pd.Series({c: I[c].objective - tab.loc[c, "cost_C_DKK"] for c in R}) > -TOL).sum()),
            "members_better_off_after_rebate": int(((bC + (-bC).clip(lower=0) + max(rent - loss, 0) / len(bC)) > -TOL).sum()
                                                   if rent >= loss - TOL else (bC > -TOL).sum())}
    rebate = (-bC).clip(lower=0) + max(rent - loss, 0) / len(bC) if rent >= loss - TOL else bC * 0
    mem = []
    for d in com.members:
        c = d.consumer_id
        base = {"consumer_id": c, "slope_class": d.slope_class, "asset_class": _asset_class(d)}
        mem.append({**base, "model": "I", "cost_DKK": I[c].objective, "benefit_vs_R_DKK": bI[c], "rebate_DKK": 0.0})
        mem.append({**base, "model": "C", "cost_DKK": tab.loc[c, "cost_C_DKK"], "benefit_vs_R_DKK": bC[c],
                    "benefit_vs_I_DKK": I[c].objective - tab.loc[c, "cost_C_DKK"], "rebate_DKK": rebate[c]})
    return [rowI, rowC], mem


def run_sweep(com: CommunityData, out: Path) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    R = solve_community_R(com.members)
    sumR = sum(r.objective for r in R.values())
    mR = community_metrics(com, R)
    rows = [{"model": "R", "parameter": "all", "value": np.nan, **{k: mR[k] for k in ("exceedance_kWh", "peak_import_kW", "hours_above_cap", "total_cost_DKK")},
             "dso_revenue_DKK": mR["cost_tariffs_DKK"] + mR["cost_penalty_DKK"], "benefit_vs_R_DKK": 0.0}]
    mrows = []
    for param, values in GRID.items():
        for v in values:
            r2, m2 = evaluate(com, R, **_kwargs(com, param, v))
            rows += [{"parameter": param, "value": v, **r} for r in r2]
            mrows += [{"parameter": param, "value": v, **r} for r in m2]
    df = pd.DataFrame(rows)
    df.to_csv(out / "sweep.csv", index=False)
    mem = pd.DataFrame(mrows)
    mem["benefit_after_rebate_DKK"] = mem.benefit_vs_R_DKK + mem.rebate_DKK
    mem.to_csv(out / "sweep_members.csv", index=False)
    keys = ["parameter", "value", "model"]
    by_type = pd.concat({k: mem.groupby(keys + [k])[["benefit_vs_R_DKK", "benefit_after_rebate_DKK"]].sum().reset_index().rename(columns={k: "group"})
                         for k in ("slope_class", "asset_class")}, names=["split"]).reset_index(level=0)
    by_type.to_csv(out / "sweep_by_type.csv", index=False)
    checks = hypothesis_checks(com, R, sumR, df)
    pd.Series(checks).to_csv(out / "sweep_checks.csv", header=["value"], index_label="check")
    _figures(df, sumR, out)
    return df


def _crossing(x, y):
    """First x at which y changes sign from positive to <= 0 (linear interpolation), or nan."""
    for a, b, ya, yb in zip(x[:-1], x[1:], y[:-1], y[1:]):
        if ya > 0 >= yb:
            return a + (b - a) * ya / (ya - yb)
    return float("nan")


def _bisect(f, lo, hi, n=14):
    """Smallest x in [lo, hi] with f(x) True, assuming f(lo) False and f(hi) True."""
    for _ in range(n):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if f(mid) else (mid, hi)
    return hi


def hypothesis_checks(com: CommunityData, R: dict, sumR: float, df: pd.DataFrame) -> dict:
    """Numbers that decide H1-H4 (the grid alone is too coarse for thresholds, so they are refined by bisection)."""
    ct = com.contract
    d = df[df.parameter == "discount"]
    out = {}
    # H1: break-even discount (community cost = R). C is cheaper than R above it.
    for m in ("I", "C"):
        s = d[d.model == m].sort_values("value")
        out[f"H1_breakeven_discount_{m}"] = _crossing(list(s.value), list(-s.benefit_vs_R_DKK))
    out["H1_I_cost_minus_R_at_discount_1"] = float(d[(d.model == "I") & (d.value == 1.0)].total_cost_DKK.iloc[0] - sumR)
    # H2: tightest cap scale at which C still has no exceedance; gap I - C by scale
    c = df[(df.parameter == "cap_scale") & (df.model == "C")].sort_values("value")
    i = df[(df.parameter == "cap_scale") & (df.model == "I")].sort_values("value")
    ok = c[c.exceedance_kWh <= 1e-3].value
    out["H2_smallest_grid_cap_scale_with_zero_exceedance_C"] = float(ok.min()) if len(ok) else float("nan")
    out["H2_I_min_exceedance_kWh_over_scales_le_1"] = float(i[i.value <= 1.0].exceedance_kWh.min())
    out["H2_gap_I_minus_C_at_scale_0.6"] = float(i.iloc[0].total_cost_DKK - c.iloc[0].total_cost_DKK)
    out["H2_gap_I_minus_C_at_scale_1.0"] = float(i[i.value == 1.0].total_cost_DKK.iloc[0] - c[c.value == 1.0].total_cost_DKK.iloc[0])
    out["H2_gap_I_minus_C_at_scale_3.0"] = float(i.iloc[-1].total_cost_DKK - c.iloc[-1].total_cost_DKK)
    out["H2_C_cost_minus_R_at_scale_3.0"] = float(c.iloc[-1].total_cost_DKK - sumR)
    out["H2_members_better_off_vs_R_C_at_scale_3.0"] = int(c.iloc[-1].members_better_off_vs_R)
    if not np.isnan(out["H2_smallest_grid_cap_scale_with_zero_exceedance_C"]):
        lo_ok = out["H2_smallest_grid_cap_scale_with_zero_exceedance_C"]
        below = c[c.value < lo_ok].value
        if len(below):
            f = lambda s: solve_community_C(com, cap_scale=s).grid["P_imp"].sub(s * np.asarray(ct.capacity_limit_kW)).clip(lower=0).sum() <= 1e-3
            out["H2_refined_tightest_cap_scale_zero_exceedance_C"] = _bisect(f, float(below.max()), lo_ok)
        else:
            out["H2_refined_tightest_cap_scale_zero_exceedance_C"] = lo_ok
    # H3: penalty threshold above which C does not change
    p = df[(df.parameter == "penalty") & (df.model == "C")].sort_values("value")
    out["H3_C_cost_range_for_penalty_ge_6"] = float(p[p.value >= 6].total_cost_DKK.max() - p[p.value >= 6].total_cost_DKK.min())
    pos = p[p.exceedance_kWh > 1e-3].value
    out["H3_largest_grid_penalty_with_exceedance_C"] = float(pos.max()) if len(pos) else float("nan")
    if len(pos) and pos.max() < p.value.max():
        hi = float(p[p.value > pos.max()].value.min())
        g = lambda a: solve_community_C(com, alpha=a).grid["P_imp"].sub(np.asarray(ct.capacity_limit_kW)).clip(lower=0).sum() <= 1e-3
        out["H3_refined_penalty_threshold_C"] = _bisect(g, float(pos.max()), hi)
    ip = df[(df.parameter == "penalty") & (df.model == "I")].sort_values("value")
    out["H3_I_exceedance_at_penalty_4_and_20"] = f"{ip[ip.value == 4].exceedance_kWh.iloc[0]:.2f} / {ip[ip.value == 20].exceedance_kWh.iloc[0]:.2f}"
    out["H3_I_cost_slope_DKK_per_penalty_unit_4_to_20"] = float((ip[ip.value == 20].total_cost_DKK.iloc[0] - ip[ip.value == 4].total_cost_DKK.iloc[0]) / 16)
    # H4: rent covers losses  <=>  C cheaper than R (and gains <= saving)
    cc = df[(df.model == "C") & (df.parameter != "all")].copy()
    cc["rent_covers_losses"] = cc.rent_covers_losses.astype(bool)
    # identity behind H4: rent >= losses  <=>  gains of the winners <= saving of the community (S = C_R - C_C)
    out["H4_identity_holds_in_all_runs"] = bool(((cc.member_gains_DKK <= cc.benefit_vs_R_DKK + 1e-6) == cc.rent_covers_losses).all())
    out["H4_runs_where_rent_covers_losses"] = int(cc.rent_covers_losses.sum())
    out["H4_runs_total"] = int(len(cc))
    out["H4_runs_C_cheaper_than_R"] = int((cc.benefit_vs_R_DKK > 0).sum())
    out["H4_runs_C_cheaper_than_R_but_rent_short"] = int(((cc.benefit_vs_R_DKK > 0) & ~cc.rent_covers_losses).sum())
    out["H4_runs_rent_covers_but_C_not_cheaper"] = int((cc.rent_covers_losses & (cc.benefit_vs_R_DKK <= 0)).sum())
    # H4 refined: how low can the discount / how tight can the cap be while the rent still covers the losses?
    def covers(**kw):
        C = solve_community_C(com, **kw)
        t = settlement(com, C)
        loss = sum(max(0.0, t.loc[c, "cost_C_DKK"] - R[c].objective) for c in R)
        return t.attrs["manager"]["budget_DKK"] >= loss - TOL
    out["H4_refined_min_discount_rent_covers_losses"] = _bisect(lambda b: covers(beta=b), 0.5, 0.75)
    out["H4_refined_min_cap_scale_rent_covers_losses"] = _bisect(lambda s_: covers(cap_scale=s_), 0.8, 0.9)
    return out


def _figures(df: pd.DataFrame, sumR: float, out: Path) -> None:
    col = {"I": "#1f4e79", "C": "#c97b1a"}
    lab = {"discount": r"Discount $\beta$", "cap_scale": "Cap scale (1 = 9.39 kW)", "penalty": "Penalty (DKK/kWh)"}
    R = df[df.model == "R"].iloc[0]
    fig, ax = plt.subplots(2, 3, figsize=(11, 6.2), sharex="col")
    for j, par in enumerate(GRID):
        for m in ("I", "C"):
            s = df[(df.parameter == par) & (df.model == m)].sort_values("value")
            ax[0, j].plot(s.value, s.total_cost_DKK, "o-", color=col[m], label=f"{m}")
            ax[1, j].plot(s.value, s.exceedance_kWh, "o-", color=col[m], label=f"{m}")
        ax[0, j].axhline(sumR, color="#888888", ls="--", label="R")
        ax[1, j].axhline(R.exceedance_kWh, color="#888888", ls="--", label="R")
        ax[1, j].set_xlabel(lab[par])
        if par == "cap_scale":
            ax[0, j].set_xscale("log"); ax[1, j].set_xscale("log")
    ax[0, 0].set_ylabel("Community cost (DKK/day)"); ax[1, 0].set_ylabel("Exceedance (kWh/day)")
    ax[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "sweep_cost.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
    for j, par in enumerate(GRID):
        s = df[(df.parameter == par) & (df.model == "C")].sort_values("value")
        i = df[(df.parameter == par) & (df.model == "I")].sort_values("value")
        ax[j].plot(s.value, s.members_better_off_vs_R, "o-", color=col["C"], label="C, internal price only")
        ax[j].plot(s.value, s.members_better_off_after_rebate, "s--", color="#2a7f62", label="C, after rent rebate")
        ax[j].plot(i.value, i.members_better_off_vs_R, "o-", color=col["I"], label="I")
        ax[j].set_xlabel(lab[par]); ax[j].set_ylim(-0.5, 16.5)
        if par == "cap_scale":
            ax[j].set_xscale("log")
    ax[0].set_ylabel("Members better off than in R (of 16)"); ax[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(out / "sweep_settlement.png", dpi=170); plt.close(fig)
