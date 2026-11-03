"""Experiments of Question 1. One function per experiment, each writes to ``results/<community>/<model>/``."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data_loader import CommunityData
from .model import aggregate, solve_community_R
from .validation import optimal_face_range


def run_R(com: CommunityData, out: Path, face: bool = True) -> dict:
    """Model R: members alone, full tariffs, no contract (the reference case).

    Writes to ``out``: per-member hourly CSVs, the aggregated profile, ``R_summary.csv`` (feeder congestion and
    cost), ``R_members.csv`` (per-member cost = the benchmark of individual rationality, shifted energy and the
    multiplier of the daily minimum) and ``R_import_vs_cap.png``. Returns the summary as a dict.
    """
    out.mkdir(parents=True, exist_ok=True)
    res = solve_community_R(com.members)
    agg = aggregate(res)
    cap = np.asarray(com.contract.capacity_limit_kW, dtype=float)
    P = agg["import"].values
    exc = np.maximum(P - cap, 0.0)
    over = exc > 1e-6
    ref = com.contract.reference_model_R

    # ---- feeder congestion and cost (community level) ----
    comp = pd.DataFrame({cid: r.cost_breakdown for cid, r in res.items()}).T
    summary = {
        "daily_import_kWh": P.sum(),
        "peak_import_kW": P.max(),
        "peak_hour": int(P.argmax()),
        "cap_kW": cap.mean(),
        "peak_over_cap": P.max() / cap.mean(),
        "hours_above_cap": int(over.sum()),
        "exceedance_kWh": exc.sum(),
        "max_exceedance_kW": exc.max(),
        "import_in_capped_hours_kWh": P[over].sum(),
        "total_cost_DKK": comp.sum().sum(),
        **{f"cost_{k}_DKK": comp[k].sum() for k in comp.columns},
        "ref_daily_import_kWh": ref["daily_import_kWh"], "ref_peak_import_kW": ref["peak_import_kW"],
        "ref_cost_DKK": ref["cost_DKK"], "ref_exceedance_kWh": ref["exceedance_kWh"],
    }
    if face:
        r = optimal_face_range(com, res)
        summary.update(exceedance_min_over_optima_kWh=r["exceedance_min_kWh"],
                       exceedance_max_over_optima_kWh=r["exceedance_max_kWh"])
    pd.Series(summary).to_csv(out / "R_summary.csv", header=["value"], index_label="metric")

    # ---- per member: benchmark of individual rationality, flexibility used ----
    rows = []
    for d in com.members:
        r = res[d.consumer_id]
        h = r.hourly
        rows.append({
            "consumer_id": d.consumer_id, "archetype": d.archetype, "slope_class": d.slope_class,
            "cost_DKK": r.objective, **{f"cost_{k}_DKK": v for k, v in r.cost_breakdown.items()},
            "net_import_kWh": h["net_import"].sum(), "import_kWh": h["import"].sum(),
            "shifted_kWh": h["deviation"].abs().sum() / 2,       # energy moved between hours (|dev| counts twice)
            "battery_throughput_kWh": h["discharge"].sum() if "discharge" in h else 0.0,
            "value_of_daily_energy_DKK_per_kWh": -r.duals["min_energy"],   # internal value of a kWh of daily need
        })
    members = pd.DataFrame(rows).set_index("consumer_id")
    members.to_csv(out / "R_members.csv")
    agg.assign(cap=cap, exceedance=exc).to_csv(out / "R_aggregate_hourly.csv", index_label="hour")
    for cid, r in res.items():
        r.hourly.to_csv(out / f"R_{cid}_hourly.csv", index_label="hour")
    by_class = members.groupby("slope_class")[["shifted_kWh", "cost_disutility_DKK", "cost_DKK"]].sum()
    by_class.to_csv(out / "R_by_slope_class.csv")

    # ---- figure: the answer is "where and how much is the congestion" ----
    price = com.energy_price
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    a1.plot(agg.index, P, color="#1f4e79", lw=2, label="Import without contract, model R")
    a1.axhline(cap.mean(), color="#c0392b", ls="--", lw=1.5, label=f"Cap {cap.mean():.1f} kW")
    a1.fill_between(agg.index, cap, P, where=over, color="#c0392b", alpha=0.25, interpolate=True, label="Congestion to reduce")
    a1.set_ylabel("Community import (kW)")
    a1.set_title(f"Without a contract the feeder is congested in {over.sum()} hours:\n"
                 f"{exc.sum():.0f} kWh above the cap, peak {P.max():.1f} kW ({P.max() / cap.mean():.1f} x cap)", fontsize=10)
    a1.legend(frameon=False, loc="upper left", fontsize=8)
    a2.step(agg.index, price, where="mid", color="#555555")
    a2.set_ylabel("Day-ahead\nprice (DKK/kWh)"); a2.set_xlabel("Hour of day (day %s)" % com.day)
    fig.tight_layout(); fig.savefig(out / "R_import_vs_cap.png", dpi=170); plt.close(fig)

    worst = {}
    for r in res.values():
        for k, v in r.checks.items():
            worst[k] = max(worst.get(k, 0.0), v)
    pd.Series(worst).to_csv(out / "R_checks.csv", header=["max_over_members"], index_label="check")
    summary["checks"] = worst
    return summary
