"""Question 3, D2: stress test of the deterministic baselines M^D on the out-of-sample scenarios (figure).

Three panels (community cost, budget, IR shortfall against C_ref), one row per baseline: every dot is one of the
100 out-of-sample scenarios, the box spans q05-q95 with the median, the diamond is the value at theta = 1 (what the
deterministic model promises). Reads the files written by  python -m src.q3_evaluate.

Run:  python -m src.q3_plot_d2              ->  results/base/Q3_baseline/D2_stress_test.pdf / .png
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")                     # file output only (the GUI backend crashes on some Windows setups)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

from src.q3_evaluate import OUT

SET = "OOS_N100"
# (file name, label in the figure, colour): reference categorical palette, slots 1-3 in fixed order
MODELS = [("ind_ir_ref", r"$\mathcal{DP}^{ind}$", "#2a78d6"),
          ("budget_balance", r"$\mathcal{DP}^{BB}$", "#eb6834"),
          ("budget_balance_relaxed_ir", r"$\mathcal{DP}^{soft}$", "#1baf7a")]
PANELS = [("community_cost_DKK", "tester_community_cost_DKK", "Community cost [DKK]"),
          ("budget_DKK", "tester_budget_DKK", "Budget: payments − meter bill [DKK]"),
          ("IR_shortfall_total_DKK", "tester_IR_shortfall_vs_Cref_DKK", r"Total IR shortfall vs $C^{ref}$ [DKK]")]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"


def main() -> None:
    data = {v: pd.read_csv(OUT / f"MD_{v}_on_{SET}.csv") for v, _, _ in MODELS}
    expected = {v: pd.read_csv(OUT / f"MD_{v}_validation.csv", index_col=0)["value"] for v, _, _ in MODELS}

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9), sharey=True)
    rng = np.random.default_rng(0)                                   # jitter only (deterministic figure)
    rows = np.arange(len(MODELS))[::-1]                              # first model on top

    for ax, (col, exp_col, title) in zip(axes, PANELS):
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        if col == "budget_DKK":
            ax.axvline(0, color=INK2, linewidth=0.8)                 # balanced budget
        for y, (v, label, colour) in zip(rows, MODELS):
            vals = data[v][col].to_numpy()
            q05, q50, q95 = np.quantile(vals, [0.05, 0.5, 0.95])
            ax.add_patch(plt.Rectangle((q05, y - 0.28), q95 - q05, 0.56, facecolor=colour, alpha=0.14,
                                       edgecolor=colour, linewidth=0.8))
            ax.plot([q50, q50], [y - 0.28, y + 0.28], color=colour, linewidth=2)
            ax.scatter(vals, y + rng.uniform(-0.2, 0.2, len(vals)), s=9, color=colour, alpha=0.55,
                       edgecolor=SURFACE, linewidth=0.3, zorder=3)
            ax.scatter([float(expected[v][exp_col])], [y], marker="D", s=42, color=INK, edgecolor=SURFACE,
                       linewidth=1, zorder=4)
        ax.set_xlabel(title)
    axes[0].set_yticks(rows, [m[1] for m in MODELS])
    for ax in axes:
        ax.tick_params(axis="y", length=0)
    axes[0].set_ylim(-0.6, len(MODELS) - 0.4)

    handles = [plt.Line2D([], [], marker="o", linestyle="", color=INK2, alpha=0.6, markersize=4,
                          label=f"one of {len(data[MODELS[0][0]])} out-of-sample scenarios"),
               plt.Rectangle((0, 0), 1, 1, facecolor=INK2, alpha=0.14, edgecolor=INK2, label="q05–q95, line = median"),
               plt.Line2D([], [], marker="D", linestyle="", color=INK, markersize=5,
                          label=r"value at $\theta = 1$ (deterministic promise)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"D2_stress_test.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved {OUT / 'D2_stress_test.pdf'} (+ .png)")


if __name__ == "__main__":
    main()
