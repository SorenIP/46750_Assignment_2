"""Figures for the Q2 sensitivity section. Reads queue_summary.csv; writes PNG + PDF."""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Run from "Question 2/code": python plot_q2_sweeps.py  (or give both paths)
src = sys.argv[1] if len(sys.argv) > 1 else "../DP_code_eleni/results/base/DP_queue/queue_summary.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "../report/figs/q2_sweeps"
raw = pd.read_csv(src); d = raw.set_index("job")
R = 398.241694
BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2c5b8a", "#d98a26", "#3a9a8a", "#222222", "#777777", "#e3e2de"

def C(job):  # community optimum from cap-only run
    return d.loc[job, "community_cost_DKK"] if job in d.index else np.nan

def row(prefix, name):
    """Collect one sweep point."""
    cap = d.loc[f"{prefix}_cap"]; bb = d.loc[f"{prefix}_bb"]; ind = d.loc[f"{prefix}_individual"]
    Cc = cap.community_cost_DKK
    r = dict(C=Cc, G=R - Cc)
    r["bb_extra"] = bb.community_cost_DKK - Cc
    r["bb_proven"] = (bb.mip_gap or 0) < 0.01
    r["pm_extra"] = ind.community_cost_DKK - Cc if ind.status != "infeasible" else np.nan
    r["pm_infeasible"] = ind.status == "infeasible"
    st = raw[raw.job == f"{prefix}_ir2"].set_index("stage")
    s1, s2 = st.loc[f"{prefix}_ir2_stage1"], st.loc[f"{prefix}_ir2_stage2"]
    r["sf"] = s2.total_IR_shortfall_DKK if pd.notna(s2.total_IR_shortfall_DKK) else s1.total_IR_shortfall_DKK
    r["sf_proven"] = (s1.status == "optimal") and r["sf"] < 0.01
    r["w_cap"] = cap.members_worse_off_vs_R; r["w_bb"] = bb.members_worse_off_vs_R
    r["w_ir"] = s1.members_worse_off_vs_R if r["sf_proven"] or pd.isna(s2.members_worse_off_vs_R) else s2.members_worse_off_vs_R
    r["cost_bb"] = bb.community_cost_DKK; r["cost_pm"] = ind.community_cost_DKK
    return r

def base():
    Cc = raw[raw.job == "proof_two_stage_uniform"].iloc[0].C_cost_DKK
    bbc = 372.184; 
    return dict(C=Cc, G=R - Cc, bb_extra=0.02, bb_proven=True, pm_extra=372.165823 - Cc, pm_infeasible=False,
                sf=35.89, sf_proven=False, w_cap=13, w_bb=6, w_ir=6, cost_bb=372.184, cost_pm=372.165823)

kap = [0.8, 0.9, 1.1, 1.25, 1.5]; bet = [0, 0.5, 1]
kr = {k: row(f"sweep_cap_scale{k}", k) for k in kap}; br = {b: row(f"sweep_beta{b:g}", b) for b in bet}
# beta=0.5 IR: use the rerun (stage 2 found a solution)
st = raw[raw.job == "sweep_beta0.5_ir2_rerun"].set_index("stage")
br[0.5]["sf"] = st.loc["sweep_beta0.5_ir2_rerun_stage2"].total_IR_shortfall_DKK; br[0.5]["w_ir"] = st.loc["sweep_beta0.5_ir2_rerun_stage2"].members_worse_off_vs_R
b0 = base(); kr[1.0] = b0; br[0.75] = b0
K = sorted(kr); B = sorted(br)


CAP0 = 9.39  # base community cap in kW (Question 1 axis)
plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#444444", "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.8})
fig, ax = plt.subplots(3, 2, figsize=(7.2, 7.0), sharex="col")
def fm(a, xs, ys, proven, color, label=None):
    a.plot(xs, ys, "-", color=color, lw=1.3, zorder=1)
    for x, y, pr in zip(xs, ys, proven):
        a.plot(x, y, "o", ms=4.5, color=color if pr else "white", mec=color, mew=1.2, zorder=2)
    a.plot([], [], "-o", color=color, ms=4.5, lw=1.3, label=label)
for col, (xs, rr, xl, base_x) in enumerate([(K, kr, "Cap of the community (kW)", CAP0), (B, br, r"Discount $\beta$", 0.75)]):
    X = [x * CAP0 for x in xs] if col == 0 else xs
    a = ax[0, col]
    a.axhline(R, color="#888888", ls="--", lw=1.1, label=r"$\mathcal{R}$")
    fm(a, X, [rr[x]["C"] for x in xs], [True]*len(xs), ORANGE, r"$\mathcal{C}$, cap only")
    fm(a, X, [rr[x]["cost_bb"] for x in xs], [rr[x]["bb_proven"] for x in xs], BLUE, "+ budget balance")
    ok = [(X[i], rr[x]["cost_pm"]) for i, x in enumerate(xs) if not rr[x]["pm_infeasible"]]
    a.plot(*zip(*ok), "-s", color=AQUA, lw=1.3, ms=4, label="+ IR, price per member", zorder=3)
    for i, x in enumerate(xs):
        if rr[x]["pm_infeasible"]: a.plot(X[i], rr[x]["C"], "x", color=AQUA, ms=7, mew=1.6, zorder=4)
    a.set_ylabel("Community cost (DKK/day)")
    a = ax[1, col]
    for key, color, lab in [("w_cap", ORANGE, "cap only"), ("w_bb", BLUE, "+ budget balance"), ("w_ir", AQUA, "+ relaxed IR, one price")]:
        fm(a, X, [rr[x][key] for x in xs], [True]*len(xs) if key != "w_ir" else [rr[x]["sf_proven"] for x in xs], color, lab)
    a.set_ylabel("Members worse off than in R"); a.set_ylim(-0.5, 16.5)
    a = ax[2, col]
    fm(a, X, [rr[x]["sf"] for x in xs], [rr[x]["sf_proven"] for x in xs], BLUE)
    a.set_ylabel("IR shortfall, one price (DKK/day)"); a.set_xlabel(xl)
    a.set_ylim(-8, 215 if col == 0 else 165)
    for r_ in range(3):
        ax[r_, col].axvline(base_x, color="#aaaaaa", lw=0.8, ls=":")
    if col == 0: ax[2, col].set_xticks([x * CAP0 for x in xs]); ax[2, col].set_xticklabels([f"{x*CAP0:.1f}" for x in xs])
ax[2, 1].annotate("no improvement\nfound", (1, 146), xytext=(0.52, 128), fontsize=7.5, color=MUTED, arrowprops=dict(arrowstyle="-", color=MUTED))
ax[0, 0].legend(frameon=False, fontsize=7, loc="upper right", ncol=1)
ax[1, 0].legend(frameon=False, fontsize=7, loc="upper right")
fig.tight_layout()
fig.savefig(out + ".png", dpi=220); fig.savefig(out + ".pdf")
print("ok")
