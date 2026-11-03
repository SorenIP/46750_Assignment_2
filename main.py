"""Entry point of the Assignment 2 starter package: load a community and summarize it.

    python main.py                                   # base case (16 members), day d1
    python main.py --community my_mix                # your own data/params_community_my_mix.json
    python main.py --day d4                          # another day of the week (and that day's cap)
    python main.py --members                         # also print every member's Assignment 1 summary
    python main.py --demo                            # also run the scenario-generation demonstration (Question 3)

No model is provided: build your Assignment 1 model class on the ``InputData`` of each member, then the community
models of the assignment (R, I, C, DP, ...) on top of it, one function per experiment, writing to ``results/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import load_community, list_communities
from src.experiments import run_R
from src.experiments_C import run_C, validation_C
from src.experiments_I import limit_case_check, run_I
from src.experiments_sweep import run_sweep
from src.community_model import CONFIGS, compare, member_table, run as run_models
from src.check_unified import check_unified

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--community", default="base", help=f"base (shipped) or your own params_community_<name>.json; available: {list_communities()}")
    ap.add_argument("--day", default="d1", help="d1 (base case) ... d7")
    ap.add_argument("--members", action="store_true", help="print the Assignment 1 summary of every member")
    ap.add_argument("--R", action="store_true", help="solve model R (no contract) and save results/<community>/R/")
    ap.add_argument("--I", action="store_true", help="solve model I (individual contracts) and compare with R, save results/<community>/I/")
    ap.add_argument("--C", action="store_true", help="solve model C (community contract), compare with R and I, save results/<community>/C/")
    ap.add_argument("--sweep", action="store_true", help="sensitivity study (discount, cap scale, penalty) for R, I and C, save results/<community>/sweep/")
    ap.add_argument("--models", nargs="+", metavar="NAME", help=f"unified model (src/community_model.py): solve these named configurations and compare them, e.g. --models R I C; available: {list(CONFIGS)}")
    ap.add_argument("--check-unified", action="store_true", help="regression test: the unified model reproduces the separate R, I and C modules")
    ap.add_argument("--demo", action="store_true", help="run the scenario-generation demonstration")
    args = ap.parse_args()

    com = load_community(args.community, day=args.day)
    print(com.summary(), "\n")
    if args.members:
        for m in com.members:
            print(m.summary(), "\n")
    print("Cap profile (kW):", " ".join(f"{x:.1f}" for x in com.contract.capacity_limit_kW))
    ic = com.contract.individual_capacity_limit_kW
    print("Individual caps in model I (kW), hour 0 / hour 11:",
          "  ".join(f"{cid} ({com.member(cid).archetype}) {ic[cid][0]:.2f}/{ic[cid][11]:.2f}" for cid in ("C1", "C1_PVB", "C4", "C4_PVB") if cid in com.member_ids))
    (RESULTS_DIR / args.community).mkdir(parents=True, exist_ok=True)
    if args.R:
        sm = run_R(com, RESULTS_DIR / args.community / "R")
        print(f"Model R, day {com.day}: import {sm['daily_import_kWh']:.1f} kWh (ref {sm['ref_daily_import_kWh']}), "
              f"peak {sm['peak_import_kW']:.1f} kW (ref {sm['ref_peak_import_kW']}), "
              f"cost {sm['total_cost_DKK']:.2f} DKK (ref {sm['ref_cost_DKK']}), "
              f"exceedance {sm['exceedance_kWh']:.1f} kWh (ref {sm['ref_exceedance_kWh']}; range over optima "
              f"{sm['exceedance_min_over_optima_kWh']:.1f}-{sm['exceedance_max_over_optima_kWh']:.1f}), "
              f"hours above cap {sm['hours_above_cap']}")
        print("Largest validation residual over members:", {k: float(f"{v:.1e}") for k, v in sm["checks"].items()})
    if args.I:
        import numpy as np
        ic = com.contract.individual_capacity_limit_kW
        gap = np.max(np.abs(sum(np.asarray(v) for v in ic.values()) - com.contract.capacity_limit_kW))
        print(f"Check: individual caps sum to the community cap (largest gap {gap:.1e} kW)")
        print("Check: I with no discount/penalty/cap equals R:", {k: float(f"{v:.1e}") for k, v in limit_case_check(com).items()})
        r = run_I(com, RESULTS_DIR / args.community / "I")
        print(r["summary"].round(2).to_string())
        print(r["members"][["archetype", "cost_R_DKK", "cost_I_DKK", "benefit_DKK", "penalty_paid_DKK"]].round(2).to_string())
    if args.C:
        print("Check: C vs R and I:", {k: float(f"{v:.2e}") for k, v in validation_C(com).items()})
        r = run_C(com, RESULTS_DIR / args.community / "C")
        print(f"Solver: {r['solver']}")
        print("Largest validation residual:", {k: float(f"{v:.1e}") for k, v in r["checks"].items()})
        print(r["summary"].round(2).to_string())
        print({k: round(v, 2) for k, v in r["manager"].items()})
        print(r["members"][["archetype", "cost_R_DKK", "cost_I_DKK", "cost_C_DKK", "benefit_vs_R_DKK", "benefit_vs_I_DKK"]].round(2).to_string())
    if args.sweep:
        import pandas as pd
        df = run_sweep(com, RESULTS_DIR / args.community / "sweep")
        pd.set_option("display.width", 250)
        print(df[["model", "parameter", "value", "exceedance_kWh", "total_cost_DKK", "dso_revenue_DKK", "benefit_vs_R_DKK", "members_better_off_vs_R"]].round(2).to_string())
        print(pd.read_csv(RESULTS_DIR / args.community / "sweep" / "sweep_checks.csv", index_col=0).to_string())
    if args.models:
        import pandas as pd
        pd.set_option("display.width", 250)
        res = run_models(com, args.models)
        ref = "R" if "R" in res else next(iter(res))
        out = RESULTS_DIR / args.community / "models"
        out.mkdir(parents=True, exist_ok=True)
        summ = compare(res, ref)
        summ.to_csv(out / "summary.csv", index_label="metric")
        mt = member_table(com, res, ref)
        mt.to_csv(out / "members.csv", index_label="consumer_id")
        pd.DataFrame({n: r.checks for n, r in res.items()}).to_csv(out / "checks.csv", index_label="check")
        for n, r in res.items():
            r.grid.to_csv(out / f"{n}_grid.csv", index_label="hour")
        print(summ.round(2).to_string())
        print(mt.filter(regex="^(archetype|cost_|benefit_)").round(2).to_string())
    if args.check_unified:
        print({k: float(f"{v:.1e}") for k, v in check_unified(com).items()})
    if args.demo:
        import runpy
        runpy.run_module("src.scenario_generation", run_name="__main__")


if __name__ == "__main__":
    main()
