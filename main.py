"""Entry point of the Assignment 2 starter package: load a community and summarize it.

    python main.py                                   # base case (16 members), day d1
    python main.py --community my_mix                # your own data/params_community_my_mix.json
    python main.py --day d4                          # another day of the week (and that day's cap)
    python main.py --members                         # also print every member's Assignment 1 summary
    python main.py --model-c                         # solve Model C and write results under results/
    python main.py --demo                            # also run the scenario-generation demonstration (Question 3)

Model C centrally dispatches the community under its shared contract. Other models (R, I, DP, ...) are not yet
implemented.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import load_community, list_communities
from src.model_c import run_model_c, save_model_c_result

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--community", default="base", help=f"base (shipped) or your own params_community_<name>.json; available: {list_communities()}")
    ap.add_argument("--day", default="d1", help="d1 (base case) ... d7")
    ap.add_argument("--members", action="store_true", help="print the Assignment 1 summary of every member")
    ap.add_argument("--model-c", action="store_true", help="solve centralized Model C and save its results")
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
    if args.model_c:
        result = run_model_c(com)
        output_dir = RESULTS_DIR / args.community / f"model_c_{args.day}"
        summary_path, dispatch_path = save_model_c_result(result, output_dir)
        print(f"Model C status: {result.status}")
        print(f"Model C welfare: {result.welfare_DKK:.2f} DKK")
        print(f"Community bill: {result.community_bill_DKK:.2f} DKK")
        print(f"Manager budget balance: {result.manager_budget_balance_DKK:.2f} DKK")
        print(f"Saved: {summary_path}")
        print(f"Saved: {dispatch_path}")
    if args.demo:
        import runpy
        runpy.run_module("src.scenario_generation", run_name="__main__")


if __name__ == "__main__":
    main()
