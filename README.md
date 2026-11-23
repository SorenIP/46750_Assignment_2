# Question 2 - Pricing problem DP (Group 12, 46750 Assignment 2)

Uses the model of `../Question 1` (data loader, `CommunityModel`). Nothing in Question 1 is changed.

## Contents
| File | What it is |
|---|---|
| `report/Q2_formulation.tex` | On paper: member best response, KKT, manager's problem as MILP (KKT + strong duality + big-M), DP = C claim |
| `report/Q2_preview.tex/.pdf` | Compile from this folder: `pdflatex report/Q2_preview.tex` (twice) |
| `code/check_q2_formulas.py` | Numerical check of the formulas on the base case (needs `../Question 1`) |

Run the check: `python code/check_q2_formulas.py` -> KKT residual ~4e-16, dual objective gap ~1e-14, best response = cost in C to ~2e-14.

## Plan
1. Best response + KKT (done)  2. MILP reformulation (done)  3. Implement DP, check it equals C, compare prices with C duals, indifferent member-hours  4. Variants: budget balance, individual rationality, infeasibility fix  5. Metrics and hypotheses  6. Sensitivity study  7. Indifference discussion  8. Q2 section and recommendation

## Change log
- 2026-11-04: folder created; formulation and check added.
- 2026-11-08: formulation rewritten in the Assignment 1 notation (group 16 report); added relation to the price ladder of Assignment 1.
