# Assignment 2 - Question 1 (46750, Fall 2026)

Working folder for Question 1. Everything needed to run it is in here (code, data, settings).
Built on the official starter package: `data/` and `src/data_loader.py`, `src/scenario_generation.py` are unchanged copies.

## Run

```bash
pip install -r requirements.txt     # or: conda env create -f environment.yaml
python main.py --R                  # model R, base community, day d1
python main.py --I                  # model I (individual contracts), compared with R, base community, day d1
python main.py --models R I C      # unified model (one class, switches): solve named configurations and compare
python main.py --check-unified      # unified model reproduces the separate R, I, C modules (all differences ~0)
python main.py --sweep              # sensitivity study (discount, cap scale, penalty) for R, I and C, ~10 s
python main.py --C                  # model C (community contract), compared with R and I, base community, day d1
python main.py --R --day d3         # another day (prices, PV, reference profiles and cap of that day)
```

R, I and C are independent: `--I` and `--C` re-solve R (and I) internally only to compare, and never write to the other models' result folders.

Output goes to `results/<community>/R/` (see below). Model R is 16 small LPs: the restricted Gurobi licence that ships
with `gurobipy` is enough. Model C (all 16 members in one model) is above the size limit of the restricted licence (2000 variables): `src/lp_solve.py`
then solves it with HiGHS (installed with scipy) and returns the same duals, so no extra licence is needed. With the academic
licence (`grbgetkey`) Gurobi is used.

## Contents

| Path | What |
|---|---|
| `main.py` | Entry point. `--R` runs model R (check against the contract file, validation residuals). `--I` runs model I against R. `--C` runs model C against R and I. |
| `src/model.py` | **Ours.** `MemberModel` (one LP per member, can also be added to a shared Gurobi model), `ContractTerms` (switches: discount, cap, penalty), `solve_community_R`, `aggregate`. |
| `src/experiments.py` | **Ours.** `run_R`: summary, per-member benchmark, figure. |
| `src/experiments_I.py`, `experiments_C.py`, `experiments_sweep.py` | First version, kept as the reference for the regression test and for the result files in `results/base/{R,I,C,sweep}/`. The unified file gives the same numbers. |
| (details) `src/experiments_I.py` | **Ours.** Model I only: `solve_community_I`, `community_metrics`, `run_I`, `limit_case_check`. Uses `MemberModel` of `src/model.py`; changes nothing in R. |
| `src/community_model.py` | **Ours, the one to use from now on.** `CommunityModel(com, Config)`: R, I and C as one parametrized class. Switches: `meter` (member / community), `beta`, `alpha`, `cap` (none / individual / community), `cap_scale`, `tariff_base` (meter / own_flow). Named configurations in `CONFIGS`; a variation is `CONFIGS['C'].with_(beta=0.5)`. Returns `Results` (metrics, benefits, manager budget, rebate, duals, checks). Also `run`, `compare`, `member_table`, `validate`. |
| `src/check_unified.py` | **Ours.** Regression test of `community_model.py` against the separate modules below (R, I, C, and variations: all differences ~0). |
| `src/lp_solve.py` | **Ours.** `solve_lp`: Gurobi, with automatic HiGHS fallback if the model is too large; returns objective, values and duals (same sign convention as Gurobi `Pi`). |
| `src/experiments_C.py` | **Ours.** Model C only: `build_community_C`, `solve_community_C`, `settlement`, `run_C`, `validation_C`. Uses `MemberModel` of `src/model.py`; changes nothing in R or I. |
| `src/validation.py` | **Ours.** `optimal_face_range` (range of the exceedance over all optimal schedules), `solve_with_highs` (HiGHS fallback for models above the restricted Gurobi licence). |
| `src/data_loader.py`, `src/scenario_generation.py`, `data/` | Starter package, unchanged. |
| `report/metrics_hypotheses.tex` | Table of all metrics (view, definition, unit, reference) and the hypotheses H1-H4 of the sensitivity study, **written before the sweep**. |
| `report/Q1_recommendations.tex` | Settlement by consumer type (table), recommendations for DSO, manager and members, limits of the study. |
| `report/sweep_section.tex` | Results of the sensitivity study against H1-H4 (verdict for each). |
| `src/experiments_sweep.py` | **Ours.** `run_sweep`, `evaluate`, `hypothesis_checks`: one-at-a-time sweep of discount, cap scale and penalty for R, I and C. |
| `report/R_section.tex`, `I_section.tex`, `C_section.tex` | Report drafts for models R, I and C: nomenclature rows, formulation (I: only what changes), problem class and duality, results. `R_preview.tex/.pdf`: both sections compiled (from this folder, twice: `pdflatex report/R_preview.tex`). |

## Outputs of `python main.py --R` (in `results/base/R/`)

| File | Content |
|---|---|
| `R_summary.csv` | Feeder congestion (import, peak, hours above cap, exceedance and its range over optimal schedules) and cost components, with the reference values of the contract file. |
| `R_members.csv` | Per member: total cost (the benchmark of individual rationality), cost components, shifted energy, battery throughput, marginal cost of the daily minimum energy. |
| `R_by_slope_class.csv` | Shifted energy and cost by slope class. |
| `R_aggregate_hourly.csv`, `R_<member>_hourly.csv` | Hourly aggregate and per-member primal values and duals. |
| `R_checks.csv` | Largest validation residual over members. |
| `R_import_vs_cap.png` | Figure: community import against the cap, with the day-ahead price. |

## Outputs of `python main.py --I` (in `results/base/I/`)

| File | Content |
|---|---|
| `I_summary.csv` | R and I side by side: import, peak, hours above cap, exceedance, cost and its components. |
| `I_members.csv` | Per member: cost in R and in I, benefit B_i = C_R - C_I, penalty paid, exceedance, shifted energy. |
| `I_by_type.csv` | Benefit by slope class and by asset class. |
| `I_aggregate_hourly.csv`, `I_<member>_hourly.csv`, `I_checks.csv` | Hourly aggregate, per-member values and duals, validation residuals. |
| `I_vs_R.png` | Figure: community import in R and I against the cap, and benefit of every member. |

## Outputs of `python main.py --C` (in `results/base/C/`)

| File | Content |
|---|---|
| `C_summary.csv` | R, I and C side by side: import, peak, hours above cap, exceedance, cost and its components. |
| `C_members.csv` | Per member: cost in R, I and C, benefit against R and against I, payment, rebate, benefit after rebate, shifted energy. |
| `C_by_type.csv` | Costs and benefits by slope class and by asset class. |
| `C_prices.csv` | Per hour: day-ahead price, community import/export, exceedance, cap, internal price pi_t, congestion rent mu_t. |
| `C_manager.csv` | Settlement: payments collected, cost at the connection point, budget (= congestion rent), losses to compensate. |
| `C_checks.csv`, `C_validation.csv` | Validation residuals; checks of C against R and I (own-flow limit case, individual meters, sum of I minus C, tariff netting). |
| `C_profiles.png`, `C_members.png` | Figures: import in R/I/C and internal price; benefit of every member. |

## Model R (no contract)

Each member alone, full tariffs, no cap, no penalty. Member i minimizes

    sum_t [ (p_t + tau_imp) q_imp_t - (p_t - tau_exp) q_exp_t + c_PV g_t + c_L (d+_t + d-_t) ]

subject to: energy balance, `l_t - l_ref_t = d+_t - d-_t`, hourly load bounds (reference +- 1 kWh, within 0-3),
daily minimum energy (= daily reference energy), PV limit, and, if present, battery dynamics and limits.
It is a linear program, so strong duality holds. The full formulation is in `report/R_section.tex`.

**Modelling choices**
- Battery: ends the day at its starting charge (50 %).
- No binaries for import/export: buying costs `p + tau_imp`, selling pays `p - tau_exp`, so importing and exporting in the same hour never pays (checked: overlap 0).
- The cost includes the PV production cost, so the total matches the reference cost of the contract file.
- Contract switches (used from model I on): the discounted tariff `(1 - beta) tau` applies to all metered import and export, and the penalty `alpha` is charged on top of it on the exceedance. **To be confirmed.**

## Validation of model R (base community)

| Day | Import kWh | Peak kW | Cost DKK | Reference cost DKK |
|---|---|---|---|---|
| d1 | 173.4 | 27.6 | 398.24 | 398.24 |
| d2 | 393.0 | 37.1 | 803.40 | 803.40 |
| d3 | 258.9 | 23.4 | 681.76 | 681.76 |
| d4 | 321.4 | 35.0 | 681.24 | 681.24 |
| d5 | 302.8 | 27.6 | 633.39 | 633.39 |
| d6 | 375.1 | 42.8 | 691.49 | 691.49 |
| d7 | 393.0 | 37.2 | 655.25 | 655.25 |

Further checks (d1, largest over members): strong-duality gap 1e-14 DKK, energy balance 2e-16 kWh, import/export and
charge/discharge overlap 0, battery ends at its start, daily minimum met, load bounds respected.

The optimal schedule of a member is not unique (it is indifferent when a price is exactly c_L from its marginal value of
daily energy), so aggregate metrics depend on the solver. Exceedance on d1: ours 66.4 kWh, range over all optimal
schedules 66.3-66.7 kWh, contract file 66.5 kWh (inside the range). On d2 the range is 82.6-85.9 kWh (file: 83.9).
Small differences in daily import on d2, d5, d6 with identical cost have the same cause.

## Results of R on d1 (summary)

9 of 24 hours above the 9.39 kW cap (hours 4, 8, 9, 10-15), peak 27.6 kW at hour 11 (2.9 x cap), exceedance 66.4 kWh
(max 18.2 kW), 87 % of the daily import in the congested hours. Only 7.3 kWh are shifted without a contract (7.19 kWh by
the c_L = 0.25 members, 0.06 kWh by c_L = 0.5, none above), disutility 3.65 DKK of 398.24 DKK. Member costs range
from 22.7 DKK (PV + battery) to 27.1 DKK (no assets).

## Model I (individual contracts)

Each member alone with its own contract: discount beta = 0.75 on the metered tariffs, penalty alpha = 10 DKK/kWh on import above
its own cap kappa_i x cap_t (shares from the contract file). Same member LP as in R, with the contract switches of
`ContractTerms` on; formulation in `report/I_section.tex`. Convention: discounted tariff on all metered import, penalty on top.
Tested: charging the full tariff above the cap instead (same as alpha = 10.375) gives identical schedules and changes the
community cost by 1.6 DKK, so the conclusions do not depend on it.

Validation: I with no discount, no penalty and no cap equals R (largest member cost difference 0 DKK); the individual caps sum to the
community cap (gap 1e-5 kW, because the shares in the file are rounded; hours above the cap are therefore counted with a 1e-3 kW
tolerance in I); same member-level residuals as R (all ~0).

Result on d1 (I against R): exceedance 66.4 -> 4.3 kWh, peak 27.6 -> 11.1 kW, hours above cap 9 -> 3; community cost +82.4 DKK
(discount saves 71.5 DKK of tariffs, disutility +103.5, penalty +42.9, energy and PV +7.6); only 3 of 16 members are better off
(C1_B, C2_B, C1_PVB), the loss grows with the disutility slope (C4: -23.4 DKK).

## Model C (community contract, central dispatch)

One meter, one contract, one LP for all 16 members: minimize community cost = energy + discounted tariffs and penalty at the
meter + all PV costs and disutilities, subject to every member's constraints and the internal balance
`sum_i (q_imp_i - q_exp_i) = P_imp - P_exp`. The dual of this balance is the internal price pi_t. Members pay `pi_t` on their
net consumption (dual pricing); the manager's budget equals the congestion rent. Formulation in `report/C_section.tex`.
We added a rent rebate (our proposal): losers against R are made whole, the rest is split equally.

Validation: with no cap, no discount, no penalty and the tariffs on every member's own gross flow (`own_flow_tariffs=True`) the community model equals R (3e-13 DKK on d1; 1e-13 on d2, d5); also with one meter per member in one model (3e-13); saved in `C_validation.csv`; sum of I costs - C cost = 108.5 >= 0
(pooling is a relaxation of I); with no contract terms C is 1.09 DKK below R (tariff netting); strong-duality gap 2e-13 DKK; internal
price = day-ahead + discounted tariff + cap rent (error 0). Duals are almost unique: simplex and interior point differ in one hour by 0.03.

Result on d1: exceedance 66.4 (R) / 4.3 (I) -> 0 kWh, peak 9.39 kW, cap reached in 18 hours; community cost 398.2 (R) / 480.7 (I) -> 372.2 DKK;
shifted energy 55.6 kWh (R: 7.3), disutility 47.4 DKK. Internal price up to 5.41 DKK/kWh (hour 11), congestion rent 163.8 DKK. With the
internal price alone only 3 of 16 members beat R; members' losses sum to 149.9 DKK < 163.8 rent, so after the rebate all 16 gain (min 0.87 DKK).

## Unified model (`src/community_model.py`)

Following the tip of the assignment: R, I and C are one class with switches, not copies. A configuration is a `Config`:

| Config | meter | cap | discount, penalty | tariffs |
|---|---|---|---|---|
| `R` | member (own meter) | none | 0, 0 | on own flow |
| `I` | member | individual (kappa_i x cap) | contract file | on own flow |
| `C` | community (one LP, internal balance) | community | contract file | on community meter |
| `C_own_flow` | community | none | 0, 0 | on each member's own flow (= R, validation) |
| `C_no_terms`, `I_no_terms` | community / member | none / individual | 0, 0 | validation cases |

```python
from src.data_loader import load_community
from src.community_model import CommunityModel, CONFIGS, run, compare
com = load_community("base", day="d1")
res = run(com, ["R", "I", "C"])                                   # same community, same interface
print(compare(res))                                               # DSO and community metrics side by side
c = CommunityModel(com, CONFIGS["C"].with_(beta=0.5, cap_scale=0.9)).solve()   # a variation = a changed switch
c.metrics(); c.benefit_vs(res["R"]); c.manager(); c.rebate(res["R"])
```
Members and contract are shared by all configurations (`MemberModel` is unchanged), so DP and Question 3 can add a switch or a `Config`
without touching R, I and C. `python main.py --models R I C` writes `results/<community>/models/` (summary, members, checks, hourly grid per model).
`python main.py --check-unified` compares with the first version: all differences are 0 up to 6e-14.
R and I are solved member by member (they do not interact), C as one LP; the user does not see the difference.

## Outputs of `python main.py --sweep` (in `results/base/sweep/`)

`sweep.csv` (one row per model and parameter value: exceedance, peak, hours above cap, DSO revenue, cost, benefit against R, members better off, rent and losses for C),
`sweep_checks.csv` (refined thresholds by bisection and the test of every hypothesis), `sweep_members.csv` (every member in every run: cost, benefit against R and I, rebate), `sweep_by_type.csv` (benefit by slope class and asset class, before and after the rebate), `sweep_cost.png`, `sweep_settlement.png`.

Result against the hypotheses (d1): H1 confirmed (C breaks even with R at beta 0.48; I never, +59.8 DKK at beta 1). H2 partly wrong:
C holds the cap down to scale 0.68 (not 0.8), the gap I-C grows as the cap tightens (20 -> 131 DKK) but shrinks again at 0.6 when C also breaks,
and I has no exceedance from scale 2. H3 confirmed: C unchanged for penalty above 4.185 DKK/kWh (= largest congestion rent 4.19).
Rent covers all losses from discount 0.60 and from cap scale 0.90 (bisection). By type: the losses of C with the internal price sit with the members without PV (no assets -74 DKK, battery only -58) and with high slope; PV+battery members gain (+5.3); the discount hardly changes the split (2-4 DKK per group from beta 0.5 to 1). H4 half right: rent covers losses iff winners' gains <= saving (identity holds in all 21 runs); fails in 6 of 21 runs, 5 where C is dearer
than R and 1 (beta 0.5) where C is cheaper by only 1.6 DKK.

## Metrics and hypotheses (before the sweep)

`report/metrics_hypotheses.tex`: DSO view (peak, hours above cap, exceedance, revenue from the connection), community view (total cost,
flexibility used, benefit), manager (budget), member (benefit against R and against I). R is the reference for all; I is the second
reference for pooling. `C_summary.csv` now has a `dso_revenue_DKK` row (tariffs + penalty). Hypotheses to check with the sweep:
H1 discount: C beats R from beta ~0.5, I never beats R; H2 cap: C exceedance stays 0 down to scale ~0.8, gap I-C widens when tighter;
H3 penalty: C unchanged for alpha above ~4.2 DKK/kWh; H4 settlement: rebate makes all members rational iff the rent covers the losses.

## Change log

- 2026-10-28: Model R implemented and validated (`src/model.py`, `main.py --R`).
- 2026-11-03: Unified model `src/community_model.py` (R, I, C as one class with switches and named configurations), `src/check_unified.py`, `--models`, `--check-unified`. Old modules and result files unchanged.
- 2026-10-28: Settlement by type over the sweep (`sweep_members.csv`, `sweep_by_type.csv`), refined rent-feasibility thresholds, `report/Q1_recommendations.tex`.
- 2026-10-28: Sensitivity sweep (`src/experiments_sweep.py`, `main.py --sweep`), results against H1-H4 in `report/sweep_section.tex`.
- 2026-10-28: Metrics table, DSO revenue row in `C_summary.csv`, hypotheses H1-H4 written before the sweep (`report/metrics_hypotheses.tex`).
- 2026-10-28: Added the own-flow-tariff limit check (C = R), `C_validation.csv`.
- 2026-10-28: Model C added in its own module (`src/experiments_C.py`, `main.py --C`), solver wrapper `src/lp_solve.py`, `report/C_section.tex`. R and I outputs byte-identical before and after.
- 2026-10-28: Model I added in its own module (`src/experiments_I.py`, `main.py --I`), LaTeX draft `report/I_section.tex`. No R file changed (R outputs byte-identical before and after).
- 2026-10-28: Added extra validation (`src/validation.py`), results outputs (`src/experiments.py`), LaTeX draft of the R section (`report/`).

## Next

1. Group review of the Q1 drafts; move `experiments_sweep.py` to the unified class when the group agrees (same numbers expected).
2. Decide the open conventions (tariff above the cap, battery end state) and other rebate rules.
3. Questions 2 and 3 on top of `CommunityModel` (new switches or new `Config`s).
