"""Question 3: draw, validate and save the preference scenarios. Run: python -m src.q3_scenarios"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.scenario_generation import (QUANTITIES, BetaRule,_catalogue_by_id, day_data, draw_preference_scenarios,
                                     draw_scaling_scenarios, load_json, rules_to_spec, set_rule)

SEEDS = {"in_sample": 1, "in_sample_crossval": 2, "out_of_sample": 1000, "large_sample": 12345}
OUT = Path(__file__).resolve().parent.parent / "results" / "base" / "Q3_scenarios"
TOL = 1e-9

def load(day="d1"):
    app, com, rules = load_json("appliance_params.json"), load_json("params_community_base.json"), load_json("preference_uncertainty_params.json")
    app, _ = day_data(app, load_json("bus_params.json"), load_json("weekly_profiles.json"), day)  # profiles of the chosen day
    return app, com, rules

def independent_draws(com, rules, n, seed):
    """The same draws that draw_preference_scenarios makes internally (to check the indices)."""
    spec = rules_to_spec(rules, [m["consumer_id"] for m in com["consumers"]])
    for q in QUANTITIES:
        set_rule(spec, q, None, n_scenarios=n if q == "profile" else 1)
    return spec, draw_scaling_scenarios(spec, seed)

def check(app, com, rules, scen, n, seed):
    cat = _catalogue_by_id(app)
    ids = [m["consumer_id"] for m in com["consumers"]]
    spec, scaling = independent_draws(com, rules, n, seed)
    rule = spec["profile"][ids[0]]
    th = np.array([[s["theta"]["profile"][c] for c in ids] for s in scen])

    r = {"n": n, "seed": seed}
    # 1) bounds
    r["theta_min"], r["theta_max"] = th.min(), th.max()
    r["bounds_ok"] = bool(((th >= rule.lo - TOL) & (th <= rule.hi + TOL)).all())
    # 2) mean
    r["mean_sample"] = th.mean()
    r["mean_z"] = (th.mean() - rule.mean) / np.sqrt(rule.variance / th.size)
    # 3) daily minimum + 4) indices
    dmin = prof = idx = 0.0
    aligned = True
    for k, s in enumerate(scen):
        for c in ids:
            j = s["index"][("profile", c)]
            aligned &= (j == k)
            idx = max(idx, abs(scaling["profile"][c][j] - s["theta"]["profile"][c]))
            for aid, a in s["members"][c].items():
                if "load_id" not in a:
                    continue
                rho = np.array(a["reference_load_capacity_factor"])
                dmin = max(dmin, abs(a["min_total_energy_per_day_kWh"] - a["max_load_KWh_per_hour"] * rho.sum()))
                rho0 = np.array(cat[aid]["reference_load_capacity_factor"])
                prof = max(prof, np.abs(rho - rho0 ** s["theta"]["profile"][c]).max())
                assert 0 <= rho.min() and rho.max() <= 1
    r["daily_min_err_kWh"], r["profile_err"], r["index_err"] = dmin, prof, idx
    r["aligned_ok"] = bool(aligned)
    r["prob_sum"] = sum(s["probability"] for s in scen)
    again = draw_preference_scenarios(app, com, rules, n=n, seed=seed, quantities=("profile",))
    r["reproducible"] = all(a["theta"] == b["theta"] for a, b in zip(scen, again))
    return r

def save(name, scen, seed, n, ids):
    ser = [{"id": s["id"], "probability": s["probability"],
            "index": {f"{q}|{c}": k for (q, c), k in s["index"].items()},
            "theta": s["theta"],
            "members": {c: {aid: {"reference_load_capacity_factor": a["reference_load_capacity_factor"],
                                  "min_total_energy_per_day_kWh": a["min_total_energy_per_day_kWh"]}
                            for aid, a in apps.items() if "load_id" in a} for c, apps in s["members"].items()}}
           for s in scen]
    json.dump({"name": name, "seed": seed, "n": n, "scenarios": ser}, open(OUT / f"{name}.json", "w"), indent=1)
    pd.DataFrame([[s["theta"]["profile"][c] for c in ids] for s in scen], columns=ids).to_csv(OUT / f"{name}_theta.csv")

def main() -> pd.DataFrame:
    """Draw, check and save all scenario sets; returns the check table (one row per set)."""
    OUT.mkdir(parents=True, exist_ok=True)
    app, com, rules = load("d1")
    ids = [m["consumer_id"] for m in com["consumers"]]
    sets = {"IS_N5": (5, SEEDS["in_sample"]), "IS_N10": (10, SEEDS["in_sample"]),
            "IS_N10_cv": (10, SEEDS["in_sample_crossval"]), "OOS_N100": (100, SEEDS["out_of_sample"])}
    drawn, rows = {}, {}
    for name, (n, seed) in sets.items():
        drawn[name] = draw_preference_scenarios(app, com, rules, n=n, seed=seed, quantities=("profile",))
        rows[name] = check(app, com, rules, drawn[name], n, seed)
        save(name, drawn[name], seed, n, ids)
    rows["IS_N5"]["nested_in_N10"] = all(a["theta"] == b["theta"] for a, b in zip(drawn["IS_N5"], drawn["IS_N10"][:5]))

    # large sample: does the distribution itself have the right mean and variance?
    p = rules["quantities"]["profile"]
    rule = BetaRule(200_000, p["lo"], p["hi"], p["mean"], p["kappa"])
    x = rule.draw(np.random.default_rng(SEEDS["large_sample"]))
    big = {"mean_set": rule.mean, "mean_sample": x.mean(), "var_set": rule.variance, "var_sample": x.var(ddof=1)}

    df = pd.DataFrame(rows).T
    df.to_csv(OUT / "scenario_checks.csv")
    json.dump({"seeds": SEEDS, "sets": sets, "large_sample_check": big}, open(OUT / "seeds.json", "w"), indent=1, default=float)
    print(df.T.to_string()); print(big)
    return df


if __name__ == "__main__":
    main()