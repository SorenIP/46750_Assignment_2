"""Regression test: ``src/community_model.py`` (one parametrized class) reproduces the first version (separate modules
``model.py`` solve_community_R, ``experiments_I.py``, ``experiments_C.py``). Run with ``python main.py --check-unified``.
Every number printed should be ~0 (differences in DKK, kWh, kW, DKK/kWh)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .community_model import CONFIGS, CommunityModel, validate
from .data_loader import CommunityData
from .experiments_C import settlement, solve_community_C
from .experiments_I import community_metrics, solve_community_I
from .model import aggregate, solve_community_R


def _common_metric_diff(new: dict, old: dict) -> float:
    return max(abs(float(new[k]) - float(old[k])) for k in old)


def check_unified(com: CommunityData) -> dict[str, float]:
    out = {}
    # ---- R and I: per-member cost and aggregate import profile
    for name, old in (("R", solve_community_R(com.members)), ("I", solve_community_I(com))):
        new = CommunityModel(com, name).solve()
        out[f"{name}_member_cost"] = max(abs(new.member_cost.loc[c, "cost_DKK"] - r.objective) for c, r in old.items())
        out[f"{name}_import_profile"] = float((new.grid["P_imp"] - aggregate(old)["import"]).abs().max())
        out[f"{name}_metrics"] = _common_metric_diff(new.metrics(), community_metrics(com, old))
    # ---- C (base) and two variations: cost, grid, internal price, rent, member costs
    for label, kw_new, kw_old in (("C", {}, {}),
                                  ("C_beta0.5", {"beta": 0.5}, {"beta": 0.5}),
                                  ("C_cap0.9", {"cap_scale": 0.9}, {"cap_scale": 0.9})):
        new = CommunityModel(com, CONFIGS["C"].with_(**kw_new)).solve()
        old = solve_community_C(com, **kw_old)
        tab = settlement(com, old)
        out[f"{label}_objective"] = abs(new.objective - old.objective)
        out[f"{label}_grid"] = float(max((new.grid[c] - old.grid[c]).abs().max() for c in ("P_imp", "P_exp", "pi", "mu")))
        out[f"{label}_member_cost"] = float((new.member_cost["cost_DKK"] - tab["cost_C_DKK"]).abs().max())
        out[f"{label}_rent"] = abs(new.manager()["budget_DKK"] - tab.attrs["manager"]["budget_DKK"])
    # ---- the sweep varies the switches of the same class (one-at-a-time, as in experiments_sweep.py)
    R = CommunityModel(com, "R").solve()
    for kw in ({"beta": 0.0}, {"alpha": 2.0}, {"cap_scale": 1.5}):
        i_new = CommunityModel(com, CONFIGS["I"].with_(**kw)).solve()
        ct = com.contract
        i_old = solve_community_I(com, beta=kw.get("beta"), alpha=kw.get("alpha"), cap_scale=kw.get("cap_scale", 1.0))
        out[f"I_{kw}_cost"] = abs(i_new.objective - sum(r.objective for r in i_old.values()))
    # ---- limit cases (new file only)
    out.update({f"validate_{k}": v for k, v in validate(com).items()})
    return out
