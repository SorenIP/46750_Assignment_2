"""Validation helpers for model R (and later I, C).

``optimal_face_range`` answers: is the community's no-contract profile unique? Every member's LP can have
several optimal schedules (a member is indifferent when a price is exactly c_L away from its internal value
of energy). The members are independent, so any combination of optimal schedules is optimal for R, and
aggregate metrics such as the cap exceedance are not unique. This function computes their range over all
combinations of optimal member schedules.
"""
from __future__ import annotations

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from .data_loader import CommunityData
from .model import ContractTerms, MemberModel, MemberResults


def optimal_face_range(com: CommunityData, res: dict[str, MemberResults], tol: float = 1e-6) -> dict[str, float]:
    """Min and max of the aggregated cap exceedance (kWh) and min of the peak (kW) over all schedules in which
    every member is optimal (member cost <= its optimum + tol). Returns a dict of floats."""
    cap = np.asarray(com.contract.capacity_limit_kW, dtype=float)
    out = {}
    for sense, label in ((GRB.MINIMIZE, "exceedance_min_kWh"), (GRB.MAXIMIZE, "exceedance_max_kWh")):
        m = gp.Model("face")
        m.Params.OutputFlag = 0
        members = [MemberModel(d, ContractTerms(), model=m).build() for d in com.members]
        for mm in members:
            m.addConstr(gp.quicksum(mm._cost.values()) <= res[mm.d.consumer_id].objective + tol)
        T = range(24)
        P = {t: gp.quicksum(mm.var["import"][t] for mm in members) for t in T}
        exc = m.addVars(T, lb=0, name="exc")
        if sense == GRB.MINIMIZE:
            m.addConstrs((exc[t] >= P[t] - float(cap[t]) for t in T))
        else:   # maximizing a convex function: exc_t = max(P_t - cap_t, 0) needs one binary per hour
            z = m.addVars(T, vtype=GRB.BINARY, name="z")
            M = 100.0
            m.addConstrs((exc[t] <= P[t] - float(cap[t]) + M * (1 - z[t]) for t in T))
            m.addConstrs((exc[t] <= M * z[t] for t in T))
        m.setObjective(gp.quicksum(exc[t] for t in T), sense)
        out[label] = solve_with_highs(m)
    return out


def solve_with_highs(m: gp.Model) -> float:
    """Solve a (MI)LP built in gurobipy with HiGHS (scipy) and return the objective value.

    Used where the model is larger than the restricted Gurobi licence (2000 variables/constraints) allows;
    with the academic licence ``m.optimize()`` can be used directly."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    m.update()
    A = m.getA().tocsr()
    v, c = m.getVars(), m.getConstrs()
    obj = np.array([x.Obj for x in v]) * (1.0 if m.ModelSense == GRB.MINIMIZE else -1.0)
    rhs = np.array([k.RHS for k in c])
    sense = np.array([k.Sense for k in c])
    lo = np.where(sense == "<", -np.inf, rhs)
    hi = np.where(sense == ">", np.inf, rhs)
    res = milp(obj, constraints=LinearConstraint(A, lo, hi),
               bounds=Bounds([x.LB for x in v], [x.UB for x in v]),
               integrality=np.array([1 if x.VType in ("B", "I") else 0 for x in v]))
    if not res.success:
        raise RuntimeError(res.message)
    return float(res.fun) * (1.0 if m.ModelSense == GRB.MINIMIZE else -1.0)
