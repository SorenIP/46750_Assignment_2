"""Solve an LP built in gurobipy, with a fallback when the model is larger than the restricted Gurobi licence.

``solve_lp(m)`` returns a ``Solution`` with the objective, the value of every variable and the dual of every
constraint, whichever solver did the work:

* Gurobi if the model fits the licence (the free academic licence has no size limit, the one that ships with
  ``gurobipy`` is limited to 2000 variables/constraints);
* otherwise HiGHS through ``scipy.optimize.linprog``.

Dual convention (both solvers): the dual of a constraint is d(objective)/d(right-hand side), as Gurobi's ``Pi``.
For a ``<=`` constraint in a minimization it is <= 0; for an equality it is free.
"""
from __future__ import annotations

from dataclasses import dataclass

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from scipy.optimize import linprog

METHOD = "highs"   # "highs-ds" (simplex) or "highs-ipm" (interior point); duals can differ when the LP is degenerate


@dataclass
class Solution:
    """Result of ``solve_lp``: objective, primal and dual values, and the solver that produced them."""

    objective: float
    solver: str
    _x: dict          # variable index -> value
    _pi: dict         # constraint index -> dual

    def x(self, var: gp.Var) -> float:
        return self._x[var.index]

    def pi(self, con: gp.Constr) -> float:
        return self._pi[con.index]


def solve_lp(m: gp.Model, solver: str = "auto") -> Solution:
    """Solve the LP ``m``. ``solver``: "auto" (Gurobi if it fits, else HiGHS), "gurobi" or "highs"."""
    m.update()
    if solver in ("auto", "gurobi"):
        try:
            m.Params.OutputFlag = 0
            m.optimize()
            if m.Status != GRB.OPTIMAL:
                raise RuntimeError(f"Gurobi status {m.Status}")
            return Solution(m.ObjVal, "gurobi", {v.index: v.X for v in m.getVars()},
                            {c.index: c.Pi for c in m.getConstrs()})
        except gp.GurobiError as e:
            if solver == "gurobi" or "too large" not in str(e):
                raise
    return _solve_highs(m)


def _solve_highs(m: gp.Model) -> Solution:
    """HiGHS (scipy ``linprog``) on the matrix form of ``m``. Variables keep their Gurobi bounds."""
    A = m.getA().tocsr()
    v, c = m.getVars(), m.getConstrs()
    sgn = 1.0 if m.ModelSense == GRB.MINIMIZE else -1.0
    cost = sgn * np.array([x.Obj for x in v])
    rhs = np.array([k.RHS for k in c])
    sense = np.array([k.Sense for k in c])
    le, ge, eq = sense == "<", sense == ">", sense == "="
    A_ub = None
    b_ub = None
    if le.any() or ge.any():
        from scipy.sparse import vstack
        A_ub = vstack([A[le], -A[ge]]).tocsr()
        b_ub = np.concatenate([rhs[le], -rhs[ge]])
    res = linprog(cost, A_ub=A_ub, b_ub=b_ub,
                  A_eq=A[eq] if eq.any() else None, b_eq=rhs[eq] if eq.any() else None,
                  bounds=[(None if x.LB <= -GRB.INFINITY else x.LB, None if x.UB >= GRB.INFINITY else x.UB) for x in v],
                  method=METHOD)
    if res.status != 0:
        raise RuntimeError(f"HiGHS: {res.message}")
    pi = np.zeros(len(c))
    n_le = int(le.sum())
    if A_ub is not None:
        marg = res.ineqlin.marginals
        pi[np.where(le)[0]] = marg[:n_le]
        pi[np.where(ge)[0]] = -marg[n_le:]          # the row was multiplied by -1
    if eq.any():
        pi[np.where(eq)[0]] = res.eqlin.marginals
    pi *= sgn                                        # marginals are for the minimization form
    obj = sgn * res.fun + m.ObjCon
    return Solution(obj, "highs", {x.index: float(res.x[i]) for i, x in enumerate(v)},
                    {k.index: float(pi[i]) for i, k in enumerate(c)})
