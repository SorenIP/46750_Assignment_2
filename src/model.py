"""Member model of Assignment 2: one parametrized LP per community member.

One class, three uses (switches, not copies):

    R : ContractTerms()                       no contract: full tariffs, no cap, no penalty
    I : ContractTerms(beta, cap_i, alpha)     individual contract on the member's own meter

(The community model C will reuse the same member data and contract terms.)

Member i minimizes, over the 24 hours,

    sum_t [ (p_t + (1-beta) tau_imp) q_imp_t - (p_t - (1-beta) tau_exp) q_exp_t  + alpha * exc_t ]
        + c_PV sum_t g_t + c_L sum_t (d+_t + d-_t)

with exc_t >= q_imp_t - cap_t, exc_t >= 0 (only if a cap is given). It is a linear program, so strong
duality holds and the duals are the member's marginal values.

Conventions follow Assignment 1: variables are stored in self.var, constraints in self.con (explicit
constraints, no variable bounds, so that every dual is read from Pi). Units: kWh/h, kWh, DKK.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from .data_loader import InputData


@dataclass
class ContractTerms:
    """Contract switches of one meter. The defaults are model R (no contract)."""

    beta: float = 0.0                       # tariff discount in [0, 1]
    cap: np.ndarray | None = None           # hourly cap on the import, kW (None = no cap)
    alpha: float = 0.0                      # penalty, DKK per kWh imported above the cap


@dataclass
class MemberResults:
    """Primal and dual values of one member."""

    consumer_id: str
    objective: float
    cost_breakdown: dict[str, float]        # energy, tariffs, penalty, pv, disutility (DKK)
    hourly: pd.DataFrame                    # primal values and hourly duals, one row per hour
    duals: dict[str, float] = field(default_factory=dict)   # duals of daily constraints
    checks: dict[str, float] = field(default_factory=dict)  # validation residuals (all should be ~0)


class MemberModel:
    """LP of one member (load with linear disutility, optional PV, optional battery)."""

    def __init__(self, data: InputData, terms: ContractTerms | None = None, verbose: bool = False,
                 model: gp.Model | None = None):
        """``model``: an existing Gurobi model to add this member to (names are then prefixed with the
        consumer id). Default: a model of its own."""
        self.d = data
        self.terms = terms or ContractTerms()
        self.T = range(data.n_hours)
        self.pre = f"{data.consumer_id}_" if model is not None else ""
        self.m = model if model is not None else gp.Model(f"member_{data.consumer_id}")
        if model is None:
            self.m.Params.OutputFlag = 1 if verbose else 0
        self.var: dict[str, gp.tupledict] = {}
        self.con: dict[str, gp.tupledict | gp.Constr] = {}

    # ----------------------------------------------------------------- build
    def build(self) -> "MemberModel":
        """Declare variables, constraints and objective."""
        self._variables()
        self._constraints()
        self._objective()
        self.m.update()
        return self

    def _variables(self) -> None:
        """All variables are free here; every bound is an explicit constraint (to get its dual)."""
        m, T, d = self.m, self.T, self.d
        free = dict(lb=-GRB.INFINITY)
        self.var["load"] = m.addVars(T, name=self.pre + "load", **free)          # l_t, kWh/h
        self.var["dev_plus"] = m.addVars(T, name=self.pre + "dev_plus", **free)  # d+_t >= 0
        self.var["dev_minus"] = m.addVars(T, name=self.pre + "dev_minus", **free)  # d-_t >= 0
        self.var["pv"] = m.addVars(T, name=self.pre + "pv", **free)              # g_t
        self.var["import"] = m.addVars(T, name=self.pre + "import", **free)      # q_imp_t
        self.var["export"] = m.addVars(T, name=self.pre + "export", **free)      # q_exp_t
        if d.battery_capacity_kWh:
            self.var["charge"] = m.addVars(T, name=self.pre + "charge", **free)      # p_ch_t
            self.var["discharge"] = m.addVars(T, name=self.pre + "discharge", **free)  # p_dis_t
            self.var["soc"] = m.addVars(T, name=self.pre + "soc", **free)            # s_t, end of hour t
        if self.terms.cap is not None:
            self.var["exceed"] = m.addVars(T, name=self.pre + "exceed", **free)  # exc_t >= 0

    def _constraints(self) -> None:
        d, m, T, tm = self.d, self.m, self.T, self.terms
        L, DP, DM = self.var["load"], self.var["dev_plus"], self.var["dev_minus"]
        G, QI, QE = self.var["pv"], self.var["import"], self.var["export"]
        has_bat = "soc" in self.var
        has_pv = d.pv_max_kW > 0

        def bat(t):
            return (self.var["charge"][t] - self.var["discharge"][t]) if has_bat else 0

        # Energy balance at the meter: imports + PV + discharge = load + charge + exports
        self.con["balance"] = m.addConstrs(
            (L[t] + bat(t) + QE[t] - G[t] - QI[t] == 0 for t in T), name=self.pre + "balance")
        # Deviation from the preferred profile: l_t - l_ref_t = d+_t - d-_t
        self.con["deviation"] = m.addConstrs(
            (L[t] - float(d.reference_load[t]) - DP[t] + DM[t] == 0 for t in T), name=self.pre + "deviation")
        self.con["dev_plus_nonneg"] = m.addConstrs((-DP[t] <= 0 for t in T), name=self.pre + "dev_plus_nonneg")
        self.con["dev_minus_nonneg"] = m.addConstrs((-DM[t] <= 0 for t in T), name=self.pre + "dev_minus_nonneg")
        # Hourly load bounds (reference +- max deviation, within [0, L_max]) and daily minimum energy
        self.con["load_min"] = m.addConstrs((float(d.load_lower_kWh[t]) - L[t] <= 0 for t in T), name=self.pre + "load_min")
        self.con["load_max"] = m.addConstrs((L[t] - float(d.load_upper_kWh[t]) <= 0 for t in T), name=self.pre + "load_max")
        self.con["min_energy"] = m.addConstr(
            float(d.min_daily_energy_kWh) - gp.quicksum(L[t] for t in T) <= 0, name=self.pre + "min_energy")
        # PV (kept as variables even when absent: the available energy is then 0)
        pv_av = d.pv_available if has_pv else np.zeros(len(T))
        self.con["pv_min"] = m.addConstrs((-G[t] <= 0 for t in T), name=self.pre + "pv_min")
        self.con["pv_max"] = m.addConstrs((G[t] - float(pv_av[t]) <= 0 for t in T), name=self.pre + "pv_max")
        # Grid exchange
        self.con["import_nonneg"] = m.addConstrs((-QI[t] <= 0 for t in T), name=self.pre + "import_nonneg")
        self.con["export_nonneg"] = m.addConstrs((-QE[t] <= 0 for t in T), name=self.pre + "export_nonneg")
        # Contract: exceedance above the cap
        if tm.cap is not None:
            X = self.var["exceed"]
            self.con["exceed_def"] = m.addConstrs(
                (QI[t] - float(tm.cap[t]) - X[t] <= 0 for t in T), name=self.pre + "exceed_def")
            self.con["exceed_nonneg"] = m.addConstrs((-X[t] <= 0 for t in T), name=self.pre + "exceed_nonneg")
        # Battery: dynamics, end-of-day state = initial state (our choice), limits
        if has_bat:
            C, D, S = self.var["charge"], self.var["discharge"], self.var["soc"]
            ec, ed = d.battery_charging_efficiency, d.battery_discharging_efficiency
            S0, Smax = d.battery_initial_soc_kWh, d.battery_capacity_kWh
            prev = lambda t: S0 if t == 0 else S[t - 1]
            self.con["soc_dynamics"] = m.addConstrs(
                (S[t] - prev(t) - ec * C[t] + D[t] / ed == 0 for t in T), name=self.pre + "soc_dynamics")
            self.con["soc_end"] = m.addConstr(S[len(T) - 1] - S0 == 0, name=self.pre + "soc_end")
            self.con["soc_min"] = m.addConstrs((-S[t] <= 0 for t in T), name=self.pre + "soc_min")
            self.con["soc_max"] = m.addConstrs((S[t] - Smax <= 0 for t in T), name=self.pre + "soc_max")
            self.con["charge_min"] = m.addConstrs((-C[t] <= 0 for t in T), name=self.pre + "charge_min")
            self.con["charge_max"] = m.addConstrs((C[t] - d.battery_max_charge_kW <= 0 for t in T), name=self.pre + "charge_max")
            self.con["discharge_min"] = m.addConstrs((-D[t] <= 0 for t in T), name=self.pre + "discharge_min")
            self.con["discharge_max"] = m.addConstrs(
                (D[t] - d.battery_max_discharge_kW <= 0 for t in T), name=self.pre + "discharge_max")

    def _cost_expressions(self) -> dict[str, gp.LinExpr]:
        """Cost components as expressions (DKK): energy, tariffs, penalty, PV, disutility."""
        d, T, tm = self.d, self.T, self.terms
        QI, QE, G = self.var["import"], self.var["export"], self.var["pv"]
        DP, DM = self.var["dev_plus"], self.var["dev_minus"]
        p = d.energy_price
        k = 1.0 - tm.beta
        out = {
            "energy": gp.quicksum(float(p[t]) * (QI[t] - QE[t]) for t in T),
            "tariffs": gp.quicksum(k * d.import_tariff * QI[t] + k * d.export_tariff * QE[t] for t in T),
            "pv": gp.quicksum(d.pv_marginal_cost * G[t] for t in T),
            "disutility": gp.quicksum(d.linear_disutility * (DP[t] + DM[t]) for t in T),
        }
        out["penalty"] = (gp.quicksum(tm.alpha * self.var["exceed"][t] for t in T)
                          if tm.cap is not None else gp.LinExpr())
        return out

    def _objective(self) -> None:
        self._cost = self._cost_expressions()
        self.m.setObjective(gp.quicksum(self._cost.values()), GRB.MINIMIZE)

    # ----------------------------------------------------------------- solve
    def solve(self) -> MemberResults:
        """Optimize and return primal values, cost components and duals."""
        self.m.optimize()
        if self.m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"{self.d.consumer_id}: status {self.m.Status}")
        T, d = list(self.T), self.d
        h = pd.DataFrame(index=pd.Index(T, name=self.pre + "hour"))
        h["price"] = d.energy_price
        h["reference_load"] = d.reference_load
        h["pv_available"] = d.pv_available if d.pv_max_kW > 0 else 0.0
        for name, v in self.var.items():
            h[name] = [v[t].X for t in T]
        h["net_import"] = h["import"] - h["export"]
        h["deviation"] = h["load"] - h["reference_load"]
        duals = {}
        for name, c in self.con.items():
            if isinstance(c, gp.tupledict):
                h[f"dual_{name}"] = [c[t].Pi for t in T]
            else:
                duals[name] = c.Pi
        cost = {k: e.getValue() for k, e in self._cost.items()}
        assert abs(sum(cost.values()) - self.m.ObjVal) < 1e-6 * max(1.0, abs(self.m.ObjVal))
        return MemberResults(d.consumer_id, self.m.ObjVal, cost, h, duals, self._checks(h))

    def _checks(self, h: pd.DataFrame) -> dict[str, float]:
        """Validation residuals of the solution (units in the key). All should be zero up to solver tolerance."""
        d = self.d
        out = {}
        # strong duality: for an LP with free variables the dual objective is sum_c Pi_c * RHS_c
        dual_obj = sum(c.Pi * c.RHS for c in self.m.getConstrs())
        out["strong_duality_gap_DKK"] = abs(self.m.ObjVal - dual_obj)
        # no simultaneous import and export (would be wasteful: buy at p+tau, sell at p-tau)
        out["import_export_overlap_kWh"] = float(np.minimum(h["import"], h["export"]).max())
        if "charge" in h:
            out["charge_discharge_overlap_kWh"] = float(np.minimum(h["charge"], h["discharge"]).max())
            out["soc_end_minus_start_kWh"] = abs(h["soc"].iloc[-1] - d.battery_initial_soc_kWh)
        out["energy_balance_kWh"] = float((h["load"] + h.get("charge", 0) - h.get("discharge", 0)
                                           + h["export"] - h["pv"] - h["import"]).abs().max())
        out["daily_energy_shortfall_kWh"] = float(max(0.0, d.min_daily_energy_kWh - h["load"].sum()))
        out["load_bound_violation_kWh"] = float(max((d.load_lower_kWh - h["load"]).max(),
                                                   (h["load"] - d.load_upper_kWh).max(), 0.0))
        return out


def solve_community_R(members: list[InputData]) -> dict[str, MemberResults]:
    """Model R: every member alone, no contract, full tariffs. Returns {consumer_id: results}."""
    return {m.consumer_id: MemberModel(m).build().solve() for m in members}


def aggregate(results: dict[str, MemberResults]) -> pd.DataFrame:
    """Sum of the members' hourly import/export/load/PV (the profile the DSO sees)."""
    cols = ["import", "export", "load", "pv", "net_import"]
    return sum(r.hourly[cols] for r in results.values())
