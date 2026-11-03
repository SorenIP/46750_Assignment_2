"""One parametrized model of the energy community: R, I and C are *configurations*, not copies.

    from src.community_model import CommunityModel, CONFIGS
    res = CommunityModel(com, CONFIGS["C"]).solve()          # or CONFIGS["R"], CONFIGS["I"]
    res = CommunityModel(com, CONFIGS["C"].with_(beta=0.5, cap_scale=0.9)).solve()   # a variation = a changed switch

What the switches mean (``Config``)
-----------------------------------
meter        "member"    every member has its own meter and schedules alone (R and I). The members do not interact, so
                         the model is solved member by member (16 small LPs).
             "community" one meter for the community and central dispatch (C): one LP for all members, linked by the
                         internal balance  sum_i q_i,t = P_imp_t - P_exp_t.  Its dual is the internal price pi_t.
beta         discount on the tariffs at the meter(s): (1 - beta) * tariff.  None = value of the contract file.
alpha        penalty on import above the cap, DKK/kWh.  None = value of the contract file.
cap          "none"       no cap (R).
             "individual" every member has its own cap kappa_i * cap_t (I; needs meter="member").
             "community"  one cap on the community import (C; needs meter="community").
cap_scale    multiplies the caps (cap tightness; 1 = contract file).
tariff_base  "meter"      tariffs on the flow at the meter (normal).
             "own_flow"   community meter, but the tariffs are charged on each member's own gross import and export.
                          With no cap, no discount and no penalty this is exactly R (validation, ``validate``).

Named configurations (``CONFIGS``): R, I, C and the validation cases C_own_flow, C_no_terms, I_no_terms.
Add a new variation by adding a ``Config`` (or by ``with_``); nothing else has to change.

What every model returns (``Results``)
--------------------------------------
objective   community cost (energy + tariffs + penalty + PV + disutility; payments between members cancel)
members     {consumer_id: hourly DataFrame}, columns load, import, export, pv, net, deviation, ...
grid        per hour: price, P_imp, P_exp, cap, exceed, and for C the internal price pi and the cap rent mu
member_cost per-member cost: own cost (member meter) or internal-price payment + PV + disutility (community meter)
checks      validation residuals (all ~0)
Methods: ``metrics()`` (DSO and community view), ``benefit_vs(ref)``, ``manager()``, ``rebate(ref)``.

Metric definitions and their references: ``report/metrics_hypotheses.tex``.
The code of the first version (separate modules ``experiments*.py``) is kept; ``src/check_unified.py`` shows that this
file reproduces its numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from .data_loader import CommunityData
from .lp_solve import Solution, solve_lp
from .model import ContractTerms, MemberModel

TOL_KW = 1e-3      # an hour counts as "above the cap" only above this (the rounded individual caps sum to cap + 1e-5)
TOL_DKK = 1e-6


# =========================================================================== configuration
@dataclass(frozen=True)
class Config:
    """Switches of the model, see the module docstring. ``None`` for beta/alpha = take the contract file."""

    name: str
    meter: str = "member"
    beta: float | None = 0.0
    alpha: float | None = 0.0
    cap: str = "none"
    cap_scale: float = 1.0
    tariff_base: str = "meter"

    def __post_init__(self):
        assert self.meter in ("member", "community"), self.meter
        assert self.cap in ("none", "individual", "community"), self.cap
        assert self.tariff_base in ("meter", "own_flow"), self.tariff_base
        assert not (self.cap == "individual" and self.meter != "member"), "individual caps need meter='member'"
        assert not (self.cap == "community" and self.meter != "community"), "community cap needs meter='community'"
        assert not (self.tariff_base == "own_flow" and self.meter != "community"), "own_flow needs meter='community'"

    def with_(self, **changes) -> "Config":
        """A copy with some switches changed (e.g. ``CONFIGS['C'].with_(beta=0.5)``)."""
        return replace(self, **changes)


CONFIGS: dict[str, Config] = {
    "R": Config("R"),                                                                   # no contract
    "I": Config("I", beta=None, alpha=None, cap="individual"),                          # individual contracts
    "C": Config("C", meter="community", beta=None, alpha=None, cap="community"),        # community contract
    "C_own_flow": Config("C_own_flow", meter="community", tariff_base="own_flow"),      # C with the terms off = R
    "C_no_terms": Config("C_no_terms", meter="community"),                              # tariff netting only
    "I_no_terms": Config("I_no_terms", cap="individual"),                               # I with beta = alpha = 0 = R
}


# =========================================================================== results
@dataclass
class Results:
    """Solution of one configuration. See the module docstring."""

    config: Config
    solver: str
    objective: float
    members: dict[str, pd.DataFrame]
    grid: pd.DataFrame
    cost: dict[str, float]
    member_cost: pd.DataFrame        # per member: payment_DKK, pv_DKK, disutility_DKK, cost_DKK
    checks: dict[str, float]
    beta: float = 0.0
    alpha: float = 0.0
    cap: np.ndarray = field(default_factory=lambda: np.zeros(24))
    duals: dict[str, pd.DataFrame] = field(default_factory=dict)   # member meter: hourly and daily duals per member

    # ---- DSO and community view
    def metrics(self) -> dict[str, float]:
        """DSO view (import, peak, hours above cap, exceedance, revenue) and community view (cost, flexibility)."""
        P = self.grid["P_imp"].values
        exc = np.maximum(P - self.cap, 0.0)
        shifted = sum(h["deviation"].abs().sum() / 2 for h in self.members.values())
        return {"daily_import_kWh": P.sum(), "peak_import_kW": P.max(), "hours_above_cap": int((exc > TOL_KW).sum()),
                "exceedance_kWh": exc.sum(), "max_exceedance_kW": exc.max(), "total_cost_DKK": self.objective,
                **{f"cost_{k}_DKK": v for k, v in self.cost.items()},
                "dso_revenue_DKK": self.cost["tariffs"] + self.cost["penalty"], "shifted_kWh": shifted}

    # ---- member view
    def benefit_vs(self, ref: "Results") -> pd.Series:
        """B_i = cost of member i in ``ref`` minus its cost here (> 0: better off than in ``ref``)."""
        return (ref.member_cost["cost_DKK"] - self.member_cost["cost_DKK"]).rename("benefit_DKK")

    # ---- manager (community meter only)
    def manager(self) -> dict[str, float]:
        """Settlement of the community meter: payments collected, cost at the meter, budget (= congestion rent)."""
        if self.config.meter != "community":
            return {}
        collected = float(self.member_cost["payment_DKK"].sum())
        at_meter = self.cost["energy"] + self.cost["tariffs"] + self.cost["penalty"]
        return {"collected_DKK": collected, "cost_at_meter_DKK": at_meter, "budget_DKK": collected - at_meter,
                "cap_rent_DKK": float((self.grid["mu"] * self.grid["cap_active"]).sum())}

    def rebate(self, ref: "Results") -> tuple[pd.Series, dict[str, float]]:
        """Rent rebate (our proposal): the manager keeps nothing; members who lose against ``ref`` are made whole first
        and the rest is split equally. Possible only if the rent covers the losses. Returns (rebate per member, info)."""
        mg = self.manager()
        b = self.benefit_vs(ref)
        loss = (-b).clip(lower=0)
        rent = mg.get("budget_DKK", 0.0)
        ok = bool(mg) and rent >= loss.sum() - TOL_DKK
        reb = loss + (rent - loss.sum()) / len(b) if ok else b * 0.0
        return reb.rename("rebate_DKK"), {"rent_DKK": rent, "losses_DKK": float(loss.sum()),
                                          "gains_DKK": float(b.clip(lower=0).sum()), "rent_covers_losses": ok}


# =========================================================================== model
class CommunityModel:
    """R, I or C depending on ``config``. ``build()`` then ``solve()``; ``solve()`` builds if needed."""

    def __init__(self, com: CommunityData, config: Config | str, solver: str = "auto"):
        self.com = com
        self.cfg = CONFIGS[config] if isinstance(config, str) else config
        self.solver = solver
        ct = com.contract
        self.beta = ct.tariff_discount_ratio if self.cfg.beta is None else self.cfg.beta
        self.alpha = ct.penalty_DKK_per_kWh if self.cfg.alpha is None else self.cfg.alpha
        self.cap = self.cfg.cap_scale * np.asarray(ct.capacity_limit_kW, dtype=float)   # community cap (also the DSO reference)
        self._built = False

    # ------------------------------------------------------------------ build
    def build(self) -> "CommunityModel":
        if self.cfg.meter == "member":
            self._build_member_meters()
        else:
            self._build_community_meter()
        self._built = True
        return self

    def _build_member_meters(self) -> None:
        """R and I: an independent LP per member (own meter, own contract terms)."""
        ct = self.com.contract
        self.members = []
        for d in self.com.members:
            cap_i = (self.cfg.cap_scale * np.asarray(ct.individual_capacity_limit_kW[d.consumer_id], dtype=float)
                     if self.cfg.cap == "individual" else None)
            terms = ContractTerms(beta=self.beta, cap=cap_i, alpha=self.alpha if cap_i is not None else 0.0)
            self.members.append(MemberModel(d, terms).build())

    def _build_community_meter(self) -> None:
        """C: all members in one model, linked by the internal balance; tariffs and penalty at the community meter."""
        com, cfg, T = self.com, self.cfg, range(24)
        self.m = m = gp.Model("community")
        self.members = [MemberModel(d, ContractTerms(), model=m).build() for d in com.members]   # unchanged member LPs
        parts = gp.quicksum(mm._cost["pv"] + mm._cost["disutility"] for mm in self.members)
        p, k = com.energy_price, 1.0 - self.beta
        ti, te = com.import_tariff, com.export_tariff
        free = dict(lb=-GRB.INFINITY)
        self.P_imp, self.P_exp = m.addVars(T, name="P_imp", **free), m.addVars(T, name="P_exp", **free)
        # internal balance: members' net consumption = metered import - export; dual = - internal price
        self.c_bal = m.addConstrs(
            (gp.quicksum(mm.var["import"][t] - mm.var["export"][t] for mm in self.members)
             - self.P_imp[t] + self.P_exp[t] == 0 for t in T), name="community_balance")
        m.addConstrs((-self.P_imp[t] <= 0 for t in T), name="P_imp_nonneg")
        m.addConstrs((-self.P_exp[t] <= 0 for t in T), name="P_exp_nonneg")
        penalty = gp.LinExpr()
        self.c_cap = None
        if cfg.cap == "community":
            self.exceed = m.addVars(T, name="exceed", **free)
            self.c_cap = m.addConstrs((self.P_imp[t] - float(self.cap[t]) - self.exceed[t] <= 0 for t in T), name="cap")
            m.addConstrs((-self.exceed[t] <= 0 for t in T), name="exceed_nonneg")
            penalty = gp.quicksum(self.alpha * self.exceed[t] for t in T)
        if cfg.tariff_base == "own_flow":
            tariffs = gp.quicksum(k * ti * mm.var["import"][t] + k * te * mm.var["export"][t]
                                  for mm in self.members for t in T)
        else:
            tariffs = gp.quicksum(k * ti * self.P_imp[t] + k * te * self.P_exp[t] for t in T)
        self.expr = {"energy": gp.quicksum(float(p[t]) * (self.P_imp[t] - self.P_exp[t]) for t in T),
                     "tariffs": tariffs, "penalty": penalty}
        m.setObjective(gp.quicksum(self.expr.values()) + parts, GRB.MINIMIZE)
        m.update()

    # ------------------------------------------------------------------ solve
    def solve(self) -> Results:
        if not self._built:
            self.build()
        return self._solve_member_meters() if self.cfg.meter == "member" else self._solve_community_meter()

    def _grid(self, P_imp, P_exp) -> pd.DataFrame:
        g = pd.DataFrame({"price": self.com.energy_price, "P_imp": P_imp, "P_exp": P_exp, "cap": self.cap},
                         index=pd.Index(range(24), name="hour"))
        g["cap_active"] = self.cap if self.cfg.cap != "none" else 0.0     # cap that is enforced (for the rent)
        g["exceed"] = np.maximum(g["P_imp"] - g["cap"], 0.0)
        g["pi"] = np.nan
        g["mu"] = np.nan
        return g

    def _solve_member_meters(self) -> Results:
        res = {mm.d.consumer_id: mm.solve() for mm in self.members}
        members = {}
        for c, r in res.items():
            h = r.hourly.copy()
            h["net"] = h["import"] - h["export"]
            members[c] = h
        P = sum(h["import"] for h in members.values())
        X = sum(h["export"] for h in members.values())
        comp = pd.DataFrame({c: r.cost_breakdown for c, r in res.items()}).T
        mc = pd.DataFrame({"payment_DKK": comp["energy"] + comp["tariffs"] + comp["penalty"],
                           "pv_DKK": comp["pv"], "disutility_DKK": comp["disutility"]})
        mc["cost_DKK"] = pd.Series({c: r.objective for c, r in res.items()})
        worst = {}
        for r in res.values():
            for k, v in r.checks.items():
                worst[k] = max(worst.get(k, 0.0), v)
        return Results(self.cfg, "gurobi", float(comp.sum().sum()), members, self._grid(P, X),
                       {k: float(comp[k].sum()) for k in comp.columns}, mc, worst, self.beta, self.alpha, self.cap,
                       {c: r.hourly.filter(like="dual_") for c, r in res.items()})

    def _solve_community_meter(self) -> Results:
        T = list(range(24))
        sol = solve_lp(self.m, self.solver)
        members, parts = {}, {}
        for mm in self.members:
            d = mm.d
            h = pd.DataFrame(index=pd.Index(T, name="hour"))
            for name, v in mm.var.items():
                h[name] = [sol.x(v[t]) for t in T]
            h["reference_load"] = d.reference_load
            h["net"] = h["import"] - h["export"]
            h["deviation"] = h["load"] - h["reference_load"]
            members[d.consumer_id] = h
            parts[d.consumer_id] = {"pv_DKK": d.pv_marginal_cost * h["pv"].sum(),
                                    "disutility_DKK": d.linear_disutility * (h["dev_plus"] + h["dev_minus"]).sum()}
        g = self._grid([sol.x(self.P_imp[t]) for t in T], [sol.x(self.P_exp[t]) for t in T])
        g["pi"] = [-sol.pi(self.c_bal[t]) for t in T]                       # internal price, DKK/kWh
        g["mu"] = [-sol.pi(self.c_cap[t]) for t in T] if self.c_cap is not None else 0.0   # congestion rent per kW of cap
        cost = {k: _value(e, sol) for k, e in self.expr.items()}
        mc = pd.DataFrame(parts).T
        mc["payment_DKK"] = pd.Series({c: float((g["pi"] * h["net"]).sum()) for c, h in members.items()})
        mc["cost_DKK"] = mc["payment_DKK"] + mc["pv_DKK"] + mc["disutility_DKK"]
        cost["pv"], cost["disutility"] = float(mc["pv_DKK"].sum()), float(mc["disutility_DKK"].sum())
        assert abs(sum(cost.values()) - sol.objective) < 1e-6 * max(1.0, abs(sol.objective))
        return Results(self.cfg, sol.solver, sol.objective, members, g, cost, mc[["payment_DKK", "pv_DKK", "disutility_DKK", "cost_DKK"]],
                       self._checks_community(sol, g, members), self.beta, self.alpha, self.cap)

    def _checks_community(self, sol: Solution, g: pd.DataFrame, members: dict) -> dict[str, float]:
        k = 1.0 - self.beta
        dual_obj = sum(sol.pi(c) * c.RHS for c in self.m.getConstrs())
        chk = {"strong_duality_gap_DKK": abs(sol.objective - dual_obj),
               "internal_balance_kWh": float((sum(h["net"] for h in members.values()) - (g["P_imp"] - g["P_exp"])).abs().max()),
               "member_energy_balance_kWh": max(float((h["load"] + h.get("charge", 0) - h.get("discharge", 0) + h["export"]
                                                      - h["pv"] - h["import"]).abs().max()) for h in members.values())}
        if self.cfg.tariff_base == "meter":
            # while the community imports, pi = price + discounted import tariff + cap rent
            imp = g["P_imp"] > 1e-9
            chk["price_vs_import_price_plus_rent"] = float((g["pi"] - (g["price"] + k * self.com.import_tariff + g["mu"]))[imp].abs().max())
        if self.cfg.cap == "community":
            slack = g["P_imp"] < g["cap"] - 1e-6
            chk["cap_rent_zero_when_cap_slack"] = float(g["mu"][slack].abs().max()) if slack.any() else 0.0
        return chk


def _value(expr: gp.LinExpr, sol: Solution) -> float:
    """Value of a linear expression at the solution (works for any solver)."""
    return sum(expr.getCoeff(i) * sol.x(expr.getVar(i)) for i in range(expr.size())) + expr.getConstant()


# =========================================================================== convenience
def run(com: CommunityData, names=("R", "I", "C"), solver: str = "auto") -> dict[str, Results]:
    """Solve several named configurations (or ``Config`` objects) on the same community."""
    out = {}
    for n in names:
        cfg = CONFIGS[n] if isinstance(n, str) else n
        out[cfg.name] = CommunityModel(com, cfg, solver).solve()
    return out


def compare(results: dict[str, Results], ref: str = "R") -> pd.DataFrame:
    """Metrics of every model side by side, and the difference to ``ref``."""
    df = pd.DataFrame({n: r.metrics() for n, r in results.items()})
    for n in results:
        if n != ref:
            df[f"{n}_minus_{ref}"] = df[n] - df[ref]
    return df


def member_table(com: CommunityData, results: dict[str, Results], ref: str = "R") -> pd.DataFrame:
    """Per member: type, cost in every model, benefit against ``ref``; for community-meter models also the rebate."""
    rows = pd.DataFrame({d.consumer_id: {"archetype": d.archetype, "slope_class": d.slope_class,
                                         "asset_class": "PV+BAT" if d.has_pv and d.has_battery else "PV" if d.has_pv
                                         else "BAT" if d.has_battery else "none"} for d in com.members}).T
    for n, r in results.items():
        rows[f"cost_{n}_DKK"] = r.member_cost["cost_DKK"]
        if n != ref:
            rows[f"benefit_{n}_vs_{ref}_DKK"] = r.benefit_vs(results[ref])
            if r.config.meter == "community":
                reb, _ = r.rebate(results[ref])
                rows[f"rebate_{n}_DKK"] = reb
                rows[f"benefit_{n}_after_rebate_DKK"] = rows[f"benefit_{n}_vs_{ref}_DKK"] + reb
    return rows


def validate(com: CommunityData, solver: str = "auto") -> dict[str, float]:
    """Limit-case checks of the configurations (differences in DKK, ~0 unless stated).

    * C_own_flow - R ~ 0: community meter, no cap/discount/penalty, tariffs on each member's own flow = R.
    * I_no_terms - R ~ 0: individual contracts without discount and penalty = R (the caps then do not matter).
    * sum of member costs I - C >= 0: pooling is a relaxation of the individual contracts.
    * C_no_terms - R <= 0: tariff netting behind one meter can only save.
    """
    r = run(com, ["R", "I", "C", "C_own_flow", "C_no_terms", "I_no_terms"], solver)
    R = r["R"].objective
    return {"C_own_flow_minus_R_DKK": r["C_own_flow"].objective - R,
            "I_no_terms_minus_R_DKK": r["I_no_terms"].objective - R,
            "I_minus_C_DKK": r["I"].objective - r["C"].objective,
            "C_no_terms_minus_R_DKK": r["C_no_terms"].objective - R}
