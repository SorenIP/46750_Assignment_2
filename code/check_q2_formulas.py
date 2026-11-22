"""Check the formulas of the Q2 derivation: member LP at posted prices x (paper notation), KKT residuals, dual objective."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Question 1"))
import numpy as np, gurobipy as gp
from gurobipy import GRB
from src.data_loader import load_community
from src.community_model import CommunityModel

def member_lp(d, x):
    T = range(24); m = gp.Model(); m.Params.OutputFlag = 0
    f = dict(lb=-GRB.INFINITY)
    l, dp, dm, g, q = (m.addVars(T, name=n, **f) for n in ("l", "dp", "dm", "g", "q"))
    bat = bool(d.battery_capacity_kWh)
    if bat: ch, dis, s = (m.addVars(T, name=n, **f) for n in ("ch", "dis", "s"))
    C = {}
    B = lambda t: (ch[t] - dis[t]) if bat else 0
    C["bal"] = m.addConstrs((l[t] + B(t) - g[t] - q[t] == 0 for t in T))
    C["dev"] = m.addConstrs((l[t] - float(d.reference_load[t]) - dp[t] + dm[t] == 0 for t in T))
    C["dp"] = m.addConstrs((-dp[t] <= 0 for t in T)); C["dm"] = m.addConstrs((-dm[t] <= 0 for t in T))
    C["llo"] = m.addConstrs((float(d.load_lower_kWh[t]) - l[t] <= 0 for t in T))
    C["lhi"] = m.addConstrs((l[t] - float(d.load_upper_kWh[t]) <= 0 for t in T))
    C["E"] = m.addConstr(float(d.min_daily_energy_kWh) - gp.quicksum(l[t] for t in T) <= 0)
    Gb = d.pv_available if d.pv_max_kW > 0 else np.zeros(24)
    C["glo"] = m.addConstrs((-g[t] <= 0 for t in T)); C["ghi"] = m.addConstrs((g[t] - float(Gb[t]) <= 0 for t in T))
    if bat:
        ec, ed, S0, Sm = d.battery_charging_efficiency, d.battery_discharging_efficiency, d.battery_initial_soc_kWh, d.battery_capacity_kWh
        prev = lambda t: S0 if t == 0 else s[t - 1]
        C["sdyn"] = m.addConstrs((s[t] - prev(t) - ec * ch[t] + dis[t] / ed == 0 for t in T))
        C["send"] = m.addConstr(s[23] - S0 == 0)
        C["slo"] = m.addConstrs((-s[t] <= 0 for t in T)); C["shi"] = m.addConstrs((s[t] - Sm <= 0 for t in T))
        C["clo"] = m.addConstrs((-ch[t] <= 0 for t in T)); C["chi"] = m.addConstrs((ch[t] - d.battery_max_charge_kW <= 0 for t in T))
        C["dlo"] = m.addConstrs((-dis[t] <= 0 for t in T)); C["dhi"] = m.addConstrs((dis[t] - d.battery_max_discharge_kW <= 0 for t in T))
    m.setObjective(gp.quicksum(float(x[t]) * q[t] + d.pv_marginal_cost * g[t] + d.linear_disutility * (dp[t] + dm[t]) for t in T), GRB.MINIMIZE)
    m.optimize()
    mult = lambda k, t=None: -(C[k][t].Pi if t is not None else C[k].Pi)
    r = {}
    cL, cP = d.linear_disutility, d.pv_marginal_cost
    for t in T:
        lam, nu = mult("bal", t), mult("dev", t)
        r.setdefault("q", []).append(x[t] - lam)
        r.setdefault("d+", []).append(cL - nu - mult("dp", t)); r.setdefault("d-", []).append(cL + nu - mult("dm", t))
        r.setdefault("l", []).append(lam + nu + mult("lhi", t) - mult("llo", t) - mult("E"))
        r.setdefault("g", []).append(cP - lam + mult("ghi", t) - mult("glo", t))
        if bat:
            sg = mult("sdyn", t)
            r.setdefault("ch", []).append(lam - ec * sg + mult("chi", t) - mult("clo", t))
            r.setdefault("dis", []).append(-lam + sg / ed + mult("dhi", t) - mult("dlo", t))
            nxt = mult("sdyn", t + 1) if t < 23 else -mult("send")
            r.setdefault("s", []).append(sg - nxt + mult("shi", t) - mult("slo", t))   # t=23: sg + eps + ...
    D = sum(mult("llo", t) * d.load_lower_kWh[t] - mult("lhi", t) * d.load_upper_kWh[t] - mult("dev", t) * d.reference_load[t]
            - mult("ghi", t) * Gb[t] for t in T) + mult("E") * d.min_daily_energy_kWh
    if bat:
        D += sum(-mult("shi", t) * Sm - mult("chi", t) * d.battery_max_charge_kW - mult("dhi", t) * d.battery_max_discharge_kW for t in T)
        D -= (mult("sdyn", 0) + mult("send")) * S0
    return m.ObjVal, D, {k: float(np.abs(v).max()) for k, v in r.items()}

com = load_community("base", day="d1")
C = CommunityModel(com, "C").solve()
x = C.grid["pi"].values
worst = {"kkt": 0, "dual": 0, "BR_vs_C": 0}
for d in com.members:
    obj, D, res = member_lp(d, x)
    worst["kkt"] = max(worst["kkt"], max(res.values())); worst["dual"] = max(worst["dual"], abs(obj - D))
    worst["BR_vs_C"] = max(worst["BR_vs_C"], abs(obj - C.member_cost.loc[d.consumer_id, "cost_DKK"]))
print(worst)
