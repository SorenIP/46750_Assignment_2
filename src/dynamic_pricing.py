"""Base-case price-response check for Question 2's dynamic-pricing model.

The centralized model C supplies candidate posted prices through its balance duals.
This module checks whether C's dispatch is supported by each member's best response
to those prices, and reports the indifference member-hours caused by linear
disutility.
"""
from __future__ import annotations

from pathlib import Path

import gurobipy as gp
import numpy as np
import pandas as pd

from .community_model import CommunityModel, Results
from .data_loader import CommunityData, InputData
from .model import ContractTerms, MemberModel


class PostedPriceMemberModel(MemberModel):
    """A member LP whose net consumption is charged at an hourly posted price."""

    def __init__(self, data: InputData, posted_price: np.ndarray, verbose: bool = False):
        prices = np.asarray(posted_price, dtype=float)
        if prices.shape != (data.n_hours,) or not np.isfinite(prices).all():
            raise ValueError(f"{data.consumer_id}: posted_price must contain {data.n_hours} finite hourly prices")
        self.posted_price = prices
        super().__init__(data, ContractTerms(), verbose=verbose)

    def _cost_expressions(self) -> dict[str, gp.LinExpr]:
        d, T = self.d, self.T
        QI, QE, G = self.var["import"], self.var["export"], self.var["pv"]
        DP, DM = self.var["dev_plus"], self.var["dev_minus"]
        return {
            "energy": gp.quicksum(float(self.posted_price[t]) * (QI[t] - QE[t]) for t in T),
            "tariffs": gp.LinExpr(),
            "pv": gp.quicksum(d.pv_marginal_cost * G[t] for t in T),
            "disutility": gp.quicksum(d.linear_disutility * (DP[t] + DM[t]) for t in T),
            "penalty": gp.LinExpr(),
        }


def _manager_cost(com: CommunityData, central: Results, members: dict[str, pd.DataFrame]) -> float:
    """Evaluate the manager's external bill and members' real costs at a given dispatch."""
    net = sum(h["import"] - h["export"] for h in members.values())
    p_imp = net.clip(lower=0.0)
    p_exp = (-net).clip(lower=0.0)
    exceed = np.maximum(p_imp.to_numpy() - central.cap, 0.0)
    energy = float((com.energy_price * (p_imp - p_exp)).sum())
    tariffs = (1.0 - central.beta) * float(com.import_tariff * p_imp.sum() + com.export_tariff * p_exp.sum())
    penalty = central.alpha * float(exceed.sum())
    pv = sum(float(members[d.consumer_id]["pv"].sum()) * d.pv_marginal_cost for d in com.members)
    disutility = sum(float((members[d.consumer_id]["dev_plus"] + members[d.consumer_id]["dev_minus"]).sum())
                     * d.linear_disutility for d in com.members)
    return energy + tariffs + penalty + pv + disutility


def run_dp_base(com: CommunityData, output_dir: str | Path | None = None,
                solver: str = "auto", indifference_tolerance: float = 1e-5) -> dict[str, object]:
    """Test C's internal prices as posted prices in the base DP case.

    The member LPs omit grid tariffs and the cap penalty: those are paid by the
    manager at the community meter. Internal payments at the posted prices are
    transfers and therefore do not enter the manager's community-cost objective.
    """
    central = CommunityModel(com, "C", solver).solve()
    posted_prices = central.grid["pi"].to_numpy(dtype=float)

    responses: dict[str, pd.DataFrame] = {}
    member_rows = []
    indifferent_rows = []
    best_response_gaps = []
    schedule_gaps = []

    for data in com.members:
        response = PostedPriceMemberModel(data, posted_prices).build().solve()
        hourly = response.hourly.copy()
        hourly["posted_price"] = posted_prices
        responses[data.consumer_id] = hourly

        central_cost_at_price = float(central.member_cost.loc[data.consumer_id, "cost_DKK"])
        gap = central_cost_at_price - response.objective
        best_response_gaps.append(abs(gap))

        central_hourly = central.members[data.consumer_id]
        comparable = [c for c in ("load", "import", "export", "pv", "charge", "discharge")
                      if c in hourly and c in central_hourly]
        schedule_gap = max(float((hourly[c] - central_hourly[c]).abs().max()) for c in comparable)
        schedule_gaps.append(schedule_gap)

        # min_daily_energy is Emin - sum(load) <= 0, whose Gurobi dual is nonpositive.
        daily_energy_value = -float(response.duals["min_energy"])
        price_margin = posted_prices - daily_energy_value
        threshold_gap = np.abs(np.abs(price_margin) - float(data.linear_disutility))
        for t in np.flatnonzero(threshold_gap <= indifference_tolerance):
            direction = "reduce_load" if price_margin[t] > 0 else "increase_load"
            room = (data.reference_load[t] - data.load_lower_kWh[t] if direction == "reduce_load"
                    else data.load_upper_kWh[t] - data.reference_load[t])
            if room > indifference_tolerance:
                indifferent_rows.append({
                    "consumer_id": data.consumer_id,
                    "hour": int(t),
                    "slope_DKK_per_kWh": float(data.linear_disutility),
                    "posted_price_DKK_per_kWh": float(posted_prices[t]),
                    "daily_energy_value_DKK_per_kWh": daily_energy_value,
                    "price_margin_DKK_per_kWh": float(price_margin[t]),
                    "indifferent_direction": direction,
                    "available_load_movement_kWh": float(room),
                    "C_load_kWh": float(central_hourly["load"].iloc[t]),
                    "response_load_kWh": float(hourly["load"].iloc[t]),
                })

        member_rows.append({
            "consumer_id": data.consumer_id,
            "archetype": data.archetype,
            "slope_DKK_per_kWh": float(data.linear_disutility),
            "daily_energy_value_DKK_per_kWh": daily_energy_value,
            "C_cost_at_posted_price_DKK": central_cost_at_price,
            "best_response_cost_DKK": response.objective,
            "C_best_response_gap_DKK": gap,
            "max_schedule_difference_kWh": schedule_gap,
            "indifferent_member_hours": sum(r["consumer_id"] == data.consumer_id for r in indifferent_rows),
        })

    manager_cost = _manager_cost(com, central, responses)
    aggregate_net = sum(h["import"] - h["export"] for h in responses.values())
    aggregate_import = aggregate_net.clip(lower=0.0)
    aggregate_export = (-aggregate_net).clip(lower=0.0)
    checks = {
        "centralized_cost_DKK": central.objective,
        "independent_LP_solver_response_manager_cost_DKK": manager_cost,
        "independent_LP_solver_response_cost_minus_C_DKK": manager_cost - central.objective,
        "C_dispatch_best_response_gap_max_DKK": max(best_response_gaps, default=0.0),
        "independent_LP_solver_member_schedule_difference_max_kWh": max(schedule_gaps, default=0.0),
        "independent_LP_solver_aggregate_import_difference_max_kWh":
            float((aggregate_import - central.grid["P_imp"]).abs().max()),
        "independent_LP_solver_aggregate_export_difference_max_kWh":
            float((aggregate_export - central.grid["P_exp"]).abs().max()),
    }
    import_hours = central.grid["P_imp"] > 1e-8
    export_hours = central.grid["P_exp"] > 1e-8
    if import_hours.any():
        expected_import_price = (com.energy_price + (1.0 - central.beta) * com.import_tariff
                                 + central.grid["mu"].to_numpy())
        checks["import_price_dual_relation_max_DKK_per_kWh"] = float(
            np.abs(posted_prices[import_hours.to_numpy()] - expected_import_price[import_hours.to_numpy()]).max())
    if export_hours.any():
        expected_export_price = com.energy_price - (1.0 - central.beta) * com.export_tariff
        checks["export_price_dual_relation_max_DKK_per_kWh"] = float(
            np.abs(posted_prices[export_hours.to_numpy()] - expected_export_price[export_hours.to_numpy()]).max())

    prices = central.grid[["price", "P_imp", "P_exp", "cap", "pi", "mu"]].copy()
    prices = prices.rename(columns={"price": "day_ahead_price", "pi": "posted_price"})
    members = pd.DataFrame(member_rows).set_index("consumer_id")
    indifferent = pd.DataFrame(indifferent_rows, columns=[
        "consumer_id", "hour", "slope_DKK_per_kWh", "posted_price_DKK_per_kWh",
        "daily_energy_value_DKK_per_kWh", "price_margin_DKK_per_kWh",
        "indifferent_direction", "available_load_movement_kWh", "C_load_kWh", "response_load_kWh",
    ])

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        prices.to_csv(out / "DP_base_prices.csv", index_label="hour")
        members.to_csv(out / "DP_base_members.csv")
        indifferent.to_csv(out / "DP_base_indifferent_hours.csv", index=False)
        pd.Series(checks, name="value").to_csv(out / "DP_base_checks.csv", index_label="check")

    return {"central": central, "responses": responses, "prices": prices, "members": members,
            "indifferent_hours": indifferent, "checks": checks}
