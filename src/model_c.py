"""Centralized dispatch and dual settlement for the community contract (Model C)."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from src.data_loader import CommunityData, InputData


@dataclass
class MemberDispatch:
    """Hourly dispatch and settlement for one member; energy is in kWh and price in DKK/kWh."""

    flexible_load_kWh: np.ndarray
    pv_generation_kWh: np.ndarray
    battery_charge_kW: np.ndarray
    battery_discharge_kW: np.ndarray
    battery_soc_kWh: np.ndarray | None
    net_consumption_kWh: np.ndarray
    disutility_DKK: float
    pv_cost_DKK: float
    settlement_DKK: float
    surplus_after_settlement_DKK: float


@dataclass
class ModelCResult:
    """Optimal Model C objective, grid exchange, prices, and member dispatch."""

    community_id: str
    day: str
    status: str
    welfare_DKK: float
    energy_cost_DKK: float
    tariff_cost_DKK: float
    penalty_cost_DKK: float
    disutility_cost_DKK: float
    pv_cost_DKK: float
    community_bill_DKK: float
    total_member_settlement_DKK: float
    manager_budget_balance_DKK: float
    community_import_kW: np.ndarray
    community_export_kW: np.ndarray
    cap_exceedance_kW: np.ndarray
    capacity_limit_kW: np.ndarray
    internal_price_DKK_per_kWh: np.ndarray
    members: dict[str, MemberDispatch]


def _hourly_array(values: np.ndarray | None, name: str, n_hours: int) -> np.ndarray:
    if values is None:
        raise ValueError(f"{name} is required by Model C")
    array = np.asarray(values, dtype=float)
    if array.shape != (n_hours,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {n_hours} finite hourly values")
    return array


def _validate_member(member: InputData, n_hours: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if member.reference_load is None or member.min_daily_energy_kWh is None:
        raise ValueError(f"{member.consumer_id} needs a reference profile and daily minimum energy")
    if member.linear_disutility is None or member.linear_disutility < 0:
        raise ValueError(f"{member.consumer_id} needs a non-negative linear disutility")
    if member.quadratic_disutility is not None or member.consumption_utility is not None:
        raise NotImplementedError(
            "Model C currently supports the assignment's flexible load with linear disutility only"
        )
    lower = _hourly_array(member.load_lower_kWh, f"{member.consumer_id}.load_lower_kWh", n_hours)
    upper = _hourly_array(member.load_upper_kWh, f"{member.consumer_id}.load_upper_kWh", n_hours)
    reference = _hourly_array(member.reference_load, f"{member.consumer_id}.reference_load", n_hours)
    if np.any(lower > upper):
        raise ValueError(f"{member.consumer_id} has hourly load lower bounds above upper bounds")
    if member.min_daily_energy_kWh < 0:
        raise ValueError(f"{member.consumer_id}.min_daily_energy_kWh cannot be negative")
    _hourly_array(member.pv_available, f"{member.consumer_id}.pv_available", n_hours)
    if member.pv_marginal_cost < 0:
        raise ValueError(f"{member.consumer_id}.pv_marginal_cost cannot be negative")
    if member.has_battery:
        required = (
            member.battery_capacity_kWh,
            member.battery_max_charge_kW,
            member.battery_max_discharge_kW,
            member.battery_charging_efficiency,
            member.battery_discharging_efficiency,
            member.battery_initial_soc_kWh,
        )
        if any(value is None for value in required):
            raise ValueError(f"{member.consumer_id} has incomplete battery parameters")
        if member.battery_capacity_kWh < 0 or member.battery_max_charge_kW < 0 or member.battery_max_discharge_kW < 0:
            raise ValueError(f"{member.consumer_id} battery capacity and power limits must be non-negative")
        if not (0 < member.battery_charging_efficiency <= 1 and 0 < member.battery_discharging_efficiency <= 1):
            raise ValueError(f"{member.consumer_id} battery efficiencies must lie in (0, 1]")
        if not (0 <= member.battery_initial_soc_kWh <= member.battery_capacity_kWh):
            raise ValueError(f"{member.consumer_id} initial battery SoC must lie within its capacity")
    return lower, upper, reference


def run_model_c(community: CommunityData) -> ModelCResult:
    """Solve centralized Model C and compute uniform internal prices from balance duals.

    Battery end-of-day state of charge equals its initial state unless an explicit
    ``battery_final_soc_kWh`` is provided. The problem is a continuous LP.
    """
    if not community.members:
        raise ValueError("Model C requires at least one community member")
    n_hours = community.members[0].n_hours
    if any(member.n_hours != n_hours for member in community.members):
        raise ValueError("All community members must have the same hourly horizon")
    prices = _hourly_array(community.energy_price, "energy_price", n_hours)
    if np.any(prices < 0):
        raise ValueError("Model C assumes non-negative day-ahead energy prices")
    caps = _hourly_array(community.contract.capacity_limit_kW, "capacity_limit_kW", n_hours)
    discount = community.contract.tariff_discount_ratio
    penalty_rate = community.contract.penalty_DKK_per_kWh
    if not 0 <= discount <= 1 or penalty_rate < 0:
        raise ValueError("The tariff discount must lie in [0, 1] and the penalty must be non-negative")
    if community.import_tariff < 0 or community.export_tariff < 0:
        raise ValueError("Model C assumes non-negative import and export tariffs")
    member_bounds = {
        member.consumer_id: _validate_member(member, n_hours)
        for member in community.members
    }

    with gp.Model(f"ModelC_{community.community_id}_{community.day}") as model:
        model.Params.OutputFlag = 0
        load: dict[str, gp.MVar] = {}
        deviation: dict[str, gp.MVar] = {}
        pv: dict[str, gp.MVar] = {}
        charge: dict[str, gp.MVar] = {}
        discharge: dict[str, gp.MVar] = {}
        soc: dict[str, gp.MVar] = {}
        net: dict[str, gp.MVar] = {}
        member_cost_terms: list[gp.LinExpr] = []

        for member in community.members:
            cid = member.consumer_id
            lower, upper, reference = member_bounds[cid]
            load[cid] = model.addMVar(n_hours, lb=lower, ub=upper, name=f"load_{cid}")
            deviation[cid] = model.addMVar(n_hours, lb=0, name=f"deviation_{cid}")
            model.addConstr(deviation[cid] >= load[cid] - reference, name=f"deviation_pos_{cid}")
            model.addConstr(deviation[cid] >= reference - load[cid], name=f"deviation_neg_{cid}")
            model.addConstr(load[cid].sum() >= member.min_daily_energy_kWh, name=f"daily_min_{cid}")

            available_pv = _hourly_array(member.pv_available, f"{cid}.pv_available", n_hours)
            pv[cid] = model.addMVar(n_hours, lb=0, ub=available_pv, name=f"pv_{cid}")
            net_terms: gp.MLinExpr = load[cid] - pv[cid]

            if member.has_battery:
                capacity = float(member.battery_capacity_kWh)
                charge[cid] = model.addMVar(
                    n_hours, lb=0, ub=float(member.battery_max_charge_kW), name=f"charge_{cid}"
                )
                discharge[cid] = model.addMVar(
                    n_hours, lb=0, ub=float(member.battery_max_discharge_kW), name=f"discharge_{cid}"
                )
                soc[cid] = model.addMVar(n_hours + 1, lb=0, ub=capacity, name=f"soc_{cid}")
                model.addConstr(soc[cid][0] == member.battery_initial_soc_kWh, name=f"soc_initial_{cid}")
                for hour in range(n_hours):
                    model.addConstr(
                        soc[cid][hour + 1]
                        == soc[cid][hour]
                        + member.battery_charging_efficiency * charge[cid][hour]
                        - discharge[cid][hour] / member.battery_discharging_efficiency,
                        name=f"soc_balance_{cid}_{hour}",
                    )
                final_soc = (
                    member.battery_initial_soc_kWh
                    if member.battery_final_soc_kWh is None
                    else member.battery_final_soc_kWh
                )
                if not 0 <= final_soc <= capacity:
                    raise ValueError(f"{cid}.battery_final_soc_kWh must lie within battery capacity")
                model.addConstr(soc[cid][n_hours] == final_soc, name=f"soc_final_{cid}")
                net_terms = net_terms + charge[cid] - discharge[cid]
            else:
                charge[cid] = model.addMVar(n_hours, lb=0, ub=0, name=f"charge_{cid}")
                discharge[cid] = model.addMVar(n_hours, lb=0, ub=0, name=f"discharge_{cid}")

            net[cid] = net_terms
            member_cost_terms.append(
                member.linear_disutility * deviation[cid].sum()
                + member.pv_marginal_cost * pv[cid].sum()
            )

        community_import = model.addMVar(n_hours, lb=0, name="community_import")
        community_export = model.addMVar(n_hours, lb=0, name="community_export")
        exceedance = model.addMVar(n_hours, lb=0, name="cap_exceedance")
        balance_constraints = model.addConstr(
            gp.quicksum(net[member.consumer_id] for member in community.members)
            == community_import - community_export,
            name="community_energy_balance",
        )
        model.addConstr(exceedance >= community_import - caps, name="soft_import_cap")
        if community.contract.export_capacity_limit_kW is not None:
            model.addConstr(
                community_export <= community.contract.export_capacity_limit_kW,
                name="community_export_limit",
            )
        physical_import_limits = {member.max_import_kW for member in community.members}
        physical_export_limits = {member.max_export_kW for member in community.members}
        if len(physical_import_limits) != 1 or len(physical_export_limits) != 1:
            raise ValueError("Members at the shared meter must have consistent physical grid limits")
        import_limit = next(iter(physical_import_limits))
        export_limit = next(iter(physical_export_limits))
        if import_limit is not None:
            model.addConstr(community_import <= import_limit, name="physical_import_limit")
        if export_limit is not None:
            model.addConstr(community_export <= export_limit, name="physical_export_limit")

        energy_cost = gp.quicksum(prices[t] * (community_import[t] - community_export[t]) for t in range(n_hours))
        tariff_cost = (1 - discount) * (
            community.import_tariff * community_import.sum()
            + community.export_tariff * community_export.sum()
        )
        penalty_cost = penalty_rate * exceedance.sum()
        model.setObjective(
            -gp.quicksum(member_cost_terms) - energy_cost - tariff_cost - penalty_cost,
            GRB.MAXIMIZE,
        )
        model.optimize()

        if model.Status != GRB.OPTIMAL:
            status_name = {
                GRB.INFEASIBLE: "INFEASIBLE",
                GRB.INF_OR_UNBD: "INFEASIBLE_OR_UNBOUNDED",
                GRB.UNBOUNDED: "UNBOUNDED",
                GRB.TIME_LIMIT: "TIME_LIMIT",
            }.get(model.Status, str(model.Status))
            raise RuntimeError(f"Model C did not reach an optimal solution (status: {status_name})")

        imports = np.asarray(community_import.X, dtype=float)
        exports = np.asarray(community_export.X, dtype=float)
        exceedance_values = np.maximum(0.0, imports - caps)
        internal_prices = np.asarray(balance_constraints.getAttr(GRB.Attr.Pi), dtype=float)
        member_results: dict[str, MemberDispatch] = {}
        for member in community.members:
            cid = member.consumer_id
            load_values = np.asarray(load[cid].X, dtype=float)
            pv_values = np.asarray(pv[cid].X, dtype=float)
            charge_values = np.asarray(charge[cid].X, dtype=float)
            discharge_values = np.asarray(discharge[cid].X, dtype=float)
            net_values = np.asarray(net[cid].getValue(), dtype=float)
            reference = member_bounds[cid][2]
            disutility = float(member.linear_disutility * np.abs(load_values - reference).sum())
            pv_cost = float(member.pv_marginal_cost * pv_values.sum())
            settlement = float(np.dot(internal_prices, net_values))
            battery_soc = np.asarray(soc[cid].X, dtype=float) if member.has_battery else None
            member_results[cid] = MemberDispatch(
                flexible_load_kWh=load_values,
                pv_generation_kWh=pv_values,
                battery_charge_kW=charge_values,
                battery_discharge_kW=discharge_values,
                battery_soc_kWh=battery_soc,
                net_consumption_kWh=net_values,
                disutility_DKK=disutility,
                pv_cost_DKK=pv_cost,
                settlement_DKK=settlement,
                surplus_after_settlement_DKK=-disutility - pv_cost - settlement,
            )

        energy_cost_value = float(np.dot(prices, imports - exports))
        tariff_cost_value = float(
            (1 - discount)
            * (community.import_tariff * imports.sum() + community.export_tariff * exports.sum())
        )
        penalty_cost_value = float(penalty_rate * exceedance_values.sum())
        disutility_value = sum(result.disutility_DKK for result in member_results.values())
        pv_cost_value = sum(result.pv_cost_DKK for result in member_results.values())
        total_settlement = sum(result.settlement_DKK for result in member_results.values())
        community_bill = energy_cost_value + tariff_cost_value + penalty_cost_value
        return ModelCResult(
            community_id=community.community_id,
            day=community.day,
            status="OPTIMAL",
            welfare_DKK=float(model.ObjVal),
            energy_cost_DKK=energy_cost_value,
            tariff_cost_DKK=tariff_cost_value,
            penalty_cost_DKK=penalty_cost_value,
            disutility_cost_DKK=disutility_value,
            pv_cost_DKK=pv_cost_value,
            community_bill_DKK=community_bill,
            total_member_settlement_DKK=total_settlement,
            manager_budget_balance_DKK=total_settlement - community_bill,
            community_import_kW=imports,
            community_export_kW=exports,
            cap_exceedance_kW=exceedance_values,
            capacity_limit_kW=caps.copy(),
            internal_price_DKK_per_kWh=internal_prices,
            members=member_results,
        )


def save_model_c_result(result: ModelCResult, output_dir: Path | str) -> tuple[Path, Path]:
    """Write Model C's summary JSON and hourly member dispatch CSV to ``output_dir``."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = directory / "summary.json"
    dispatch_path = directory / "dispatch.csv"
    summary = {
        key: value
        for key, value in asdict(result).items()
        if key not in {
            "community_import_kW", "community_export_kW", "cap_exceedance_kW",
            "capacity_limit_kW", "internal_price_DKK_per_kWh", "members",
        }
    }
    summary["hourly"] = {
        "community_import_kW": result.community_import_kW.tolist(),
        "community_export_kW": result.community_export_kW.tolist(),
        "cap_exceedance_kW": result.cap_exceedance_kW.tolist(),
        "capacity_limit_kW": result.capacity_limit_kW.tolist(),
        "internal_price_DKK_per_kWh": result.internal_price_DKK_per_kWh.tolist(),
    }
    summary["members"] = {
        cid: {
            key: value
            for key, value in asdict(dispatch).items()
            if key not in {
                "flexible_load_kWh", "pv_generation_kWh", "battery_charge_kW",
                "battery_discharge_kW", "battery_soc_kWh", "net_consumption_kWh",
            }
        }
        for cid, dispatch in result.members.items()
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fields = (
        "hour", "consumer_id", "community_import_kW", "community_export_kW",
        "capacity_limit_kW", "cap_exceedance_kW", "internal_price_DKK_per_kWh",
        "net_consumption_kWh", "flexible_load_kWh", "pv_generation_kWh",
        "battery_charge_kW", "battery_discharge_kW", "battery_soc_end_kWh",
        "settlement_DKK",
    )
    with dispatch_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for cid, member in result.members.items():
            for hour, net in enumerate(member.net_consumption_kWh):
                writer.writerow({
                    "hour": hour,
                    "consumer_id": cid,
                    "community_import_kW": result.community_import_kW[hour],
                    "community_export_kW": result.community_export_kW[hour],
                    "capacity_limit_kW": result.capacity_limit_kW[hour],
                    "cap_exceedance_kW": result.cap_exceedance_kW[hour],
                    "internal_price_DKK_per_kWh": result.internal_price_DKK_per_kWh[hour],
                    "net_consumption_kWh": net,
                    "flexible_load_kWh": member.flexible_load_kWh[hour],
                    "pv_generation_kWh": member.pv_generation_kWh[hour],
                    "battery_charge_kW": member.battery_charge_kW[hour],
                    "battery_discharge_kW": member.battery_discharge_kW[hour],
                    "battery_soc_end_kWh": (
                        "" if member.battery_soc_kWh is None else member.battery_soc_kWh[hour + 1]
                    ),
                    "settlement_DKK": (
                        result.internal_price_DKK_per_kWh[hour] * net
                    ),
                })
    return summary_path, dispatch_path
