import csv
import json

import numpy as np

from src.data_loader import load_community
from src.model_c import run_model_c, save_model_c_result


def test_model_c_dispatch_settlement_and_exports(tmp_path):
    community = load_community("base", day="d1")
    result = run_model_c(community)

    assert result.status == "OPTIMAL"
    assert result.community_import_kW.shape == (24,)
    assert result.community_export_kW.shape == (24,)
    np.testing.assert_allclose(
        sum(member.net_consumption_kWh for member in result.members.values()),
        result.community_import_kW - result.community_export_kW,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result.cap_exceedance_kW,
        np.maximum(0, result.community_import_kW - result.capacity_limit_kW),
        atol=1e-6,
    )

    for member_data in community.members:
        dispatch = result.members[member_data.consumer_id]
        assert np.all(dispatch.flexible_load_kWh >= member_data.load_lower_kWh - 1e-6)
        assert np.all(dispatch.flexible_load_kWh <= member_data.load_upper_kWh + 1e-6)
        assert dispatch.flexible_load_kWh.sum() >= member_data.min_daily_energy_kWh - 1e-6
        assert np.all(dispatch.pv_generation_kWh >= -1e-6)
        assert np.all(dispatch.pv_generation_kWh <= member_data.pv_available + 1e-6)
        np.testing.assert_allclose(
            dispatch.settlement_DKK,
            np.dot(result.internal_price_DKK_per_kWh, dispatch.net_consumption_kWh),
            atol=1e-6,
        )
        if member_data.has_battery:
            assert dispatch.battery_soc_kWh is not None
            assert np.all(dispatch.battery_soc_kWh >= -1e-6)
            assert np.all(dispatch.battery_soc_kWh <= member_data.battery_capacity_kWh + 1e-6)
            assert abs(dispatch.battery_soc_kWh[0] - member_data.battery_initial_soc_kWh) < 1e-6
            assert abs(dispatch.battery_soc_kWh[-1] - member_data.battery_initial_soc_kWh) < 1e-6

    assert abs(
        result.total_member_settlement_DKK
        - sum(member.settlement_DKK for member in result.members.values())
    ) < 1e-6
    assert abs(
        result.manager_budget_balance_DKK
        - (result.total_member_settlement_DKK - result.community_bill_DKK)
    ) < 1e-6
    total_cost = (
        result.energy_cost_DKK
        + result.tariff_cost_DKK
        + result.penalty_cost_DKK
        + result.disutility_cost_DKK
        + result.pv_cost_DKK
    )
    assert abs(result.welfare_DKK + total_cost) < 1e-5

    summary_path, dispatch_path = save_model_c_result(result, tmp_path)
    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved_summary["status"] == "OPTIMAL"
    assert len(saved_summary["hourly"]["internal_price_DKK_per_kWh"]) == 24
    with dispatch_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 24 * len(community.members)
