from __future__ import annotations

import pandas as pd

from src.adaptive_target_optimizer import (
    _net_pnl_for_debit,
    evaluate_adaptive_policies,
    session_entry_date,
    summarize_oos,
    target_debit_for_net_profit,
    walk_forward_adaptive,
)
from src.core import Costs


def _zero_costs() -> Costs:
    return Costs(
        brokerage_per_order=0.0,
        stt_sell_option_pct=0.0,
        exchange_txn_pct=0.0,
        sebi_turnover_pct=0.0,
        stamp_duty_buy_option_pct=0.0,
        gst_pct=0.0,
        slippage_points_per_leg=0.0,
    )


def test_session_entry_date_uses_trading_sessions_not_calendar_days():
    x = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-06-08 10:00",
                    "2026-06-09 10:00",
                    "2026-06-10 10:00",
                    "2026-06-12 10:00",
                    "2026-06-15 10:00",
                    "2026-06-16 10:00",
                ]
            )
        }
    )
    expiry = pd.Timestamp("2026-06-16")
    assert session_entry_date(x, expiry, 1) == pd.Timestamp("2026-06-15")
    assert session_entry_date(x, expiry, 2) == pd.Timestamp("2026-06-12")
    assert session_entry_date(x, expiry, 4) == pd.Timestamp("2026-06-10")
    assert session_entry_date(x, expiry, 6) is None


def test_target_debit_matches_fixed_rupee_profit():
    costs = _zero_costs()
    credit = 30.0
    quantity = 325
    target = 6500.0
    debit = target_debit_for_net_profit(
        credit,
        target_rupees=target,
        quantity=quantity,
        costs=costs,
        trade_date=pd.Timestamp("2026-10-01"),
    )
    assert debit is not None
    assert abs(debit - 10.0) < 1e-8
    net = _net_pnl_for_debit(
        credit,
        debit,
        quantity=quantity,
        costs=costs,
        trade_date=pd.Timestamp("2026-10-01"),
    )
    assert abs(net - target) < 1e-6


def test_target_is_none_when_entry_credit_cannot_support_target():
    costs = _zero_costs()
    debit = target_debit_for_net_profit(
        10.0,
        target_rupees=4000.0,
        quantity=325,
        costs=costs,
        trade_date=pd.Timestamp("2026-10-01"),
    )
    assert debit is None


def test_adaptive_policy_schema_when_no_group_meets_sample_size():
    x = pd.DataFrame(
        {
            "session_offset": [2, 2],
            "sd": [2.0, 2.0],
            "target_rupees": [7000.0, 7000.0],
            "stop_multiple": [2.5, 2.5],
            "entry_timestamp": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "net_pnl_current": [7000.0, -7000.0],
            "target_hit": [True, False],
        }
    )
    out = evaluate_adaptive_policies(x, min_trades=10, min_half_trades=3)
    assert out.empty
    assert "expectancy" in out.columns
    assert "target_hit_rate" in out.columns


def test_walk_forward_adaptive_does_not_fallback_to_unstable_policy():
    dates = pd.date_range("2024-01-01", periods=30, freq="14D")
    x = pd.DataFrame(
        {
            "entry_timestamp": dates,
            "session_offset": [2] * 30,
            "sd": [2.0] * 30,
            "target_rupees": [7000.0] * 30,
            "stop_multiple": [2.5] * 30,
            "net_pnl_current": [1000.0] * 15 + [-1000.0] * 15,
            "target_hit": [True] * 30,
        }
    )
    selected, oos = walk_forward_adaptive(
        x,
        train_months=6,
        test_months=3,
        rebalance_months=3,
        min_train_trades=6,
    )
    assert isinstance(selected, pd.DataFrame)
    assert isinstance(oos, pd.DataFrame)
    if not selected.empty:
        assert set(selected["selection_status"].dropna().unique()) <= {
            "NO_STABLE_POLICY",
            "SELECTED",
        }
        assert not ((selected["selection_status"] == "NO_STABLE_POLICY") & (selected["test_trades"] > 0)).any()


def test_summarize_oos_counts_profitable_folds_not_nonempty_folds():
    base = pd.DataFrame(
        {
            "entry_timestamp": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-04-01", "2026-04-02"]
            ),
            "wf_test_start": pd.to_datetime(
                ["2026-01-01", "2026-01-01", "2026-04-01", "2026-04-01"]
            ),
            "net_pnl_current": [7000.0, -2000.0, -5000.0, -5000.0],
            "target_hit": [True, False, False, False],
        }
    )
    out = summarize_oos(base)
    assert out["positive_test_folds"] == 1
    assert out["total_test_folds"] == 2
