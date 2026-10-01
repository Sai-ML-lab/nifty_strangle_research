from __future__ import annotations

import pandas as pd


def test_entry_optimizer_5lot_scaling_and_target_band():
    from src.core import Costs
    from src.entry_timing_optimizer import add_lot_metrics

    trades = pd.DataFrame(
        [
            {
                "initial_credit_points": 30.0,
                "exit_debit_points": 7.5,
                "profit_capture": 0.75,
                "lot_size": 65,
                "entry_timestamp": pd.Timestamp("2026-10-07 10:00"),
            },
            {
                "initial_credit_points": 30.0,
                "exit_debit_points": 75.0,
                "profit_capture": 0.75,
                "lot_size": 65,
                "entry_timestamp": pd.Timestamp("2026-10-07 10:00"),
            },
        ]
    )
    out = add_lot_metrics(
        trades,
        Costs(slippage_points_per_leg=0.50),
        lots=5,
        target_low=6000,
        target_high=8000,
    )
    assert len(out) == 2
    assert out.loc[0, "net_pnl_5lot"] > 6000
    assert out.loc[0, "net_pnl_5lot_current"] > 6000
    assert bool(out.loc[0, "target_band_hit_5lot"]) is True
    assert bool(out.loc[0, "target_band_hit_5lot_current"]) is True
    assert out.loc[1, "net_pnl_5lot"] < 0


def test_entry_optimizer_weekday_policy_is_exclusion():
    from src.entry_timing_optimizer import apply_policy

    x = pd.DataFrame(
        {
            "entry_weekday": ["Tuesday", "Wednesday", "Friday"],
            "net_pnl_5lot": [1.0, 2.0, 3.0],
        }
    )
    out = apply_policy(x, frozenset({"Tuesday", "Friday"}))
    assert out["entry_weekday"].tolist() == ["Wednesday"]


def test_entry_optimizer_merges_existing_vendor_spot_with_reference():
    from src.entry_timing_optimizer import _merge_spot_reference

    options = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-10-01 10:00", "2026-10-01 10:01"]),
            "spot": [22000.0, float("nan")],
        }
    )
    reference = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-10-01 10:00", "2026-10-01 10:01"]),
            "spot": [21999.0, 21998.0],
        }
    )
    out = _merge_spot_reference(options, reference)
    assert "spot_spot_ref" not in out.columns
    assert out["spot"].tolist() == [22000.0, 21998.0]


def test_entry_optimizer_current_size_target_band_ignores_historical_lot_size():
    from src.core import Costs
    from src.entry_timing_optimizer import add_lot_metrics

    trades = pd.DataFrame(
        [
            {
                "initial_credit_points": 30.0,
                "exit_debit_points": 7.5,
                "profit_capture": 0.75,
                "lot_size": 25,
                "entry_timestamp": pd.Timestamp("2024-10-07 10:00"),
            }
        ]
    )
    out = add_lot_metrics(
        trades,
        Costs(slippage_points_per_leg=0.50),
        lots=5,
        target_low=6000,
        target_high=8000,
        reference_lot_size=65,
    )
    assert bool(out.loc[0, "target_band_hit_5lot_current"]) is True


def test_entry_optimizer_empty_policy_result_is_schema_safe():
    from src.entry_timing_optimizer import evaluate_policies, POLICY_COLUMNS

    x = pd.DataFrame(
        {
            "dte": [6],
            "sd": [2.0],
            "profit_capture": [0.75],
            "stop_multiple": [2.5],
            "entry_timestamp": pd.to_datetime(["2026-10-01 10:00"]),
            "entry_weekday": ["Thursday"],
            "net_pnl_5lot_current": [7000.0],
            "target_band_hit_5lot_current": [True],
        }
    )
    out = evaluate_policies(
        x,
        min_trades=40,
        pnl_column="net_pnl_5lot_current",
        target_column="target_band_hit_5lot_current",
    )
    assert out.empty
    assert list(out.columns) == POLICY_COLUMNS


def test_entry_optimizer_targetable_rows_are_not_double_counted_in_explanation_columns():
    from src.core import Costs
    from src.entry_timing_optimizer import add_lot_metrics

    trades = pd.DataFrame(
        [
            {
                "initial_credit_points": 30.0,
                "exit_debit_points": 7.5,
                "profit_capture": 0.75,
                "lot_size": 25,
                "entry_timestamp": pd.Timestamp("2024-10-07 10:00"),
            },
            {
                "initial_credit_points": 20.0,
                "exit_debit_points": 5.0,
                "profit_capture": 0.75,
                "lot_size": 25,
                "entry_timestamp": pd.Timestamp("2024-10-08 10:00"),
            },
        ]
    )
    out = add_lot_metrics(
        trades,
        Costs(slippage_points_per_leg=0.50),
        lots=5,
        reference_lot_size=65,
        target_low=6000,
        target_high=8000,
    )
    assert "target_band_hit_5lot_current" in out.columns
    assert out["target_band_hit_5lot_current"].dtype == bool


def test_entry_optimizer_empty_walk_forward_selection_preserves_schema():
    from src.entry_timing_optimizer import walk_forward_select, WALK_FORWARD_SELECTION_COLUMNS

    x = pd.DataFrame(
        {
            "entry_timestamp": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "dte": [6, 6],
            "sd": [2.0, 2.0],
            "profit_capture": [0.75, 0.75],
            "stop_multiple": [2.5, 2.5],
            "entry_weekday": ["Thursday", "Friday"],
            "net_pnl_5lot_current": [7000.0, 7000.0],
            "target_band_hit_5lot_current": [True, True],
        }
    )
    selected, oos = walk_forward_select(
        x,
        train_months=12,
        test_months=3,
        rebalance_months=3,
        min_train_trades=25,
    )
    assert selected.empty
    assert list(selected.columns) == WALK_FORWARD_SELECTION_COLUMNS
    assert oos.empty
