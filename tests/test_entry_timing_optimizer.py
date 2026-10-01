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
    assert out.loc[0, "target_band_hit_5lot"] is True
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
