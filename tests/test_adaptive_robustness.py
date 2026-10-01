from __future__ import annotations

import numpy as np
import pandas as pd

from src.adaptive_robustness import (
    apply_baseline_selection,
    grouped_oos_stats,
    pair_adaptive_with_control,
    pnl_stats,
    winner_concentration,
)


def _sample_oos() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_timestamp": pd.to_datetime(
                [
                    "2025-01-02 10:00",
                    "2025-01-09 10:00",
                    "2025-01-16 10:00",
                    "2025-01-23 10:00",
                    "2025-01-30 10:00",
                ]
            ),
            "expiry": pd.to_datetime(
                [
                    "2025-01-08",
                    "2025-01-15",
                    "2025-01-22",
                    "2025-01-29",
                    "2025-02-05",
                ]
            ),
            "session_offset": [6, 6, 5, 5, 2],
            "sd": [1.5, 1.5, 1.5, 1.75, 1.5],
            "target_rupees": [6000.0, 6000.0, 7000.0, 7000.0, 8000.0],
            "stop_multiple": [2.5, 2.5, 2.5, 2.0, 2.5],
            "net_pnl_current": [7000.0, -12000.0, 6500.0, -18000.0, 8000.0],
            "target_hit": [True, False, True, False, True],
        }
    )


def test_pnl_stats_and_winner_concentration():
    x = _sample_oos()
    stats = pnl_stats(x)
    assert stats["trades"] == 5
    assert np.isclose(stats["win_rate"], 0.6)
    assert np.isclose(stats["expectancy"], -1700.0)

    c = winner_concentration(x, removals=[1, 3])
    assert c["removed_top_winners"].tolist() == [1, 3]
    assert np.isclose(c.loc[0, "total_pnl_remaining"], -16500.0)
    assert c.loc[1, "trades_remaining"] == 2


def test_grouped_oos_stats():
    x = _sample_oos()
    out = grouped_oos_stats(x, ["target_rupees"])
    assert out["target_rupees"].tolist() == [6000.0, 7000.0, 8000.0]
    assert out.loc[out["target_rupees"] == 6000.0, "trades"].item() == 2


def test_baseline_selection_applies_each_fold():
    x = _sample_oos()
    selected = pd.DataFrame(
        {
            "train_start": pd.to_datetime(["2024-01-01", "2024-04-01"]),
            "test_start": pd.to_datetime(["2025-01-01", "2025-01-16"]),
            "test_end": pd.to_datetime(["2025-01-15", "2025-01-31"]),
            "selected_session_offset": [6, 5],
            "selected_sd": [1.5, 1.5],
            "selected_target_rupees": [6000.0, 7000.0],
            "selected_stop_multiple": [2.5, 2.5],
            "selection_status": ["SELECTED", "SELECTED"],
        }
    )
    out = apply_baseline_selection(x, selected)
    assert len(out) == 3
    assert out["wf_test_start"].dt.strftime("%Y-%m-%d").tolist() == [
        "2025-01-01",
        "2025-01-01",
        "2025-01-16",
    ]


def test_pair_adaptive_with_control_reports_date_and_expiry_matches():
    adaptive = _sample_oos().iloc[[0, 1, 2]].copy()
    adaptive["put_strike"] = [24000.0, 24100.0, 24200.0]
    adaptive["call_strike"] = [26000.0, 26100.0, 26200.0]

    control = pd.DataFrame(
        {
            "entry_timestamp": pd.to_datetime(
                ["2025-01-02 10:00", "2025-01-09 10:00", "2025-01-16 10:00"]
            ),
            "expiry": pd.to_datetime(
                ["2025-01-08", "2025-01-20", "2025-01-22"]
            ),
            "net_pnl_5lot_current": [5000.0, -10000.0, 4000.0],
            "put_strike": [23900.0, 24000.0, 24100.0],
            "call_strike": [26100.0, 26200.0, 26300.0],
        }
    )
    merged, summary = pair_adaptive_with_control(adaptive, control)
    assert len(merged) == 3
    assert summary["matched_dates"] == 3
    assert summary["same_expiry_matches"] == 2
    assert np.isclose(summary["mean_pnl_delta"], 833.3333333333)


def test_pair_without_matches_is_schema_safe():
    adaptive = _sample_oos().iloc[[0]].copy()
    control = pd.DataFrame(
        {
            "entry_timestamp": pd.to_datetime(["2025-02-06 10:00"]),
            "expiry": pd.to_datetime(["2025-02-12"]),
            "net_pnl_5lot_current": [1000.0],
            "put_strike": [23000.0],
            "call_strike": [27000.0],
        }
    )
    merged, summary = pair_adaptive_with_control(adaptive, control)
    assert len(merged) == 1
    assert summary["matched_dates"] == 0
    assert np.isclose(summary["date_match_rate"], 0.0)
