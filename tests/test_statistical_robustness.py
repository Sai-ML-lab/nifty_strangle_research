from __future__ import annotations

import pandas as pd


def test_robustness_report_detects_positive_bootstrap_mean():
    from tools_statistical_robustness import robustness_report
    t = pd.DataFrame({
        "entry_timestamp": pd.date_range("2026-01-02", periods=12, freq="7D"),
        "net_pnl": [100.0] * 12,
        "exit_reason": ["profit_target"] * 12,
    })
    out = robustness_report(t, seed=1, n_boot=1000)
    row = out[out["metric"] == "trade_mean_pnl"].iloc[0]
    assert row["estimate"] == 100.0
    assert row["ci_low_2_5"] == 100.0
    assert row["probability_positive_bootstrap"] == 1.0
