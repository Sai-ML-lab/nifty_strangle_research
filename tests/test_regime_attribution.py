from __future__ import annotations

import pandas as pd


def test_regime_attribution_uses_fixed_diagnostic_buckets():
    from src.regime_attribution import enrich_regime_features, summarize_dimensions

    t = pd.DataFrame({
        "entry_timestamp": pd.date_range("2025-01-03", periods=4, freq="90D"),
        "exit_timestamp": pd.date_range("2025-01-06", periods=4, freq="90D"),
        "entry_spot": [23000, 23000, 23000, 23000],
        "exit_spot": [23100, 22500, 23500, 23050],
        "rv20": [0.08, 0.12, 0.18, 0.26],
        "iv_rv_spread": [-0.01, 0.005, 0.015, 0.035],
        "initial_credit_points": [2.0, 4.0, 6.0, 14.0],
        "initial_credit_rupees": [100.0, 200.0, 300.0, 700.0],
        "net_pnl": [20.0, -100.0, 60.0, -300.0],
        "exit_reason": ["profit_target", "stop", "profit_target", "stop"],
    })
    x = enrich_regime_features(t)
    assert list(x["iv_rv_bucket"]) == ["<0", "0-1", "1-2", "3+"]
    assert list(x["rv20_bucket"]) == ["<10%", "10-15%", "15-20%", "25%+"]
    assert list(x["credit_bucket"]) == ["<3", "3-5", "5-8", "12+"]
    assert float(x.loc[1, "loss_to_credit_multiple"]) == 0.5
    s = summarize_dimensions(x)
    assert {"dimension", "bucket", "trades", "total_net_pnl"}.issubset(s.columns)


def test_spot_diagnostics_compute_entry_gap():
    from src.regime_attribution import add_spot_diagnostics

    dates = pd.date_range("2026-01-01 09:15", periods=6, freq="D")
    spot = pd.DataFrame({"timestamp": dates, "close": [100, 101, 100, 102, 101, 103]})
    trades = pd.DataFrame({
        "entry_timestamp": [dates[2]],
        "exit_timestamp": [dates[4]],
        "entry_spot": [100.0],
        "exit_spot": [101.0],
        "net_pnl": [10.0],
    })
    x = add_spot_diagnostics(trades, spot)
    assert round(float(x.loc[0, "entry_gap_abs_pct"]), 6) == 0.990099
