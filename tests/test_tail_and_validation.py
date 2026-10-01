from __future__ import annotations

import pandas as pd


def test_loss_to_credit_uses_rupee_credit():
    from tools_tail_analysis import analyze_tails
    t = pd.DataFrame({
        "entry_timestamp": pd.to_datetime(["2026-01-01"]),
        "exit_timestamp": pd.to_datetime(["2026-01-02"]),
        "initial_credit_points": [100.0],
        "lot_size": [65],
        "net_pnl": [-6500.0],
        "exit_reason": ["stop"],
    })
    x, _ = analyze_tails(t)
    assert x.iloc[0]["loss_to_credit"] == 1.0


def test_independent_compare_uses_entry_date():
    from tools_independent_validation import compare_ledgers
    a = pd.DataFrame({
        "entry_timestamp":["2026-01-01 10:00"],"expiry":["2026-01-07"],
        "put_strike":[23000],"call_strike":[25000],"net_pnl":[100],
    })
    b = pd.DataFrame({
        "entry_timestamp":["2026-01-01 10:01"],"expiry":["2026-01-07"],
        "put_strike":[23000],"call_strike":[25000],"net_pnl":[90],
    })
    _, s = compare_ledgers(a,b)
    assert s.iloc[0]["matched_trades"] == 1
