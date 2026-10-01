import pandas as pd


def test_condor_payoff_is_bounded():
    from src.iron_condor import _condor_intrinsic

    values = [
        _condor_intrinsic(s, 23000, 23500, 24500, 25000)
        for s in [22000, 23000, 23500, 24000, 24500, 25000, 26000]
    ]
    assert all(0 <= x <= 500 for x in values)
    assert max(values) == 500


def test_condor_wing_selection_is_at_least_requested_width():
    from src.iron_condor import _pick_wing

    chain = pd.DataFrame(
        {
            "strike": [22000, 22500, 23000, 23500, 24500, 25000, 25500],
            "option_type": ["PE"] * 4 + ["CE"] * 3,
        }
    )
    put_wing, put_width = _pick_wing(chain, 23500, "PE", 500)
    call_wing, call_width = _pick_wing(chain, 24500, "CE", 500)
    assert put_wing == 23000
    assert put_width == 500
    assert call_wing == 25000
    assert call_width == 500


def test_condor_intrinsic_is_never_above_wing_width():
    from src.iron_condor import _condor_intrinsic
    assert _condor_intrinsic(22000, 23000, 23500, 24500, 25000) == 500
    assert _condor_intrinsic(26000, 23000, 23500, 24500, 25000) == 500
from __future__ import annotations

import pandas as pd


def test_tail_analysis_loss_buckets():
    from tools_tail_analysis import analyze_tails
    t = pd.DataFrame({
        "entry_timestamp": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        "exit_timestamp": pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-04"]),
        "entry_spot": [24000,24000,24000],
        "exit_spot": [24000,24500,23000],
        "initial_credit_points": [100,100,100],
        "net_pnl": [50,-150,-350],
        "gross_pnl": [60,-140,-340],
        "transaction_cost": [10,10,10],
        "exit_reason": ["profit_target","stop","stop"],
    })
    x, s = analyze_tails(t)
    assert set(s["section"]) == {"exit_reason","loss_to_credit_bucket"}
    assert x.loc[x["net_pnl"] == -350, "loss_to_credit"].iloc[0] == 3.5
from __future__ import annotations

import pandas as pd


def test_independent_compare_matches_by_trade_keys():
    from tools_independent_validation import compare_ledgers
    a = pd.DataFrame({"entry_timestamp":["2026-01-01"],"expiry":["2026-01-07"],"put_strike":[23000],"call_strike":[25000],"net_pnl":[100]})
    b = pd.DataFrame({"entry_timestamp":["2026-01-01"],"expiry":["2026-01-07"],"put_strike":[23000],"call_strike":[25000],"net_pnl":[90]})
    d, s = compare_ledgers(a,b)
    assert s.iloc[0]["matched_trades"] == 1
    assert s.iloc[0]["mean_abs_pnl_diff"] == 10

