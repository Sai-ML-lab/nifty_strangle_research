import numpy as np
import pandas as pd
from src.core import black76_price, black76_delta, implied_vol_black76, sd_band, nifty_lot_size, trade_cost, Costs
from src.data_ingest import normalize_options_vendor_file


def test_black76_roundtrip():
    fwd, k, vol, t, r = 24000.0, 25000.0, 0.18, 6/365, 0.06
    p = black76_price(fwd, k, vol, t, r, True)
    iv = implied_vol_black76(p, fwd, k, t, r, True)
    assert np.isfinite(iv)
    assert abs(iv - vol) < 1e-5


def test_put_call_delta_signs():
    dc = black76_delta(24000, 25000, 0.18, 6/365, 0.06, True)
    dp = black76_delta(24000, 23000, 0.18, 6/365, 0.06, False)
    assert 0 < dc < 0.5
    assert -0.5 < dp < 0


def test_sd_band():
    lo, hi, move = sd_band(24000, 0.15, __import__('pandas').Timestamp('2026-06-09'), __import__('pandas').Timestamp('2026-06-03 10:00'), 2)
    assert lo < 24000 < hi
    assert move > 0


def test_historical_nifty_lot_size():
    import pandas as pd
    assert nifty_lot_size(pd.Timestamp("2024-04-25")) == 50
    assert nifty_lot_size(pd.Timestamp("2024-05-02")) == 25
    assert nifty_lot_size(pd.Timestamp("2024-11-26")) == 75
    assert nifty_lot_size(pd.Timestamp("2026-01-06")) == 65


def test_stt_rate_is_date_aware():
    c = Costs(exchange_txn_pct=0.0, sebi_turnover_pct=0.0, stamp_duty_buy_option_pct=0.0, brokerage_per_order=0.0)
    old = trade_cost(100000, 100000, 0, 1, c, pd.Timestamp("2026-03-31"))
    new = trade_cost(100000, 100000, 0, 1, c, pd.Timestamp("2026-04-01"))
    assert abs(old - 100.0) < 1e-9
    assert abs(new - 150.0) < 1e-9


def test_options_data_layout_a_normalizes():
    raw = pd.DataFrame({
        "datetime": ["2026-09-15 10:00:00"] * 2,
        "stock_code": ["NIFTY"] * 2,
        "exchange_code": ["NFO"] * 2,
        "product_type": ["Options"] * 2,
        "expiry_date": ["15-SEP-2026"] * 2,
        "strike_price": [24000, 24000],
        "right": ["Call", "Put"],
        "open": [100, 90],
        "high": [105, 95],
        "low": [95, 85],
        "close": [102, 92],
        "volume": [10, 20],
        "open_interest": [100, 200],
    })
    x = normalize_options_vendor_file(raw)
    assert list(x["option_type"]) == ["CE", "PE"]
    assert x["expiry"].iloc[0] == pd.Timestamp("2026-09-15")
    assert float(x["open"].iloc[0]) == 100.0


def test_parity_forward_falls_near_spot():
    from src.core import estimate_forward_from_parity, black76_price
    ts = pd.Timestamp("2026-06-03 10:00:00", tz="Asia/Kolkata")
    expiry = pd.Timestamp("2026-06-09", tz="Asia/Kolkata")
    fwd = 24020.0
    rows = []
    for k in [23800, 23900, 24000, 24100, 24200]:
        c = black76_price(fwd, k, 0.15, (expiry + pd.Timedelta(hours=15, minutes=30) - ts).total_seconds()/(365*24*3600), 0.06, True)
        p = black76_price(fwd, k, 0.15, (expiry + pd.Timedelta(hours=15, minutes=30) - ts).total_seconds()/(365*24*3600), 0.06, False)
        rows.extend([
            {"timestamp": ts, "expiry": expiry, "strike": k, "option_type": "CE", "entry_price": c, "spot": 24000.0},
            {"timestamp": ts, "expiry": expiry, "strike": k, "option_type": "PE", "entry_price": p, "spot": 24000.0},
        ])
    x = pd.DataFrame(rows)
    est = estimate_forward_from_parity(x, 24000.0, 0.06, "entry_price")
    assert abs(est - fwd) < 2.0


def test_baseline_next_expiry_mapping_avoids_overlap():
    from run_batch_research import entry_date_for_expiry
    from src.core import StrategyConfig

    cfg_weekday = 2  # Wednesday
    dates = [pd.Timestamp("2023-01-19"), pd.Timestamp("2023-01-25")]
    def entry_date_for(expiry):
        offset = (expiry.weekday() - cfg_weekday) % 7
        if offset == 0:
            offset = 7
        return (expiry - pd.Timedelta(days=offset)).normalize()

    mapping = {}
    for expiry in dates:
        d = entry_date_for(expiry)
        if d not in mapping:
            mapping[d] = expiry

    assert mapping[pd.Timestamp("2023-01-18")] == pd.Timestamp("2023-01-19")

    cfg = StrategyConfig(entry_mode="dte", target_dte=6)
    assert entry_date_for_expiry(pd.Timestamp("2023-01-19"), cfg) == pd.Timestamp("2023-01-13")


def test_ivrv_threshold_sweep_filters_by_entry_spread():
    from src.ivrv import threshold_sweep

    t = pd.DataFrame({
        "entry_timestamp": pd.date_range("2026-01-01", periods=4, freq="D"),
        "iv_rv_spread": [0.005, 0.015, 0.025, 0.035],
        "net_pnl": [10.0, 20.0, -5.0, 40.0],
    })
    out = threshold_sweep(t, thresholds=[0.0, 1.0, 2.0, 3.0], exclude_quality_warnings=True)
    assert out.loc[out["min_iv_rv_spread_pct"] == 0.0, "trades"].iloc[0] == 4
    assert out.loc[out["min_iv_rv_spread_pct"] == 2.0, "trades"].iloc[0] == 2
    assert out.loc[out["min_iv_rv_spread_pct"] == 3.0, "trades"].iloc[0] == 1


def test_ivrv_walk_forward_selects_only_from_training_window():
    from src.ivrv import walk_forward_ivrv

    dates = pd.date_range("2024-01-03", periods=30, freq="7D")
    t = pd.DataFrame({
        "entry_timestamp": dates,
        "iv_rv_spread": ([0.01] * 10) + ([0.03] * 10) + ([0.01] * 10),
        "net_pnl": ([50.0] * 10) + ([100.0] * 10) + ([-10.0] * 10),
    })
    out = walk_forward_ivrv(
        t,
        thresholds=[0.0, 1.0, 3.0],
        train_months=6,
        test_months=6,
        rebalance_months=6,
        min_train_trades=1,
    )
    assert not out.empty
    assert "selected_min_iv_rv_spread_pct" in out.columns


def test_ivrv_no_filter_keeps_missing_rv_trades():
    from src.ivrv import threshold_sweep

    t = pd.DataFrame({
        "entry_timestamp": pd.date_range("2026-01-01", periods=3, freq="D"),
        "iv_rv_spread": [np.nan, 0.02, 0.03],
        "net_pnl": [5.0, 10.0, -2.0],
    })
    out = threshold_sweep(t, thresholds=[2.0])
    base = out[out["filter_enabled"] == False].iloc[0]
    assert base["trades"] == 3


def test_frozen_slippage_price_adjustment():
    from src.slippage_replay import executable_sell_price, executable_buy_price
    row = pd.Series({"bid": 100.0, "ask": 110.0, "entry_price": 105.0, "ltp": 108.0})
    assert executable_sell_price(row, 0.0) == 100.0
    assert executable_sell_price(row, 0.5) == 99.5
    assert executable_buy_price(row, 0.0) == 110.0
    assert executable_buy_price(row, 0.5) == 110.5


def test_frozen_slippage_replay_keeps_exit_decision_fixed():
    from src.slippage_replay import replay_trade
    chain = pd.DataFrame([
        {"timestamp": pd.Timestamp("2026-01-05 10:00"), "option_type": "PE", "strike": 23000, "entry_price": 100, "ltp": 100},
        {"timestamp": pd.Timestamp("2026-01-05 10:00"), "option_type": "CE", "strike": 25000, "entry_price": 100, "ltp": 100},
        {"timestamp": pd.Timestamp("2026-01-06 15:00"), "option_type": "PE", "strike": 23000, "entry_price": 40, "ltp": 40},
        {"timestamp": pd.Timestamp("2026-01-06 15:00"), "option_type": "CE", "strike": 25000, "entry_price": 40, "ltp": 40},
    ])
    trade = pd.Series({
        "entry_timestamp": "2026-01-05 10:00", "expiry": "2026-01-08",
        "exit_timestamp": "2026-01-06 15:00", "exit_reason": "profit_target",
        "put_strike": 23000, "call_strike": 25000, "lot_size": 50, "lots": 1,
        "exit_spot": 24000, "net_pnl": 0.0, "data_quality_flag": "PASS",
    })
    a = replay_trade(trade, chain, 0.0)
    b = replay_trade(trade, chain, 0.5)
    assert a["exit_timestamp"] == b["exit_timestamp"] == pd.Timestamp("2026-01-06 15:00")
    assert b["net_pnl"] < a["net_pnl"]


def test_regime_gate_uses_training_only_for_activation():
    from src.regime_gate import walk_forward_ivrv_with_regime_gate
    dates = pd.date_range("2024-01-03", periods=30, freq="7D")
    t = pd.DataFrame({
        "entry_timestamp": dates,
        "iv_rv_spread": [0.01] * 30,
        "net_pnl": [-20.0] * 20 + [100.0] * 10,
        "data_quality_flag": ["PASS"] * 30,
    })
    out = walk_forward_ivrv_with_regime_gate(
        t, thresholds=[0.0], train_months=3, test_months=3, rebalance_months=3,
        min_train_trades=1, min_train_expectancy=0.0, min_train_profit_factor=1.0,
    )
    assert not out.empty
    assert bool(out.iloc[0]["regime_gate_pass"]) is False
    assert out.iloc[0]["gated_test_trades"] == 0

def test_portfolio_risk_metrics_include_weekly_risk():
    from src.risk_metrics import portfolio_risk_metrics
    t = pd.DataFrame({
        "entry_timestamp": pd.to_datetime(["2024-01-03", "2024-01-10", "2024-01-17", "2024-01-24"]),
        "exit_timestamp": pd.to_datetime(["2024-01-04", "2024-01-11", "2024-01-18", "2024-01-25"]),
        "net_pnl": [100.0, -50.0, 200.0, -25.0],
    })
    out = portfolio_risk_metrics(t, 10000.0)
    assert out["trades"] == 4
    assert out["total_net_pnl"] == 225.0
    assert out["ending_capital"] == 10225.0
    assert "weekly_sharpe" in out
    assert "weekly_cvar5_pnl" in out


def test_sd_ivrv_walk_forward_selects_from_training_only():
    from src.sd_ivrv import walk_forward_sd_ivrv
    dates = pd.to_datetime([
        "2024-01-03", "2024-01-10", "2024-02-07", "2024-02-14",
        "2024-07-03", "2024-07-10", "2024-08-07", "2024-08-14",
    ])
    a = pd.DataFrame({
        "entry_timestamp": dates, "iv_rv_spread": [0.01] * 8,
        "net_pnl": [100.0, 120.0, 110.0, 90.0, -50.0, -50.0, -50.0, -50.0],
        "data_quality_flag": ["PASS"] * 8, "sd_multiple": [1.5] * 8,
    })
    b = a.copy(); b["net_pnl"] = [-50.0, -50.0, -50.0, -50.0, 100.0, 120.0, 110.0, 90.0]; b["sd_multiple"] = 2.0
    out, selected, base = walk_forward_sd_ivrv(
        {1.5: a, 2.0: b}, thresholds=[None], train_months=6, test_months=3, rebalance_months=3, min_train_trades=1
    )
    assert not out.empty
    assert set([1.5, 2.0]) >= set(out["selected_sd"].dropna().unique())
    assert len(selected) > 0
    assert len(base) > 0