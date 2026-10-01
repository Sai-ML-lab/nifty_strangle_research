from __future__ import annotations

import pandas as pd
import pytest


def test_frozen_config_rejects_parameter_change():
    from src.paper_trading import assert_frozen_config
    from src.core import StrategyConfig

    cfg = StrategyConfig(
        entry_mode="dte",
        target_dte=6,
        entry_time="10:00",
        sd_multiple=2.0,
        strike_method="pure_sd",
        profit_capture=0.75,
        stop_multiple=2.5,
        time_exit_mode="days_before_expiry",
        time_exit_dte=1,
        time_exit_time="15:00",
        min_iv_rv_spread=None,
        lots=1,
        min_entry_credit_points=1.0,
    )
    assert_frozen_config(cfg)
    with pytest.raises(ValueError):
        assert_frozen_config(
            StrategyConfig(
                entry_mode="dte",
                target_dte=6,
                entry_time="10:00",
                sd_multiple=1.75,
                strike_method="pure_sd",
                profit_capture=0.75,
                stop_multiple=2.5,
                time_exit_mode="days_before_expiry",
                time_exit_dte=1,
                time_exit_time="15:00",
                min_iv_rv_spread=None,
                lots=1,
                min_entry_credit_points=1.0,
            )
        )


def test_paper_signal_does_not_need_future_prices():
    from src.paper_trading import build_paper_signal
    from src.core import Costs, StrategyConfig

    expiry = pd.Timestamp("2026-10-15")
    as_of = pd.Timestamp("2026-10-09 10:00")
    strikes = [22000, 22250, 22500, 22750, 23000, 23250, 23500, 23750, 24000]
    rows = []
    for k in strikes:
        for ot in ["PE", "CE"]:
            rows.append({
                "timestamp": as_of,
                "expiry": expiry,
                "underlying": "NIFTY",
                "spot": 23000.0,
                "future": 23000.0,
                "strike": k,
                "option_type": ot,
                "open": 10.0 if ((ot == "PE" and k <= 22500) or (ot == "CE" and k >= 23500)) else 100.0,
                "ltp": 10.0 if ((ot == "PE" and k <= 22500) or (ot == "CE" and k >= 23500)) else 100.0,
                "bid": 9.5,
                "ask": 10.5,
                "iv": 0.15,
                "volume": 100,
                "oi": 100,
            })
    options = pd.DataFrame(rows)
    spot_dates = pd.date_range("2026-09-01", as_of.normalize(), freq="D")
    spot = pd.DataFrame({"timestamp": spot_dates, "close": 23000.0})
    cfg = StrategyConfig(
        entry_mode="dte",
        target_dte=6,
        entry_time="10:00",
        sd_multiple=2.0,
        strike_method="pure_sd",
        profit_capture=0.75,
        stop_multiple=2.5,
        time_exit_mode="days_before_expiry",
        time_exit_dte=1,
        time_exit_time="15:00",
        min_iv_rv_spread=None,
        lots=1,
        min_entry_credit_points=1.0,
    )
    signal = build_paper_signal(options, spot, expiry, as_of, cfg, Costs(slippage_points_per_leg=0.0))
    assert signal.loc[0, "signal_status"] == "READY"
    assert float(signal.loc[0, "planned_credit_points"]) > 0
    assert signal.loc[0, "profit_target_debit_points"] < signal.loc[0, "stop_debit_points"]


def test_paper_ledger_entry_and_exit_calculate_net_pnl():
    from src.paper_trading import record_entry_fill, record_exit_fill
    from src.core import Costs

    ledger = pd.DataFrame([{
        "strategy_id": "NIFTY_DTE6_2SD_PC75_STOP2_5_V1",
        "strategy_spec_hash": "dummy",
        "signal_status": "READY",
        "signal_id": "S1",
        "ledger_status": "SIGNAL",
        "lot_size": 65,
        "put_bid": 10.0,
        "call_bid": 10.0,
        "profit_target_debit_points": 5.0,
        "stop_debit_points": 50.0,
        "scheduled_time_exit_timestamp": "2026-10-14 15:00",
        "actual_entry_timestamp": pd.NaT,
        "actual_entry_credit_rupees": pd.NA,
    }])
    # record_entry_fill intentionally requires the frozen strategy hash.
    from src.paper_trading import FROZEN_SPEC_HASH
    ledger["strategy_spec_hash"] = FROZEN_SPEC_HASH

    ledger = record_entry_fill(ledger, "S1", pd.Timestamp("2026-10-09 10:00"), 10.0, 10.0)
    ledger = record_exit_fill(
        ledger, "S1", pd.Timestamp("2026-10-10 11:00"), 2.0, 2.0, "profit_target", Costs()
    )
    assert ledger.loc[0, "ledger_status"] == "CLOSED"
    assert ledger.loc[0, "gross_pnl"] == (20.0 - 4.0) * 65
    assert ledger.loc[0, "net_pnl"] < ledger.loc[0, "gross_pnl"]
