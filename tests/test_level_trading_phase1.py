from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from src.level_trading.levels import (
    confirmed_swing_levels,
    opening_range_levels,
    previous_day_levels,
)
from src.level_trading.upstox_market_data import completed_candles, normalize_candles

IST = ZoneInfo("Asia/Kolkata")


def test_normalize_candles_sorts_and_deduplicates():
    rows = [
        ["2026-11-27T09:20:00+05:30", 2, 4, 1, 3, 20, 0],
        ["2026-11-27T09:15:00+05:30", 1, 3, 0, 2, 10, 0],
        ["2026-11-27T09:20:00+05:30", 2, 5, 1, 4, 30, 0],
    ]
    df = normalize_candles(rows)
    assert len(df) == 2
    assert df.iloc[0]["close"] == 2
    assert df.iloc[1]["close"] == 4


def test_completed_candles_excludes_forming_bar():
    df = normalize_candles([
        ["2026-11-27T09:15:00+05:30", 1, 2, 1, 2, 10, 0],
        ["2026-11-27T09:20:00+05:30", 2, 3, 2, 3, 10, 0],
    ])
    result = completed_candles(
        df, interval_minutes=5,
        as_of=datetime(2026, 11, 27, 9, 24, tzinfo=IST),
    )
    assert len(result) == 1


def test_previous_day_levels_use_strictly_prior_session():
    daily = pd.DataFrame([
        {"timestamp": pd.Timestamp("2026-11-26", tz=IST), "high": 100, "low": 90, "close": 95},
        {"timestamp": pd.Timestamp("2026-11-27", tz=IST), "high": 110, "low": 99, "close": 105},
    ])
    levels = previous_day_levels(daily, "2026-11-27")
    assert {x["name"]: x["price"] for x in levels} == {"PDH": 100, "PDL": 90, "PDC": 95}


def test_opening_range_waits_for_three_five_minute_bars():
    candles = pd.DataFrame([
        {"timestamp": pd.Timestamp(f"2026-11-27T09:{minute}:00+05:30"), "high": high, "low": low}
        for minute, high, low in [("15", 100, 95), ("20", 105, 94), ("25", 103, 93)]
    ])
    levels = opening_range_levels(candles, session_date="2026-11-27", minutes=15)
    assert {x["name"]: x["price"] for x in levels} == {"ORH": 105.0, "ORL": 93.0}


def test_swing_pivot_waits_for_right_confirmation_bars():
    candles = pd.DataFrame({
        "timestamp": pd.date_range("2026-11-27 09:15", periods=7, freq="5min", tz=IST),
        "high": [1, 2, 5, 3, 2, 2, 1],
        "low": [0, 1, 2, 1, 0, 1, 0],
    })
    levels = confirmed_swing_levels(candles, left=2, right=2)
    highs = [x for x in levels if x["name"] == "SWING_HIGH"]
    assert len(highs) == 1
    assert highs[0]["price"] == 5.0
    assert highs[0]["confirmed_at"] == candles.iloc[4]["timestamp"]
