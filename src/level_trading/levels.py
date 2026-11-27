from __future__ import annotations

from typing import Any
import pandas as pd


def previous_day_levels(daily_candles: pd.DataFrame,
                        session_date: str | pd.Timestamp) -> list[dict[str, Any]]:
    """Use the most recent daily candle strictly before the current session."""
    if daily_candles.empty:
        return []
    df = daily_candles.copy()
    df["session_date"] = pd.to_datetime(df["timestamp"]).dt.date
    target = pd.Timestamp(session_date).date()
    prior = df.loc[df["session_date"] < target].sort_values("timestamp")
    if prior.empty:
        return []
    row = prior.iloc[-1]
    return [
        {"name": "PDH", "price": float(row["high"]), "side": "RESISTANCE", "source": "PREVIOUS_DAY"},
        {"name": "PDL", "price": float(row["low"]), "side": "SUPPORT", "source": "PREVIOUS_DAY"},
        {"name": "PDC", "price": float(row["close"]), "side": "REFERENCE", "source": "PREVIOUS_DAY"},
    ]


def opening_range_levels(candles: pd.DataFrame, *,
                         session_date: str | pd.Timestamp,
                         minutes: int = 15) -> list[dict[str, Any]]:
    """Calculate ORH/ORL after all three 5-minute bars from 09:15 to 09:30 close."""
    if candles.empty or minutes % 5:
        return []
    df = candles.copy()
    ts = pd.to_datetime(df["timestamp"])
    df["timestamp"] = ts.dt.tz_localize("Asia/Kolkata") if ts.dt.tz is None else ts.dt.tz_convert("Asia/Kolkata")
    target = pd.Timestamp(session_date).date()
    start = pd.Timestamp(f"{target} 09:15:00", tz="Asia/Kolkata")
    end = start + pd.Timedelta(minutes=minutes)
    opening = df.loc[(df["timestamp"] >= start) & (df["timestamp"] < end)]
    if len(opening) < minutes // 5:
        return []
    return [
        {"name": "ORH", "price": float(opening["high"].max()), "side": "RESISTANCE", "source": "OPENING_RANGE"},
        {"name": "ORL", "price": float(opening["low"].min()), "side": "SUPPORT", "source": "OPENING_RANGE"},
    ]


def confirmed_swing_levels(candles: pd.DataFrame, *,
                           left: int = 3, right: int = 3) -> list[dict[str, Any]]:
    """Emit a pivot only after its right-side confirmation bars have closed."""
    if left < 1 or right < 1:
        raise ValueError("left and right must be >= 1")
    if len(candles) < left + right + 1:
        return []
    df = candles.sort_values("timestamp").reset_index(drop=True)
    highs = pd.to_numeric(df["high"], errors="coerce")
    lows = pd.to_numeric(df["low"], errors="coerce")
    out = []
    for i in range(left, len(df) - right):
        pivot_ts, confirmed_ts = df.iloc[i]["timestamp"], df.iloc[i + right]["timestamp"]
        if highs.iloc[i] > highs.iloc[i-left:i].max() and highs.iloc[i] >= highs.iloc[i+1:i+right+1].max():
            out.append({"name": "SWING_HIGH", "price": float(highs.iloc[i]), "side": "RESISTANCE",
                        "source": "SWING", "timestamp": pivot_ts, "confirmed_at": confirmed_ts})
        if lows.iloc[i] < lows.iloc[i-left:i].min() and lows.iloc[i] <= lows.iloc[i+1:i+right+1].min():
            out.append({"name": "SWING_LOW", "price": float(lows.iloc[i]), "side": "SUPPORT",
                        "source": "SWING", "timestamp": pivot_ts, "confirmed_at": confirmed_ts})
    return out


def build_level_frame(*, daily_candles: pd.DataFrame,
                      intraday_candles: pd.DataFrame,
                      session_date: str | pd.Timestamp,
                      manual_levels: list[dict[str, Any]] | None = None) -> pd.DataFrame:
    levels = previous_day_levels(daily_candles, session_date)
    levels += opening_range_levels(intraday_candles, session_date=session_date)
    levels += confirmed_swing_levels(intraday_candles)
    levels += manual_levels or []
    return pd.DataFrame(levels) if levels else pd.DataFrame(
        columns=["name", "price", "side", "source", "timestamp", "confirmed_at"]
    )
