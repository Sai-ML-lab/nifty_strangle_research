from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any
from urllib.parse import quote
from zoneinfo import ZoneInfo

import pandas as pd
import requests

IST = ZoneInfo("Asia/Kolkata")
NIFTY_INDEX_KEY = "NSE_INDEX|Nifty 50"
UPSTOX_BASE_URL = "https://api.upstox.com"
CANDLE_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "oi"]


def _timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize(IST) if ts.tzinfo is None else ts.tz_convert(IST)


def normalize_candles(rows: list[list[Any]]) -> pd.DataFrame:
    records = []
    for row in rows:
        if len(row) < 6:
            continue
        records.append({
            "timestamp": _timestamp(row[0]),
            "open": float(row[1]), "high": float(row[2]),
            "low": float(row[3]), "close": float(row[4]),
            "volume": float(row[5] or 0),
            "oi": float(row[6]) if len(row) > 6 and row[6] is not None else None,
        })
    if not records:
        return pd.DataFrame(columns=CANDLE_COLUMNS)
    return (pd.DataFrame(records)
            .drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp").reset_index(drop=True))


def completed_candles(candles: pd.DataFrame, *, interval_minutes: int,
                      as_of: datetime | pd.Timestamp | None = None) -> pd.DataFrame:
    """Upstox timestamps candles by start time; discard bars not yet fully closed."""
    if candles.empty:
        return candles.copy()
    now = _timestamp(as_of or datetime.now(IST))
    end_times = candles["timestamp"] + pd.Timedelta(minutes=int(interval_minutes))
    return candles.loc[end_times <= now].copy().reset_index(drop=True)


class UpstoxMarketDataClient:
    """Read-only market-data client. Deliberately exposes no order-placement methods."""

    def __init__(self, token: str, *, base_url: str = UPSTOX_BASE_URL,
                 timeout_seconds: float = 15.0, session: requests.Session | None = None,
                 instrument_key: str = NIFTY_INDEX_KEY):
        if not str(token).strip():
            raise ValueError("UPSTOX_ANALYTICS_TOKEN is required")
        self.base_url = base_url.rstrip("/")
        self.instrument_key = instrument_key
        self.timeout_seconds = float(timeout_seconds)
        self.session = session or requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {str(token).strip()}",
        })

    def _get(self, path: str) -> dict[str, Any]:
        response = self.session.get(
            f"{self.base_url}{path}", timeout=self.timeout_seconds
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") not in {None, "success"}:
            raise RuntimeError(f"Upstox API returned status={payload.get('status')}")
        return payload

    def fetch_intraday_candles(self, *, interval_minutes: int = 5,
                               as_of: datetime | pd.Timestamp | None = None,
                               only_completed: bool = True) -> pd.DataFrame:
        if not 1 <= int(interval_minutes) <= 300:
            raise ValueError("interval_minutes must be between 1 and 300")
        instrument = quote(self.instrument_key, safe="")
        path = f"/v3/historical-candle/intraday/{instrument}/minutes/{int(interval_minutes)}"
        payload = self._get(path)
        candles = normalize_candles(payload.get("data", {}).get("candles") or [])
        if only_completed:
            candles = completed_candles(
                candles, interval_minutes=interval_minutes, as_of=as_of
            )
        return candles

    def fetch_daily_candles(self, *, to_date: date | str | None = None,
                            lookback_days: int = 45) -> pd.DataFrame:
        end = pd.Timestamp(to_date or datetime.now(IST).date()).date()
        start = end - timedelta(days=int(lookback_days))
        instrument = quote(self.instrument_key, safe="")
        path = f"/v3/historical-candle/{instrument}/days/1/{end:%Y-%m-%d}/{start:%Y-%m-%d}"
        payload = self._get(path)
        return normalize_candles(payload.get("data", {}).get("candles") or [])


def save_candles(candles: pd.DataFrame, path: str) -> None:
    from pathlib import Path
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    candles.to_parquet(temporary, index=False)
    temporary.replace(destination)


def load_candles(path: str) -> pd.DataFrame:
    from pathlib import Path
    source = Path(path)
    return pd.read_parquet(source) if source.exists() else pd.DataFrame(columns=CANDLE_COLUMNS)
