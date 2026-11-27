from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.level_trading.upstox_market_data import UpstoxMarketDataClient, save_candles

IST = ZoneInfo("Asia/Kolkata")
DATA_DIR = Path("data/level_trading")


def run(*, once: bool, poll_seconds: int) -> None:
    token = os.getenv("UPSTOX_ANALYTICS_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Set UPSTOX_ANALYTICS_TOKEN in your local environment")
    client = UpstoxMarketDataClient(token)
    last_daily_date = None
    while True:
        now = datetime.now(IST)
        intraday = client.fetch_intraday_candles(interval_minutes=5, as_of=now)
        save_candles(intraday, str(DATA_DIR / "nifty_5m.parquet"))
        if last_daily_date != now.date():
            daily = client.fetch_daily_candles(to_date=now.date(), lookback_days=45)
            save_candles(daily, str(DATA_DIR / "nifty_daily.parquet"))
            last_daily_date = now.date()
        latest = intraday.iloc[-1]["timestamp"] if not intraday.empty else "none"
        print(f"{now.isoformat()} completed_5m={len(intraday)} latest={latest}", flush=True)
        if once:
            return
        time.sleep(max(15, int(poll_seconds)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only NIFTY 5-minute market-data collector")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    run(once=args.once, poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    main()
