from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from .core import black76_delta, black76_price, Costs, StrategyConfig
from .strangle_backtest import performance_report, run_backtest


def synthetic_chain() -> pd.DataFrame:
    # Deliberately synthetic data used only to verify the mechanics.
    # It is NOT market data and must never be interpreted as strategy evidence.
    start = pd.Timestamp("2026-06-03 10:00")
    expiries = [pd.Timestamp("2026-06-09"), pd.Timestamp("2026-06-16"), pd.Timestamp("2026-06-23"), pd.Timestamp("2026-06-30"), pd.Timestamp("2026-07-07")]
    rows = []
    strikes = np.arange(21000, 27501, 100)
    for ei, expiry in enumerate(expiries):
        dates = pd.date_range(expiry - pd.Timedelta(days=6), expiry, freq="60min")
        dates = dates[(dates.weekday < 5) & (dates.hour >= 10) & (dates.hour <= 15)]
        for ts in dates:
            # small random-free deterministic path with occasional moderate moves
            base = 24200 + 230 * np.sin((ts - start).total_seconds() / 86400 / 3.2) + 80 * ei
            iv = 0.145 + 0.012 * (1 + np.sin((ts - start).total_seconds() / 86400 / 4.0))
            fwd = base
            t = max((expiry - ts).total_seconds() / (365 * 24 * 3600), 1e-6)
            for k in strikes:
                for opt in ("CE", "PE"):
                    mid = black76_price(fwd, float(k), iv, t, 0.06, opt == "CE")
                    spread = max(0.4, 0.006 * max(mid, 1.0))
                    rows.append({
                        "timestamp": ts,
                        "expiry": expiry,
                        "underlying": "NIFTY",
                        "spot": base,
                        "future": fwd,
                        "strike": float(k),
                        "option_type": opt,
                        "bid": max(0, mid - spread/2),
                        "ask": mid + spread/2,
                        "ltp": mid,
                        "volume": 100000,
                        "oi": 100000,
                        "iv": iv,
                        "delta": black76_delta(fwd, float(k), iv, t, 0.06, opt == "CE"),
                    })
    return pd.DataFrame(rows)


def main() -> None:
    cfg = StrategyConfig()
    costs = Costs()
    data = synthetic_chain()
    from .strangle_backtest import prepare_data
    data = prepare_data(data, cfg, slippage_points=costs.slippage_points_per_leg)
    trades = run_backtest(data, cfg, costs)
    report = performance_report(trades, cfg.starting_capital)
    print("Synthetic smoke test completed.")
    print(pd.Series(report).to_string())
    assert report["trades"] >= 1
    assert np.isfinite(report["ending_capital"])


if __name__ == "__main__":
    main()
