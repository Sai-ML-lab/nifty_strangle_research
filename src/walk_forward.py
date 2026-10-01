from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import numpy as np
import pandas as pd

from .core import Costs, StrategyConfig
from .research import parameter_sweep
from .strangle_backtest import performance_report, run_backtest


def _month_start(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts.year, ts.month, 1)


def _add_months(ts: pd.Timestamp, months: int) -> pd.Timestamp:
    return ts + pd.offsets.MonthBegin(months)


def walk_forward(
    data: pd.DataFrame,
    base_cfg: StrategyConfig,
    costs: Costs,
    train_months: int = 12,
    test_months: int = 3,
    rebalance_months: int = 3,
    sweep_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Walk-forward selection: optimize only in train window, evaluate once in next test window."""
    if data.empty:
        return pd.DataFrame()
    sweep_kwargs = sweep_kwargs or {
        "sd_values": (1.5, 1.75, 2.0),
        "delta_values": (0.05, 0.10),
        "profit_capture_values": (0.40, 0.60),
        "stop_multiple_values": (2.0, 3.0),
        "strike_methods": ("pure_sd", "delta", "hybrid"),
    }
    start = _month_start(pd.to_datetime(data["timestamp"]).min())
    end = pd.to_datetime(data["timestamp"]).max().normalize()
    rows = []
    train_start = start

    while True:
        train_end = _add_months(train_start, train_months)
        test_end = _add_months(train_end, test_months)
        if test_end > end + pd.Timedelta(days=1):
            break

        train = data[(data["timestamp"] >= train_start) & (data["timestamp"] < train_end)].copy()
        test = data[(data["timestamp"] >= train_end) & (data["timestamp"] < test_end)].copy()
        if train.empty or test.empty:
            train_start = _add_months(train_start, rebalance_months)
            continue

        sweep = parameter_sweep(train, base_cfg, costs, **sweep_kwargs)
        if sweep.empty or not np.isfinite(sweep.iloc[0].get("expectancy", np.nan)):
            train_start = _add_months(train_start, rebalance_months)
            continue

        best = sweep.iloc[0]
        tuned = replace(
            base_cfg,
            strike_method=str(best["strike_method"]),
            sd_multiple=float(best["sd"]),
            target_delta=float(best["target_delta"]),
            profit_capture=float(best["profit_capture"]),
            stop_multiple=float(best["stop_multiple"]),
        )
        test_trades = run_backtest(test, tuned, costs)
        report = performance_report(test_trades, tuned.starting_capital)
        rows.append({
            "train_start": train_start,
            "train_end": train_end - pd.Timedelta(days=1),
            "test_start": train_end,
            "test_end": test_end - pd.Timedelta(days=1),
            "selected_strike_method": tuned.strike_method,
            "selected_sd": tuned.sd_multiple,
            "selected_delta": tuned.target_delta,
            "selected_profit_capture": tuned.profit_capture,
            "selected_stop_multiple": tuned.stop_multiple,
            "train_expectancy": float(best.get("expectancy", np.nan)),
            "train_profit_factor": float(best.get("profit_factor", np.nan)),
            **{f"test_{k}": v for k, v in report.items()},
        })
        train_start = _add_months(train_start, rebalance_months)
    return pd.DataFrame(rows)
