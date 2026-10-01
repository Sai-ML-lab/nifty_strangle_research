from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Iterable

import pandas as pd

from .core import Costs, StrategyConfig
from .strangle_backtest import performance_report, run_backtest


def parameter_sweep(
    data: pd.DataFrame,
    base_cfg: StrategyConfig,
    costs: Costs,
    sd_values: Iterable[float] = (1.5, 1.75, 2.0),
    delta_values: Iterable[float] = (0.03, 0.05, 0.10),
    profit_capture_values: Iterable[float] = (0.25, 0.40, 0.50, 0.60),
    stop_multiple_values: Iterable[float] = (1.5, 2.0, 2.5, 3.0),
    strike_methods: Iterable[str] = ("pure_sd", "delta", "hybrid"),
) -> pd.DataFrame:
    """Run an intentionally simple research grid.

    This is exploratory: use a separate out-of-sample/walk-forward period before
    interpreting the best row as a candidate trading rule.
    """
    rows = []
    for method, sd, delta, profit_capture, stop_multiple in product(
        strike_methods, sd_values, delta_values, profit_capture_values, stop_multiple_values
    ):
        cfg = replace(
            base_cfg,
            strike_method=method,
            sd_multiple=sd,
            target_delta=delta,
            profit_capture=profit_capture,
            stop_multiple=stop_multiple,
        )
        trades = run_backtest(data, cfg, costs)
        report = performance_report(trades, cfg.starting_capital)
        rows.append({
            "strike_method": method,
            "sd": sd,
            "target_delta": delta,
            "profit_capture": profit_capture,
            "stop_multiple": stop_multiple,
            **report,
        })
    return pd.DataFrame(rows).sort_values(
        ["expectancy", "profit_factor", "max_drawdown_pct"], ascending=[False, False, False]
    ).reset_index(drop=True)


def trade_regime_report(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    x = trades.copy()
    x["vol_regime"] = pd.cut(
        x["atm_iv"],
        bins=[-float("inf"), 0.12, 0.18, 0.25, float("inf")],
        labels=["low", "normal", "high", "extreme"],
    )
    return (
        x.groupby("vol_regime", observed=True)["net_pnl"]
        .agg(["count", "mean", "median", "sum"])
        .reset_index()
    )
