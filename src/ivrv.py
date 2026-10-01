from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _clean_trades(trades: pd.DataFrame, exclude_quality_warnings: bool = True) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    x = trades.copy()
    if "entry_timestamp" in x:
        x["entry_timestamp"] = pd.to_datetime(x["entry_timestamp"], errors="coerce")
    if exclude_quality_warnings and "data_quality_flag" in x:
        x = x[x["data_quality_flag"].fillna("PASS") == "PASS"].copy()
    return x.dropna(subset=["net_pnl", "entry_timestamp"]).sort_values("entry_timestamp")


def _stats(x: pd.DataFrame) -> dict:
    p = pd.to_numeric(x["net_pnl"], errors="coerce").dropna()
    if p.empty:
        return {"trades": 0}
    wins = p[p > 0]
    losses = p[p < 0]
    return {
        "trades": int(len(p)),
        "win_rate": float((p > 0).mean()),
        "avg_pnl": float(p.mean()),
        "median_pnl": float(p.median()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) else np.inf,
        "total_net_pnl": float(p.sum()),
        "best_trade": float(p.max()),
        "worst_trade": float(p.min()),
    }


def threshold_sweep(
    trades: pd.DataFrame,
    thresholds: Iterable[float] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
    exclude_quality_warnings: bool = True,
) -> pd.DataFrame:
    x = _clean_trades(trades, exclude_quality_warnings=exclude_quality_warnings)
    rows = [{"filter_enabled": False, "min_iv_rv_spread_pct": np.nan, **_stats(x)}]
    for threshold in thresholds:
        y = x[
            x["iv_rv_spread"].notna()
            & (x["iv_rv_spread"] >= float(threshold) / 100.0)
        ]
        rows.append({
            "filter_enabled": True,
            "min_iv_rv_spread_pct": float(threshold),
            **_stats(y),
        })
    return pd.DataFrame(rows)


def walk_forward_ivrv(
    trades: pd.DataFrame,
    thresholds: Iterable[float] = (0.0, 1.0, 2.0, 3.0),
    train_months: int = 12,
    test_months: int = 3,
    rebalance_months: int = 3,
    min_train_trades: int = 20,
    exclude_quality_warnings: bool = True,
) -> pd.DataFrame:
    x = _clean_trades(trades, exclude_quality_warnings=exclude_quality_warnings)
    if x.empty:
        return pd.DataFrame()

    start = pd.Timestamp(x["entry_timestamp"].min()).to_period("M").to_timestamp()
    end = pd.Timestamp(x["entry_timestamp"].max()).normalize()
    thresholds = [float(t) for t in thresholds]
    rows = []
    train_start = start

    while True:
        train_end = train_start + pd.offsets.MonthBegin(train_months)
        test_end = train_end + pd.offsets.MonthBegin(test_months)
        if test_end > end + pd.Timedelta(days=1):
            break

        train = x[(x["entry_timestamp"] >= train_start) & (x["entry_timestamp"] < train_end)]
        test = x[(x["entry_timestamp"] >= train_end) & (x["entry_timestamp"] < test_end)]
        if train.empty or test.empty:
            train_start = train_start + pd.offsets.MonthBegin(rebalance_months)
            continue

        candidates = []
        for threshold in thresholds:
            tr = train[
                train["iv_rv_spread"].notna()
                & (train["iv_rv_spread"] >= threshold / 100.0)
            ]
            if len(tr) < min_train_trades:
                continue
            s = _stats(tr)
            candidates.append((threshold, s))

        if not candidates:
            train_start = train_start + pd.offsets.MonthBegin(rebalance_months)
            continue

        # Maximize training expectancy, then profit factor, then sample size.
        best_threshold, best_stats = sorted(
            candidates,
            key=lambda z: (
                z[1]["avg_pnl"],
                z[1]["profit_factor"],
                z[1]["trades"],
                -z[0],
            ),
            reverse=True,
        )[0]

        te = test[test["iv_rv_spread"] >= best_threshold / 100.0]
        te_stats = _stats(te)
        base_test_stats = _stats(test)

        rows.append({
            "train_start": train_start.date(),
            "train_end": (train_end - pd.Timedelta(days=1)).date(),
            "test_start": train_end.date(),
            "test_end": (test_end - pd.Timedelta(days=1)).date(),
            "selected_min_iv_rv_spread_pct": best_threshold,
            "train_trades": best_stats["trades"],
            "train_expectancy": best_stats["avg_pnl"],
            "train_profit_factor": best_stats["profit_factor"],
            "test_trades": te_stats.get("trades", 0),
            "test_expectancy": te_stats.get("avg_pnl", np.nan),
            "test_profit_factor": te_stats.get("profit_factor", np.nan),
            "test_net_pnl": te_stats.get("total_net_pnl", 0.0),
            "test_win_rate": te_stats.get("win_rate", np.nan),
            "test_worst_trade": te_stats.get("worst_trade", np.nan),
            "base_test_trades": base_test_stats.get("trades", 0),
            "base_test_expectancy": base_test_stats.get("avg_pnl", np.nan),
            "base_test_net_pnl": base_test_stats.get("total_net_pnl", 0.0),
        })

        train_start = train_start + pd.offsets.MonthBegin(rebalance_months)

    return pd.DataFrame(rows)
