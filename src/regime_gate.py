from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.ivrv import _clean_trades, _stats

def walk_forward_ivrv_with_regime_gate(
    trades: pd.DataFrame,
    thresholds: Iterable[float] = (0.0, 1.0, 2.0, 3.0),
    train_months: int = 12,
    test_months: int = 3,
    rebalance_months: int = 3,
    min_train_trades: int = 20,
    min_train_expectancy: float = 0.0,
    min_train_profit_factor: float = 1.0,
    exclude_quality_warnings: bool = True,
) -> pd.DataFrame:
    x = _clean_trades(trades, exclude_quality_warnings=exclude_quality_warnings)
    if x.empty:
        return pd.DataFrame()
    x["entry_timestamp"] = pd.to_datetime(x["entry_timestamp"], errors="coerce")
    start = x["entry_timestamp"].min().to_period("M").to_timestamp()
    end = x["entry_timestamp"].max().normalize()
    threshold_list = [float(t) for t in thresholds]
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
            train_start += pd.offsets.MonthBegin(rebalance_months)
            continue
        candidates = []
        for threshold in threshold_list:
            tr = train[train["iv_rv_spread"].notna() & (train["iv_rv_spread"] >= threshold / 100.0)]
            if len(tr) < min_train_trades:
                continue
            candidates.append((threshold, _stats(tr)))
        if not candidates:
            train_start += pd.offsets.MonthBegin(rebalance_months)
            continue
        best_threshold, best_stats = sorted(
            candidates,
            key=lambda z: (z[1]["avg_pnl"], z[1]["profit_factor"], z[1]["trades"], -z[0]),
            reverse=True,
        )[0]
        gate_pass = best_stats["avg_pnl"] >= float(min_train_expectancy) and best_stats["profit_factor"] >= float(min_train_profit_factor)
        gate_reason = "PASS" if gate_pass else "TRAIN_EXPECTANCY_OR_PF_BELOW_GATE"
        selected_test = test[test["iv_rv_spread"].notna() & (test["iv_rv_spread"] >= best_threshold / 100.0)]
        gated_test = selected_test if gate_pass else selected_test.iloc[0:0]
        base_test_stats = _stats(test)
        selected_stats = _stats(selected_test)
        gated_stats = _stats(gated_test)
        rows.append({
            "train_start": train_start.date(),
            "train_end": (train_end - pd.Timedelta(days=1)).date(),
            "test_start": train_end.date(),
            "test_end": (test_end - pd.Timedelta(days=1)).date(),
            "selected_min_iv_rv_spread_pct": best_threshold,
            "train_trades": best_stats["trades"],
            "train_expectancy": best_stats["avg_pnl"],
            "train_profit_factor": best_stats["profit_factor"],
            "regime_gate_pass": gate_pass,
            "regime_gate_reason": gate_reason,
            "selected_test_trades": selected_stats.get("trades", 0),
            "selected_test_expectancy": selected_stats.get("avg_pnl", np.nan),
            "selected_test_profit_factor": selected_stats.get("profit_factor", np.nan),
            "selected_test_net_pnl": selected_stats.get("total_net_pnl", 0.0),
            "gated_test_trades": gated_stats.get("trades", 0),
            "gated_test_expectancy": gated_stats.get("avg_pnl", np.nan),
            "gated_test_profit_factor": gated_stats.get("profit_factor", np.nan),
            "gated_test_net_pnl": gated_stats.get("total_net_pnl", 0.0),
            "gated_test_worst_trade": gated_stats.get("worst_trade", np.nan),
            "base_test_trades": base_test_stats.get("trades", 0),
            "base_test_expectancy": base_test_stats.get("avg_pnl", np.nan),
            "base_test_net_pnl": base_test_stats.get("total_net_pnl", 0.0),
        })
        train_start += pd.offsets.MonthBegin(rebalance_months)
    return pd.DataFrame(rows)