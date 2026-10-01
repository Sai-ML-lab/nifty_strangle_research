from __future__ import annotations

import numpy as np
import pandas as pd

def portfolio_risk_metrics(trades: pd.DataFrame, starting_capital: float = 1_000_000.0) -> dict:
    if trades is None or trades.empty:
        return {"trades": 0}
    x = trades.copy()
    x["entry_timestamp"] = pd.to_datetime(x["entry_timestamp"], errors="coerce")
    if "exit_timestamp" in x:
        x["exit_timestamp"] = pd.to_datetime(x["exit_timestamp"], errors="coerce")
    x["net_pnl"] = pd.to_numeric(x["net_pnl"], errors="coerce")
    x = x.dropna(subset=["entry_timestamp", "net_pnl"]).sort_values("entry_timestamp").copy()
    if x.empty:
        return {"trades": 0}
    cap = float(starting_capital)
    p = x["net_pnl"]
    equity = cap + p.cumsum()
    drawdown = equity / equity.cummax() - 1.0
    first_date = x["entry_timestamp"].min().normalize()
    last_date = (x["exit_timestamp"].dropna().max() if "exit_timestamp" in x and x["exit_timestamp"].notna().any() else x["entry_timestamp"].max()).normalize()
    years = max((last_date - first_date).days / 365.25, 1 / 365.25)
    ending = float(equity.iloc[-1])
    cagr = (ending / cap) ** (1.0 / years) - 1.0 if ending > 0 else -1.0
    weekly = x.set_index("entry_timestamp")["net_pnl"].resample("W-FRI").sum()
    full_weeks = pd.date_range(weekly.index.min(), weekly.index.max(), freq="W-FRI")
    weekly = weekly.reindex(full_weeks, fill_value=0.0)
    weekly_ret = weekly / cap
    mean_w = float(weekly_ret.mean()) if len(weekly_ret) else np.nan
    std_w = float(weekly_ret.std(ddof=1)) if len(weekly_ret) > 1 else np.nan
    down = weekly_ret[weekly_ret < 0]
    downside_std = float(down.std(ddof=1)) if len(down) > 1 else np.nan
    sharpe = float(mean_w / std_w * np.sqrt(52)) if np.isfinite(std_w) and std_w > 0 else np.nan
    sortino = float(mean_w / downside_std * np.sqrt(52)) if np.isfinite(downside_std) and downside_std > 0 else np.nan
    worst_week = float(weekly_ret.min() * cap) if len(weekly_ret) else np.nan
    q05 = float(weekly_ret.quantile(0.05)) if len(weekly_ret) else np.nan
    tail = weekly_ret[weekly_ret <= q05]
    cvar5 = float(tail.mean() * cap) if len(tail) else np.nan
    p = p.to_numpy(dtype=float)
    wins, losses = p[p > 0], p[p < 0]
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    return {
        "trades": int(len(p)),
        "win_rate": float((p > 0).mean()),
        "total_net_pnl": float(p.sum()),
        "avg_trade_pnl": float(p.mean()),
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) else np.inf,
        "ending_capital": ending,
        "total_return_pct": float((ending / cap - 1.0) * 100.0),
        "cagr_pct": float(cagr * 100.0),
        "max_drawdown_pct": float(drawdown.min() * 100.0),
        "calmar": float((cagr / abs(drawdown.min()))) if drawdown.min() < 0 else np.inf,
        "weekly_sharpe": sharpe,
        "weekly_sortino": sortino,
        "worst_week_pnl": worst_week,
        "weekly_cvar5_pnl": cvar5,
        "best_trade": float(p.max()),
        "worst_trade": float(p.min()),
        "avg_loss": avg_loss,
        "max_losing_streak": int(_max_losing_streak(p)),
    }

def _max_losing_streak(p: np.ndarray) -> int:
    best = cur = 0
    for v in p:
        if v < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best