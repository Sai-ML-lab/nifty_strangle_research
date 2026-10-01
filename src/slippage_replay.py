from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dataclasses import replace

from src.core import Costs, nifty_lot_size, option_value_at_expiry, trade_cost
from src.data_ingest import normalize_options_vendor_file

def expiry_from_filename(path: Path) -> pd.Timestamp | None:
    stem = path.stem.replace("expiry=", "")
    try:
        return pd.Timestamp(stem).normalize()
    except Exception:
        return None

def _numeric(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")

def executable_sell_price(row: pd.Series, slippage_points: float) -> float:
    if "bid" in row and pd.notna(row["bid"]):
        base = _numeric(row["bid"])
    elif "entry_price" in row and pd.notna(row["entry_price"]):
        base = _numeric(row["entry_price"])
    else:
        base = _numeric(row.get("ltp"))
    return max(base - float(slippage_points), 0.0)

def executable_buy_price(row: pd.Series, slippage_points: float) -> float:
    if "ask" in row and pd.notna(row["ask"]):
        base = _numeric(row["ask"])
    else:
        base = _numeric(row.get("ltp"))
    return max(base + float(slippage_points), 0.0)

def _quote_at(chain: pd.DataFrame, timestamp, strike: float, option_type: str) -> pd.Series | None:
    ts = pd.Timestamp(timestamp)
    y = chain[(chain["timestamp"] == ts) & (pd.to_numeric(chain["strike"], errors="coerce") == float(strike)) & (chain["option_type"] == option_type)].copy()
    if y.empty:
        return None
    return y.sort_index().iloc[0]

def _normalise_chain(raw: pd.DataFrame) -> pd.DataFrame:
    x = normalize_options_vendor_file(raw)
    if x["timestamp"].dt.tz is not None:
        x["timestamp"] = x["timestamp"].dt.tz_localize(None)
    return x

def replay_trade(trade: pd.Series, chain: pd.DataFrame, slippage_points: float, base_costs: Costs | None = None) -> dict:
    base_costs = base_costs or Costs()
    expiry = pd.Timestamp(trade["expiry"]).normalize()
    entry_ts = pd.Timestamp(trade["entry_timestamp"])
    exit_ts = pd.Timestamp(trade["exit_timestamp"])
    put_strike = _numeric(trade["put_strike"])
    call_strike = _numeric(trade["call_strike"])
    lot_size = int(trade["lot_size"]) if pd.notna(trade.get("lot_size")) else nifty_lot_size(expiry)
    lots = int(trade["lots"]) if "lots" in trade and pd.notna(trade["lots"]) else 1
    multiplier = lot_size * lots

    p_entry = _quote_at(chain, entry_ts, put_strike, "PE")
    c_entry = _quote_at(chain, entry_ts, call_strike, "CE")
    if p_entry is None or c_entry is None:
        raise ValueError(f"Missing entry quote for {entry_ts} {put_strike}/{call_strike}")

    put_entry = executable_sell_price(p_entry, slippage_points)
    call_entry = executable_sell_price(c_entry, slippage_points)
    credit_points = put_entry + call_entry
    sell_premium = credit_points * multiplier

    exit_reason = str(trade["exit_reason"])
    if exit_reason == "expiry":
        spot_exit = _numeric(trade["exit_spot"])
        exit_debit_points = option_value_at_expiry(put_strike, spot_exit, "PE") + option_value_at_expiry(call_strike, spot_exit, "CE")
    else:
        p_exit = _quote_at(chain, exit_ts, put_strike, "PE")
        c_exit = _quote_at(chain, exit_ts, call_strike, "CE")
        if p_exit is None or c_exit is None:
            raise ValueError(f"Missing exit quote for {exit_ts} {put_strike}/{call_strike}")
        exit_debit_points = executable_buy_price(p_exit, slippage_points) + executable_buy_price(c_exit, slippage_points)

    buy_premium = exit_debit_points * multiplier
    gross_pnl = sell_premium - buy_premium
    costs_rupees = trade_cost(sell_premium + buy_premium, sell_premium, buy_premium, orders=4, costs=Costs(slippage_points_per_leg=float(slippage_points)), trade_date=entry_ts)
    return {
        "entry_timestamp": entry_ts, "expiry": expiry, "exit_timestamp": exit_ts, "exit_reason": exit_reason,
        "put_strike": put_strike, "call_strike": call_strike, "lot_size": lot_size, "lots": lots,
        "slippage_points_per_leg": float(slippage_points), "initial_credit_points": credit_points,
        "exit_debit_points": float(exit_debit_points), "gross_pnl": float(gross_pnl),
        "transaction_cost": float(costs_rupees), "net_pnl": float(gross_pnl - costs_rupees),
        "reference_net_pnl": _numeric(trade.get("net_pnl")), "data_quality_flag": str(trade.get("data_quality_flag", "PASS")),
    }

def replay_slippage(trades: pd.DataFrame, options_dir: str | Path, slippages: list[float], base_costs: Costs | None = None, exclude_quality_warnings: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_costs = base_costs or Costs()
    required = {"entry_timestamp", "expiry", "exit_timestamp", "exit_reason", "put_strike", "call_strike"}
    missing = sorted(required - set(trades.columns))
    if missing:
        raise ValueError(f"Trade ledger missing required columns: {missing}")
    x = trades.copy()
    x["entry_timestamp"] = pd.to_datetime(x["entry_timestamp"], errors="coerce")
    x["exit_timestamp"] = pd.to_datetime(x["exit_timestamp"], errors="coerce")
    x["expiry"] = pd.to_datetime(x["expiry"], errors="coerce").dt.normalize()
    x = x.dropna(subset=["entry_timestamp", "exit_timestamp", "expiry"])
    if exclude_quality_warnings and "data_quality_flag" in x.columns:
        x = x[x["data_quality_flag"].fillna("PASS") == "PASS"].copy()
    files_by_expiry: dict[pd.Timestamp, Path] = {}
    for path in sorted(Path(options_dir).rglob("*.parquet")):
        expiry = expiry_from_filename(path)
        if expiry is not None:
            files_by_expiry[expiry] = path
    all_rows = []
    for expiry, group in x.groupby("expiry", sort=True):
        path = files_by_expiry.get(pd.Timestamp(expiry).normalize())
        if path is None:
            raise FileNotFoundError(f"No option parquet found for expiry {expiry.date()}: {options_dir}")
        chain = _normalise_chain(pd.read_parquet(path))
        relevant_ts = set(group["entry_timestamp"]).union(set(group["exit_timestamp"]))
        chain = chain[chain["timestamp"].isin(relevant_ts)].copy()
        for _, trade in group.iterrows():
            for slippage in slippages:
                row = replay_trade(trade, chain, float(slippage), base_costs=base_costs)
                row["trade_index"] = trade.name
                all_rows.append(row)
    detail = pd.DataFrame(all_rows)
    if detail.empty:
        return detail, pd.DataFrame()
    summary_rows = []
    for slippage, group in detail.groupby("slippage_points_per_leg", sort=True):
        p = pd.to_numeric(group["net_pnl"], errors="coerce").dropna()
        wins, losses = p[p > 0], p[p < 0]
        summary_rows.append({
            "slippage_points_per_leg": float(slippage), "trades": int(len(p)),
            "win_rate": float((p > 0).mean()) if len(p) else np.nan,
            "avg_pnl": float(p.mean()) if len(p) else np.nan, "median_pnl": float(p.median()) if len(p) else np.nan,
            "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) else np.inf,
            "total_net_pnl": float(p.sum()) if len(p) else 0.0, "worst_trade": float(p.min()) if len(p) else np.nan,
        })
    return detail, pd.DataFrame(summary_rows)