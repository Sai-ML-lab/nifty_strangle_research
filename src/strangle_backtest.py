from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yaml

from .core import (
    Costs,
    StrategyConfig,
    add_executable_prices,
    ensure_iv_delta,
    normalize_columns,
    option_value_at_expiry,
    pick_atm_iv,
    estimate_forward_from_parity,
    realized_vol,
    select_strikes,
    trade_cost,
    nifty_lot_size,
)


def load_config(path: str | Path) -> tuple[StrategyConfig, Costs, dict]:
    raw = yaml.safe_load(Path(path).read_text())
    cfg = StrategyConfig(
        lot_size=raw.get("lot_size", 65),
        risk_free_rate=raw.get("risk_free_rate", 0.06),
        calendar_days_per_year=raw.get("calendar_days_per_year", 365),
        trading_days_per_year=raw.get("trading_days_per_year", 252),
        entry_mode=raw["entry"].get("mode", "weekday"),
        entry_weekday=raw["entry"].get("weekday", 2),
        target_dte=raw["entry"].get("target_dte", 6),
        entry_time=raw["entry"].get("time", "10:00"),
        sd_multiple=raw["strike_selection"].get("sd_multiple", 2.0),
        target_delta=raw["strike_selection"].get("target_delta", 0.05),
        strike_method=raw["strike_selection"].get("method", "hybrid"),
        atm_band=raw["strike_selection"].get("atm_band", 3),
        minimum_oi=raw["strike_selection"].get("minimum_oi", 0),
        minimum_volume=raw["strike_selection"].get("minimum_volume", 0),
        profit_capture=raw["exit"].get("profit_capture", 0.5),
        stop_multiple=raw["exit"].get("stop_multiple", 2.0),
        time_exit_mode=raw["exit"].get("time_exit_mode", "weekday"),
        time_exit_weekday=raw["exit"].get("time_exit_weekday", 0),
        time_exit_dte=raw["exit"].get("time_exit_dte", 1),
        time_exit_time=raw["exit"].get("time_exit_time", "15:00"),
        settle_at_expiry=raw["exit"].get("settle_at_expiry", True),
        lots=raw["backtest"].get("lots", 1),
        min_entry_credit_points=raw["backtest"].get("min_entry_credit_points", 1.0),
        starting_capital=raw["backtest"].get("starting_capital", 1_000_000),
    )
    c = raw.get("costs", {})
    costs = Costs(**{k: c[k] for k in asdict(Costs()) if k in c})
    return cfg, costs, raw


def prepare_data(path: str | Path | pd.DataFrame, cfg: StrategyConfig, slippage_points: float = 0.0) -> pd.DataFrame:
    if isinstance(path, pd.DataFrame):
        raw = path.copy()
    else:
        p = Path(path)
        if p.suffix.lower() == ".parquet":
            raw = pd.read_parquet(p)
        elif p.suffix.lower() in {".csv", ".txt"}:
            raw = pd.read_csv(p)
        else:
            raise ValueError("Supported data formats: CSV, Parquet")
    x = normalize_columns(raw)
    x = add_executable_prices(x, slippage_points=slippage_points)
    x["date"] = x["timestamp"].dt.normalize()
    # Compute RV from completed DAILY closes, then shift it so today's entry never
    # uses any same-day price information (avoids intraday/look-ahead contamination).
    intraday_spot = x[["timestamp", "spot", "date"]].drop_duplicates("timestamp").sort_values("timestamp")
    daily = (intraday_spot.groupby("date", as_index=False)["spot"].last().sort_values("date"))
    daily["rv20"] = realized_vol(daily["spot"], 20)
    daily["rv20_available_at_open"] = daily["rv20"].shift(1)
    x = x.merge(daily[["date", "rv20_available_at_open"]], on="date", how="left")
    x["rv20"] = x["rv20_available_at_open"]
    x = x.drop(columns=["rv20_available_at_open"])
    return x.sort_values(["timestamp", "expiry", "strike", "option_type"]).reset_index(drop=True)


def _entry_snapshots(x: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    hh, mm = map(int, cfg.entry_time.split(":"))
    date_mask = (x["timestamp"].dt.weekday == cfg.entry_weekday)
    time_mask = (x["timestamp"].dt.hour > hh) | ((x["timestamp"].dt.hour == hh) & (x["timestamp"].dt.minute >= mm))
    return x.loc[date_mask & time_mask]


def _get_expiry(ts: pd.Timestamp, expiries: np.ndarray) -> Optional[pd.Timestamp]:
    future = expiries[expiries > ts.normalize()]
    return pd.Timestamp(future[0]) if len(future) else None


def _chain_at(x: pd.DataFrame, ts: pd.Timestamp, expiry: pd.Timestamp) -> pd.DataFrame:
    return x[(x["timestamp"] == ts) & (x["expiry"] == expiry)].copy()


def _quote(x: pd.DataFrame, strike: float, option_type: str) -> pd.Series:
    q = x[(x["strike"] == strike) & (x["option_type"] == option_type)]
    if q.empty:
        raise KeyError(f"Missing {option_type} strike {strike} in snapshot")
    return q.iloc[0]


def _close_debit(snapshot: pd.DataFrame, put_k: float, call_k: float) -> float:
    p = _quote(snapshot, put_k, "PE")
    c = _quote(snapshot, call_k, "CE")
    return float(p["buy_exec"] + c["buy_exec"])


def _entry_credit(snapshot: pd.DataFrame, put_k: float, call_k: float) -> float:
    p = _quote(snapshot, put_k, "PE")
    c = _quote(snapshot, call_k, "CE")
    return float(p["entry_sell_exec"] + c["entry_sell_exec"])


def run_backtest(x: pd.DataFrame, cfg: StrategyConfig, costs: Costs) -> pd.DataFrame:
    expiries = np.sort(x["expiry"].dropna().unique())
    entries = _entry_snapshots(x, cfg)
    # One candidate trade per Wednesday. Choose earliest usable snapshot per date.
    first_by_date = entries.groupby(entries["timestamp"].dt.normalize())["timestamp"].min()
    trades = []
    used_expiries = set()

    for entry_date, entry_ts in first_by_date.items():
        expiry = _get_expiry(pd.Timestamp(entry_ts), expiries)
        if expiry is None or expiry in used_expiries:
            continue
        entry_chain = _chain_at(x, pd.Timestamp(entry_ts), expiry)
        if entry_chain.empty:
            continue
        spot = float(entry_chain["spot"].iloc[0])
        future_value = float(entry_chain["future"].iloc[0]) if pd.notna(entry_chain["future"].iloc[0]) and float(entry_chain["future"].iloc[0]) > 0 else np.nan
        forward = future_value if np.isfinite(future_value) else estimate_forward_from_parity(entry_chain, spot, cfg.risk_free_rate, price_column="entry_price")
        entry_chain["forward"] = forward
        try:
            entry_chain = ensure_iv_delta(entry_chain, cfg.risk_free_rate, allow_iv_calc=True, price_column="entry_price")
            atm_iv = pick_atm_iv(entry_chain, forward, cfg.atm_band)
            put, call = select_strikes(entry_chain, forward, atm_iv, expiry, pd.Timestamp(entry_ts), cfg)
        except Exception:
            continue

        put_k, call_k = float(put["strike"]), float(call["strike"])
        credit = _entry_credit(entry_chain, put_k, call_k)
        if not np.isfinite(credit) or credit < cfg.min_entry_credit_points:
            continue

        contract_lot = nifty_lot_size(expiry)
        lot_multiplier = cfg.lots * contract_lot
        exit_ts = None
        exit_reason = None
        exit_debit = None

        path = x[(x["timestamp"] >= entry_ts) & (x["timestamp"].dt.normalize() <= expiry) & (x["expiry"] == expiry)].copy()
        # Reduce the path to the two held contracts and vectorize the exit scan.
        held = path[((path["strike"] == put_k) & (path["option_type"] == "PE")) |
                    ((path["strike"] == call_k) & (path["option_type"] == "CE"))]
        marks = (held.pivot_table(index="timestamp", columns="option_type", values="buy_exec", aggfunc="first")
                 .reindex(columns=["PE", "CE"]).dropna())
        if not marks.empty:
            marks["debit"] = marks["PE"] + marks["CE"]
            marks["weekday"] = marks.index.weekday
            profit_cut = credit * (1 - cfg.profit_capture)
            stop_cut = credit * cfg.stop_multiple
            hit = marks.index[(marks["debit"] <= profit_cut) | (marks["debit"] >= stop_cut)]
            time_mask = marks["weekday"].eq(cfg.time_exit_weekday)
            hh, mm = map(int, cfg.time_exit_time.split(":"))
            time_mask &= ((marks.index.hour > hh) | ((marks.index.hour == hh) & (marks.index.minute >= mm)))
            time_hits = marks.index[time_mask]
            candidates = []
            if len(hit):
                candidates.append((hit[0], "profit_target" if marks.loc[hit[0], "debit"] <= profit_cut else "stop"))
            if len(time_hits):
                candidates.append((time_hits[0], "time_exit"))
            if candidates:
                exit_ts, exit_reason = min(candidates, key=lambda z: z[0])
                exit_debit = float(marks.loc[exit_ts, "debit"])

        p_entry = float(_quote(entry_chain, put_k, "PE")["entry_sell_exec"])
        c_entry = float(_quote(entry_chain, call_k, "CE")["entry_sell_exec"])
        sell_premium = (p_entry + c_entry) * lot_multiplier

        if exit_ts is None:
            # Expiry settlement uses intrinsic value rather than an end-of-day quoted price.
            expiry_spot_rows = x[(x["timestamp"].dt.normalize() == expiry) & (x["expiry"] == expiry)]
            if expiry_spot_rows.empty:
                # Fall back to last available snapshot on/before expiry.
                expiry_spot_rows = x[(x["timestamp"] <= expiry + pd.Timedelta(days=1)) & (x["expiry"] == expiry)]
            spot_exit = float(expiry_spot_rows.sort_values("timestamp")["spot"].iloc[-1])
            intrinsic = option_value_at_expiry(put_k, spot_exit, "PE") + option_value_at_expiry(call_k, spot_exit, "CE")
            exit_premium_total = intrinsic * lot_multiplier
            exit_ts = expiry
            exit_reason = "expiry"
            buy_premium_for_cost = exit_premium_total
            gross = sell_premium - exit_premium_total
            spot_at_exit = spot_exit
        else:
            buy_premium_for_cost = float(exit_debit) * lot_multiplier
            gross = sell_premium - buy_premium_for_cost
            spot_at_exit = float(path[path["timestamp"] == exit_ts]["spot"].iloc[0])

        # Costs are expressed in premium rupees across the complete trade.
        transaction_cost = trade_cost(
            notional_premium=sell_premium + buy_premium_for_cost,
            sell_premium=sell_premium,
            buy_premium=buy_premium_for_cost,
            orders=4,
            costs=costs,
            trade_date=pd.Timestamp(entry_ts),
        )
        net = gross - transaction_cost

        trades.append({
            "trade_id": len(trades) + 1,
            "entry_timestamp": pd.Timestamp(entry_ts),
            "expiry": expiry,
            "lot_size": contract_lot,
            "entry_spot": spot,
            "exit_spot": spot_at_exit,
            "forward_entry": forward,
            "atm_iv": atm_iv,
            "rv20": float(entry_chain["rv20"].iloc[0]) if pd.notna(entry_chain["rv20"].iloc[0]) else np.nan,
            "iv_rv_spread": atm_iv - float(entry_chain["rv20"].iloc[0]) if pd.notna(entry_chain["rv20"].iloc[0]) else np.nan,
            "put_strike": put_k,
            "call_strike": call_k,
            "put_delta": float(put["delta"]),
            "call_delta": float(call["delta"]),
            "initial_credit_points": credit,
            "put_entry_price": p_entry,
            "call_entry_price": c_entry,
            "initial_credit_rupees": sell_premium,
            "exit_timestamp": exit_ts,
            "exit_reason": exit_reason,
            "exit_debit_points": exit_debit if exit_debit is not None else np.nan,
            "gross_pnl": gross,
            "transaction_cost": transaction_cost,
            "net_pnl": net,
        })
        used_expiries.add(expiry)

    return pd.DataFrame(trades)


def performance_report(trades: pd.DataFrame, starting_capital: float) -> Dict[str, float]:
    if trades.empty:
        return {"trades": 0}
    p = trades["net_pnl"].astype(float)
    wins = p[p > 0]
    losses = p[p < 0]
    equity = starting_capital + p.cumsum()
    dd = equity / equity.cummax() - 1
    loss_mask = p < 0
    groups = (loss_mask != loss_mask.shift(fill_value=False)).cumsum()
    losing_streaks = p[loss_mask].groupby(groups[loss_mask]).size()
    return {
        "trades": int(len(p)),
        "win_rate": float((p > 0).mean()),
        "avg_pnl": float(p.mean()),
        "median_pnl": float(p.median()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "avg_win_loss_ratio": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else np.inf,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) else np.inf,
        "expectancy": float(p.mean()),
        "p05_trade": float(p.quantile(0.05)),
        "p01_trade": float(p.quantile(0.01)),
        "total_net_pnl": float(p.sum()),
        "total_return_pct": float((equity.iloc[-1] / starting_capital - 1.0) * 100.0),
        "max_drawdown_pct": float(dd.min()),
        "max_losing_streak": int(losing_streaks.max()) if len(losing_streaks) else 0,
        "best_trade": float(p.max()),
        "worst_trade": float(p.min()),
        "ending_capital": float(equity.iloc[-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest NIFTY weekly 2SD short strangle")
    parser.add_argument("--data", required=True, help="CSV or Parquet long-form option chain")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="results/trades.csv")
    args = parser.parse_args()

    cfg, costs, raw = load_config(args.config)
    data = prepare_data(args.data, cfg, slippage_points=costs.slippage_points_per_leg)
    trades = run_backtest(data, cfg, costs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out, index=False)
    report = performance_report(trades, cfg.starting_capital)
    print(pd.Series(report).to_string())
    print(f"\nSaved trades to: {out}")


if __name__ == "__main__":
    main()
