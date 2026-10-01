from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.core import Costs, add_executable_prices, realized_vol
from src.data_ingest import normalize_options_vendor_file, normalize_spot_file
from src.strangle_backtest import load_config, performance_report
from src.core import ensure_iv_delta, estimate_forward_from_parity, pick_atm_iv, select_strikes, nifty_lot_size, option_value_at_expiry, trade_cost


def load_spot(path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    x = normalize_spot_file(raw).sort_values("timestamp")
    if x["timestamp"].dt.tz is not None:
        x["timestamp"] = x["timestamp"].dt.tz_localize(None)
    x["date"] = x["timestamp"].dt.normalize()
    daily = x.groupby("date", as_index=False)["spot"].last().sort_values("date")
    daily["rv20"] = realized_vol(daily["spot"], 20)
    daily["rv20_available_at_open"] = daily["rv20"].shift(1)
    x = x.merge(daily[["date", "rv20_available_at_open"]], on="date", how="left")
    return x.drop(columns=["date"]).rename(columns={"rv20_available_at_open": "rv20"})


def expiry_from_filename(path: Path) -> pd.Timestamp | None:
    # Accept expiry=YYYY-MM-DD.parquet and YYYY-MM-DD.parquet.
    stem = path.stem.replace("expiry=", "")
    try:
        return pd.Timestamp(stem).normalize()
    except Exception:
        return None


def entry_date_for_expiry(expiry: pd.Timestamp, cfg) -> pd.Timestamp:
    mode = getattr(cfg, "entry_mode", "weekday")
    if mode == "dte":
        return (expiry - pd.Timedelta(days=int(cfg.target_dte))).normalize()
    offset = (expiry.weekday() - cfg.entry_weekday) % 7
    if offset == 0:
        offset = 7
    return (expiry - pd.Timedelta(days=offset)).normalize()


def entry_timestamp_for_expiry(
    x: pd.DataFrame,
    expiry: pd.Timestamp,
    cfg,
) -> pd.Timestamp | None:
    entry_date = entry_date_for_expiry(expiry, cfg)
    hh, mm = map(int, cfg.entry_time.split(":"))
    m = (
        (x["timestamp"].dt.normalize() == entry_date)
        & (
            (x["timestamp"].dt.hour > hh)
            | ((x["timestamp"].dt.hour == hh) & (x["timestamp"].dt.minute >= mm))
        )
    )
    ts = x.loc[m, "timestamp"]
    return ts.min() if not ts.empty else None


def single_expiry_trade(x: pd.DataFrame, spot: pd.DataFrame, cfg, costs: Costs, expiry: pd.Timestamp) -> dict | None:
    if x.empty:
        return None
    x = normalize_options_vendor_file(x)
    if x["timestamp"].dt.tz is not None:
        x["timestamp"] = x["timestamp"].dt.tz_localize(None)
    spot_for_join = spot.drop(columns=["spot"]) if "spot" in x.columns else spot
    x = x.merge(spot_for_join, on="timestamp", how="left", validate="many_to_one")
    x["future"] = np.nan
    entry_column = "open" if "open" in x.columns else "entry_price"
    x["entry_price"] = pd.to_numeric(x[entry_column], errors="coerce")
    x = add_executable_prices(x, slippage_points=costs.slippage_points_per_leg)
    x["expiry"] = expiry
    entry_ts = entry_timestamp_for_expiry(x, expiry, cfg)
    if entry_ts is None:
        return None
    entry_chain = x[x["timestamp"] == entry_ts].copy()
    entry_chain = entry_chain.dropna(subset=["spot", "entry_price"])
    if entry_chain.empty:
        return None
    spot0 = float(entry_chain["spot"].iloc[0])
    forward = estimate_forward_from_parity(entry_chain, spot0, cfg.risk_free_rate, price_column="entry_price")
    entry_chain["forward"] = forward
    try:
        entry_chain = ensure_iv_delta(entry_chain, cfg.risk_free_rate, allow_iv_calc=True, price_column="entry_price")
        atm_iv = pick_atm_iv(entry_chain, forward, cfg.atm_band)
        put, call = select_strikes(entry_chain, forward, atm_iv, expiry, pd.Timestamp(entry_ts), cfg)
    except Exception:
        return None

    pk, ck = float(put["strike"]), float(call["strike"])
    p = entry_chain[(entry_chain.strike == pk) & (entry_chain.option_type == "PE")].iloc[0]
    c = entry_chain[(entry_chain.strike == ck) & (entry_chain.option_type == "CE")].iloc[0]
    credit_points = float(p.entry_sell_exec + c.entry_sell_exec)
    if not np.isfinite(credit_points) or credit_points < cfg.min_entry_credit_points:
        return None

    lot = nifty_lot_size(expiry)
    multiplier = cfg.lots * lot
    sell_premium = credit_points * multiplier

    path = x[(x.timestamp >= entry_ts) & (x.timestamp.dt.date <= expiry.date())].copy()
    held = path[((path.strike == pk) & (path.option_type == "PE")) | ((path.strike == ck) & (path.option_type == "CE"))]
    marks = held.pivot_table(index="timestamp", columns="option_type", values="buy_exec", aggfunc="first").reindex(columns=["PE","CE"]).dropna()
    exit_ts = None
    exit_reason = None
    exit_debit = None
    if not marks.empty:
        marks["debit"] = marks["PE"] + marks["CE"]
        profit_cut = credit_points * (1 - cfg.profit_capture)
        stop_cut = credit_points * cfg.stop_multiple
        hit = marks.index[(marks.debit <= profit_cut) | (marks.debit >= stop_cut)]
        hh, mm = map(int, cfg.time_exit_time.split(":"))
        time_mask = (marks.index.weekday == cfg.time_exit_weekday) & (((marks.index.hour > hh) | ((marks.index.hour == hh) & (marks.index.minute >= mm))))
        th = marks.index[time_mask]
        candidates = []
        if len(hit):
            exit_ts = hit[0]
            exit_reason = "profit_target" if marks.loc[exit_ts, "debit"] <= profit_cut else "stop"
        if len(th):
            candidates.append((th[0], "time_exit"))
        if exit_ts is not None:
            candidates.append((exit_ts, exit_reason))
        if candidates:
            exit_ts, exit_reason = min(candidates, key=lambda z: z[0])
            exit_debit = float(marks.loc[exit_ts, "debit"])

    if exit_ts is None:
        spot_exit_rows = spot[spot.timestamp.dt.date == expiry.date()]
        if spot_exit_rows.empty:
            return None
        spot_exit = float(spot_exit_rows.iloc[-1].spot)
        intrinsic = option_value_at_expiry(pk, spot_exit, "PE") + option_value_at_expiry(ck, spot_exit, "CE")
        buy_premium = intrinsic * multiplier
        spot_final = spot_exit
        exit_ts = expiry
        exit_reason = "expiry"
        exit_debit_points = intrinsic
    else:
        buy_premium = exit_debit * multiplier
        spot_final = float(path[path.timestamp == exit_ts].spot.iloc[0])
        exit_debit_points = exit_debit

    gross = sell_premium - buy_premium
    costs_rupees = trade_cost(sell_premium + buy_premium, sell_premium, buy_premium, 4, costs, pd.Timestamp(entry_ts))
    return {
        "entry_timestamp": entry_ts,
        "expiry": expiry,
        "lot_size": lot,
        "entry_spot": spot0,
        "exit_spot": spot_final,
        "forward_entry": forward,
        "atm_iv": atm_iv,
        "rv20": float(entry_chain["rv20"].iloc[0]) if "rv20" in entry_chain.columns and pd.notna(entry_chain["rv20"].iloc[0]) else np.nan,
        "iv_rv_spread": float(atm_iv - entry_chain["rv20"].iloc[0]) if "rv20" in entry_chain.columns and pd.notna(entry_chain["rv20"].iloc[0]) else np.nan,
        "put_strike": pk,
        "call_strike": ck,
        "put_delta": float(put.delta),
        "call_delta": float(call.delta),
        "initial_credit_points": credit_points,
        "initial_credit_rupees": sell_premium,
        "exit_timestamp": exit_ts,
        "exit_reason": exit_reason,
        "exit_debit_points": exit_debit_points,
        "gross_pnl": gross,
        "transaction_cost": costs_rupees,
        "net_pnl": gross - costs_rupees,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest NIFTY strategy expiry-by-expiry without loading the entire chain into RAM")
    ap.add_argument("--options-dir", required=True, help="Directory of NIFTY expiry Parquets")
    ap.add_argument("--spot", required=True, help="NIFTY spot parquet/csv")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out-dir", default="results/real")
    args = ap.parse_args()

    cfg, costs, _ = load_config(args.config)
    spot = load_spot(Path(args.spot))
    files = sorted(Path(args.options_dir).rglob("*.parquet"))
    # The baseline must hold only the NEXT available expiry for a given entry date.
    # Multiple expiry files can map to the same entry date. Keep the earliest expiry
    # so the control strategy opens at most one position per entry date.
    parsed = []
    for f in files:
        expiry = expiry_from_filename(f)
        if expiry is not None:
            parsed.append((expiry, f))
    parsed.sort(key=lambda z: z[0])

    next_expiry_by_entry_date = {}
    for expiry, _ in parsed:
        entry_date = entry_date_for_expiry(expiry, cfg)
        if entry_date not in next_expiry_by_entry_date:
            next_expiry_by_entry_date[entry_date] = expiry



    rows = []
    skipped_not_next = 0
    for i, (expiry, f) in enumerate(parsed, 1):
        entry_date = entry_date_for_expiry(expiry)
        if next_expiry_by_entry_date.get(entry_date) != expiry:
            skipped_not_next += 1
            continue

        raw = pd.read_parquet(f)
        row = single_expiry_trade(raw, spot, cfg, costs, expiry)
        if row is not None:
            rows.append(row)

        if i % 25 == 0:
            print(
                f"processed {i}/{len(parsed)} expiries; trades={len(rows)}; "
                f"skipped_non_next={skipped_not_next}"
            )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trades = pd.DataFrame(rows).sort_values("entry_timestamp") if rows else pd.DataFrame()
    trades.to_csv(out / "baseline_trades.csv", index=False)
    pd.Series(performance_report(trades, cfg.starting_capital)).to_csv(out / "baseline_report.csv")
    print(pd.Series(performance_report(trades, cfg.starting_capital)).to_string())


if __name__ == "__main__":
    main()
