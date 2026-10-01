from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.core import (
    Costs,
    add_executable_prices,
    ensure_iv_delta,
    estimate_forward_from_parity,
    nifty_lot_size,
    option_value_at_expiry,
    pick_atm_iv,
    select_strikes,
    trade_cost,
)
from src.data_ingest import normalize_options_vendor_file, normalize_spot_file
from src.strangle_backtest import load_config


DEFAULT_DTES = (4, 5, 6, 7, 8)
DEFAULT_SDS = (1.50, 1.75, 2.00, 2.25)
DEFAULT_PROFIT_CAPTURES = (0.50, 0.75)
DEFAULT_STOP_MULTIPLES = (2.0, 2.5)
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")


def load_spot(path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    x = normalize_spot_file(raw).sort_values("timestamp").copy()
    if getattr(x["timestamp"].dt, "tz", None) is not None:
        x["timestamp"] = x["timestamp"].dt.tz_localize(None)
    return x


def expiry_from_filename(path: Path) -> pd.Timestamp | None:
    stem = path.stem.replace("expiry=", "")
    try:
        return pd.Timestamp(stem).normalize()
    except Exception:
        return None


def _entry_timestamp(x: pd.DataFrame, entry_date: pd.Timestamp, entry_time: str) -> pd.Timestamp | None:
    hh, mm = map(int, entry_time.split(":"))
    m = (
        (x["timestamp"].dt.normalize() == entry_date.normalize())
        & (
            (x["timestamp"].dt.hour > hh)
            | ((x["timestamp"].dt.hour == hh) & (x["timestamp"].dt.minute >= mm))
        )
    )
    values = x.loc[m, "timestamp"]
    return values.min() if not values.empty else None


def _daily_rv20(spot: pd.DataFrame) -> pd.Series:
    y = spot.copy()
    y["date"] = y["timestamp"].dt.normalize()
    daily = y.groupby("date", as_index=False)["spot"].last().sort_values("date")
    log_ret = np.log(daily["spot"]).diff()
    daily["rv20"] = log_ret.rolling(20).std() * np.sqrt(252)
    daily["rv20_available_at_open"] = daily["rv20"].shift(1)
    return daily.set_index("date")["rv20_available_at_open"]


def _quote(snapshot: pd.DataFrame, strike: float, option_type: str) -> pd.Series | None:
    q = snapshot[(snapshot["strike"] == float(strike)) & (snapshot["option_type"] == option_type)]
    return q.iloc[0] if not q.empty else None


def _entry_credit(snapshot: pd.DataFrame, put_k: float, call_k: float) -> float:
    p = _quote(snapshot, put_k, "PE")
    c = _quote(snapshot, call_k, "CE")
    if p is None or c is None:
        return np.nan
    return float(p["entry_sell_exec"] + c["entry_sell_exec"])


def _exit_trade(
    *,
    x: pd.DataFrame,
    entry_ts: pd.Timestamp,
    expiry: pd.Timestamp,
    put_k: float,
    call_k: float,
    credit_points: float,
    profit_capture: float,
    stop_multiple: float,
    lots: int,
    costs: Costs,
    time_exit_dte: int,
    time_exit_time: str,
) -> dict:
    contract_lot = nifty_lot_size(expiry)
    multiplier = int(lots) * contract_lot
    path = x[
        (x["timestamp"] >= entry_ts)
        & (x["timestamp"].dt.normalize() <= expiry)
        & (x["expiry"] == expiry)
    ].copy()
    held = path[
        ((path["strike"] == put_k) & (path["option_type"] == "PE"))
        | ((path["strike"] == call_k) & (path["option_type"] == "CE"))
    ]
    marks = (
        held.pivot_table(index="timestamp", columns="option_type", values="buy_exec", aggfunc="first")
        .reindex(columns=["PE", "CE"])
        .dropna()
    )

    exit_ts = None
    exit_reason = None
    exit_debit = np.nan
    if not marks.empty:
        marks["debit"] = marks["PE"] + marks["CE"]
        profit_cut = credit_points * (1.0 - float(profit_capture))
        stop_cut = credit_points * float(stop_multiple)
        hit = marks.index[(marks["debit"] <= profit_cut) | (marks["debit"] >= stop_cut)]
        if len(hit):
            exit_ts = hit[0]
            exit_reason = "profit_target" if marks.loc[exit_ts, "debit"] <= profit_cut else "stop"

        exit_date = (expiry - pd.Timedelta(days=int(time_exit_dte))).normalize()
        hh, mm = map(int, time_exit_time.split(":"))
        time_mask = (marks.index.normalize() == exit_date) & (
            (marks.index.hour > hh)
            | ((marks.index.hour == hh) & (marks.index.minute >= mm))
        )
        time_hits = marks.index[time_mask]
        candidates = []
        if exit_ts is not None:
            candidates.append((exit_ts, exit_reason))
        if len(time_hits):
            candidates.append((time_hits[0], "time_exit"))
        if candidates:
            exit_ts, exit_reason = min(candidates, key=lambda z: z[0])
            exit_debit = float(marks.loc[exit_ts, "debit"])

    p_entry = _quote(x[x["timestamp"] == entry_ts], put_k, "PE")
    c_entry = _quote(x[x["timestamp"] == entry_ts], call_k, "CE")
    if p_entry is None or c_entry is None:
        raise ValueError("Missing entry quote after strike selection")
    p_entry_px = float(p_entry["entry_sell_exec"])
    c_entry_px = float(c_entry["entry_sell_exec"])
    sell_premium = (p_entry_px + c_entry_px) * multiplier

    if exit_ts is None:
        spot_rows = x[x["timestamp"].dt.normalize() == expiry]
        if spot_rows.empty:
            spot_rows = x[x["timestamp"] <= expiry + pd.Timedelta(days=1)]
            spot_rows = spot_rows[spot_rows["expiry"] == expiry]
        if spot_rows.empty:
            raise ValueError("No expiry spot available")
        spot_exit = float(spot_rows.sort_values("timestamp")["spot"].iloc[-1])
        intrinsic = option_value_at_expiry(put_k, spot_exit, "PE") + option_value_at_expiry(
            call_k, spot_exit, "CE"
        )
        exit_debit = float(intrinsic)
        buy_premium = exit_debit * multiplier
        exit_ts = expiry
        exit_reason = "expiry"
        intrinsic_gap = 0.0
    else:
        buy_premium = float(exit_debit) * multiplier
        spot_slice = path[path["timestamp"] == exit_ts]["spot"]
        if spot_slice.empty:
            raise ValueError("Missing spot at exit timestamp")
        spot_exit = float(spot_slice.iloc[0])
        intrinsic = option_value_at_expiry(put_k, spot_exit, "PE") + option_value_at_expiry(
            call_k, spot_exit, "CE"
        )
        intrinsic_gap = float(exit_debit - intrinsic)

    gross = sell_premium - buy_premium
    txn_cost = trade_cost(
        sell_premium + buy_premium,
        sell_premium,
        buy_premium,
        4,
        costs,
        pd.Timestamp(entry_ts),
    )
    return {
        "entry_timestamp": pd.Timestamp(entry_ts),
        "expiry": pd.Timestamp(expiry),
        "entry_weekday": pd.Timestamp(entry_ts).day_name(),
        "lot_size": int(contract_lot),
        "put_strike": float(put_k),
        "call_strike": float(call_k),
        "initial_credit_points": float(credit_points),
        "initial_credit_rupees": float(sell_premium),
        "exit_timestamp": pd.Timestamp(exit_ts),
        "exit_reason": str(exit_reason),
        "exit_debit_points": float(exit_debit),
        "exit_intrinsic_points": float(intrinsic),
        "exit_intrinsic_gap_points": float(intrinsic_gap),
        "gross_pnl": float(gross),
        "transaction_cost": float(txn_cost),
        "net_pnl": float(gross - txn_cost),
    }


def build_candidate_trades(
    *,
    options_dir: Path,
    spot: pd.DataFrame,
    base_cfg,
    costs: Costs,
    dtes: Iterable[int],
    sds: Iterable[float],
    profit_captures: Iterable[float],
    stop_multiples: Iterable[float],
) -> pd.DataFrame:
    files = []
    for f in sorted(options_dir.rglob("*.parquet")):
        expiry = expiry_from_filename(f)
        if expiry is not None:
            files.append((expiry, f))
    if not files:
        raise ValueError(f"No expiry Parquets found under {options_dir}")

    rv20 = _daily_rv20(spot)
    rows: list[dict] = []
    earliest_by_dte_entry: dict[tuple[int, pd.Timestamp], pd.Timestamp] = {}
    for expiry, _ in files:
        for dte in dtes:
            entry_date = (expiry - pd.Timedelta(days=int(dte))).normalize()
            key = (int(dte), entry_date)
            previous = earliest_by_dte_entry.get(key)
            if previous is None or expiry < previous:
                earliest_by_dte_entry[key] = expiry

    for idx, (expiry, path) in enumerate(files, 1):
        raw = pd.read_parquet(path)
        x = normalize_options_vendor_file(raw)
        if getattr(x["timestamp"].dt, "tz", None) is not None:
            x["timestamp"] = x["timestamp"].dt.tz_localize(None)
        x = x.merge(spot, on="timestamp", how="left", validate="many_to_one")
        x["future"] = np.nan
        if "open" in x.columns:
            x["entry_price"] = pd.to_numeric(x["open"], errors="coerce")
        else:
            x["entry_price"] = pd.to_numeric(x["ltp"], errors="coerce")
        x = add_executable_prices(x, slippage_points=costs.slippage_points_per_leg)
        x["expiry"] = expiry

        for dte in dtes:
            entry_date = (expiry - pd.Timedelta(days=int(dte))).normalize()
            if earliest_by_dte_entry.get((int(dte), entry_date)) != expiry:
                continue
            entry_ts = _entry_timestamp(x, entry_date, base_cfg.entry_time)
            if entry_ts is None:
                continue

            entry_chain = x[x["timestamp"] == entry_ts].dropna(subset=["spot", "entry_price"]).copy()
            if entry_chain.empty:
                continue

            spot0 = float(entry_chain["spot"].iloc[0])
            forward = estimate_forward_from_parity(
                entry_chain, spot0, base_cfg.risk_free_rate, price_column="entry_price"
            )
            entry_chain["forward"] = forward
            try:
                entry_chain = ensure_iv_delta(
                    entry_chain,
                    base_cfg.risk_free_rate,
                    allow_iv_calc=True,
                    price_column="entry_price",
                )
                atm_iv = pick_atm_iv(entry_chain, forward, base_cfg.atm_band)
                rv = rv20.get(entry_date, np.nan)
            except Exception:
                continue

            for sd in sds:
                cfg = replace(base_cfg, target_dte=int(dte), sd_multiple=float(sd))
                try:
                    put, call = select_strikes(entry_chain, forward, atm_iv, expiry, entry_ts, cfg)
                except Exception:
                    continue
                put_k, call_k = float(put["strike"]), float(call["strike"])
                credit = _entry_credit(entry_chain, put_k, call_k)
                if not np.isfinite(credit) or credit < float(base_cfg.min_entry_credit_points):
                    continue

                base = {
                    "dte": int(dte),
                    "sd": float(sd),
                    "entry_timestamp": pd.Timestamp(entry_ts),
                    "expiry": pd.Timestamp(expiry),
                    "entry_weekday": pd.Timestamp(entry_ts).day_name(),
                    "entry_spot": float(spot0),
                    "forward_entry": float(forward),
                    "atm_iv": float(atm_iv),
                    "rv20": float(rv) if pd.notna(rv) else np.nan,
                    "iv_rv_spread": float(atm_iv - rv) if pd.notna(rv) else np.nan,
                    "put_strike": put_k,
                    "call_strike": call_k,
                    "put_delta": float(put["delta"]),
                    "call_delta": float(call["delta"]),
                }
                for capture in profit_captures:
                    for stop in stop_multiples:
                        trade = _exit_trade(
                            x=x,
                            entry_ts=entry_ts,
                            expiry=expiry,
                            put_k=put_k,
                            call_k=call_k,
                            credit_points=credit,
                            profit_capture=float(capture),
                            stop_multiple=float(stop),
                            lots=1,
                            costs=costs,
                            time_exit_dte=base_cfg.time_exit_dte,
                            time_exit_time=base_cfg.time_exit_time,
                        )
                        row = {
                            **base,
                            "profit_capture": float(capture),
                            "stop_multiple": float(stop),
                            **trade,
                        }
                        # The spread/strategy fields are restored because _exit_trade also
                        # returns the common trade fields.
                        row["dte"] = int(dte)
                        row["sd"] = float(sd)
                        row["profit_capture"] = float(capture)
                        row["stop_multiple"] = float(stop)
                        rows.append(row)

        if idx % 25 == 0:
            print(f"processed {idx}/{len(files)} expiries; candidate trades={len(rows):,}")

    out = pd.DataFrame(rows).sort_values(["entry_timestamp", "expiry", "dte", "sd", "profit_capture", "stop_multiple"])
    return out.reset_index(drop=True)


def _policy_keys(max_excluded_days: int = 2) -> list[frozenset[str]]:
    from itertools import combinations

    policies: list[frozenset[str]] = [frozenset()]
    for size in range(1, max(1, int(max_excluded_days)) + 1):
        for combo in combinations(WEEKDAYS, size):
            policies.append(frozenset(combo))
    return policies


def add_lot_metrics(
    trades: pd.DataFrame,
    costs: Costs,
    lots: int = 5,
    target_low: float = 6000.0,
    target_high: float = 8000.0,
) -> pd.DataFrame:
    x = trades.copy()
    if x.empty:
        return x
    qty = int(lots)
    x["gross_pnl_points"] = x["initial_credit_points"] - x["exit_debit_points"]
    lot_qty = qty * x["lot_size"].astype(int)
    sell = x["initial_credit_points"] * lot_qty
    buy = x["exit_debit_points"] * lot_qty
    x["net_pnl_5lot"] = [
        float(
            gross
            - trade_cost(
                s + b,
                s,
                b,
                4,
                costs,
                pd.Timestamp(ts),
            )
        )
        for gross, s, b, ts in zip(
            x["gross_pnl_points"] * lot_qty,
            sell,
            buy,
            x["entry_timestamp"],
        )
    ]
    x["profit_target_pnl_5lot"] = (
        x["initial_credit_points"] * x["profit_capture"] * lot_qty
    )
    x["target_band_hit_5lot"] = x["net_pnl_5lot"].between(
        float(target_low), float(target_high), inclusive="both"
    )
    return x


def apply_policy(trades: pd.DataFrame, exclude_weekdays: frozenset[str]) -> pd.DataFrame:
    if not exclude_weekdays:
        return trades
    return trades[~trades["entry_weekday"].isin(exclude_weekdays)]


def report(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "trades": 0,
            "win_rate": np.nan,
            "expectancy": np.nan,
            "median_pnl": np.nan,
            "avg_win": np.nan,
            "avg_loss": np.nan,
            "profit_factor": np.nan,
            "max_drawdown": np.nan,
            "positive_trade_rate": np.nan,
            "median_win": np.nan,
            "target_band_rate_all": np.nan,
            "target_band_rate_wins": np.nan,
        }
    p = trades["net_pnl_5lot"].astype(float)
    wins = p[p > 0]
    losses = p[p < 0]
    ordered = trades.sort_values("entry_timestamp")
    p_ordered = ordered["net_pnl_5lot"].astype(float)
    equity = p_ordered.cumsum()
    dd = equity - equity.cummax()
    return {
        "trades": int(len(p)),
        "win_rate": float((p > 0).mean()),
        "expectancy": float(p.mean()),
        "median_pnl": float(p.median()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) else np.inf,
        "max_drawdown": float(dd.min()),
        "positive_trade_rate": float((p > 0).mean()),
        "median_win": float(wins.median()) if len(wins) else np.nan,
        "target_band_rate_all": float(trades["target_band_hit_5lot"].mean()),
        "target_band_rate_wins": float(trades.loc[p > 0, "target_band_hit_5lot"].mean()) if len(wins) else np.nan,
    }


def evaluate_policies(
    trades: pd.DataFrame,
    *,
    min_trades: int = 40,
    train_only: bool = False,
    max_excluded_days: int = 2,
) -> pd.DataFrame:
    rows = []
    base_groups = trades.groupby(["dte", "sd", "profit_capture", "stop_multiple"], dropna=False)
    for key, group in base_groups:
        dte, sd, capture, stop = key
        for excluded in _policy_keys(max_excluded_days=max_excluded_days):
            g = apply_policy(group, excluded).sort_values("entry_timestamp")
            if len(g) < min_trades:
                continue
            r = report(g)
            mid = g["entry_timestamp"].min() + (
                g["entry_timestamp"].max() - g["entry_timestamp"].min()
            ) / 2
            first_half = g[g["entry_timestamp"] <= mid]
            second_half = g[g["entry_timestamp"] > mid]
            r1 = report(first_half) if len(first_half) >= max(10, min_trades // 3) else {"expectancy": np.nan, "profit_factor": np.nan}
            r2 = report(second_half) if len(second_half) >= max(10, min_trades // 3) else {"expectancy": np.nan, "profit_factor": np.nan}
            stable_train = (
                np.isfinite(r1["expectancy"])
                and np.isfinite(r2["expectancy"])
                and r1["expectancy"] > 0
                and r2["expectancy"] > 0
                and r1["profit_factor"] >= 0.9
                and r2["profit_factor"] >= 0.9
            )
            if train_only and (
                not np.isfinite(r["expectancy"])
                or r["expectancy"] <= 0
                or r["profit_factor"] < 1.0
                or not stable_train
            ):
                continue
            rows.append({
                "dte": int(dte),
                "sd": float(sd),
                "profit_capture": float(capture),
                "stop_multiple": float(stop),
                "excluded_weekdays": ",".join(sorted(excluded)) if excluded else "NONE",
                "excluded_count": len(excluded),
                "first_half_expectancy": float(r1["expectancy"]),
                "second_half_expectancy": float(r2["expectancy"]),
                "stable_train": bool(stable_train),
                **r,
            })
    return pd.DataFrame(rows)


def walk_forward_select(
    trades: pd.DataFrame,
    *,
    train_months: int = 12,
    test_months: int = 3,
    rebalance_months: int = 3,
    min_train_trades: int = 25,
    target_low: float = 6000.0,
    target_high: float = 8000.0,
    max_excluded_days: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame()
    x = trades.copy()
    x["entry_timestamp"] = pd.to_datetime(x["entry_timestamp"])
    start = pd.Timestamp(x["entry_timestamp"].min().year, x["entry_timestamp"].min().month, 1)
    end = pd.Timestamp(x["entry_timestamp"].max()).normalize()
    selected_rows = []
    oos_rows = []
    train_start = start

    def month_add(ts: pd.Timestamp, n: int) -> pd.Timestamp:
        return ts + pd.offsets.MonthBegin(n)

    while True:
        train_end = month_add(train_start, train_months)
        test_end = month_add(train_end, test_months)
        if test_end > end + pd.Timedelta(days=1):
            break

        train = x[(x["entry_timestamp"] >= train_start) & (x["entry_timestamp"] < train_end)]
        test = x[(x["entry_timestamp"] >= train_end) & (x["entry_timestamp"] < test_end)]
        if train.empty or test.empty:
            train_start = month_add(train_start, rebalance_months)
            continue

        policies = evaluate_policies(
            train,
            min_trades=min_train_trades,
            train_only=True,
            max_excluded_days=max_excluded_days,
        )
        if policies.empty:
            train_start = month_add(train_start, rebalance_months)
            continue

        midpoint = (target_low + target_high) / 2.0
        band = policies["median_win"].sub(midpoint).abs()
        policies["target_gap"] = band
        in_band = policies[
            policies["median_win"].between(target_low, target_high, inclusive="both")
        ].copy()
        pool = in_band if not in_band.empty else policies
        pool = pool.sort_values(
            ["expectancy", "profit_factor", "target_gap"],
            ascending=[False, False, True],
        )
        best = pool.iloc[0]

        key_mask = (
            x["dte"].eq(best["dte"])
            & x["sd"].eq(best["sd"])
            & x["profit_capture"].eq(best["profit_capture"])
            & x["stop_multiple"].eq(best["stop_multiple"])
        )
        excluded = frozenset(
            s for s in str(best["excluded_weekdays"]).split(",") if s and s != "NONE"
        )
        test_cfg = apply_policy(x[key_mask], excluded)
        test_slice = test_cfg[
            (test_cfg["entry_timestamp"] >= train_end)
            & (test_cfg["entry_timestamp"] < test_end)
        ].copy()

        selected_rows.append({
            "train_start": train_start,
            "train_end": train_end - pd.Timedelta(days=1),
            "test_start": train_end,
            "test_end": test_end - pd.Timedelta(days=1),
            "selected_dte": int(best["dte"]),
            "selected_sd": float(best["sd"]),
            "selected_profit_capture": float(best["profit_capture"]),
            "selected_stop_multiple": float(best["stop_multiple"]),
            "excluded_weekdays": str(best["excluded_weekdays"]),
            "train_trades": int(best["trades"]),
            "train_expectancy_5lot": float(best["expectancy"]),
            "train_profit_factor": float(best["profit_factor"]),
            "train_median_win_5lot": float(best["median_win"]),
            "train_target_band_rate_wins": float(best["target_band_rate_wins"]),
            "test_trades": int(len(test_slice)),
        })
        if not test_slice.empty:
            test_slice = test_slice.copy()
            test_slice["wf_train_start"] = train_start
            test_slice["wf_test_start"] = train_end
            test_slice["selected_excluded_weekdays"] = str(best["excluded_weekdays"])
            oos_rows.append(test_slice)

        train_start = month_add(train_start, rebalance_months)

    oos = pd.concat(oos_rows, ignore_index=True) if oos_rows else pd.DataFrame()
    return pd.DataFrame(selected_rows), oos
