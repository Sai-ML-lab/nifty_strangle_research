from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import brentq

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
from src.data_ingest import normalize_options_vendor_file
from src.entry_timing_optimizer import (
    _entry_timestamp,
    _merge_spot_reference,
    report,
)

SESSION_OFFSETS = (2, 3, 4, 5, 6)
DEFAULT_SDS = (1.50, 1.75, 2.00, 2.25)
DEFAULT_TARGETS = (6000.0, 7000.0, 8000.0)
DEFAULT_STOPS = (2.0, 2.5)

POLICY_COLUMNS = [
    "session_offset",
    "sd",
    "target_rupees",
    "stop_multiple",
    "trades",
    "win_rate",
    "expectancy",
    "median_pnl",
    "avg_win",
    "avg_loss",
    "profit_factor",
    "max_drawdown",
    "positive_trade_rate",
    "median_win",
    "target_hit_rate",
    "target_hit_rate_wins",
    "first_half_expectancy",
    "second_half_expectancy",
    "stable_train",
]


def _trading_dates_before_expiry(x: pd.DataFrame, expiry: pd.Timestamp) -> list[pd.Timestamp]:
    dates = (
        x.loc[x["timestamp"].dt.normalize() < pd.Timestamp(expiry).normalize(), "timestamp"]
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    return [pd.Timestamp(d) for d in dates]


def session_entry_date(
    x: pd.DataFrame,
    expiry: pd.Timestamp,
    session_offset: int,
) -> pd.Timestamp | None:
    offset = int(session_offset)
    if offset < 1:
        raise ValueError("session_offset must be >= 1")
    dates = _trading_dates_before_expiry(x, expiry)
    if len(dates) < offset:
        return None
    return dates[-offset]


def _net_pnl_for_debit(
    credit_points: float,
    debit_points: float,
    *,
    quantity: int,
    costs: Costs,
    trade_date: pd.Timestamp,
) -> float:
    sell = max(float(credit_points), 0.0) * int(quantity)
    buy = max(float(debit_points), 0.0) * int(quantity)
    gross = (float(credit_points) - float(debit_points)) * int(quantity)
    return float(
        gross
        - trade_cost(
            sell + buy,
            sell,
            buy,
            4,
            costs,
            pd.Timestamp(trade_date),
        )
    )


def target_debit_for_net_profit(
    credit_points: float,
    *,
    target_rupees: float,
    quantity: int,
    costs: Costs,
    trade_date: pd.Timestamp,
) -> float | None:
    credit = max(float(credit_points), 0.0)
    target = float(target_rupees)
    if credit <= 0 or int(quantity) <= 0 or target <= 0:
        return None

    max_net = _net_pnl_for_debit(
        credit,
        0.0,
        quantity=int(quantity),
        costs=costs,
        trade_date=pd.Timestamp(trade_date),
    )
    if max_net < target:
        return None

    net_at_credit = _net_pnl_for_debit(
        credit,
        credit,
        quantity=int(quantity),
        costs=costs,
        trade_date=pd.Timestamp(trade_date),
    )
    if net_at_credit > target:
        return credit

    try:
        return float(
            brentq(
                lambda debit: _net_pnl_for_debit(
                    credit,
                    debit,
                    quantity=int(quantity),
                    costs=costs,
                    trade_date=pd.Timestamp(trade_date),
                )
                - target,
                0.0,
                credit,
            )
        )
    except ValueError:
        return None


def _quote(snapshot: pd.DataFrame, strike: float, option_type: str) -> pd.Series | None:
    q = snapshot[
        (snapshot["strike"] == float(strike))
        & (snapshot["option_type"] == option_type)
    ]
    return q.iloc[0] if not q.empty else None


def _entry_credit(snapshot: pd.DataFrame, put_k: float, call_k: float) -> float:
    p = _quote(snapshot, put_k, "PE")
    c = _quote(snapshot, call_k, "CE")
    if p is None or c is None:
        return np.nan
    return float(p["entry_sell_exec"] + c["entry_sell_exec"])


def _simulate_trade(
    *,
    x: pd.DataFrame,
    entry_ts: pd.Timestamp,
    expiry: pd.Timestamp,
    put_k: float,
    call_k: float,
    credit_points: float,
    target_rupees: float,
    stop_multiple: float,
    quantity: int,
    costs: Costs,
    time_exit_dte: int,
    time_exit_time: str,
) -> dict | None:
    target_debit = target_debit_for_net_profit(
        credit_points,
        target_rupees=target_rupees,
        quantity=quantity,
        costs=costs,
        trade_date=entry_ts,
    )
    if target_debit is None:
        return None

    path = x[
        (x["timestamp"] >= entry_ts)
        & (x["timestamp"].dt.normalize() <= expiry)
        & (x["expiry"] == expiry)
    ].copy()
    held = path[
        (
            (path["strike"] == float(put_k))
            & (path["option_type"] == "PE")
        )
        | (
            (path["strike"] == float(call_k))
            & (path["option_type"] == "CE")
        )
    ]
    marks = (
        held.pivot_table(
            index="timestamp",
            columns="option_type",
            values="buy_exec",
            aggfunc="first",
        )
        .reindex(columns=["PE", "CE"])
        .dropna()
    )

    exit_ts = None
    exit_reason = None
    exit_debit = None
    if not marks.empty:
        marks["debit"] = marks["PE"] + marks["CE"]
        stop_cut = float(credit_points) * float(stop_multiple)

        target_hits = marks.index[marks["debit"] <= float(target_debit)]
        stop_hits = marks.index[marks["debit"] >= stop_cut]
        target_ts = target_hits[0] if len(target_hits) else None
        stop_ts = stop_hits[0] if len(stop_hits) else None

        exit_date = (pd.Timestamp(expiry) - pd.Timedelta(days=int(time_exit_dte))).normalize()
        hh, mm = map(int, time_exit_time.split(":"))
        time_mask = (marks.index.normalize() == exit_date) & (
            (marks.index.hour > hh)
            | ((marks.index.hour == hh) & (marks.index.minute >= mm))
        )
        time_hits = marks.index[time_mask]
        time_ts = time_hits[0] if len(time_hits) else None

        candidates: list[tuple[pd.Timestamp, str]] = []
        if target_ts is not None:
            candidates.append((target_ts, "rupee_target"))
        if stop_ts is not None:
            candidates.append((stop_ts, "stop"))
        if time_ts is not None:
            candidates.append((time_ts, "time_exit"))
        if candidates:
            exit_ts, exit_reason = min(candidates, key=lambda z: z[0])
            exit_debit = float(marks.loc[exit_ts, "debit"])

    entry_snapshot = x[x["timestamp"] == entry_ts]
    p_entry = _quote(entry_snapshot, put_k, "PE")
    c_entry = _quote(entry_snapshot, call_k, "CE")
    if p_entry is None or c_entry is None:
        return None
    entry_put = float(p_entry["entry_sell_exec"])
    entry_call = float(c_entry["entry_sell_exec"])

    sell_premium = (entry_put + entry_call) * int(quantity)

    if exit_ts is None:
        expiry_rows = x[
            (x["timestamp"].dt.normalize() == pd.Timestamp(expiry).normalize())
            & (x["expiry"] == expiry)
        ]
        if expiry_rows.empty:
            expiry_rows = x[
                (x["timestamp"] <= pd.Timestamp(expiry) + pd.Timedelta(days=1))
                & (x["expiry"] == expiry)
            ]
        if expiry_rows.empty:
            return None
        spot_exit = float(expiry_rows.sort_values("timestamp")["spot"].iloc[-1])
        intrinsic = option_value_at_expiry(put_k, spot_exit, "PE") + option_value_at_expiry(
            call_k, spot_exit, "CE"
        )
        buy_premium = intrinsic * int(quantity)
        exit_ts = pd.Timestamp(expiry)
        exit_reason = "expiry"
        exit_debit = float(intrinsic)
        intrinsic_gap = 0.0
    else:
        buy_premium = float(exit_debit) * int(quantity)
        spot_rows = path[path["timestamp"] == exit_ts]["spot"]
        if spot_rows.empty:
            return None
        spot_exit = float(spot_rows.iloc[0])
        intrinsic = option_value_at_expiry(put_k, spot_exit, "PE") + option_value_at_expiry(
            call_k, spot_exit, "CE"
        )
        intrinsic_gap = float(exit_debit) - float(intrinsic)

    gross = sell_premium - buy_premium
    txn_cost = trade_cost(
        sell_premium + buy_premium,
        sell_premium,
        buy_premium,
        4,
        costs,
        pd.Timestamp(entry_ts),
    )
    net = float(gross - txn_cost)

    return {
        "entry_timestamp": pd.Timestamp(entry_ts),
        "expiry": pd.Timestamp(expiry),
        "entry_weekday": pd.Timestamp(entry_ts).day_name(),
        "session_offset": np.nan,
        "lot_size_historical": int(nifty_lot_size(expiry)),
        "quantity_current": int(quantity),
        "put_strike": float(put_k),
        "call_strike": float(call_k),
        "initial_credit_points": float(credit_points),
        "initial_credit_rupees_current": float(sell_premium),
        "target_rupees": float(target_rupees),
        "target_debit_points": float(target_debit),
        "target_capture_pct": float(
            100.0 * (credit_points - target_debit) / credit_points
        ),
        "exit_timestamp": pd.Timestamp(exit_ts),
        "exit_reason": str(exit_reason),
        "exit_debit_points": float(exit_debit),
        "exit_intrinsic_points": float(intrinsic),
        "exit_intrinsic_gap_points": float(intrinsic_gap),
        "gross_pnl_current": float(gross),
        "transaction_cost_current": float(txn_cost),
        "net_pnl_current": net,
        "target_hit": bool(exit_reason == "rupee_target"),
    }


def build_adaptive_target_trades(
    *,
    options_dir: Path,
    spot: pd.DataFrame,
    base_cfg,
    costs: Costs,
    session_offsets: Iterable[int] = SESSION_OFFSETS,
    sds: Iterable[float] = DEFAULT_SDS,
    target_rupees: Iterable[float] = DEFAULT_TARGETS,
    stop_multiples: Iterable[float] = DEFAULT_STOPS,
    reference_lot_size: int = 65,
    lots: int = 5,
) -> pd.DataFrame:
    files: list[tuple[pd.Timestamp, Path]] = []
    for f in sorted(options_dir.rglob("*.parquet")):
        stem = f.stem.replace("expiry=", "")
        try:
            expiry = pd.Timestamp(stem).normalize()
        except Exception:
            continue
        files.append((expiry, f))
    if not files:
        raise ValueError(f"No expiry Parquets found under {options_dir}")

    earliest: dict[tuple[int, pd.Timestamp], pd.Timestamp] = {}
    rows: list[dict] = []
    quantity = int(reference_lot_size) * int(lots)

    for idx, (expiry, path) in enumerate(files, 1):
        raw = pd.read_parquet(path)
        x = normalize_options_vendor_file(raw)
        if getattr(x["timestamp"].dt, "tz", None) is not None:
            x["timestamp"] = x["timestamp"].dt.tz_localize(None)
        x = _merge_spot_reference(x, spot)
        x["future"] = np.nan
        if "open" in x.columns:
            x["entry_price"] = pd.to_numeric(x["open"], errors="coerce")
        else:
            x["entry_price"] = pd.to_numeric(x["ltp"], errors="coerce")
        x = add_executable_prices(x, slippage_points=costs.slippage_points_per_leg)
        x["expiry"] = expiry

        dates = _trading_dates_before_expiry(x, expiry)
        if not dates:
            continue

        for offset in session_offsets:
            entry_date = session_entry_date(x, expiry, int(offset))
            if entry_date is None:
                continue
            key = (int(offset), entry_date)
            prev = earliest.get(key)
            if prev is None or expiry < prev:
                earliest[key] = expiry

        for offset in session_offsets:
            entry_date = session_entry_date(x, expiry, int(offset))
            if entry_date is None or earliest.get((int(offset), entry_date)) != expiry:
                continue
            entry_ts = _entry_timestamp(x, entry_date, base_cfg.entry_time)
            if entry_ts is None:
                continue

            entry_chain = x[
                x["timestamp"].eq(entry_ts)
            ].dropna(subset=["spot", "entry_price"]).copy()
            if entry_chain.empty:
                continue

            spot0 = float(entry_chain["spot"].iloc[0])
            forward = estimate_forward_from_parity(
                entry_chain,
                spot0,
                base_cfg.risk_free_rate,
                price_column="entry_price",
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
                entry_chain["forward"] = forward
            except Exception:
                continue

            for sd in sds:
                cfg = replace(
                    base_cfg,
                    target_dte=int(offset),
                    sd_multiple=float(sd),
                    strike_method="pure_sd",
                )
                try:
                    put, call = select_strikes(
                        entry_chain,
                        forward,
                        atm_iv,
                        expiry,
                        entry_ts,
                        cfg,
                    )
                except Exception:
                    continue

                put_k = float(put["strike"])
                call_k = float(call["strike"])
                credit = _entry_credit(entry_chain, put_k, call_k)
                if not np.isfinite(credit) or credit < base_cfg.min_entry_credit_points:
                    continue

                for target in target_rupees:
                    for stop in stop_multiples:
                        trade = _simulate_trade(
                            x=x,
                            entry_ts=entry_ts,
                            expiry=expiry,
                            put_k=put_k,
                            call_k=call_k,
                            credit_points=credit,
                            target_rupees=float(target),
                            stop_multiple=float(stop),
                            quantity=quantity,
                            costs=costs,
                            time_exit_dte=base_cfg.time_exit_dte,
                            time_exit_time=base_cfg.time_exit_time,
                        )
                        if trade is None:
                            continue
                        trade.update(
                            {
                                "session_offset": int(offset),
                                "sd": float(sd),
                                "target_rupees": float(target),
                                "stop_multiple": float(stop),
                                "entry_spot": float(spot0),
                                "forward_entry": float(forward),
                                "atm_iv": float(atm_iv),
                                "rv20": (
                                    float(entry_chain["rv20"].iloc[0])
                                    if "rv20" in entry_chain.columns
                                    and pd.notna(entry_chain["rv20"].iloc[0])
                                    else np.nan
                                ),
                                "iv_rv_spread": np.nan,
                                "put_delta": float(put["delta"]),
                                "call_delta": float(call["delta"]),
                            }
                        )
                        trade["iv_rv_spread"] = (
                            trade["atm_iv"] - trade["rv20"]
                            if pd.notna(trade["rv20"])
                            else np.nan
                        )
                        rows.append(trade)

        if idx % 25 == 0:
            print(
                f"processed {idx}/{len(files)} expiries; "
                f"adaptive target trades={len(rows):,}"
            )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values(
        [
            "entry_timestamp",
            "expiry",
            "session_offset",
            "sd",
            "target_rupees",
            "stop_multiple",
        ]
    )
    return out.reset_index(drop=True)


def evaluate_adaptive_policies(
    trades: pd.DataFrame,
    *,
    min_trades: int = 30,
    min_half_trades: int = 10,
) -> pd.DataFrame:
    rows: list[dict] = []
    for key, group in trades.groupby(
        ["session_offset", "sd", "target_rupees", "stop_multiple"],
        dropna=False,
    ):
        offset, sd, target, stop = key
        g = group.sort_values("entry_timestamp").copy()
        if len(g) < int(min_trades):
            continue

        p = g["net_pnl_current"].astype(float)
        target_hit = g["target_hit"].astype(bool)
        wins = p[p > 0]
        losses = p[p < 0]
        mid = g["entry_timestamp"].min() + (
            g["entry_timestamp"].max() - g["entry_timestamp"].min()
        ) / 2
        first = g[g["entry_timestamp"] <= mid]
        second = g[g["entry_timestamp"] > mid]

        def simple_stats(h: pd.DataFrame) -> tuple[float, float]:
            if len(h) < int(min_half_trades):
                return np.nan, np.nan
            hp = h["net_pnl_current"].astype(float)
            hw = hp[hp > 0]
            hl = hp[hp < 0]
            pf = (
                float(hw.sum() / abs(hl.sum()))
                if len(hl) and abs(hl.sum()) > 0
                else (np.inf if len(hw) else np.nan)
            )
            return float(hp.mean()), pf

        first_exp, first_pf = simple_stats(first)
        second_exp, second_pf = simple_stats(second)
        stable = (
            np.isfinite(first_exp)
            and np.isfinite(second_exp)
            and first_exp > 0
            and second_exp > 0
            and (first_pf >= 1.0 or np.isinf(first_pf))
            and (second_pf >= 1.0 or np.isinf(second_pf))
        )
        pf = float(wins.sum() / abs(losses.sum())) if len(losses) and abs(losses.sum()) > 0 else np.inf
        rows.append(
            {
                "session_offset": int(offset),
                "sd": float(sd),
                "target_rupees": float(target),
                "stop_multiple": float(stop),
                "trades": int(len(g)),
                "win_rate": float((p > 0).mean()),
                "expectancy": float(p.mean()),
                "median_pnl": float(p.median()),
                "avg_win": float(wins.mean()) if len(wins) else 0.0,
                "avg_loss": float(losses.mean()) if len(losses) else 0.0,
                "profit_factor": pf,
                "max_drawdown": float((p.cumsum() - p.cumsum().cummax()).min()),
                "positive_trade_rate": float((p > 0).mean()),
                "median_win": float(wins.median()) if len(wins) else np.nan,
                "target_hit_rate": float(target_hit.mean()),
                "target_hit_rate_wins": (
                    float(g.loc[p > 0, "target_hit"].mean()) if len(wins) else np.nan
                ),
                "first_half_expectancy": float(first_exp),
                "second_half_expectancy": float(second_exp),
                "stable_train": bool(stable),
            }
        )
    if not rows:
        return pd.DataFrame(columns=POLICY_COLUMNS)
    return pd.DataFrame(rows, columns=POLICY_COLUMNS)


def walk_forward_adaptive(
    trades: pd.DataFrame,
    *,
    train_months: int = 12,
    test_months: int = 3,
    rebalance_months: int = 3,
    min_train_trades: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "train_start", "train_end", "test_start", "test_end",
                "selected_session_offset", "selected_sd", "selected_target_rupees",
                "selected_stop_multiple", "train_trades", "train_expectancy",
                "train_profit_factor", "train_target_hit_rate", "test_trades",
                "selection_status",
            ]
        ), pd.DataFrame()

    x = trades.copy()
    x["entry_timestamp"] = pd.to_datetime(x["entry_timestamp"])
    start = pd.Timestamp(x["entry_timestamp"].min().year, x["entry_timestamp"].min().month, 1)
    end = pd.Timestamp(x["entry_timestamp"].max()).normalize()
    selected_rows: list[dict] = []
    oos_rows: list[pd.DataFrame] = []
    train_start = start

    def month_add(ts: pd.Timestamp, n: int) -> pd.Timestamp:
        return ts + pd.offsets.MonthBegin(n)

    while True:
        train_end = month_add(train_start, train_months)
        test_end = month_add(train_end, test_months)
        if test_end > end + pd.Timedelta(days=1):
            break

        train = x[
            (x["entry_timestamp"] >= train_start)
            & (x["entry_timestamp"] < train_end)
        ]
        test = x[
            (x["entry_timestamp"] >= train_end)
            & (x["entry_timestamp"] < test_end)
        ]
        if train.empty or test.empty:
            train_start = month_add(train_start, rebalance_months)
            continue

        policies = evaluate_adaptive_policies(
            train,
            min_trades=min_train_trades,
            min_half_trades=max(10, min_train_trades // 3),
        )
        if policies.empty:
            train_start = month_add(train_start, rebalance_months)
            continue

        eligible = policies[policies["stable_train"]].copy()
        if eligible.empty:
            selected_rows.append(
                {
                    "train_start": train_start,
                    "train_end": train_end - pd.Timedelta(days=1),
                    "test_start": train_end,
                    "test_end": test_end - pd.Timedelta(days=1),
                    "selected_session_offset": np.nan,
                    "selected_sd": np.nan,
                    "selected_target_rupees": np.nan,
                    "selected_stop_multiple": np.nan,
                    "train_trades": int(len(train)),
                    "train_expectancy": np.nan,
                    "train_profit_factor": np.nan,
                    "train_target_hit_rate": np.nan,
                    "test_trades": 0,
                    "selection_status": "NO_STABLE_POLICY",
                }
            )
            train_start = month_add(train_start, rebalance_months)
            continue

        pool = eligible.sort_values(
            ["expectancy", "profit_factor", "target_hit_rate"],
            ascending=[False, False, False],
        )
        best = pool.iloc[0]

        mask = (
            x["session_offset"].eq(best["session_offset"])
            & x["sd"].eq(best["sd"])
            & x["target_rupees"].eq(best["target_rupees"])
            & x["stop_multiple"].eq(best["stop_multiple"])
        )
        test_slice = x[
            mask
            & (x["entry_timestamp"] >= train_end)
            & (x["entry_timestamp"] < test_end)
        ].copy()
        selected_rows.append(
            {
                "train_start": train_start,
                "train_end": train_end - pd.Timedelta(days=1),
                "test_start": train_end,
                "test_end": test_end - pd.Timedelta(days=1),
                "selected_session_offset": int(best["session_offset"]),
                "selected_sd": float(best["sd"]),
                "selected_target_rupees": float(best["target_rupees"]),
                "selected_stop_multiple": float(best["stop_multiple"]),
                "train_trades": int(best["trades"]),
                "train_expectancy": float(best["expectancy"]),
                "train_profit_factor": float(best["profit_factor"]),
                "train_target_hit_rate": float(best["target_hit_rate"]),
                "test_trades": int(len(test_slice)),
                "selection_status": "SELECTED",
            }
        )
        if not test_slice.empty:
            test_slice["wf_train_start"] = train_start
            test_slice["wf_test_start"] = train_end
            oos_rows.append(test_slice)

        train_start = month_add(train_start, rebalance_months)

    selected_columns = [
        "train_start", "train_end", "test_start", "test_end",
        "selected_session_offset", "selected_sd", "selected_target_rupees",
        "selected_stop_multiple", "train_trades", "train_expectancy",
        "train_profit_factor", "train_target_hit_rate", "test_trades",
        "selection_status",
    ]
    selected = pd.DataFrame(selected_rows, columns=selected_columns)
    oos = pd.concat(oos_rows, ignore_index=True) if oos_rows else pd.DataFrame()
    return selected, oos


def summarize_oos(oos: pd.DataFrame) -> dict:
    if oos.empty:
        return {"trades": 0}
    result = report(
        oos.rename(columns={"net_pnl_current": "net_pnl_5lot_current"}),
        pnl_column="net_pnl_5lot_current",
        target_column="target_hit",
    )
    fold_pnl = oos.groupby("wf_test_start")["net_pnl_current"].sum()
    result["positive_test_folds"] = int((fold_pnl > 0).sum())
    result["total_test_folds"] = int(len(fold_pnl))
    result["target_hit_rate"] = float(oos["target_hit"].mean())
    result["target_hit_rate_wins"] = float(
        oos.loc[oos["net_pnl_current"] > 0, "target_hit"].mean()
    )
    result["oos_start"] = str(pd.Timestamp(oos["entry_timestamp"].min()).date())
    result["oos_end"] = str(pd.Timestamp(oos["entry_timestamp"].max()).date())
    return result
