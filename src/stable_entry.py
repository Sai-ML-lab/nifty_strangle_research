from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from run_batch_research import entry_date_for_expiry, expiry_from_filename, load_spot, single_expiry_trade
from src.core import Costs, StrategyConfig
from src.risk_metrics import portfolio_risk_metrics


def _stats(x: pd.DataFrame) -> dict:
    p = pd.to_numeric(x.get("net_pnl", pd.Series(dtype=float)), errors="coerce").dropna()
    if p.empty:
        return {"trades": 0, "expectancy": np.nan, "profit_factor": np.nan, "net_pnl": 0.0}
    wins, losses = p[p > 0], p[p < 0]
    return {
        "trades": int(len(p)),
        "expectancy": float(p.mean()),
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) else np.inf,
        "net_pnl": float(p.sum()),
    }


def _filter_trades(x: pd.DataFrame, threshold: float | None) -> pd.DataFrame:
    if x.empty:
        return x.copy()
    if "data_quality_flag" in x.columns:
        x = x[x["data_quality_flag"].fillna("PASS") == "PASS"].copy()
    if threshold is None:
        return x
    return x[x["iv_rv_spread"].notna() & (x["iv_rv_spread"] >= threshold / 100.0)].copy()


def build_sd_ledgers(
    options_dir: str | Path,
    spot_path: str | Path,
    base_cfg: StrategyConfig,
    costs: Costs,
    sd_values: Iterable[float],
) -> dict[float, pd.DataFrame]:
    """Build one clean, unfiltered trade ledger for each SD value."""
    spot = load_spot(Path(spot_path))
    files = []
    for f in sorted(Path(options_dir).rglob("*.parquet")):
        expiry = expiry_from_filename(f)
        if expiry is not None:
            files.append((expiry, f))
    files.sort(key=lambda z: z[0])
    chosen = {}
    for expiry, f in files:
        d = entry_date_for_expiry(expiry, base_cfg)
        chosen.setdefault(d, (expiry, f))
    ledgers = {float(sd): [] for sd in sd_values}
    for i, (_, (expiry, f)) in enumerate(sorted(chosen.items()), 1):
        raw = pd.read_parquet(f)
        for sd in ledgers:
            cfg = replace(base_cfg, sd_multiple=sd, min_iv_rv_spread=None)
            row = single_expiry_trade(raw, spot, cfg, costs, expiry)
            if row is not None:
                row["sd_multiple"] = float(sd)
                ledgers[sd].append(row)
        if i % 25 == 0:
            print(f"processed {i}/{len(chosen)} entry dates")
    return {sd: pd.DataFrame(rows).sort_values("entry_timestamp") if rows else pd.DataFrame() for sd, rows in ledgers.items()}


def _candidate_score(
    train: pd.DataFrame,
    threshold: float | None,
    min_train_trades: int,
    min_subperiod_trades: int,
) -> tuple[bool, dict]:
    y = _filter_trades(train, threshold)
    if len(y) < min_train_trades:
        return False, {"trades": len(y)}
    y = y.sort_values("entry_timestamp")
    mid = y["entry_timestamp"].min() + (y["entry_timestamp"].max() - y["entry_timestamp"].min()) / 2
    first = y[y["entry_timestamp"] <= mid]
    second = y[y["entry_timestamp"] > mid]
    total = _stats(y)
    first_s = _stats(first)
    second_s = _stats(second)
    eligible = (
        total["expectancy"] >= 0.0
        and total["profit_factor"] >= 1.0
        and first_s["trades"] >= min_subperiod_trades
        and second_s["trades"] >= min_subperiod_trades
        and first_s["expectancy"] >= 0.0
        and second_s["expectancy"] >= 0.0
    )
    score = {
        **total,
        "first_half_trades": first_s["trades"],
        "first_half_expectancy": first_s["expectancy"],
        "first_half_profit_factor": first_s["profit_factor"],
        "second_half_trades": second_s["trades"],
        "second_half_expectancy": second_s["expectancy"],
        "second_half_profit_factor": second_s["profit_factor"],
    }
    return eligible, score


def _select_stable_candidate(
    ledgers: dict[float, pd.DataFrame],
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    allowed_sds: set[float],
    thresholds: Iterable[float | None],
    min_train_trades: int,
    min_subperiod_trades: int,
) -> tuple[float | None, float | None, dict, str]:
    candidates = []
    for sd, ledger in ledgers.items():
        if float(sd) not in allowed_sds or ledger.empty:
            continue
        e = pd.to_datetime(ledger["entry_timestamp"])
        train = ledger[(e >= train_start) & (e < train_end)].copy()
        for threshold in thresholds:
            eligible, score = _candidate_score(train, threshold, min_train_trades, min_subperiod_trades)
            if eligible:
                candidates.append((float(sd), threshold, score))
    if not candidates:
        return None, None, {}, "NO_TRADE_NO_STABLE_CANDIDATE"
    selected = sorted(
        candidates,
        key=lambda z: (
            z[2]["expectancy"],
            min(z[2]["first_half_expectancy"], z[2]["second_half_expectancy"]),
            z[2]["profit_factor"],
            z[2]["trades"],
            -abs(z[0]),
        ),
        reverse=True,
    )[0]
    return selected[0], selected[1], selected[2], "PASS_STABLE_TRAINING"


def _slice(ledger: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if ledger.empty:
        return ledger.copy()
    e = pd.to_datetime(ledger["entry_timestamp"])
    return ledger[(e >= start) & (e < end)].copy()


def walk_forward_stable_entry(
    ledgers: dict[float, pd.DataFrame],
    thresholds: Iterable[float | None] = (None, 0.0, 1.0, 2.0),
    train_months: int = 12,
    test_months: int = 3,
    rebalance_months: int = 3,
    min_train_trades: int = 20,
    min_subperiod_trades: int = 8,
    fixed_sd: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nonempty = [x for x in ledgers.values() if not x.empty]
    if not nonempty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    combined = pd.concat(nonempty, ignore_index=True)
    combined["entry_timestamp"] = pd.to_datetime(combined["entry_timestamp"])
    start = combined["entry_timestamp"].min().to_period("M").to_timestamp()
    end = combined["entry_timestamp"].max().normalize()
    thresholds = [None if t is None else float(t) for t in thresholds]
    rows = []
    adaptive_parts, fixed_parts, base_parts = [], [], []
    train_start = start
    while True:
        train_end = train_start + pd.offsets.MonthBegin(train_months)
        test_end = train_end + pd.offsets.MonthBegin(test_months)
        if test_end > end + pd.Timedelta(days=1):
            break

        adaptive_sd, adaptive_threshold, a_train, a_reason = _select_stable_candidate(
            ledgers, train_start, train_end, set(float(s) for s in ledgers), thresholds,
            min_train_trades, min_subperiod_trades
        )
        fixed_sd_selected, fixed_threshold, f_train, f_reason = _select_stable_candidate(
            ledgers, train_start, train_end, {float(fixed_sd)}, thresholds,
            min_train_trades, min_subperiod_trades
        )

        adaptive_test = pd.DataFrame()
        if adaptive_sd is not None:
            adaptive_test = _filter_trades(_slice(ledgers[adaptive_sd], train_end, test_end), adaptive_threshold)
        fixed_test = pd.DataFrame()
        if fixed_sd_selected is not None:
            fixed_test = _filter_trades(_slice(ledgers[fixed_sd_selected], train_end, test_end), fixed_threshold)
        base_test = _slice(ledgers.get(float(fixed_sd), pd.DataFrame()), train_end, test_end)

        a_stats = _stats(adaptive_test)
        f_stats = _stats(fixed_test)
        b_stats = _stats(base_test)
        fold = len(rows) + 1
        for part, kind, sd, th in [
            (adaptive_test, "adaptive_sd", adaptive_sd, adaptive_threshold),
            (fixed_test, "fixed_2sd_adaptive_ivrv", fixed_sd_selected, fixed_threshold),
        ]:
            if not part.empty:
                part = part.copy()
                part["wf_fold"] = fold
                part["model"] = kind
                part["selected_sd"] = np.nan if sd is None else sd
                part["selected_iv_rv_threshold_pct"] = np.nan if th is None else th
                if kind == "adaptive_sd":
                    adaptive_parts.append(part)
                else:
                    fixed_parts.append(part)
        b = base_test.copy()
        if not b.empty:
            b["wf_fold"] = fold
            b["model"] = "fixed_2sd_no_filter"
            base_parts.append(b)

        rows.append({
            "fold": fold,
            "train_start": train_start.date(),
            "train_end": (train_end - pd.Timedelta(days=1)).date(),
            "test_start": train_end.date(),
            "test_end": (test_end - pd.Timedelta(days=1)).date(),
            "adaptive_sd": adaptive_sd,
            "adaptive_iv_rv_threshold_pct": adaptive_threshold,
            "adaptive_gate_reason": a_reason,
            "adaptive_train_trades": a_train.get("trades", 0),
            "adaptive_train_expectancy": a_train.get("expectancy", np.nan),
            "adaptive_train_profit_factor": a_train.get("profit_factor", np.nan),
            "adaptive_first_half_expectancy": a_train.get("first_half_expectancy", np.nan),
            "adaptive_second_half_expectancy": a_train.get("second_half_expectancy", np.nan),
            "adaptive_test_trades": a_stats["trades"],
            "adaptive_test_expectancy": a_stats["expectancy"],
            "adaptive_test_profit_factor": a_stats["profit_factor"],
            "adaptive_test_net_pnl": a_stats["net_pnl"],
            "fixed_2sd_threshold_pct": fixed_threshold,
            "fixed_2sd_gate_reason": f_reason,
            "fixed_2sd_train_trades": f_train.get("trades", 0),
            "fixed_2sd_train_expectancy": f_train.get("expectancy", np.nan),
            "fixed_2sd_train_profit_factor": f_train.get("profit_factor", np.nan),
            "fixed_2sd_first_half_expectancy": f_train.get("first_half_expectancy", np.nan),
            "fixed_2sd_second_half_expectancy": f_train.get("second_half_expectancy", np.nan),
            "fixed_2sd_test_trades": f_stats["trades"],
            "fixed_2sd_test_expectancy": f_stats["expectancy"],
            "fixed_2sd_test_profit_factor": f_stats["profit_factor"],
            "fixed_2sd_test_net_pnl": f_stats["net_pnl"],
            "base_test_trades": b_stats["trades"],
            "base_test_expectancy": b_stats["expectancy"],
            "base_test_profit_factor": b_stats["profit_factor"],
            "base_test_net_pnl": b_stats["net_pnl"],
        })
        train_start += pd.offsets.MonthBegin(rebalance_months)

    wf = pd.DataFrame(rows)
    adaptive = pd.concat(adaptive_parts, ignore_index=True) if adaptive_parts else pd.DataFrame()
    fixed = pd.concat(fixed_parts, ignore_index=True) if fixed_parts else pd.DataFrame()
    base = pd.concat(base_parts, ignore_index=True) if base_parts else pd.DataFrame()
    return wf, adaptive, fixed, base


def compare_oos_models(adaptive: pd.DataFrame, fixed: pd.DataFrame, base: pd.DataFrame, starting_capital: float = 1_000_000.0) -> pd.DataFrame:
    rows = []
    for name, trades in [
        ("adaptive_stable_sd_ivrv", adaptive),
        ("fixed_2sd_stable_ivrv", fixed),
        ("fixed_2sd_no_filter", base),
    ]:
        m = portfolio_risk_metrics(trades, starting_capital)
        m["model"] = name
        rows.append(m)
    return pd.DataFrame(rows)