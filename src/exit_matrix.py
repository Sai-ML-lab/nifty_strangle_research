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


def build_exit_ledgers(
    options_dir: str | Path,
    spot_path: str | Path,
    base_cfg: StrategyConfig,
    costs: Costs,
    profit_capture_values: Iterable[float],
    stop_multiple_values: Iterable[float],
) -> dict[tuple[float, float], pd.DataFrame]:
    """Build one ledger per exit rule, keeping entry logic fixed at base_cfg."""
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
    ledgers = {(float(pc), float(sm)): [] for pc in profit_capture_values for sm in stop_multiple_values}
    for i, (_, (expiry, f)) in enumerate(sorted(chosen.items()), 1):
        raw = pd.read_parquet(f)
        for (pc, sm) in ledgers:
            cfg = replace(base_cfg, profit_capture=pc, stop_multiple=sm, min_iv_rv_spread=None)
            row = single_expiry_trade(raw, spot, cfg, costs, expiry)
            if row is not None and getattr(cfg, "exclude_quality_warnings", False) and row.get("data_quality_flag") != "PASS":
                row = None
            if row is not None:
                row["profit_capture"] = pc
                row["stop_multiple"] = sm
                ledgers[(pc, sm)].append(row)
        if i % 25 == 0:
            print(f"processed {i}/{len(chosen)} entry dates")
    return {k: pd.DataFrame(v).sort_values("entry_timestamp") if v else pd.DataFrame() for k, v in ledgers.items()}


def _stable_candidate_stats(train: pd.DataFrame, min_train_trades: int, min_subperiod_trades: int) -> tuple[bool, dict]:
    total = _stats(train)
    if total["trades"] < min_train_trades:
        return False, total
    y = train.sort_values("entry_timestamp")
    start = pd.Timestamp(y["entry_timestamp"].min()).normalize()
    end = pd.Timestamp(y["entry_timestamp"].max()).normalize() + pd.Timedelta(days=1)
    mid = start + (end - start) / 2
    first = y[(y["entry_timestamp"] >= start) & (y["entry_timestamp"] < mid)]
    second = y[(y["entry_timestamp"] >= mid) & (y["entry_timestamp"] < end)]
    fs, ss = _stats(first), _stats(second)
    score = {
        **total,
        "first_half_trades": fs["trades"], "first_half_expectancy": fs["expectancy"], "first_half_profit_factor": fs["profit_factor"],
        "second_half_trades": ss["trades"], "second_half_expectancy": ss["expectancy"], "second_half_profit_factor": ss["profit_factor"],
    }
    eligible = total["expectancy"] >= 0.0 and total["profit_factor"] >= 1.0 and fs["trades"] >= min_subperiod_trades and ss["trades"] >= min_subperiod_trades and fs["expectancy"] >= 0.0 and ss["expectancy"] >= 0.0
    return eligible, score


def walk_forward_exit_matrix(
    ledgers: dict[tuple[float, float], pd.DataFrame],
    train_months: int = 12,
    test_months: int = 3,
    rebalance_months: int = 3,
    min_train_trades: int = 20,
    min_subperiod_trades: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nonempty = [x for x in ledgers.values() if not x.empty]
    if not nonempty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    combined = pd.concat(nonempty, ignore_index=True)
    combined["entry_timestamp"] = pd.to_datetime(combined["entry_timestamp"])
    start = combined["entry_timestamp"].min().to_period("M").to_timestamp()
    end = combined["entry_timestamp"].max().normalize()
    keys = list(ledgers.keys())
    rows, selected_parts, base_parts = [], [], []
    train_start = start
    base_key = (0.50, 2.00)
    while True:
        train_end = train_start + pd.offsets.MonthBegin(train_months)
        test_end = train_end + pd.offsets.MonthBegin(test_months)
        if test_end > end + pd.Timedelta(days=1):
            break
        candidates = []
        for key in keys:
            ledger = ledgers[key]
            if ledger.empty:
                continue
            e = pd.to_datetime(ledger["entry_timestamp"])
            train = ledger[(e >= train_start) & (e < train_end)]
            eligible, score = _stable_candidate_stats(train, min_train_trades, min_subperiod_trades)
            if eligible:
                candidates.append((key, score))
        if candidates:
            selected_key, train_stats = sorted(candidates, key=lambda z: (z[1]["expectancy"], min(z[1]["first_half_expectancy"], z[1]["second_half_expectancy"]), z[1]["profit_factor"], z[1]["trades"]), reverse=True)[0]
            selection_reason = "PASS_STABLE_TRAINING"
            selected = ledgers[selected_key]
            e = pd.to_datetime(selected["entry_timestamp"])
            selected_test = selected[(e >= train_end) & (e < test_end)].copy()
        else:
            selected_key, train_stats, selection_reason, selected_test = None, {}, "NO_TRADE_NO_STABLE_EXIT", pd.DataFrame()
        base = ledgers.get(base_key, pd.DataFrame()).copy()
        if not base.empty:
            be = pd.to_datetime(base["entry_timestamp"])
            base = base[(be >= train_end) & (be < test_end)].copy()
        s = _stats(selected_test); b = _stats(base)
        fold = len(rows) + 1
        if not selected_test.empty:
            selected_test["wf_fold"] = fold; selected_test["selected_profit_capture"] = np.nan if selected_key is None else selected_key[0]; selected_test["selected_stop_multiple"] = np.nan if selected_key is None else selected_key[1]; selected_parts.append(selected_test)
        if not base.empty:
            base["wf_fold"] = fold; base_parts.append(base)
        rows.append({
            "fold": fold, "train_start": train_start.date(), "train_end": (train_end - pd.Timedelta(days=1)).date(), "test_start": train_end.date(), "test_end": (test_end - pd.Timedelta(days=1)).date(),
            "selected_profit_capture": np.nan if selected_key is None else selected_key[0], "selected_stop_multiple": np.nan if selected_key is None else selected_key[1], "selection_reason": selection_reason,
            "train_trades": train_stats.get("trades", 0), "train_expectancy": train_stats.get("expectancy", np.nan), "train_profit_factor": train_stats.get("profit_factor", np.nan),
            "first_half_expectancy": train_stats.get("first_half_expectancy", np.nan), "second_half_expectancy": train_stats.get("second_half_expectancy", np.nan),
            "test_trades": s["trades"], "test_expectancy": s["expectancy"], "test_profit_factor": s["profit_factor"], "test_net_pnl": s["net_pnl"],
            "base_test_trades": b["trades"], "base_test_expectancy": b["expectancy"], "base_test_profit_factor": b["profit_factor"], "base_test_net_pnl": b["net_pnl"],
        })
        train_start += pd.offsets.MonthBegin(rebalance_months)
    return pd.DataFrame(rows), (pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()), (pd.concat(base_parts, ignore_index=True) if base_parts else pd.DataFrame())


def full_matrix_summary(ledgers: dict[tuple[float, float], pd.DataFrame], starting_capital: float) -> pd.DataFrame:
    rows = []
    for (pc, sm), ledger in sorted(ledgers.items()):
        m = portfolio_risk_metrics(ledger, starting_capital)
        m["profit_capture"] = pc
        m["stop_multiple"] = sm
        rows.append(m)
    return pd.DataFrame(rows)


def oos_risk_comparison(selected: pd.DataFrame, base: pd.DataFrame, starting_capital: float) -> pd.DataFrame:
    rows=[]
    for name, trades in [("selected_exit", selected), ("fixed_2sd_50pct_2x", base)]:
        m=portfolio_risk_metrics(trades, starting_capital); m["model"]=name; rows.append(m)
    return pd.DataFrame(rows)