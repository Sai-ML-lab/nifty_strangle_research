from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from run_batch_research import entry_date_for_expiry, expiry_from_filename, single_expiry_trade, load_spot
from src.core import StrategyConfig, Costs
from src.risk_metrics import portfolio_risk_metrics

def _stats(x: pd.DataFrame) -> dict:
    p = pd.to_numeric(x.get("net_pnl", pd.Series(dtype=float)), errors="coerce").dropna()
    if p.empty:
        return {"trades": 0, "expectancy": np.nan, "profit_factor": np.nan, "net_pnl": 0.0}
    w, l = p[p > 0], p[p < 0]
    return {
        "trades": int(len(p)),
        "expectancy": float(p.mean()),
        "profit_factor": float(w.sum() / abs(l.sum())) if len(l) else np.inf,
        "net_pnl": float(p.sum()),
    }

def _filter_trades(x: pd.DataFrame, threshold: float | None) -> pd.DataFrame:
    if x.empty:
        return x.copy()
    if "data_quality_flag" in x:
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
    for i, (entry_date, (expiry, f)) in enumerate(sorted(chosen.items()), 1):
        raw = pd.read_parquet(f)
        for sd in ledgers:
            cfg = base_cfg.__class__(**{**base_cfg.__dict__, "sd_multiple": sd, "min_iv_rv_spread": None})
            row = single_expiry_trade(raw, spot, cfg, costs, expiry)
            if row is not None:
                row["sd_multiple"] = sd
                ledgers[sd].append(row)
        if i % 25 == 0:
            print(f"processed {i}/{len(chosen)} entry dates")
    return {sd: pd.DataFrame(rows).sort_values("entry_timestamp") if rows else pd.DataFrame() for sd, rows in ledgers.items()}

def walk_forward_sd_ivrv(
    ledgers: dict[float, pd.DataFrame],
    thresholds: Iterable[float | None] = (None, 0.0, 1.0, 2.0),
    train_months: int = 12,
    test_months: int = 3,
    rebalance_months: int = 3,
    min_train_trades: int = 20,
    starting_capital: float = 1_000_000.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([x for x in ledgers.values() if not x.empty], ignore_index=True)
    if combined.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    start = pd.to_datetime(combined["entry_timestamp"]).min().to_period("M").to_timestamp()
    end = pd.to_datetime(combined["entry_timestamp"]).max().normalize()
    thresholds = [None if t is None else float(t) for t in thresholds]
    rows, selected_parts, base_parts = [], [], []
    train_start = start
    while True:
        train_end = train_start + pd.offsets.MonthBegin(train_months)
        test_end = train_end + pd.offsets.MonthBegin(test_months)
        if test_end > end + pd.Timedelta(days=1):
            break
        candidates = []
        for sd, ledger in ledgers.items():
            if ledger.empty:
                continue
            e = pd.to_datetime(ledger["entry_timestamp"])
            train = ledger[(e >= train_start) & (e < train_end)]
            for threshold in thresholds:
                tr = _filter_trades(train, threshold)
                if len(tr) < min_train_trades:
                    continue
                s = _stats(tr)
                candidates.append((sd, threshold, s))
        if not candidates:
            train_start += pd.offsets.MonthBegin(rebalance_months)
            continue
        sd, threshold, train_stats = sorted(candidates, key=lambda z: (z[2]["expectancy"], z[2]["profit_factor"], z[2]["trades"], -abs(z[0])), reverse=True)[0]
        ledger = ledgers[sd]
        e = pd.to_datetime(ledger["entry_timestamp"])
        test = ledger[(e >= train_end) & (e < test_end)]
        selected = _filter_trades(test, threshold)
        base = _filter_trades(ledgers.get(2.0, pd.DataFrame()), threshold=None)
        be = pd.to_datetime(base["entry_timestamp"]) if not base.empty else pd.Series(dtype="datetime64[ns]")
        base = base[(be >= train_end) & (be < test_end)] if not base.empty else base
        ts = _stats(selected)
        bs = _stats(base)
        selected = selected.copy(); selected["wf_fold"] = len(rows) + 1; selected["selected_sd"] = sd; selected["selected_iv_rv_threshold_pct"] = np.nan if threshold is None else threshold
        selected_parts.append(selected)
        base = base.copy(); base["wf_fold"] = len(rows) + 1; base_parts.append(base)
        rows.append({
            "train_start": train_start.date(), "train_end": (train_end - pd.Timedelta(days=1)).date(),
            "test_start": train_end.date(), "test_end": (test_end - pd.Timedelta(days=1)).date(),
            "selected_sd": sd, "selected_iv_rv_threshold_pct": np.nan if threshold is None else threshold,
            "train_trades": train_stats["trades"], "train_expectancy": train_stats["expectancy"], "train_profit_factor": train_stats["profit_factor"],
            "test_trades": ts["trades"], "test_expectancy": ts["expectancy"], "test_profit_factor": ts["profit_factor"], "test_net_pnl": ts["net_pnl"],
            "base_test_trades": bs["trades"], "base_test_expectancy": bs["expectancy"], "base_test_profit_factor": bs["profit_factor"], "base_test_net_pnl": bs["net_pnl"],
        })
        train_start += pd.offsets.MonthBegin(rebalance_months)
    wf = pd.DataFrame(rows)
    selected_trades = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    base_trades = pd.concat(base_parts, ignore_index=True) if base_parts else pd.DataFrame()
    return wf, selected_trades, base_trades

def oos_risk_report(selected_trades: pd.DataFrame, base_trades: pd.DataFrame, starting_capital: float = 1_000_000.0) -> pd.DataFrame:
    rows = []
    for name, trades in [("selected_sd_ivrv", selected_trades), ("dte6_sd2_no_filter", base_trades)]:
        m = portfolio_risk_metrics(trades, starting_capital)
        m["strategy"] = name
        rows.append(m)
    return pd.DataFrame(rows)