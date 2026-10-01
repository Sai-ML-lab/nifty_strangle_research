from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.risk_metrics import portfolio_risk_metrics


def analyze_tails(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = trades.copy()
    x["entry_timestamp"] = pd.to_datetime(x["entry_timestamp"], errors="coerce")
    x["exit_timestamp"] = pd.to_datetime(x["exit_timestamp"], errors="coerce")
    for c in ["net_pnl", "gross_pnl", "transaction_cost", "initial_credit_points", "entry_spot", "exit_spot"]:
        if c in x:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["entry_timestamp", "net_pnl"]).sort_values("entry_timestamp").copy()

    x["is_loss"] = x["net_pnl"] < 0
    credit_rupees = pd.to_numeric(x.get("initial_credit_rupees", pd.Series(np.nan, index=x.index)), errors="coerce")
    if credit_rupees.isna().all() and "initial_credit_points" in x.columns:
        lot = pd.to_numeric(x.get("lot_size", pd.Series(1, index=x.index)), errors="coerce").fillna(1.0)
        lots = pd.to_numeric(x.get("lots", pd.Series(1, index=x.index)), errors="coerce").fillna(1.0)
        credit_rupees = pd.to_numeric(x["initial_credit_points"], errors="coerce") * lot * lots
    x["initial_credit_rupees_for_ratio"] = credit_rupees
    x["loss_to_credit"] = np.where(
        credit_rupees.gt(0),
        -x["net_pnl"] / credit_rupees,
        np.nan,
    )
    x["spot_move_pct"] = (
        (x["exit_spot"] / x["entry_spot"] - 1.0) * 100.0
        if {"entry_spot", "exit_spot"}.issubset(x.columns)
        else np.nan
    )
    x["abs_spot_move_pct"] = x["spot_move_pct"].abs()
    x["loss_bucket"] = pd.cut(
        x["loss_to_credit"],
        bins=[-np.inf, 0, 1, 2, 3, 5, 10, np.inf],
        labels=["profit", "<=1x", "1-2x", "2-3x", "3-5x", "5-10x", ">10x"],
        right=False,
    )

    losses = x[x["is_loss"]].copy()
    by_reason = (
        losses.groupby("exit_reason", dropna=False)
        .agg(
            trades=("net_pnl", "size"),
            total_net_pnl=("net_pnl", "sum"),
            avg_loss=("net_pnl", "mean"),
            worst_loss=("net_pnl", "min"),
            median_abs_spot_move_pct=("abs_spot_move_pct", "median"),
        )
        .reset_index()
        .sort_values("total_net_pnl")
        if "exit_reason" in losses.columns
        else pd.DataFrame()
    )

    by_bucket = (
        x.groupby("loss_bucket", observed=False)
        .agg(
            trades=("net_pnl", "size"),
            total_net_pnl=("net_pnl", "sum"),
            avg_net_pnl=("net_pnl", "mean"),
        )
        .reset_index()
    )
    return x, pd.concat(
        [
            by_reason.assign(section="exit_reason"),
            by_bucket.assign(section="loss_to_credit_bucket"),
        ],
        ignore_index=True,
        sort=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Tail-loss attribution for a frozen NIFTY trade ledger")
    ap.add_argument("--trades", required=True)
    ap.add_argument("--out-dir", default="results/tail_analysis")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--starting-capital", type=float, default=1_000_000)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trades = pd.read_csv(args.trades)
    enriched, summary = analyze_tails(trades)
    enriched.to_csv(out / "tail_enriched_trades.csv", index=False)
    summary.to_csv(out / "tail_summary.csv", index=False)
    top = enriched.nsmallest(args.top_n, "net_pnl")
    top.to_csv(out / "worst_trades.csv", index=False)
    risk = pd.DataFrame([portfolio_risk_metrics(enriched, args.starting_capital)])
    risk.to_csv(out / "risk_metrics.csv", index=False)
    print("\nTail summary:")
    print(summary.to_string(index=False))
    print("\nWorst trades:")
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
