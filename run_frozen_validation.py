from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.risk_metrics import portfolio_risk_metrics


def _load(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path)
    if "entry_timestamp" not in x.columns or "net_pnl" not in x.columns:
        raise ValueError(f"{path} must contain entry_timestamp and net_pnl")
    x["entry_timestamp"] = pd.to_datetime(x["entry_timestamp"], errors="coerce")
    return x.dropna(subset=["entry_timestamp"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Split frozen trade ledgers into development and post-research holdout")
    ap.add_argument("--trades", nargs="+", required=True, help="CSV trade ledgers")
    ap.add_argument("--labels", nargs="*", default=None, help="Optional labels matching --trades")
    ap.add_argument("--holdout-start", required=True, help="First date included in the frozen holdout")
    ap.add_argument("--starting-capital", type=float, default=1_000_000)
    ap.add_argument("--out", default="results/frozen_validation_summary.csv")
    args = ap.parse_args()

    labels = args.labels or [Path(p).stem for p in args.trades]
    if len(labels) != len(args.trades):
        raise ValueError("--labels must have the same number of items as --trades")

    cut = pd.Timestamp(args.holdout_start)
    rows = []
    for path, label in zip(args.trades, labels):
        t = _load(Path(path))
        for period, subset in [
            ("pre_holdout", t[t["entry_timestamp"] < cut]),
            ("holdout", t[t["entry_timestamp"] >= cut]),
        ]:
            m = portfolio_risk_metrics(subset, args.starting_capital)
            m["model"] = label
            m["period"] = period
            rows.append(m)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(out, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
