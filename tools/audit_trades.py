from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit a completed NIFTY strangle trade ledger for overlap/data issues")
    ap.add_argument("--trades", required=True, help="Trade CSV produced by the backtester")
    args = ap.parse_args()

    t = pd.read_csv(args.trades)
    required = {"entry_timestamp", "expiry", "net_pnl"}
    missing = required - set(t.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")

    t["entry_timestamp"] = pd.to_datetime(t["entry_timestamp"], errors="coerce")
    t["expiry"] = pd.to_datetime(t["expiry"], errors="coerce")
    t["entry_date"] = t["entry_timestamp"].dt.normalize()

    dup = t[t["entry_date"].duplicated(keep=False)].sort_values(["entry_date", "expiry"])
    print(f"trades={len(t):,}")
    print(f"unique_entry_dates={t['entry_date'].nunique():,}")
    print(f"duplicate_entry_dates={dup['entry_date'].nunique():,}")
    print(f"duplicate_trade_rows={len(dup):,}")

    if not dup.empty:
        print("\nOVERLAPPING ENTRY DATES:")
        print(dup[["entry_timestamp", "expiry", "net_pnl"]].to_string(index=False))

    dte = (t["expiry"] - t["entry_date"]).dt.days
    print("\nDTE distribution:")
    print(dte.value_counts().sort_index().to_string())

    print("\nExit reasons:")
    if "exit_reason" in t.columns:
        print(t["exit_reason"].value_counts(dropna=False).to_string())

    print("\nPnL:")
    print(t["net_pnl"].describe().to_string())


if __name__ == "__main__":
    main()
