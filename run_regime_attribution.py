from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.regime_attribution import run_attribution


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Diagnostic regime/event attribution for the frozen NIFTY 75%/2.5x strategy"
    )
    ap.add_argument("--trades", required=True, help="Frozen trade ledger CSV")
    ap.add_argument("--spot", default=None, help="Optional minute/daily NIFTY spot CSV/Parquet")
    ap.add_argument("--out-dir", default="results/regime_attribution")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    trades = pd.read_csv(args.trades)
    enriched, summary = run_attribution(trades, args.spot)
    enriched.to_csv(out / "attribution_trades.csv", index=False)
    summary.to_csv(out / "regime_summary.csv", index=False)

    print("\nRegime/event attribution (diagnostic only; no parameter selection):")
    if summary.empty:
        print("No valid attribution groups.")
    else:
        print(summary.to_string(index=False))

    print(f"\nSaved: {out / 'attribution_trades.csv'}")
    print(f"Saved: {out / 'regime_summary.csv'}")


if __name__ == "__main__":
    main()
