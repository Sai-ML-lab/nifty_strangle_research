from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.slippage_replay import replay_slippage
from src.strangle_backtest import load_config

def main() -> None:
    ap = argparse.ArgumentParser(description="Replay a frozen NIFTY trade ledger under different execution slippage assumptions")
    ap.add_argument("--trades", required=True, help="Trade ledger whose strikes and exit timestamps are frozen")
    ap.add_argument("--options-dir", required=True, help="Directory of per-expiry option Parquets")
    ap.add_argument("--out-dir", default="results/slippage_replay")
    ap.add_argument("--config", default="config_dte6_ivrv2.yaml", help="Cost configuration used for the replay")
    ap.add_argument("--slippages", nargs="+", type=float, default=[0.0, 0.10, 0.25, 0.50, 0.75, 1.00], help="Per-leg slippage in option points")
    ap.add_argument("--include-quality-warnings", action="store_true")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trades = pd.read_csv(args.trades)
    _, costs, _ = load_config(args.config)
    detail, summary = replay_slippage(trades, args.options_dir, args.slippages, base_costs=costs, exclude_quality_warnings=not args.include_quality_warnings)
    detail.to_csv(out / "slippage_replay_trades.csv", index=False)
    summary.to_csv(out / "slippage_sensitivity.csv", index=False)
    print("\nFrozen-decision slippage sensitivity:")
    print(summary.to_string(index=False) if not summary.empty else "No replay rows")
    print(f"\nSaved: {out / 'slippage_replay_trades.csv'}")
    print(f"Saved: {out / 'slippage_sensitivity.csv'}")

if __name__ == "__main__":
    main()