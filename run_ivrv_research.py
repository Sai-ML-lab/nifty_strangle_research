from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.ivrv import threshold_sweep, walk_forward_ivrv


def main() -> None:
    ap = argparse.ArgumentParser(description="Research IV-RV filters on an existing DTE6 NIFTY trade ledger")
    ap.add_argument("--trades", required=True, help="CSV from the DTE6 baseline run")
    ap.add_argument("--out-dir", default="results/ivrv")
    ap.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
        help="Minimum IV-RV spread in percentage points",
    )
    ap.add_argument("--train-months", type=int, default=12)
    ap.add_argument("--test-months", type=int, default=3)
    ap.add_argument("--rebalance-months", type=int, default=3)
    ap.add_argument("--min-train-trades", type=int, default=20)
    ap.add_argument("--include-quality-warnings", action="store_true")
    args = ap.parse_args()

    trades = pd.read_csv(args.trades)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    exclude_quality = not args.include_quality_warnings

    sweep = threshold_sweep(
        trades,
        thresholds=args.thresholds,
        exclude_quality_warnings=exclude_quality,
    )
    sweep.to_csv(out / "ivrv_threshold_sweep.csv", index=False)

    wf = walk_forward_ivrv(
        trades,
        thresholds=args.thresholds,
        train_months=args.train_months,
        test_months=args.test_months,
        rebalance_months=args.rebalance_months,
        min_train_trades=args.min_train_trades,
        exclude_quality_warnings=exclude_quality,
    )
    wf.to_csv(out / "ivrv_walk_forward.csv", index=False)

    print("IV-RV threshold sweep")
    print(sweep.to_string(index=False))

    print("\nIV-RV walk-forward")
    print(wf.to_string(index=False) if not wf.empty else "No valid folds")

    if not wf.empty:
        print("\nOOS aggregate:")
        selected = []
        for _, row in wf.iterrows():
            selected.append(
                {
                    "net_pnl": row["test_net_pnl"],
                    "trades": row["test_trades"],
                    "base_net_pnl": row["base_test_net_pnl"],
                    "base_trades": row["base_test_trades"],
                }
            )
        s = pd.DataFrame(selected)
        print(f"folds={len(s)}")
        print(f"oos_net_pnl={s['net_pnl'].sum():.2f}")
        print(f"base_oos_net_pnl={s['base_net_pnl'].sum():.2f}")
        print(f"positive_oos_folds={(s['net_pnl'] > 0).sum()}/{len(s)}")
        print(f"oos_trades={s['trades'].sum():.0f}")
        print(f"base_oos_trades={s['base_trades'].sum():.0f}")

    print(f"\nSaved results to: {out}")


if __name__ == "__main__":
    main()
