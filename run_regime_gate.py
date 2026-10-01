from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.regime_gate import walk_forward_ivrv_with_regime_gate

def main() -> None:
    ap = argparse.ArgumentParser(description="Run a leakage-safe IV-RV walk-forward with a train-only trade/no-trade regime gate")
    ap.add_argument("--trades", required=True, help="Clean DTE6 trade ledger CSV")
    ap.add_argument("--out-dir", default="results/ivrv_regime_gate")
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.0, 1.0, 2.0, 3.0])
    ap.add_argument("--train-months", type=int, default=12)
    ap.add_argument("--test-months", type=int, default=3)
    ap.add_argument("--rebalance-months", type=int, default=3)
    ap.add_argument("--min-train-trades", type=int, default=20)
    ap.add_argument("--min-train-expectancy", type=float, default=0.0)
    ap.add_argument("--min-train-profit-factor", type=float, default=1.0)
    ap.add_argument("--include-quality-warnings", action="store_true")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trades = pd.read_csv(args.trades)
    wf = walk_forward_ivrv_with_regime_gate(trades, thresholds=args.thresholds, train_months=args.train_months, test_months=args.test_months, rebalance_months=args.rebalance_months, min_train_trades=args.min_train_trades, min_train_expectancy=args.min_train_expectancy, min_train_profit_factor=args.min_train_profit_factor, exclude_quality_warnings=not args.include_quality_warnings)
    wf.to_csv(out / "ivrv_regime_gate.csv", index=False)
    if wf.empty:
        print("No valid folds")
    else:
        print("\nIV-RV walk-forward + regime gate:")
        print(wf.to_string(index=False))
        print("\nOOS aggregate:")
        print(f"folds={len(wf)}")
        print(f"base_oos_net_pnl={wf['base_test_net_pnl'].sum():.2f}")
        print(f"selected_threshold_oos_net_pnl={wf['selected_test_net_pnl'].sum():.2f}")
        print(f"gated_oos_net_pnl={wf['gated_test_net_pnl'].sum():.2f}")
        print(f"gate_pass_folds={int(wf['regime_gate_pass'].sum())}/{len(wf)}")
        print(f"selected_oos_trades={wf['selected_test_trades'].sum():.0f}")
        print(f"gated_oos_trades={wf['gated_test_trades'].sum():.0f}")
        print(f"positive_gated_folds={(wf['gated_test_net_pnl'] > 0).sum()}/{len(wf)}")
    print(f"\nSaved: {out / 'ivrv_regime_gate.csv'}")

if __name__ == "__main__":
    main()