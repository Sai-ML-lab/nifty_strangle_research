from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.exit_matrix import build_exit_ledgers, full_matrix_summary, oos_risk_comparison, walk_forward_exit_matrix
from src.strangle_backtest import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description="Leakage-safe exit matrix for the fixed 2-SD DTE6 strangle")
    ap.add_argument("--options-dir", required=True)
    ap.add_argument("--spot", required=True)
    ap.add_argument("--config", default="config_dte6.yaml")
    ap.add_argument("--out-dir", default="results/exit_matrix")
    ap.add_argument("--profit-capture", nargs="+", type=float, default=[0.25, 0.50, 0.75])
    ap.add_argument("--stop-multiple", nargs="+", type=float, default=[1.5, 2.0, 2.5])
    ap.add_argument("--train-months", type=int, default=12)
    ap.add_argument("--test-months", type=int, default=3)
    ap.add_argument("--rebalance-months", type=int, default=3)
    ap.add_argument("--min-train-trades", type=int, default=20)
    ap.add_argument("--min-subperiod-trades", type=int, default=8)
    args = ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    cfg, costs, _ = load_config(args.config)
    ledgers=build_exit_ledgers(args.options_dir,args.spot,cfg,costs,args.profit_capture,args.stop_multiple)
    for key,ledger in ledgers.items():
        ledger.to_csv(out / f"pc_{key[0]:g}_stop_{key[1]:g}_trades.csv", index=False)
    matrix=full_matrix_summary(ledgers,cfg.starting_capital)
    matrix.to_csv(out/"full_matrix_summary.csv",index=False)
    wf,selected,base=walk_forward_exit_matrix(ledgers,train_months=args.train_months,test_months=args.test_months,rebalance_months=args.rebalance_months,min_train_trades=args.min_train_trades,min_subperiod_trades=args.min_subperiod_trades)
    wf.to_csv(out/"exit_walk_forward.csv",index=False)
    selected.to_csv(out/"selected_oos_trades.csv",index=False)
    base.to_csv(out/"baseline_oos_trades.csv",index=False)
    risk=oos_risk_comparison(selected,base,cfg.starting_capital)
    risk.to_csv(out/"exit_oos_risk.csv",index=False)
    print("\nFull exit matrix:")
    print(matrix.to_string(index=False))
    print("\nWalk-forward exit selection:")
    print(wf.to_string(index=False) if not wf.empty else "No valid folds")
    print("\nOOS risk:")
    print(risk.to_string(index=False) if not risk.empty else "No trades")
    if not wf.empty:
        print(f"\nselected_oos_net_pnl={wf['test_net_pnl'].sum():.2f}")
        print(f"base_oos_net_pnl={wf['base_test_net_pnl'].sum():.2f}")
        print(f"selected_oos_trades={wf['test_trades'].sum():.0f}")
        print(f"base_oos_trades={wf['base_test_trades'].sum():.0f}")
        print(f"no_trade_folds={(wf['selection_reason']=='NO_TRADE_NO_STABLE_EXIT').sum()}/{len(wf)}")
    print(f"\nSaved results to: {out}")


if __name__ == "__main__":
    main()