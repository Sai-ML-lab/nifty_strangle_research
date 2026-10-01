from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.risk_metrics import portfolio_risk_metrics
from src.strangle_backtest import load_config
from src.sd_ivrv import build_sd_ledgers, walk_forward_sd_ivrv, oos_risk_report

def main() -> None:
    ap = argparse.ArgumentParser(description="Leakage-safe SD x IV-RV walk-forward research")
    ap.add_argument("--options-dir", required=True)
    ap.add_argument("--spot", required=True)
    ap.add_argument("--config", default="config_dte6.yaml")
    ap.add_argument("--out-dir", default="results/sd_ivrv")
    ap.add_argument("--sd-values", nargs="+", type=float, default=[1.5, 1.75, 2.0, 2.5, 3.0])
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.0, 1.0, 2.0])
    ap.add_argument("--train-months", type=int, default=12)
    ap.add_argument("--test-months", type=int, default=3)
    ap.add_argument("--rebalance-months", type=int, default=3)
    ap.add_argument("--min-train-trades", type=int, default=20)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    cfg, costs, _ = load_config(args.config)
    ledgers = build_sd_ledgers(args.options_dir, args.spot, cfg, costs, args.sd_values)
    for sd, ledger in ledgers.items():
        ledger.to_csv(out / f"sd_{sd:g}_trades.csv", index=False)
    thresholds = [None] + args.thresholds
    wf, selected, base = walk_forward_sd_ivrv(ledgers, thresholds=thresholds, train_months=args.train_months, test_months=args.test_months, rebalance_months=args.rebalance_months, min_train_trades=args.min_train_trades, starting_capital=cfg.starting_capital)
    wf.to_csv(out / "sd_ivrv_walk_forward.csv", index=False)
    selected.to_csv(out / "selected_oos_trades.csv", index=False)
    base.to_csv(out / "baseline_oos_trades.csv", index=False)
    risk = oos_risk_report(selected, base, cfg.starting_capital)
    risk.to_csv(out / "oos_risk_report.csv", index=False)
    print("\nSD x IV-RV walk-forward:")
    print(wf.to_string(index=False) if not wf.empty else "No valid folds")
    print("\nOOS risk report:")
    print(risk.to_string(index=False) if not risk.empty else "No trades")
    if not wf.empty:
        print(f"\nfolds={len(wf)}")
        print(f"selected_oos_net_pnl={wf['test_net_pnl'].sum():.2f}")
        print(f"base_oos_net_pnl={wf['base_test_net_pnl'].sum():.2f}")
        print(f"selected_oos_trades={wf['test_trades'].sum():.0f}")
        print(f"base_oos_trades={wf['base_test_trades'].sum():.0f}")
    print(f"\nSaved results to: {out}")

if __name__ == "__main__":
    main()