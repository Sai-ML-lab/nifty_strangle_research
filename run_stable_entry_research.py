from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.strangle_backtest import load_config
from src.stable_entry import build_sd_ledgers, compare_oos_models, walk_forward_stable_entry


def main() -> None:
    ap = argparse.ArgumentParser(description="Stability-constrained SD x IV-RV walk-forward research")
    ap.add_argument("--options-dir", required=True)
    ap.add_argument("--spot", required=True)
    ap.add_argument("--config", default="config_dte6.yaml")
    ap.add_argument("--out-dir", default="results/stable_entry")
    ap.add_argument("--sd-values", nargs="+", type=float, default=[1.5, 1.75, 2.0, 2.5, 3.0])
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.0, 1.0, 2.0])
    ap.add_argument("--train-months", type=int, default=12)
    ap.add_argument("--test-months", type=int, default=3)
    ap.add_argument("--rebalance-months", type=int, default=3)
    ap.add_argument("--min-train-trades", type=int, default=20)
    ap.add_argument("--min-subperiod-trades", type=int, default=8)
    ap.add_argument("--fixed-sd", type=float, default=2.0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg, costs, _ = load_config(args.config)

    ledgers = build_sd_ledgers(args.options_dir, args.spot, cfg, costs, args.sd_values)
    for sd, ledger in ledgers.items():
        ledger.to_csv(out / f"sd_{sd:g}_trades.csv", index=False)

    wf, adaptive, fixed, base = walk_forward_stable_entry(
        ledgers,
        thresholds=[None] + [float(x) for x in args.thresholds],
        train_months=args.train_months,
        test_months=args.test_months,
        rebalance_months=args.rebalance_months,
        min_train_trades=args.min_train_trades,
        min_subperiod_trades=args.min_subperiod_trades,
        fixed_sd=args.fixed_sd,
    )

    wf.to_csv(out / "stable_entry_walk_forward.csv", index=False)
    adaptive.to_csv(out / "adaptive_stable_oos_trades.csv", index=False)
    fixed.to_csv(out / "fixed_2sd_stable_ivrv_oos_trades.csv", index=False)
    base.to_csv(out / "fixed_2sd_no_filter_oos_trades.csv", index=False)

    risk = compare_oos_models(adaptive, fixed, base, cfg.starting_capital)
    risk.to_csv(out / "stable_entry_oos_risk.csv", index=False)

    print("\nStability-constrained walk-forward:")
    print(wf.to_string(index=False) if not wf.empty else "No valid folds")
    print("\nOOS risk comparison:")
    print(risk.to_string(index=False) if not risk.empty else "No trades")
    if not wf.empty:
        print("\nOOS aggregate:")
        for col in ["adaptive_test_net_pnl", "fixed_2sd_test_net_pnl", "base_test_net_pnl"]:
            print(f"{col}={wf[col].sum():.2f}")
        for col in ["adaptive_test_trades", "fixed_2sd_test_trades", "base_test_trades"]:
            print(f"{col}={wf[col].sum():.0f}")
        print(f"adaptive_no_trade_folds={(wf['adaptive_gate_reason'] == 'NO_TRADE_NO_STABLE_CANDIDATE').sum()}/{len(wf)}")
        print(f"fixed_2sd_no_trade_folds={(wf['fixed_2sd_gate_reason'] == 'NO_TRADE_NO_STABLE_CANDIDATE').sum()}/{len(wf)}")
    print(f"\nSaved results to: {out}")


if __name__ == "__main__":
    main()