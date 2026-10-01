from pathlib import Path
import argparse
import pandas as pd

from src.strangle_backtest import load_config, prepare_data, run_backtest, performance_report
from src.research import parameter_sweep, trade_regime_report

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description="Run NIFTY short-strangle research")
    ap.add_argument("--data", default=str(ROOT / "data" / "sample_chain.csv"))
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--run-sweep", action="store_true", help="Run exploratory parameter grid")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg, costs, _ = load_config(args.config)
    data = prepare_data(args.data, cfg, slippage_points=costs.slippage_points_per_leg)

    trades = run_backtest(data, cfg, costs)
    trades.to_csv(out / "baseline_trades.csv", index=False)
    report = performance_report(trades, cfg.starting_capital)
    pd.Series(report).to_csv(out / "baseline_report.csv")
    print("Baseline report:")
    print(pd.Series(report).to_string())

    regimes = trade_regime_report(trades)
    regimes.to_csv(out / "baseline_regime_report.csv", index=False)

    if args.run_sweep:
        sweep = parameter_sweep(data, cfg, costs)
        sweep.to_csv(out / "parameter_sweep.csv", index=False)
        print("\nTop 10 exploratory parameter rows:")
        print(sweep.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
