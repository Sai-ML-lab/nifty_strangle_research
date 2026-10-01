from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run exact NIFTY backtest repeatedly across per-leg slippage assumptions"
    )
    ap.add_argument("--options-dir", required=True)
    ap.add_argument("--spot", required=True)
    ap.add_argument("--config", default="config_dte6_ivrv2.yaml")
    ap.add_argument("--out-dir", default="results/slippage_sensitivity")
    ap.add_argument(
        "--slippages",
        nargs="+",
        type=float,
        default=[0.0, 0.10, 0.25, 0.50, 0.75, 1.00],
    )
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for slippage in args.slippages:
        run_dir = out / f"slippage_{slippage:.2f}".replace(".", "p")
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "run_batch_research.py",
            "--options-dir", args.options_dir,
            "--spot", args.spot,
            "--config", args.config,
            "--out-dir", str(run_dir),
            "--slippage", str(slippage),
        ]
        print("\n$", " ".join(cmd))
        subprocess.run(cmd, check=True)

        report_path = run_dir / "baseline_report.csv"
        report = pd.read_csv(report_path, header=None, names=["metric", "value"])
        d = dict(zip(report["metric"], report["value"]))
        rows.append({
            "slippage_points_per_leg": slippage,
            "trades": float(d.get("trades", 0)),
            "win_rate": float(d.get("win_rate", float("nan"))),
            "avg_pnl": float(d.get("avg_pnl", float("nan"))),
            "profit_factor": float(d.get("profit_factor", float("nan"))),
            "total_net_pnl": float(d.get("total_net_pnl", float("nan"))),
            "max_drawdown_pct": float(d.get("max_drawdown_pct", float("nan"))),
            "worst_trade": float(d.get("worst_trade", float("nan"))),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "slippage_sensitivity.csv", index=False)
    print("\nSlippage sensitivity:")
    print(summary.to_string(index=False))
    print(f"\nSaved: {out / 'slippage_sensitivity.csv'}")


if __name__ == "__main__":
    main()
