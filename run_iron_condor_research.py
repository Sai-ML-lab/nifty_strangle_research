from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.iron_condor import build_condor_ledgers, diagnose_condor_coverage, summarize_condors
from src.strangle_backtest import load_config
from src.risk_metrics import portfolio_risk_metrics


def _period_split(ledger: pd.DataFrame, holdout_start: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if ledger.empty or not holdout_start:
        return ledger, pd.DataFrame()
    d = pd.to_datetime(ledger["entry_timestamp"])
    cut = pd.Timestamp(holdout_start)
    return ledger[d < cut].copy(), ledger[d >= cut].copy()


def main() -> None:
    ap = argparse.ArgumentParser(description="Frozen DTE6 + 2-SD iron-condor comparison")
    ap.add_argument("--options-dir", required=True)
    ap.add_argument("--spot", required=True)
    ap.add_argument("--config", default="config_dte6_iron_condor.yaml")
    ap.add_argument("--out-dir", default="results/iron_condor")
    ap.add_argument("--wing-width", nargs="+", type=float, default=[100.0, 200.0, 300.0])
    ap.add_argument("--profit-capture", nargs="+", type=float, default=[0.50, 0.75])
    ap.add_argument("--stop-multiple", nargs="+", type=float, default=[2.0, 2.5])
    ap.add_argument("--holdout-start", default=None, help="Optional frozen post-research holdout start; no parameter selection is performed.")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg, costs, _ = load_config(args.config)

    ledgers = build_condor_ledgers(
        args.options_dir, args.spot, cfg, costs,
        args.wing_width, args.profit_capture, args.stop_multiple,
    )

    summary = summarize_condors(ledgers, cfg.starting_capital)
    summary.to_csv(out / "iron_condor_summary.csv", index=False)

    coverage = diagnose_condor_coverage(args.options_dir, args.spot, cfg, costs, args.wing_width)
    coverage.to_csv(out / "condor_coverage.csv", index=False)

    for (width, pc, sm), ledger in ledgers.items():
        ledger.to_csv(out / f"width_{width:g}_pc_{pc:g}_stop_{sm:g}_trades.csv", index=False)

    if args.holdout_start:
        rows = []
        for (width, pc, sm), ledger in ledgers.items():
            _, holdout = _period_split(ledger, args.holdout_start)
            m = portfolio_risk_metrics(holdout, cfg.starting_capital)
            m.update({"requested_wing_width": width, "profit_capture": pc, "stop_multiple": sm, "period": "holdout"})
            rows.append(m)
        pd.DataFrame(rows).to_csv(out / "iron_condor_holdout_risk.csv", index=False)
        print("\nFrozen post-research holdout:")
        print(pd.DataFrame(rows).to_string(index=False))

    print("\nIron-condor frozen suite:")
    print(summary.to_string(index=False))
    print(f"\nSaved results to: {out}")


if __name__ == "__main__":
    main()
