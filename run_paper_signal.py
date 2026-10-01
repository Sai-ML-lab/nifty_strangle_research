from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data_ingest import read_any
from src.paper_trading import add_signal_ids, build_paper_signal, FROZEN_STRATEGY_ID, FROZEN_SPEC_HASH
from src.strangle_backtest import load_config


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate one frozen NIFTY paper-trading signal from an as-of option snapshot"
    )
    ap.add_argument("--options", required=True, help="Option snapshot CSV/Parquet containing the as-of chain")
    ap.add_argument("--spot", required=True, help="NIFTY spot CSV/Parquet; include history for RV20")
    ap.add_argument("--expiry", required=True, help="Target expiry YYYY-MM-DD")
    ap.add_argument("--as-of", required=True, help="Exact entry snapshot timestamp, e.g. 2026-10-01 10:00")
    ap.add_argument("--config", default="config_dte6_frozen_75_25.yaml")
    ap.add_argument("--out", default="results/paper/signal.csv")
    ap.add_argument("--slippage", type=float, default=None)
    args = ap.parse_args()

    cfg, costs, _ = load_config(args.config)
    options = read_any(args.options)
    spot = read_any(args.spot)

    signal = build_paper_signal(
        options=options,
        spot_history=spot,
        expiry=pd.Timestamp(args.expiry),
        as_of=pd.Timestamp(args.as_of),
        cfg=cfg,
        costs=costs,
        slippage_points=args.slippage,
    )
    signal = add_signal_ids(signal)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    signal.to_csv(out, index=False)

    row = signal.iloc[0]
    print(f"strategy_id={FROZEN_STRATEGY_ID}")
    print(f"strategy_spec_hash={FROZEN_SPEC_HASH}")
    print(f"signal_id={row['signal_id']}")
    print(f"signal_status={row['signal_status']}")
    print(f"signal_reason={row['signal_reason']}")
    if row["signal_status"] == "READY":
        print(
            f"strikes={row['put_strike']}/{row['call_strike']} "
            f"planned_credit={row['planned_credit_points']:.2f} "
            f"target_debit={row['profit_target_debit_points']:.2f} "
            f"stop_debit={row['stop_debit_points']:.2f}"
        )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
