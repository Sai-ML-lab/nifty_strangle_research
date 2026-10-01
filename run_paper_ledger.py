from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.core import Costs
from src.paper_trading import (
    add_signal_ids,
    initialize_ledger,
    record_entry_fill,
    record_exit_fill,
)


def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Initialize or update the frozen-strategy paper-trading ledger"
    )
    ap.add_argument("--action", choices=["init", "entry", "exit"], required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--signals", default=None, help="Signal CSV for --action init")
    ap.add_argument("--signal-id", default=None)
    ap.add_argument("--fill-timestamp", default=None)
    ap.add_argument("--put-fill", type=float, default=None)
    ap.add_argument("--call-fill", type=float, default=None)
    ap.add_argument("--exit-reason", choices=["profit_target", "stop", "time_exit", "expiry", "manual"], default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--brokerage-per-order", type=float, default=20.0)
    ap.add_argument("--exchange-txn-pct", type=float, default=0.0003553)
    ap.add_argument("--sebi-turnover-pct", type=float, default=0.000001)
    ap.add_argument("--stamp-duty-buy-option-pct", type=float, default=0.00003)
    ap.add_argument("--gst-pct", type=float, default=0.18)
    args = ap.parse_args()

    out = Path(args.out or args.ledger)
    if args.action == "init":
        if not args.signals:
            ap.error("--signals is required for --action init")
        signals = add_signal_ids(_read_csv(args.signals))
        ledger = initialize_ledger(signals)
        ledger.to_csv(out, index=False)
        print(f"Initialized {len(ledger)} signal rows -> {out}")
        return

    ledger = _read_csv(args.ledger)
    if args.signal_id is None or args.fill_timestamp is None or args.put_fill is None or args.call_fill is None:
        ap.error("--signal-id, --fill-timestamp, --put-fill and --call-fill are required")

    if args.action == "entry":
        ledger = record_entry_fill(
            ledger,
            signal_id=args.signal_id,
            fill_timestamp=pd.Timestamp(args.fill_timestamp),
            put_fill=args.put_fill,
            call_fill=args.call_fill,
        )
    else:
        if args.exit_reason is None:
            ap.error("--exit-reason is required for --action exit")
        costs = Costs(
            brokerage_per_order=args.brokerage_per_order,
            exchange_txn_pct=args.exchange_txn_pct,
            sebi_turnover_pct=args.sebi_turnover_pct,
            stamp_duty_buy_option_pct=args.stamp_duty_buy_option_pct,
            gst_pct=args.gst_pct,
        )
        ledger = record_exit_fill(
            ledger,
            signal_id=args.signal_id,
            fill_timestamp=pd.Timestamp(args.fill_timestamp),
            put_fill=args.put_fill,
            call_fill=args.call_fill,
            exit_reason=args.exit_reason,
            costs=costs,
        )

    ledger.to_csv(out, index=False)
    print(f"Updated ledger -> {out}")


if __name__ == "__main__":
    main()
