from __future__ import annotations

import argparse
import os

from src.upstox_paper import load_env_file, run_paper_daemon


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Always-on Upstox-backed paper trader for the frozen NIFTY strategy"
    )
    ap.add_argument("--config", default="config_dte6_frozen_75_25.yaml")
    ap.add_argument("--ledger", default="results/paper/upstox_paper_ledger.csv")
    ap.add_argument("--quote-log", default="results/paper/upstox_quote_log.csv")
    ap.add_argument("--poll-seconds", type=int, default=30)
    ap.add_argument("--once", action="store_true", help="Run one polling cycle and exit")
    ap.add_argument(
        "--env-file",
        default=".env.paper",
        help="Optional file containing UPSTOX_ANALYTICS_TOKEN and alert settings",
    )
    args = ap.parse_args()

    load_env_file(args.env_file)
    token = os.getenv("UPSTOX_ANALYTICS_TOKEN")
    if not token:
        raise SystemExit(
            "Missing UPSTOX_ANALYTICS_TOKEN. Generate an Upstox Analytics Token and "
            "put it in the env file or environment."
        )

    run_paper_daemon(
        token=token,
        config_path=args.config,
        ledger_path=args.ledger,
        quote_log_path=args.quote_log,
        poll_seconds=args.poll_seconds,
        once=args.once,
    )


if __name__ == "__main__":
    main()
