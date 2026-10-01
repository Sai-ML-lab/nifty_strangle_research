from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "thetrademarkk/india-index-options-1m"
PATTERN = re.compile(r"options/NIFTY/(\d{4}-\d{2}-\d{2})\.parquet$")


def parse_date(s: str) -> date:
    return date.fromisoformat(s)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download selected free TradeMarkk NIFTY 1-minute option history from Hugging Face"
    )
    ap.add_argument("--start", required=True, help="First expiry YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="Last expiry YYYY-MM-DD")
    ap.add_argument("--out", default="data/trademarkk")
    ap.add_argument("--no-spot", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="List matching files without downloading")
    args = ap.parse_args()

    start, end = parse_date(args.start), parse_date(args.end)
    if end < start:
        raise SystemExit("--end must be >= --start")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    files = api.list_repo_files(REPO, repo_type="dataset")
    selected = []
    for f in files:
        m = PATTERN.fullmatch(f)
        if not m:
            continue
        d = parse_date(m.group(1))
        if start <= d <= end:
            selected.append((d, f))

    if not selected:
        raise SystemExit("No NIFTY option expiry files matched the requested range")

    print(f"Matched {len(selected)} expiry files from {selected[0][0]} to {selected[-1][0]}")
    if args.dry_run:
        for d, f in sorted(selected):
            print(f"{d}: {f}")
        return

    for d, f in sorted(selected):
        local = hf_hub_download(
            REPO,
            f,
            repo_type="dataset",
            local_dir=str(out),
            local_dir_use_symlinks=False,
        )
        print(f"{d}: {local}")

    if not args.no_spot:
        local = hf_hub_download(
            REPO,
            "index/NIFTY.parquet",
            repo_type="dataset",
            local_dir=str(out),
            local_dir_use_symlinks=False,
        )
        print(f"spot: {local}")

    print(f"Downloaded {len(selected)} expiry files")


if __name__ == "__main__":
    main()
