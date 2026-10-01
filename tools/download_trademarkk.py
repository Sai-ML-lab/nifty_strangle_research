from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "thetrademarkk/india-index-options-1m"
PREFIX = "options/NIFTY/"
PATTERN = re.compile(r"options/NIFTY/(\d{4}-\d{2}-\d{2})\.parquet$")


def parse_date(s: str) -> date:
    return date.fromisoformat(s)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download free TradeMarkk NIFTY 1-minute option history from Hugging Face")
    ap.add_argument("--start", required=True, help="First expiry YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="Last expiry YYYY-MM-DD")
    ap.add_argument("--out", default="data/trademarkk", help="Local destination")
    ap.add_argument("--no-spot", action="store_true", help="Skip index/NIFTY.parquet")
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

    for d, f in sorted(selected):
        local = hf_hub_download(REPO, f, repo_type="dataset", local_dir=str(out))
        print(f"{d}: {local}")

    if not args.no_spot:
        local = hf_hub_download(REPO, "index/NIFTY.parquet", repo_type="dataset", local_dir=str(out))
        print(f"spot: {local}")

    print(f"Downloaded {len(selected)} expiry files")


if __name__ == "__main__":
    main()
