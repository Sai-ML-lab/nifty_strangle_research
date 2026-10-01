from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

# Allow this script to be run directly from the repository checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_ingest import normalize_options_vendor_file, normalize_spot_file


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalize TradeMarkk option files + NIFTY spot into the project's long schema")
    ap.add_argument("--options-dir", required=True)
    ap.add_argument("--spot", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spot = normalize_spot_file(pd.read_parquet(args.spot) if str(args.spot).endswith(".parquet") else pd.read_csv(args.spot))
    option_paths = sorted(Path(args.options_dir).rglob("*.parquet"))
    option_paths = [p for p in option_paths if "index" not in p.parts]
    if not option_paths:
        raise SystemExit("No option parquet files found")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    first = True
    for p in option_paths:
        x = normalize_options_vendor_file(pd.read_parquet(p))
        x = x.merge(spot, on="timestamp", how="left", validate="many_to_one")
        x["entry_price"] = pd.to_numeric(x["open"], errors="coerce")
        x["future"] = pd.NA
        cols = ["timestamp","expiry","underlying","spot","future","strike","option_type","entry_price","ltp","volume","oi","iv"]
        x = x[[c for c in cols if c in x.columns]]
        # One file per expiry keeps memory bounded and is ideal for expiry-by-expiry backtests.
        dest = out / f"expiry={x['expiry'].iloc[0].date()}.parquet"
        x.to_parquet(dest, index=False)
        print(f"{p.name}: {len(x):,} rows -> {dest}")
        first = False


if __name__ == "__main__":
    main()
