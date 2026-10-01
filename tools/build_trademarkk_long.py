from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow direct execution from repo checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_ingest import normalize_options_vendor_file, normalize_spot_file


def normalize_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], errors="coerce")
    # Vendor timestamps are documented as IST. Keep them naive so all research
    # timestamps share one timezone convention without accidental UTC shifts.
    if getattr(x["timestamp"].dt, "tz", None) is not None:
        x["timestamp"] = x["timestamp"].dt.tz_localize(None)
    return x


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Normalize TradeMarkk NIFTY option files + spot into per-expiry research files"
    )
    ap.add_argument("--options-dir", required=True)
    ap.add_argument("--spot", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    option_dir = Path(args.options_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    spot_raw = pd.read_parquet(args.spot) if str(args.spot).endswith(".parquet") else pd.read_csv(args.spot)
    spot = normalize_spot_file(spot_raw)
    spot = normalize_timestamp_column(spot)
    spot = spot.drop_duplicates("timestamp").sort_values("timestamp")

    option_paths = sorted(option_dir.rglob("*.parquet"))
    option_paths = [p for p in option_paths if p.name != "NIFTY.parquet" and "index" not in p.parts]
    if not option_paths:
        raise SystemExit(f"No option parquet files found below {option_dir}")

    written = 0
    skipped = 0

    for p in option_paths:
        raw = pd.read_parquet(p)
        x = normalize_options_vendor_file(raw)
        x = normalize_timestamp_column(x)

        x = x.merge(spot, on="timestamp", how="left", validate="many_to_one")
        x["entry_price"] = pd.to_numeric(x["open"], errors="coerce")
        x["future"] = pd.NA

        cols = [
            "timestamp", "expiry", "underlying", "spot", "future",
            "strike", "option_type", "entry_price", "ltp",
            "volume", "oi", "iv", "bid", "ask", "delta",
        ]
        keep = [c for c in cols if c in x.columns]
        x = x[keep].sort_values(["timestamp", "strike", "option_type"])

        if x.empty or x["expiry"].isna().all():
            skipped += 1
            print(f"SKIP {p.name}: no valid expiry rows")
            continue

        expiry_values = x["expiry"].dropna().dt.normalize().unique()
        if len(expiry_values) != 1:
            raise ValueError(f"{p}: expected one expiry per file, found {len(expiry_values)}")

        expiry = pd.Timestamp(expiry_values[0])
        dest = out / f"expiry={expiry.date()}.parquet"
        x.to_parquet(dest, index=False)
        written += 1
        print(f"{p.name}: {len(x):,} rows -> {dest}")

    print(f"Written {written} expiry files; skipped {skipped}")


if __name__ == "__main__":
    main()
