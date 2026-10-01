from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from src.data_ingest import normalize_options_vendor_file, normalize_spot_file

def _load(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input: {path}")

def build_rissin_long(source_path, spot_path, out_dir, start_date=None, end_date=None):
    source = _load(Path(source_path))
    source.columns = [str(c).strip().lower() for c in source.columns]
    if "underlying" not in source.columns:
        raise ValueError("Input must contain underlying")
    source = source[source["underlying"].astype(str).str.upper().eq("NIFTY")]
    if "granularity" in source.columns:
        g = source["granularity"].astype(str).str.lower()
        source = source[g.str.contains("min", na=False)]
    x = normalize_options_vendor_file(source)
    x["timestamp"] = pd.to_datetime(x["timestamp"], errors="coerce")
    if getattr(x["timestamp"].dt, "tz", None) is not None:
        x["timestamp"] = x["timestamp"].dt.tz_localize(None)
    if start_date:
        x = x[x["timestamp"] >= pd.Timestamp(start_date)]
    if end_date:
        x = x[x["timestamp"] < pd.Timestamp(end_date) + pd.Timedelta(days=1)]
    spot = normalize_spot_file(_load(Path(spot_path)))
    if getattr(spot["timestamp"].dt, "tz", None) is not None:
        spot["timestamp"] = spot["timestamp"].dt.tz_localize(None)
    x = x.merge(spot.drop_duplicates("timestamp"), on="timestamp", how="left", validate="many_to_one")
    x["future"] = pd.NA
    x["entry_price"] = pd.to_numeric(x["open"], errors="coerce")
    x["ltp"] = pd.to_numeric(x["ltp"] if "ltp" in x.columns else x["close"], errors="coerce")
    keep = ["timestamp","expiry","underlying","spot","future","strike","option_type","entry_price","ltp","volume","oi","iv","bid","ask","delta"]
    x = x[[c for c in keep if c in x.columns]].dropna(subset=["timestamp","expiry","strike","option_type"])
    x = x.sort_values(["timestamp","expiry","strike","option_type"])
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for expiry, g in x.groupby(x["expiry"].dt.normalize(), sort=True):
        dest = out / f"expiry={pd.Timestamp(expiry).date()}.parquet"
        g.to_parquet(dest, index=False)
        print(f"{pd.Timestamp(expiry).date()}: {len(g):,} rows -> {dest}")
    print(f"Built {x['expiry'].dt.normalize().nunique():,} expiries / {len(x):,} rows")

def main():
    ap=argparse.ArgumentParser(description="Build NIFTY 1-minute per-expiry files from Rissin")
    ap.add_argument("--source", required=True)
    ap.add_argument("--spot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start-date", default="2024-10-01")
    ap.add_argument("--end-date", default=None)
    a=ap.parse_args()
    build_rissin_long(a.source,a.spot,a.out,a.start_date,a.end_date)

if __name__ == "__main__":
    main()
