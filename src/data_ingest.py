from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


OPTION_ALIASES = {
    "datetime": "timestamp",
    "timestamp": "timestamp",
    "expiry_date": "expiry",
    "expirydate": "expiry",
    "strike_price": "strike",
    "strikeprice": "strike",
    "right": "option_type",
    "optiontype": "option_type",
    "stock_code": "underlying",
    "open_interest": "oi",
    "openinterest": "oi",
    "close": "ltp",
}


def read_any(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(p)
    raise ValueError(f"Unsupported file: {p}")


def _rename_common(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    ren = {k: v for k, v in OPTION_ALIASES.items() if k in x.columns and v not in x.columns}
    x = x.rename(columns=ren)
    return x


def normalize_options_vendor_file(df: pd.DataFrame) -> pd.DataFrame:
    x = _rename_common(df)
    if "option_type" in x:
        x["option_type"] = x["option_type"].astype(str).str.upper().str.strip()
        x["option_type"] = x["option_type"].replace({"CALL": "CE", "PUT": "PE"})
    if "expiry" not in x:
        raise ValueError("Option file has no expiry column")
    x["timestamp"] = pd.to_datetime(x["timestamp"], errors="coerce")
    raw_exp = x["expiry"].astype(str).str.strip()
    x["expiry"] = pd.to_datetime(raw_exp, errors="coerce", format="%d-%b-%Y").dt.normalize()
    missing = x["expiry"].isna()
    if missing.any():
        x.loc[missing, "expiry"] = pd.to_datetime(raw_exp[missing], errors="coerce", format="%Y-%m-%d").dt.normalize()
    x["strike"] = pd.to_numeric(x["strike"], errors="coerce")
    for c in ["open", "high", "low", "ltp", "volume", "oi", "iv"]:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    # Build required fields for the backtest; spot/future are joined later.
    if "underlying" not in x:
        x["underlying"] = "NIFTY"
    x["underlying"] = x["underlying"].astype(str).str.upper()
    x = x[(x["option_type"].isin(["CE", "PE"])) & x["timestamp"].notna() & x["expiry"].notna() & x["strike"].notna()]
    return x


def normalize_spot_file(df: pd.DataFrame) -> pd.DataFrame:
    x = _rename_common(df)
    if "timestamp" not in x or "close" not in x and "ltp" not in x:
        raise ValueError("Spot file must contain timestamp and close")
    x["timestamp"] = pd.to_datetime(x["timestamp"], errors="coerce")
    if "close" in x:
        x["spot"] = pd.to_numeric(x["close"], errors="coerce")
    else:
        x["spot"] = pd.to_numeric(x["ltp"], errors="coerce")
    return x[["timestamp", "spot"]].dropna().drop_duplicates("timestamp").sort_values("timestamp")


def normalize_futures_file(df: pd.DataFrame) -> pd.DataFrame:
    x = _rename_common(df)
    if "timestamp" not in x:
        raise ValueError("Futures file must contain timestamp")
    x["timestamp"] = pd.to_datetime(x["timestamp"], errors="coerce")
    if "expiry" in x:
        x["expiry"] = pd.to_datetime(x["expiry"], errors="coerce").dt.normalize()
    x["future"] = pd.to_numeric(x["close"], errors="coerce")
    # Prefer nearest-expiry future when several expiries are present at a timestamp.
    if "expiry" in x:
        x = x.sort_values(["timestamp", "expiry"])
        x = x.groupby("timestamp", as_index=False).first()
    else:
        x = x.groupby("timestamp", as_index=False)["future"].first()
    return x[["timestamp", "future"]].dropna().drop_duplicates("timestamp").sort_values("timestamp")


def read_files(paths: Iterable[str | Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frames.append(normalize_options_vendor_file(read_any(path)))
    if not frames:
        raise ValueError("No option files supplied")
    return pd.concat(frames, ignore_index=True)


def build_long_chain(
    option_paths: Iterable[str | Path],
    spot_path: str | Path,
    future_path: str | Path | None = None,
) -> pd.DataFrame:
    opts = read_files(option_paths)
    spot = normalize_spot_file(read_any(spot_path))
    x = opts.merge(spot, on="timestamp", how="left", validate="many_to_one")
    if future_path:
        fut = normalize_futures_file(read_any(future_path))
        x = x.merge(fut, on="timestamp", how="left", validate="many_to_one")
    else:
        x["future"] = np.nan

    # Use the bar OPEN for entry information to avoid look-ahead when the vendor timestamps
    # each 1-minute bar at the bar start. Keep LTP=close for later marking/exits.
    x["ltp"] = pd.to_numeric(x["ltp"], errors="coerce") if "ltp" in x else pd.to_numeric(x["close"], errors="coerce")
    if "close" in x:
        x["ltp"] = pd.to_numeric(x["close"], errors="coerce")
    x["entry_price"] = pd.to_numeric(x["open"], errors="coerce") if "open" in x else x["ltp"]

    keep = [
        "timestamp", "expiry", "underlying", "spot", "future", "strike", "option_type",
        "entry_price", "ltp", "volume", "oi", "iv",
    ]
    for c in ["bid", "ask", "delta"]:
        if c in x:
            keep.append(c)
    x = x[[c for c in keep if c in x.columns]]
    x = x.dropna(subset=["timestamp", "expiry", "spot", "strike", "option_type"])
    return x.sort_values(["timestamp", "expiry", "strike", "option_type"]).reset_index(drop=True)


def validate_chain(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame([{"check": "non_empty", "status": "FAIL", "value": 0}])
    def add(name, ok, value):
        rows.append({"check": name, "status": "PASS" if ok else "WARN", "value": value})
    add("non_empty", len(df) > 0, len(df))
    add("unique_contract_snapshots", not df.duplicated(["timestamp", "expiry", "strike", "option_type"]).any(), int(df.duplicated(["timestamp", "expiry", "strike", "option_type"]).sum()))
    add("valid_option_types", df["option_type"].isin(["CE", "PE"]).all(), int((~df["option_type"].isin(["CE", "PE"])).sum()))
    add("non_negative_prices", (df[[c for c in ["entry_price", "ltp"] if c in df]].fillna(0) >= 0).all().all(), float(df[[c for c in ["entry_price", "ltp"] if c in df]].min().min()))
    add("spot_coverage_pct", df["spot"].notna().mean() >= 0.999, round(float(df["spot"].notna().mean()), 6))
    add("expiry_after_timestamp_pct", (df["expiry"] >= df["timestamp"].dt.normalize()).mean() >= 0.999, round(float((df["expiry"] >= df["timestamp"].dt.normalize()).mean()), 6))
    add("timestamp_coverage_days", df["timestamp"].dt.normalize().nunique() >= 3, int(df["timestamp"].dt.normalize().nunique()))
    return pd.DataFrame(rows)


def cli() -> None:
    parser = argparse.ArgumentParser(description="Normalize vendor NIFTY option files into the research long format")
    parser.add_argument("--options", nargs="*", default=None, help="Option Parquet/CSV files")
    parser.add_argument("--options-dir", default=None, help="Directory containing option Parquet/CSV files (searched recursively)")
    parser.add_argument("--pattern", default="*.parquet", help="Filename pattern used with --options-dir")
    parser.add_argument("--spot", required=True, help="Spot CSV/Parquet")
    parser.add_argument("--future", default=None, help="Optional futures CSV/Parquet")
    parser.add_argument("--out", required=True, help="Output Parquet/CSV")
    parser.add_argument("--validate-out", default=None, help="Optional validation report CSV")
    args = parser.parse_args()

    option_paths = list(args.options or [])
    if args.options_dir:
        option_paths.extend(sorted(Path(args.options_dir).rglob(args.pattern)))
    if not option_paths:
        parser.error("Provide --options or --options-dir")
    data = build_long_chain(option_paths, args.spot, args.future)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".parquet":
        data.to_parquet(out, index=False)
    else:
        data.to_csv(out, index=False)
    report = validate_chain(data)
    print(report.to_string(index=False))
    if args.validate_out:
        Path(args.validate_out).parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.validate_out, index=False)
    print(f"Saved {len(data):,} rows -> {out}")


if __name__ == "__main__":
    cli()
