from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["entry_timestamp", "expiry"]


def compare_ledgers(primary: pd.DataFrame, secondary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = primary.copy()
    b = secondary.copy()
    for x in (a, b):
        x["entry_timestamp"] = pd.to_datetime(x["entry_timestamp"], errors="coerce")
        x["expiry"] = pd.to_datetime(x["expiry"], errors="coerce").dt.normalize()

    extra_keys = [c for c in ["put_strike", "call_strike"] if c in a.columns and c in b.columns]
    keys = KEYS + extra_keys
    keep_a = keys + [c for c in ["initial_credit_points", "exit_timestamp", "exit_reason", "net_pnl", "data_quality_flag"] if c in a.columns]
    keep_b = keys + [c for c in ["initial_credit_points", "exit_timestamp", "exit_reason", "net_pnl", "data_quality_flag"] if c in b.columns]
    a = a[keep_a].rename(columns={c: f"primary_{c}" for c in keep_a if c not in keys})
    b = b[keep_b].rename(columns={c: f"secondary_{c}" for c in keep_b if c not in keys})
    m = a.merge(b, on=keys, how="outer", indicator=True)

    if "primary_net_pnl" in m and "secondary_net_pnl" in m:
        m["net_pnl_diff"] = m["secondary_net_pnl"] - m["primary_net_pnl"]
    if "primary_initial_credit_points" in m and "secondary_initial_credit_points" in m:
        m["entry_credit_diff_points"] = m["secondary_initial_credit_points"] - m["primary_initial_credit_points"]
    if "primary_exit_timestamp" in m and "secondary_exit_timestamp" in m:
        m["exit_timestamp_diff_minutes"] = (
            pd.to_datetime(m["secondary_exit_timestamp"], errors="coerce")
            - pd.to_datetime(m["primary_exit_timestamp"], errors="coerce")
        ).dt.total_seconds() / 60.0

    matched = m[m["_merge"] == "both"].copy()
    summary = pd.DataFrame(
        [{
            "primary_trades": len(a),
            "secondary_trades": len(b),
            "matched_trades": len(matched),
            "primary_only": int((m["_merge"] == "left_only").sum()),
            "secondary_only": int((m["_merge"] == "right_only").sum()),
            "match_rate_primary": float(len(matched) / len(a)) if len(a) else np.nan,
            "match_rate_secondary": float(len(matched) / len(b)) if len(b) else np.nan,
            "mean_abs_pnl_diff": float(matched["net_pnl_diff"].abs().mean()) if "net_pnl_diff" in matched else np.nan,
            "median_abs_pnl_diff": float(matched["net_pnl_diff"].abs().median()) if "net_pnl_diff" in matched else np.nan,
            "mean_abs_entry_credit_diff_points": float(matched["entry_credit_diff_points"].abs().mean()) if "entry_credit_diff_points" in matched else np.nan,
        }]
    )
    return m, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare two frozen NIFTY trade ledgers from different data sources")
    ap.add_argument("--primary", required=True)
    ap.add_argument("--secondary", required=True)
    ap.add_argument("--out-dir", default="results/independent_validation")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    a, b = pd.read_csv(args.primary), pd.read_csv(args.secondary)
    detail, summary = compare_ledgers(a, b)
    detail.to_csv(out / "ledger_comparison.csv", index=False)
    summary.to_csv(out / "ledger_comparison_summary.csv", index=False)
    print("\nIndependent ledger comparison:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
