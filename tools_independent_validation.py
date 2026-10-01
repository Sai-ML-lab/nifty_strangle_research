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

    a["entry_date"] = a["entry_timestamp"].dt.normalize()
    b["entry_date"] = b["entry_timestamp"].dt.normalize()
    # Match on trading date + expiry only. Independent vendors often timestamp
    # the same market bar differently and may round/encode strikes differently.
    keys = ["entry_date", "expiry"]
    keep_a = keys + [c for c in ["initial_credit_points", "exit_timestamp", "exit_reason", "net_pnl", "data_quality_flag"] if c in a.columns]
    keep_b = keys + [c for c in ["initial_credit_points", "exit_timestamp", "exit_reason", "net_pnl", "data_quality_flag"] if c in b.columns]
    a = a[keep_a].rename(columns={c: f"primary_{c}" for c in keep_a if c not in keys})
    b = b[keep_b].rename(columns={c: f"secondary_{c}" for c in keep_b if c not in keys})
    m = a.merge(b, on=keys, how="outer", indicator=True)

    if "primary_net_pnl" in m and "secondary_net_pnl" in m:
        m["net_pnl_diff"] = m["secondary_net_pnl"] - m["primary_net_pnl"]
    if "primary_initial_credit_points" in m and "secondary_initial_credit_points" in m:
        m["entry_credit_diff_points"] = m["secondary_initial_credit_points"] - m["primary_initial_credit_points"]
    if {"primary_net_pnl", "secondary_net_pnl"}.issubset(matched.columns):
        matched["abs_pnl_diff"] = matched["net_pnl_diff"].abs()
    if "primary_exit_timestamp" in m and "secondary_exit_timestamp" in m:
        m["exit_timestamp_diff_minutes"] = (
            pd.to_datetime(m["secondary_exit_timestamp"], errors="coerce")
            - pd.to_datetime(m["primary_exit_timestamp"], errors="coerce")
        ).dt.total_seconds() / 60.0

    matched = m[m["_merge"] == "both"].copy()
    overlap_start = max(a["entry_date"].min(), b["entry_date"].min()) if len(a) and len(b) else pd.NaT
    overlap_end = min(a["entry_date"].max(), b["entry_date"].max()) if len(a) and len(b) else pd.NaT
    a_overlap = a[(a["entry_date"] >= overlap_start) & (a["entry_date"] <= overlap_end)] if pd.notna(overlap_start) else a.iloc[0:0]
    b_overlap = b[(b["entry_date"] >= overlap_start) & (b["entry_date"] <= overlap_end)] if pd.notna(overlap_start) else b.iloc[0:0]
    overlap_matched = matched[
        (matched["entry_date"] >= overlap_start) & (matched["entry_date"] <= overlap_end)
    ] if pd.notna(overlap_start) else matched.iloc[0:0]
    summary = pd.DataFrame(
        [{
            "primary_trades": len(a),
            "secondary_trades": len(b),
            "matched_trades": len(matched),
            "primary_only": int((m["_merge"] == "left_only").sum()),
            "secondary_only": int((m["_merge"] == "right_only").sum()),
            "match_rate_primary": float(len(matched) / len(a)) if len(a) else np.nan,
            "match_rate_secondary": float(len(matched) / len(b)) if len(b) else np.nan,
            "overlap_start": overlap_start.date().isoformat() if pd.notna(overlap_start) else None,
            "overlap_end": overlap_end.date().isoformat() if pd.notna(overlap_end) else None,
            "primary_overlap_trades": len(a_overlap),
            "secondary_overlap_trades": len(b_overlap),
            "matched_overlap_trades": len(overlap_matched),
            "match_rate_primary_overlap": float(len(overlap_matched) / len(a_overlap)) if len(a_overlap) else np.nan,
            "match_rate_secondary_overlap": float(len(overlap_matched) / len(b_overlap)) if len(b_overlap) else np.nan,
            "mean_abs_pnl_diff": float(matched["net_pnl_diff"].abs().mean()) if "net_pnl_diff" in matched else np.nan,
            "median_abs_pnl_diff": float(matched["net_pnl_diff"].abs().median()) if "net_pnl_diff" in matched else np.nan,
            "mean_abs_entry_credit_diff_points": float(matched["entry_credit_diff_points"].abs().mean()) if "entry_credit_diff_points" in matched else np.nan,
            "primary_strikes_match_rate": float(
                (
                    (matched["primary_put_strike"] == matched["secondary_put_strike"])
                    & (matched["primary_call_strike"] == matched["secondary_call_strike"])
                ).mean()
            ) if {"primary_put_strike", "secondary_put_strike", "primary_call_strike", "secondary_call_strike"}.issubset(matched.columns) else np.nan,
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
