from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_ingest import normalize_spot_file, read_any


IV_RV_BINS = [-np.inf, 0.0, 1.0, 2.0, 3.0, np.inf]
IV_RV_LABELS = ["<0", "0-1", "1-2", "2-3", "3+"]

RV20_BINS = [-np.inf, 10.0, 15.0, 20.0, 25.0, np.inf]
RV20_LABELS = ["<10%", "10-15%", "15-20%", "20-25%", "25%+"]

CREDIT_BINS = [-np.inf, 3.0, 5.0, 8.0, 12.0, np.inf]
CREDIT_LABELS = ["<3", "3-5", "5-8", "8-12", "12+"]

GAP_BINS = [-np.inf, 0.25, 0.50, 1.00, 1.50, np.inf]
GAP_LABELS = ["<0.25%", "0.25-0.5%", "0.5-1%", "1-1.5%", "1.5%+"]

MOVE_BINS = [-np.inf, 0.50, 1.00, 1.50, 2.00, np.inf]
MOVE_LABELS = ["<0.5%", "0.5-1%", "1-1.5%", "1.5-2%", "2%+"]


def _require(trades: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in trades.columns]
    if missing:
        raise ValueError(f"Trade ledger missing required columns: {missing}")


def _fixed_bucket(values: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    ).astype("string")


def _stats(x: pd.DataFrame) -> dict:
    p = pd.to_numeric(x["net_pnl"], errors="coerce").dropna()
    if p.empty:
        return {
            "trades": 0,
            "wins": 0,
            "win_rate": np.nan,
            "total_net_pnl": 0.0,
            "avg_trade_pnl": np.nan,
            "median_trade_pnl": np.nan,
            "profit_factor": np.nan,
            "avg_initial_credit_points": np.nan,
            "stop_share": np.nan,
        }

    wins = p[p > 0]
    losses = p[p < 0]
    stop_share = np.nan
    if "exit_reason" in x.columns:
        stop_loss = x.loc[x["exit_reason"].eq("stop") & x["net_pnl"].lt(0), "net_pnl"].sum()
        total_loss = abs(losses.sum())
        stop_share = abs(float(stop_loss)) / total_loss if total_loss else np.nan

    return {
        "trades": int(len(p)),
        "wins": int((p > 0).sum()),
        "win_rate": float((p > 0).mean()),
        "total_net_pnl": float(p.sum()),
        "avg_trade_pnl": float(p.mean()),
        "median_trade_pnl": float(p.median()),
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) else np.inf,
        "avg_initial_credit_points": (
            float(pd.to_numeric(x.get("initial_credit_points"), errors="coerce").mean())
            if "initial_credit_points" in x
            else np.nan
        ),
        "stop_share": stop_share,
    }


def add_spot_diagnostics(
    trades: pd.DataFrame,
    spot: pd.DataFrame | str | Path | None = None,
) -> pd.DataFrame:
    x = trades.copy()
    _require(x, ["entry_timestamp", "entry_spot", "exit_spot", "net_pnl"])
    x["entry_timestamp"] = pd.to_datetime(x["entry_timestamp"], errors="coerce")
    if "exit_timestamp" in x.columns:
        x["exit_timestamp"] = pd.to_datetime(x["exit_timestamp"], errors="coerce")
    x["entry_date"] = x["entry_timestamp"].dt.normalize()

    if spot is None:
        x["entry_gap_abs_pct"] = np.nan
        x["max_abs_spot_move_pct"] = (
            (pd.to_numeric(x["exit_spot"], errors="coerce") / pd.to_numeric(x["entry_spot"], errors="coerce") - 1.0)
            .abs()
            * 100.0
        )
        x["max_up_spot_move_pct"] = x["max_abs_spot_move_pct"]
        x["max_down_spot_move_pct"] = x["max_abs_spot_move_pct"]
        return x

    raw = read_any(spot) if isinstance(spot, (str, Path)) else spot.copy()
    s = normalize_spot_file(raw)
    s["date"] = s["timestamp"].dt.normalize()

    daily = s.groupby("date", as_index=False)["spot"].last().sort_values("date")
    daily["prev_close"] = daily["spot"].shift(1)
    prev_close = daily.set_index("date")["prev_close"]
    x["prev_close_spot"] = x["entry_date"].map(prev_close)
    x["entry_gap_abs_pct"] = (
        (pd.to_numeric(x["entry_spot"], errors="coerce") / x["prev_close_spot"] - 1.0).abs() * 100.0
    )

    max_abs = []
    max_up = []
    max_down = []
    for _, row in x.iterrows():
        start = row["entry_timestamp"]
        end = row.get("exit_timestamp", pd.NaT)
        if pd.isna(start):
            max_abs.append(np.nan)
            max_up.append(np.nan)
            max_down.append(np.nan)
            continue
        if pd.isna(end):
            end = start
        path = s[(s["timestamp"] >= start) & (s["timestamp"] <= end)]
        entry_spot = float(row["entry_spot"]) if pd.notna(row["entry_spot"]) else np.nan
        if path.empty or not np.isfinite(entry_spot) or entry_spot <= 0:
            fallback = (
                abs(float(row["exit_spot"]) / entry_spot - 1.0) * 100.0
                if pd.notna(row["exit_spot"]) and np.isfinite(entry_spot) and entry_spot > 0
                else np.nan
            )
            max_abs.append(fallback)
            max_up.append(fallback)
            max_down.append(fallback)
            continue
        ret = path["spot"] / entry_spot - 1.0
        max_abs.append(float(ret.abs().max() * 100.0))
        max_up.append(float(max(0.0, ret.max()) * 100.0))
        max_down.append(float(abs(min(0.0, ret.min())) * 100.0))

    x["max_abs_spot_move_pct"] = max_abs
    x["max_up_spot_move_pct"] = max_up
    x["max_down_spot_move_pct"] = max_down
    return x


def enrich_regime_features(trades: pd.DataFrame, spot: pd.DataFrame | str | Path | None = None) -> pd.DataFrame:
    x = add_spot_diagnostics(trades, spot)
    x["year"] = x["entry_timestamp"].dt.year
    x["quarter"] = "Q" + x["entry_timestamp"].dt.quarter.astype("string")

    if "iv_rv_spread" in x.columns:
        x["iv_rv_spread_points"] = pd.to_numeric(x["iv_rv_spread"], errors="coerce") * 100.0
        x["iv_rv_bucket"] = _fixed_bucket(x["iv_rv_spread_points"], IV_RV_BINS, IV_RV_LABELS)
    else:
        x["iv_rv_spread_points"] = np.nan
        x["iv_rv_bucket"] = pd.Series("missing", index=x.index, dtype="string")

    if "rv20" in x.columns:
        x["rv20_pct"] = pd.to_numeric(x["rv20"], errors="coerce") * 100.0
        x["rv20_bucket"] = _fixed_bucket(x["rv20_pct"], RV20_BINS, RV20_LABELS)
    else:
        x["rv20_pct"] = np.nan
        x["rv20_bucket"] = pd.Series("missing", index=x.index, dtype="string")

    credit = pd.to_numeric(x.get("initial_credit_points"), errors="coerce")
    x["credit_bucket"] = _fixed_bucket(credit, CREDIT_BINS, CREDIT_LABELS)

    x["entry_gap_bucket"] = _fixed_bucket(
        pd.to_numeric(x["entry_gap_abs_pct"], errors="coerce"), GAP_BINS, GAP_LABELS
    )
    x["max_move_bucket"] = _fixed_bucket(
        pd.to_numeric(x["max_abs_spot_move_pct"], errors="coerce"), MOVE_BINS, MOVE_LABELS
    )

    pnl = pd.to_numeric(x["net_pnl"], errors="coerce")
    rupee_credit = pd.to_numeric(
        x.get("initial_credit_rupees", pd.Series(np.nan, index=x.index)),
        errors="coerce",
    )
    x["loss_to_credit_multiple"] = np.where(
        pnl < 0,
        (-pnl) / rupee_credit.replace(0, np.nan),
        np.nan,
    )
    x["loss_size_bucket"] = pd.cut(
        x["loss_to_credit_multiple"],
        bins=[-np.inf, 1.0, 2.0, 3.0, 5.0, np.inf],
        labels=["<=1x", "1-2x", "2-3x", "3-5x", ">5x"],
        right=True,
    ).astype("string")
    return x


def summarize_dimensions(trades: pd.DataFrame) -> pd.DataFrame:
    x = trades.copy()
    dimensions = [
        "year",
        "quarter",
        "iv_rv_bucket",
        "rv20_bucket",
        "credit_bucket",
        "entry_gap_bucket",
        "max_move_bucket",
        "exit_reason",
        "loss_size_bucket",
    ]
    rows = []
    for dimension in dimensions:
        if dimension not in x.columns:
            continue
        y = x[x[dimension].notna()].copy()
        if y.empty:
            continue
        for bucket, group in y.groupby(dimension, observed=True, sort=False):
            rows.append({
                "dimension": dimension,
                "bucket": str(bucket),
                **_stats(group),
            })
    return pd.DataFrame(rows)


def run_attribution(
    trades: pd.DataFrame,
    spot: pd.DataFrame | str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    enriched = enrich_regime_features(trades, spot)
    summary = summarize_dimensions(enriched)
    return enriched, summary
