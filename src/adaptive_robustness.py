from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.adaptive_target_optimizer import summarize_oos, walk_forward_adaptive
from src.core import (
    Costs,
    add_executable_prices,
    ensure_iv_delta,
    estimate_forward_from_parity,
    pick_atm_iv,
    select_strikes,
)
from src.data_ingest import normalize_options_vendor_file
from src.entry_timing_optimizer import _entry_credit, _exit_trade, _merge_spot_reference, add_lot_metrics


ROBUSTNESS_PNL = "net_pnl_current"
DEFAULT_STRESS_SLIPPAGES = (0.50, 0.75, 1.00, 1.25, 1.50)
DEFAULT_WINNER_REMOVALS = (1, 3, 5, 10)


def pnl_stats(trades: pd.DataFrame, pnl_column: str = ROBUSTNESS_PNL) -> dict:
    if trades.empty:
        return {
            "trades": 0,
            "win_rate": np.nan,
            "expectancy": np.nan,
            "median_pnl": np.nan,
            "avg_win": np.nan,
            "avg_loss": np.nan,
            "profit_factor": np.nan,
            "max_drawdown": np.nan,
        }

    p = trades[pnl_column].astype(float)
    wins = p[p > 0]
    losses = p[p < 0]
    ordered = trades.sort_values("entry_timestamp")
    equity = ordered[pnl_column].astype(float).cumsum()
    dd = equity - equity.cummax()
    return {
        "trades": int(len(p)),
        "win_rate": float((p > 0).mean()),
        "expectancy": float(p.mean()),
        "median_pnl": float(p.median()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": (
            float(wins.sum() / abs(losses.sum()))
            if len(losses) and abs(losses.sum()) > 0
            else np.inf
        ),
        "max_drawdown": float(dd.min()),
    }


def winner_concentration(
    trades: pd.DataFrame,
    *,
    pnl_column: str = ROBUSTNESS_PNL,
    removals: Sequence[int] = DEFAULT_WINNER_REMOVALS,
) -> pd.DataFrame:
    columns = [
        "removed_top_winners",
        "trades_remaining",
        "total_pnl_remaining",
        "pnl_change_vs_base",
        "expectancy_remaining",
        "win_rate_remaining",
        "profit_factor_remaining",
        "max_drawdown_remaining",
        "top_winners_pnl",
        "top_winners_share_of_base_pnl_pct",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns)

    base_pnl = float(trades[pnl_column].sum())
    ranked = trades.sort_values(pnl_column, ascending=False)
    rows = []
    for n in removals:
        n = int(n)
        n = max(0, min(n, len(ranked)))
        top = ranked.head(n)
        remaining = ranked.iloc[n:].copy()
        stats = pnl_stats(remaining, pnl_column=pnl_column)
        top_pnl = float(top[pnl_column].sum())
        share = np.nan if base_pnl == 0 else 100.0 * top_pnl / base_pnl
        rows.append(
            {
                "removed_top_winners": n,
                "trades_remaining": int(len(remaining)),
                "total_pnl_remaining": float(remaining[pnl_column].sum()),
                "pnl_change_vs_base": float(remaining[pnl_column].sum() - base_pnl),
                "expectancy_remaining": stats["expectancy"],
                "win_rate_remaining": stats["win_rate"],
                "profit_factor_remaining": stats["profit_factor"],
                "max_drawdown_remaining": stats["max_drawdown"],
                "top_winners_pnl": top_pnl,
                "top_winners_share_of_base_pnl_pct": share,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def grouped_oos_stats(
    trades: pd.DataFrame,
    group_columns: Iterable[str],
    *,
    pnl_column: str = ROBUSTNESS_PNL,
) -> pd.DataFrame:
    base_columns = list(group_columns)
    output_columns = base_columns + list(pnl_stats(pd.DataFrame(columns=[pnl_column])).keys())
    if trades.empty:
        return pd.DataFrame(columns=output_columns)

    rows = []
    for key, group in trades.groupby(base_columns, dropna=False):
        values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(base_columns, values))
        row.update(pnl_stats(group, pnl_column=pnl_column))
        rows.append(row)
    return pd.DataFrame(rows, columns=output_columns)


def pair_adaptive_with_control(
    adaptive_oos: pd.DataFrame,
    control_trades: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    if adaptive_oos.empty or control_trades.empty:
        return pd.DataFrame(), {
            "adaptive_trades": int(len(adaptive_oos)),
            "control_trades": int(len(control_trades)),
            "matched_dates": 0,
            "date_match_rate": np.nan,
            "same_expiry_matches": 0,
            "same_expiry_rate": np.nan,
        }

    a = adaptive_oos.copy()
    c = control_trades.copy()
    a["entry_date"] = pd.to_datetime(a["entry_timestamp"]).dt.normalize()
    c["entry_date"] = pd.to_datetime(c["entry_timestamp"]).dt.normalize()

    c = c.sort_values("entry_timestamp").drop_duplicates("entry_date", keep="first")
    merged = a.merge(
        c[["entry_date", "expiry", "net_pnl_5lot_current", "put_strike", "call_strike"]],
        on="entry_date",
        how="left",
        suffixes=("_adaptive", "_control"),
        validate="many_to_one",
    )
    merged["pnl_delta_adaptive_minus_control"] = (
        merged[ROBUSTNESS_PNL] - merged["net_pnl_5lot_current"]
    )
    matched = merged[merged["net_pnl_5lot_current"].notna()].copy()
    same_expiry = matched["expiry_adaptive"].eq(matched["expiry_control"])

    stats = {
        "adaptive_trades": int(len(a)),
        "control_trades": int(len(c)),
        "matched_dates": int(len(matched)),
        "date_match_rate": float(len(matched) / len(a)) if len(a) else np.nan,
        "same_expiry_matches": int(same_expiry.sum()),
        "same_expiry_rate": float(same_expiry.mean()) if len(matched) else np.nan,
        "mean_adaptive_pnl": float(matched[ROBUSTNESS_PNL].mean()) if len(matched) else np.nan,
        "mean_control_pnl": float(matched["net_pnl_5lot_current"].mean()) if len(matched) else np.nan,
        "mean_pnl_delta": float(matched["pnl_delta_adaptive_minus_control"].mean()) if len(matched) else np.nan,
        "median_pnl_delta": float(matched["pnl_delta_adaptive_minus_control"].median()) if len(matched) else np.nan,
        "adaptive_positive_rate": float((matched[ROBUSTNESS_PNL] > 0).mean()) if len(matched) else np.nan,
        "control_positive_rate": float((matched["net_pnl_5lot_current"] > 0).mean()) if len(matched) else np.nan,
    }
    return merged, stats


def apply_baseline_selection(
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    if candidates.empty or selected.empty:
        return pd.DataFrame()

    rows = []
    for _, fold in selected.iterrows():
        if str(fold.get("selection_status", "SELECTED")) != "SELECTED":
            continue
        mask = (
            candidates["session_offset"].eq(float(fold["selected_session_offset"]))
            & candidates["sd"].eq(float(fold["selected_sd"]))
            & candidates["target_rupees"].eq(float(fold["selected_target_rupees"]))
            & candidates["stop_multiple"].eq(float(fold["selected_stop_multiple"]))
            & (candidates["entry_timestamp"] >= pd.Timestamp(fold["test_start"]))
            & (candidates["entry_timestamp"] < pd.Timestamp(fold["test_end"]) + pd.Timedelta(days=1))
        )
        test_slice = candidates.loc[mask].copy()
        if test_slice.empty:
            continue
        test_slice["wf_train_start"] = pd.Timestamp(fold["train_start"])
        test_slice["wf_test_start"] = pd.Timestamp(fold["test_start"])
        rows.append(test_slice)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_slippage_stress(
    *,
    options_dir: Path,
    spot: pd.DataFrame,
    base_cfg,
    costs: Costs,
    session_offsets: Iterable[int],
    sds: Iterable[float],
    targets: Iterable[float],
    stop_multiples: Iterable[float],
    reference_lot_size: int,
    lots: int,
    baseline_selected: pd.DataFrame,
    baseline_oos: pd.DataFrame,
    stress_slippages: Iterable[float],
    train_months: int = 12,
    test_months: int = 3,
    rebalance_months: int = 3,
    min_train_trades: int = 20,
) -> tuple[pd.DataFrame, dict[float, pd.DataFrame]]:
    pipeline_rows = []
    frozen_rows = []
    frozen_trades: dict[float, pd.DataFrame] = {}

    for slippage in stress_slippages:
        stress_costs = replace(costs, slippage_points_per_leg=float(slippage))
        candidates = build_adaptive_target_trades_local(
            options_dir=options_dir,
            spot=spot,
            base_cfg=base_cfg,
            costs=stress_costs,
            session_offsets=session_offsets,
            sds=sds,
            target_rupees=targets,
            stop_multiples=stop_multiples,
            reference_lot_size=reference_lot_size,
            lots=lots,
        )
        selected, oos = walk_forward_adaptive(
            candidates,
            train_months=train_months,
            test_months=test_months,
            rebalance_months=rebalance_months,
            min_train_trades=min_train_trades,
        )
        summary = summarize_oos(oos) if not oos.empty else {"trades": 0}
        pipeline_rows.append(
            {
                "slippage_points_per_leg": float(slippage),
                "mode": "reoptimized_walk_forward",
                **summary,
            }
        )

        frozen = apply_baseline_selection(candidates, baseline_selected)
        frozen_trades[float(slippage)] = frozen
        frozen_summary = summarize_oos(frozen) if not frozen.empty else {"trades": 0}
        frozen_rows.append(
            {
                "slippage_points_per_leg": float(slippage),
                "mode": "baseline_selection_frozen",
                **frozen_summary,
            }
        )

    stress = pd.DataFrame(pipeline_rows + frozen_rows)
    return stress, frozen_trades


def build_adaptive_target_trades_local(**kwargs) -> pd.DataFrame:
    # Local wrapper keeps the public robustness API independent of the module import name.
    from src.adaptive_target_optimizer import build_adaptive_target_trades

    return build_adaptive_target_trades(**kwargs)


def build_frozen_control_trades(
    *,
    options_dir: Path,
    spot: pd.DataFrame,
    adaptive_oos: pd.DataFrame,
    base_cfg,
    costs: Costs,
    reference_lot_size: int,
    lots: int,
) -> pd.DataFrame:
    """Build a frozen 2SD/75%/2.5x control at the adaptive trade's exact entry+expiry.

    This deliberately does not require the control entry date to be exactly six
    calendar days before expiry. The comparison is same timestamp + same expiry,
    so it isolates strike/exit mechanics from the adaptive session-timing choice.
    """
    if adaptive_oos.empty:
        return pd.DataFrame()

    target_expiries = pd.to_datetime(adaptive_oos["expiry"]).dt.normalize().drop_duplicates()
    requested = adaptive_oos.copy()
    rows: list[dict] = []

    for expiry in target_expiries:
        expiry = pd.Timestamp(expiry).normalize()
        matches = requested[
            pd.to_datetime(requested["expiry"]).dt.normalize().eq(expiry)
        ].copy()
        path = None
        for candidate in sorted(options_dir.rglob("*.parquet")):
            stem = candidate.stem.replace("expiry=", "")
            try:
                candidate_expiry = pd.Timestamp(stem).normalize()
            except Exception:
                continue
            if candidate_expiry == expiry:
                path = candidate
                break
        if path is None:
            continue

        raw = pd.read_parquet(path)
        x = normalize_options_vendor_file(raw)
        if getattr(x["timestamp"].dt, "tz", None) is not None:
            x["timestamp"] = x["timestamp"].dt.tz_localize(None)
        x = _merge_spot_reference(x, spot)
        x["future"] = np.nan
        if "open" in x.columns:
            x["entry_price"] = pd.to_numeric(x["open"], errors="coerce")
        else:
            x["entry_price"] = pd.to_numeric(x["ltp"], errors="coerce")
        x = add_executable_prices(x, slippage_points=costs.slippage_points_per_leg)
        x["expiry"] = expiry

        for _, requested_row in matches.iterrows():
            entry_ts = pd.Timestamp(requested_row["entry_timestamp"])
            entry_chain = x[x["timestamp"].eq(entry_ts)].dropna(
                subset=["spot", "entry_price"]
            ).copy()
            if entry_chain.empty:
                continue
            spot0 = float(entry_chain["spot"].iloc[0])
            forward = estimate_forward_from_parity(
                entry_chain,
                spot0,
                base_cfg.risk_free_rate,
                price_column="entry_price",
            )
            entry_chain["forward"] = forward
            try:
                entry_chain = ensure_iv_delta(
                    entry_chain,
                    base_cfg.risk_free_rate,
                    allow_iv_calc=True,
                    price_column="entry_price",
                )
                atm_iv = pick_atm_iv(entry_chain, forward, base_cfg.atm_band)
                cfg = replace(
                    base_cfg,
                    target_dte=6,
                    sd_multiple=2.0,
                    strike_method="pure_sd",
                )
                put, call = select_strikes(
                    entry_chain,
                    forward,
                    atm_iv,
                    expiry,
                    entry_ts,
                    cfg,
                )
                trade = _exit_trade(
                    x=x,
                    entry_ts=entry_ts,
                    expiry=expiry,
                    put_k=float(put["strike"]),
                    call_k=float(call["strike"]),
                    credit_points=_entry_credit(entry_chain, float(put["strike"]), float(call["strike"])),
                    profit_capture=0.75,
                    stop_multiple=2.5,
                    lots=1,
                    costs=costs,
                    time_exit_dte=base_cfg.time_exit_dte,
                    time_exit_time=base_cfg.time_exit_time,
                )
            except Exception:
                continue
            row = {
                **trade,
                "entry_spot": float(spot0),
                "forward_entry": float(forward),
                "atm_iv": float(atm_iv),
                "session_offset": requested_row.get("session_offset", np.nan),
                "sd": 2.0,
                "profit_capture": 0.75,
                "stop_multiple": 2.5,
            }
            rows.append(row)

    control = pd.DataFrame(rows)
    if control.empty:
        return control
    return add_lot_metrics(
        control,
        costs,
        lots=lots,
        reference_lot_size=reference_lot_size,
    )
