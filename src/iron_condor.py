from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from run_batch_research import entry_date_for_expiry, entry_timestamp_for_expiry, expiry_from_filename, load_spot
from src.core import (
    Costs,
    StrategyConfig,
    add_executable_prices,
    ensure_iv_delta,
    estimate_forward_from_parity,
    nifty_lot_size,
    pick_atm_iv,
    option_value_at_expiry,
    select_strikes,
    trade_cost,
)
from src.risk_metrics import portfolio_risk_metrics


def _quote(chain: pd.DataFrame, strike: float, option_type: str) -> pd.Series:
    q = chain[(chain["strike"] == strike) & (chain["option_type"] == option_type)]
    if q.empty:
        raise KeyError(f"Missing {option_type} strike {strike}")
    return q.iloc[0]


def _pick_wing(chain: pd.DataFrame, short_strike: float, option_type: str, requested_width: float) -> tuple[float, float]:
    """Choose the nearest available strike at least requested_width beyond the short strike."""
    if option_type == "PE":
        candidates = chain.loc[
            (chain["option_type"] == "PE")
            & (chain["strike"] <= short_strike - requested_width),
            "strike",
        ].dropna().unique()
        if len(candidates) == 0:
            raise KeyError(f"No PE wing >= {requested_width:g} points below {short_strike:g}")
        wing = float(np.max(candidates))
    else:
        candidates = chain.loc[
            (chain["option_type"] == "CE")
            & (chain["strike"] >= short_strike + requested_width),
            "strike",
        ].dropna().unique()
        if len(candidates) == 0:
            raise KeyError(f"No CE wing >= {requested_width:g} points above {short_strike:g}")
        wing = float(np.min(candidates))
    return wing, abs(short_strike - wing)


def _condor_intrinsic(spot: float, put_wing: float, put_short: float, call_short: float, call_wing: float) -> float:
    put_spread = max(put_short - spot, 0.0) - max(put_wing - spot, 0.0)
    call_spread = max(spot - call_short, 0.0) - max(spot - call_wing, 0.0)
    return float(put_spread + call_spread)


def single_expiry_condor_trade(
    raw: pd.DataFrame,
    spot: pd.DataFrame,
    cfg: StrategyConfig,
    costs: Costs,
    expiry: pd.Timestamp,
    wing_width: float,
) -> dict | None:
    """Backtest one fixed-width iron condor with the same DTE6 + 2-SD entry logic."""
    if raw.empty:
        return None

    from src.data_ingest import normalize_options_vendor_file

    x = normalize_options_vendor_file(raw)
    if x["timestamp"].dt.tz is not None:
        x["timestamp"] = x["timestamp"].dt.tz_localize(None)

    spot_for_join = spot.drop(columns=["spot"]) if "spot" in x.columns else spot
    x = x.merge(spot_for_join, on="timestamp", how="left", validate="many_to_one")
    x["future"] = np.nan
    entry_column = "open" if "open" in x.columns else "entry_price"
    x["entry_price"] = pd.to_numeric(x[entry_column], errors="coerce")
    x = add_executable_prices(x, slippage_points=costs.slippage_points_per_leg)
    x["expiry"] = pd.Timestamp(expiry).normalize()

    entry_ts = entry_timestamp_for_expiry(x, expiry, cfg)
    if entry_ts is None:
        return None

    entry_chain = x[x["timestamp"] == entry_ts].dropna(subset=["spot", "entry_price"]).copy()
    if entry_chain.empty:
        return None

    spot0 = float(entry_chain["spot"].iloc[0])
    forward = estimate_forward_from_parity(entry_chain, spot0, cfg.risk_free_rate, price_column="entry_price")
    entry_chain["forward"] = forward

    try:
        entry_chain = ensure_iv_delta(entry_chain, cfg.risk_free_rate, allow_iv_calc=True, price_column="entry_price")
        atm_iv = pick_atm_iv(entry_chain, forward, cfg.atm_band)
        rv20_value = (
            float(entry_chain["rv20"].iloc[0])
            if "rv20" in entry_chain.columns and pd.notna(entry_chain["rv20"].iloc[0])
            else np.nan
        )
        put, call = select_strikes(entry_chain, forward, atm_iv, expiry, pd.Timestamp(entry_ts), cfg)
    except Exception:
        return None

    put_k, call_k = float(put["strike"]), float(call["strike"])
    try:
        put_wing, actual_put_width = _pick_wing(entry_chain, put_k, "PE", wing_width)
        call_wing, actual_call_width = _pick_wing(entry_chain, call_k, "CE", wing_width)
    except KeyError:
        return None

    try:
        ps = _quote(entry_chain, put_k, "PE")
        cs = _quote(entry_chain, call_k, "CE")
        pw = _quote(entry_chain, put_wing, "PE")
        cw = _quote(entry_chain, call_wing, "CE")
    except KeyError:
        return None

    # Entry: sell the central PE/CE and buy the protective wings.
    short_credit = float(ps["entry_sell_exec"] + cs["entry_sell_exec"])
    wing_debit = float(pw["buy_exec"] + cw["buy_exec"])
    net_credit = short_credit - wing_debit
    if not np.isfinite(net_credit) or net_credit < cfg.min_entry_credit_points:
        return None

    lot = nifty_lot_size(expiry)
    multiplier = cfg.lots * lot
    entry_short_sell = short_credit * multiplier
    entry_long_buy = wing_debit * multiplier

    path = x[
        (x["timestamp"] >= entry_ts)
        & (x["timestamp"].dt.date <= pd.Timestamp(expiry).date())
        & (x["expiry"] == pd.Timestamp(expiry).normalize())
    ].copy()

    held = path[
        ((path["strike"] == put_k) & (path["option_type"] == "PE"))
        | ((path["strike"] == call_k) & (path["option_type"] == "CE"))
        | ((path["strike"] == put_wing) & (path["option_type"] == "PE"))
        | ((path["strike"] == call_wing) & (path["option_type"] == "CE"))
    ].copy()

    marks = held.pivot_table(
        index="timestamp",
        columns=["strike", "option_type"],
        values=["buy_exec", "sell_exec"],
        aggfunc="first",
    )

    exit_ts = None
    exit_reason = None
    exit_debit = None
    max_loss_points = max(actual_put_width, actual_call_width) - net_credit
    if not np.isfinite(max_loss_points) or max_loss_points <= 0:
        return None

    if not marks.empty:
        def col(kind: str, strike: float, typ: str) -> pd.Series:
            return marks[(kind, strike, typ)]

        # Closing debit = buy back shorts - sell the long wings.
        debit = (
            col("buy_exec", put_k, "PE")
            + col("buy_exec", call_k, "CE")
            - col("sell_exec", put_wing, "PE")
            - col("sell_exec", call_wing, "CE")
        ).rename("debit").dropna()

        profit_cut = net_credit * (1.0 - cfg.profit_capture)
        stop_cut = net_credit * cfg.stop_multiple
        hit = debit.index[(debit <= profit_cut) | (debit >= stop_cut)]

        hh, mm = map(int, cfg.time_exit_time.split(":"))
        if getattr(cfg, "time_exit_mode", "days_before_expiry") == "days_before_expiry":
            exit_date = (pd.Timestamp(expiry).normalize() - pd.Timedelta(days=int(cfg.time_exit_dte))).normalize()
            date_mask = debit.index.normalize() == exit_date
        else:
            date_mask = debit.index.weekday == cfg.time_exit_weekday
        time_mask = date_mask & (
            (debit.index.hour > hh)
            | ((debit.index.hour == hh) & (debit.index.minute >= mm))
        )
        th = debit.index[time_mask]

        candidates = []
        if len(hit):
            exit_ts = hit[0]
            exit_reason = "profit_target" if debit.loc[exit_ts] <= profit_cut else "stop"
        if len(th):
            candidates.append((th[0], "time_exit"))
        if exit_ts is not None:
            candidates.append((exit_ts, exit_reason))
        if candidates:
            exit_ts, exit_reason = min(candidates, key=lambda z: z[0])
            exit_debit = float(debit.loc[exit_ts])

    if exit_ts is None:
        spot_rows = spot[spot["timestamp"].dt.date == pd.Timestamp(expiry).date()]
        if spot_rows.empty:
            return None
        spot_exit = float(spot_rows.iloc[-1]["spot"])
        exit_debit_points = _condor_intrinsic(spot_exit, put_wing, put_k, call_k, call_wing)
        exit_ts = pd.Timestamp(expiry)
        exit_reason = "expiry"
        intrinsic_gap_points = 0.0
        data_quality_flag = "PASS"
    else:
        spot_row = path[path["timestamp"] == exit_ts]
        if spot_row.empty or pd.isna(spot_row.iloc[0]["spot"]):
            return None
        spot_exit = float(spot_row.iloc[0]["spot"])
        exit_debit_points = float(exit_debit)
        intrinsic = _condor_intrinsic(spot_exit, put_wing, put_k, call_k, call_wing)
        intrinsic_gap_points = exit_debit_points - intrinsic
        upper_bound = max(actual_put_width, actual_call_width) + 1.0
        data_quality_flag = (
            "WARN_CONDOR_PAYOFF_BOUND"
            if intrinsic_gap_points < -1.0 or exit_debit_points > upper_bound
            else "PASS"
        )

    if exit_reason == "expiry":
        sell_premium_exit = 0.0
        buy_premium_exit = exit_debit_points * multiplier
    else:
        exit_chain = path[path["timestamp"] == exit_ts]
        try:
            p = _quote(exit_chain, put_k, "PE")
            c = _quote(exit_chain, call_k, "CE")
            pw = _quote(exit_chain, put_wing, "PE")
            cw = _quote(exit_chain, call_wing, "CE")
        except KeyError:
            return None
        buy_short = float(p["buy_exec"] + c["buy_exec"]) * multiplier
        sell_long = float(pw["sell_exec"] + cw["sell_exec"]) * multiplier
        sell_premium_exit = sell_long
        buy_premium_exit = buy_short

    sell_premium_total = entry_short_sell + sell_premium_exit
    buy_premium_total = entry_long_buy + buy_premium_exit
    gross = sell_premium_total - buy_premium_total
    transaction_cost = trade_cost(
        sell_premium_total + buy_premium_total,
        sell_premium_total,
        buy_premium_total,
        8,
        costs,
        pd.Timestamp(entry_ts),
    )

    return {
        "strategy": "iron_condor",
        "entry_timestamp": entry_ts,
        "expiry": pd.Timestamp(expiry).normalize(),
        "lot_size": lot,
        "entry_spot": spot0,
        "exit_spot": spot_exit,
        "forward_entry": forward,
        "atm_iv": atm_iv,
        "rv20": rv20_value,
        "iv_rv_spread": (atm_iv - rv20_value) if np.isfinite(rv20_value) else np.nan,
        "put_wing": put_wing,
        "put_strike": put_k,
        "call_strike": call_k,
        "call_wing": call_wing,
        "put_wing_width": actual_put_width,
        "call_wing_width": actual_call_width,
        "put_delta": float(put["delta"]),
        "call_delta": float(call["delta"]),
        "initial_credit_points": net_credit,
        "initial_credit_rupees": net_credit * multiplier,
        "exit_timestamp": exit_ts,
        "exit_reason": exit_reason,
        "exit_debit_points": exit_debit_points,
        "exit_intrinsic_points": _condor_intrinsic(spot_exit, put_wing, put_k, call_k, call_wing),
        "exit_intrinsic_gap_points": intrinsic_gap_points,
        "data_quality_flag": data_quality_flag,
        "max_loss_points": max_loss_points,
        "max_loss_rupees": max_loss_points * multiplier,
        "credit_to_max_loss_pct": (net_credit / max_loss_points * 100.0) if max_loss_points > 0 else np.nan,
        "stop_threshold_exceeds_max_loss": bool(net_credit * cfg.stop_multiple > max(actual_put_width, actual_call_width)),
        "gross_pnl": gross,
        "transaction_cost": transaction_cost,
        "net_pnl": gross - transaction_cost,
        "profit_capture": cfg.profit_capture,
        "stop_multiple": cfg.stop_multiple,
        "requested_wing_width": wing_width,
    }


def _stats(trades: pd.DataFrame) -> dict:
    p = pd.to_numeric(trades.get("net_pnl", pd.Series(dtype=float)), errors="coerce").dropna()
    if p.empty:
        return {"trades": 0, "net_pnl": 0.0, "expectancy": np.nan, "profit_factor": np.nan}
    wins, losses = p[p > 0], p[p < 0]
    return {
        "trades": int(len(p)),
        "net_pnl": float(p.sum()),
        "expectancy": float(p.mean()),
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) else np.inf,
    }


def build_condor_ledgers(
    options_dir: str | Path,
    spot_path: str | Path,
    base_cfg: StrategyConfig,
    costs: Costs,
    wing_widths: Iterable[float],
    profit_capture_values: Iterable[float],
    stop_multiple_values: Iterable[float],
) -> dict[tuple[float, float, float], pd.DataFrame]:
    spot = load_spot(Path(spot_path))
    files = []
    for f in sorted(Path(options_dir).rglob("*.parquet")):
        expiry = expiry_from_filename(f)
        if expiry is not None:
            files.append((expiry, f))
    files.sort(key=lambda z: z[0])

    chosen: dict[pd.Timestamp, tuple[pd.Timestamp, Path]] = {}
    for expiry, f in files:
        d = entry_date_for_expiry(expiry, base_cfg)
        chosen.setdefault(d, (expiry, f))

    keys = [
        (float(width), float(pc), float(sm))
        for width in wing_widths
        for pc in profit_capture_values
        for sm in stop_multiple_values
    ]
    ledgers = {k: [] for k in keys}

    # Read each expiry once; only the pre-declared structure/exit candidates vary.
    for i, (_, (expiry, f)) in enumerate(sorted(chosen.items()), 1):
        raw = pd.read_parquet(f)
        for width, pc, sm in keys:
            cfg = replace(base_cfg, profit_capture=pc, stop_multiple=sm, min_iv_rv_spread=None)
            row = single_expiry_condor_trade(raw, spot, cfg, costs, expiry, width)
            if row is not None and getattr(cfg, "exclude_quality_warnings", False) and row.get("data_quality_flag") != "PASS":
                row = None
            if row is not None:
                ledgers[(width, pc, sm)].append(row)
        if i % 25 == 0:
            print(f"processed {i}/{len(chosen)} entry dates")

    return {
        k: pd.DataFrame(v).sort_values("entry_timestamp") if v else pd.DataFrame()
        for k, v in ledgers.items()
    }


def summarize_condors(
    ledgers: dict[tuple[float, float, float], pd.DataFrame],
    starting_capital: float,
) -> pd.DataFrame:
    rows = []
    for (width, pc, sm), ledger in sorted(ledgers.items()):
        m = portfolio_risk_metrics(ledger, starting_capital)
        s = _stats(ledger)
        m.update(
            {
                "requested_wing_width": width,
                "profit_capture": pc,
                "stop_multiple": sm,
                "avg_initial_credit_points": float(ledger["initial_credit_points"].mean()) if not ledger.empty else np.nan,
                "avg_max_loss_rupees": float(ledger["max_loss_rupees"].mean()) if not ledger.empty else np.nan,
                "avg_credit_to_max_loss_pct": float(ledger["credit_to_max_loss_pct"].mean()) if not ledger.empty else np.nan,
                "data_quality_warnings": int((ledger["data_quality_flag"] != "PASS").sum()) if not ledger.empty else 0,
            }
        )
        rows.append({**s, **m})
    return pd.DataFrame(rows)
