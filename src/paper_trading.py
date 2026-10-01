from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core import (
    add_executable_prices,
    ensure_iv_delta,
    estimate_forward_from_parity,
    nifty_lot_size,
    pick_atm_iv,
    realized_vol,
    select_strikes,
    trade_cost,
)
from src.data_ingest import normalize_options_vendor_file, normalize_spot_file, read_any


FROZEN_STRATEGY_ID = "NIFTY_DTE6_2SD_PC75_STOP2_5_V1"
FROZEN_SPEC = {
    "entry_mode": "dte",
    "target_dte": 6,
    "entry_time": "10:00",
    "sd_multiple": 2.0,
    "strike_method": "pure_sd",
    "profit_capture": 0.75,
    "stop_multiple": 2.5,
    "time_exit_mode": "days_before_expiry",
    "time_exit_dte": 1,
    "time_exit_time": "15:00",
    "iv_rv_filter": None,
    "lots": 1,
    "min_entry_credit_points": 1.0,
}
FROZEN_SPEC_HASH = hashlib.sha256(
    json.dumps(FROZEN_SPEC, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


def assert_frozen_config(cfg) -> None:
    actual = {
        "entry_mode": cfg.entry_mode,
        "target_dte": int(cfg.target_dte),
        "entry_time": cfg.entry_time,
        "sd_multiple": float(cfg.sd_multiple),
        "strike_method": cfg.strike_method,
        "profit_capture": float(cfg.profit_capture),
        "stop_multiple": float(cfg.stop_multiple),
        "time_exit_mode": cfg.time_exit_mode,
        "time_exit_dte": int(cfg.time_exit_dte),
        "time_exit_time": cfg.time_exit_time,
        "iv_rv_filter": None if cfg.min_iv_rv_spread is None else float(cfg.min_iv_rv_spread),
        "lots": int(cfg.lots),
        "min_entry_credit_points": float(cfg.min_entry_credit_points),
    }
    for key, expected in FROZEN_SPEC.items():
        if actual[key] != expected:
            raise ValueError(
                f"Frozen strategy mismatch for {key}: expected {expected!r}, got {actual[key]!r}"
            )


def _daily_rv20_available_at_open(spot: pd.DataFrame) -> pd.Series:
    s = spot.copy()
    s["timestamp"] = pd.to_datetime(s["timestamp"], errors="coerce")
    s["date"] = s["timestamp"].dt.normalize()
    daily = s.groupby("date", as_index=False)["spot"].last().sort_values("date")
    daily["rv20"] = realized_vol(daily["spot"], 20)
    daily["rv20_available_at_open"] = daily["rv20"].shift(1)
    return daily.set_index("date")["rv20_available_at_open"]


def _planned_time_exit(expiry: pd.Timestamp, cfg) -> pd.Timestamp:
    exit_date = (pd.Timestamp(expiry).normalize() - pd.Timedelta(days=int(cfg.time_exit_dte))).normalize()
    hh, mm = map(int, cfg.time_exit_time.split(":"))
    return exit_date + pd.Timedelta(hours=hh, minutes=mm)


def _signal_row(
    *,
    status: str,
    reason: str,
    as_of: pd.Timestamp,
    expiry: pd.Timestamp,
) -> dict:
    return {
        "strategy_id": FROZEN_STRATEGY_ID,
        "strategy_spec_hash": FROZEN_SPEC_HASH,
        "signal_status": status,
        "signal_reason": reason,
        "as_of_timestamp": as_of,
        "entry_timestamp": as_of,
        "entry_date": as_of.normalize(),
        "expiry": expiry,
    }


def build_paper_signal(
    options: pd.DataFrame,
    spot_history: pd.DataFrame,
    expiry: pd.Timestamp,
    as_of: pd.Timestamp,
    cfg,
    costs,
    slippage_points: float | None = None,
) -> pd.DataFrame:
    """Create one frozen-strategy paper signal without looking at future option prices.

    The output contains planned entry/exit thresholds only. It never computes a realized
    P&L or uses data after the supplied as_of timestamp.
    """
    assert_frozen_config(cfg)
    expiry = pd.Timestamp(expiry).normalize()
    as_of = pd.Timestamp(as_of)

    expected_entry_date = (expiry - pd.Timedelta(days=int(cfg.target_dte))).normalize()
    row = _signal_row(status="NO_TRADE", reason="UNINITIALIZED", as_of=as_of, expiry=expiry)
    if as_of.normalize() != expected_entry_date:
        row["signal_reason"] = "WRONG_ENTRY_DATE"
        return pd.DataFrame([row])

    opt = normalize_options_vendor_file(options.copy())
    if slippage_points is None:
        slippage_points = float(costs.slippage_points_per_leg)
    if "future" not in opt.columns:
        opt["future"] = np.nan
    entry_col = "open" if "open" in opt.columns else "ltp"
    opt["entry_price"] = pd.to_numeric(opt[entry_col], errors="coerce")

    spot = normalize_spot_file(spot_history.copy())
    opt = opt.merge(spot, on="timestamp", how="left", suffixes=("", "_spot"))
    if "spot_spot" in opt.columns:
        opt["spot"] = opt["spot"].where(opt["spot"].notna(), opt["spot_spot"])
        opt = opt.drop(columns=["spot_spot"])
    opt = add_executable_prices(opt, slippage_points=float(slippage_points))

    chain = opt[(opt["timestamp"] == as_of) & (opt["expiry"] == expiry)].copy()
    chain = chain.dropna(subset=["spot", "entry_price"])
    if chain.empty:
        row["signal_reason"] = "NO_ENTRY_CHAIN"
        return pd.DataFrame([row])

    spot0 = float(chain["spot"].iloc[0])
    future0 = pd.to_numeric(chain["future"], errors="coerce")
    future = float(future0.dropna().iloc[0]) if not future0.dropna().empty and future0.dropna().iloc[0] > 0 else np.nan
    forward = future if np.isfinite(future) else estimate_forward_from_parity(
        chain, spot0, cfg.risk_free_rate, price_column="entry_price"
    )

    try:
        chain["forward"] = forward
        chain = ensure_iv_delta(
            chain, cfg.risk_free_rate, allow_iv_calc=True, price_column="entry_price"
        )
        atm_iv = pick_atm_iv(chain, forward, cfg.atm_band)
        lower = forward - forward * atm_iv * np.sqrt(
            max((pd.Timestamp(expiry) + pd.Timedelta(hours=15, minutes=30) - as_of).total_seconds(), 1.0)
            / (365.0 * 24 * 3600)
        ) * cfg.sd_multiple
        upper = forward + (forward - lower)
        rv20 = _daily_rv20_available_at_open(spot).get(as_of.normalize(), np.nan)
        put, call = select_strikes(chain, forward, atm_iv, expiry, as_of, cfg)
    except Exception as exc:
        row["signal_reason"] = f"MODEL_ERROR:{type(exc).__name__}"
        return pd.DataFrame([row])

    pk, ck = float(put["strike"]), float(call["strike"])
    p = chain[(chain["strike"] == pk) & (chain["option_type"] == "PE")].iloc[0]
    c = chain[(chain["strike"] == ck) & (chain["option_type"] == "CE")].iloc[0]

    put_bid = float(p["bid"]) if pd.notna(p["bid"]) else float(p["ltp"])
    call_bid = float(c["bid"]) if pd.notna(c["bid"]) else float(c["ltp"])
    put_ask = float(p["ask"]) if pd.notna(p["ask"]) else float(p["ltp"])
    call_ask = float(c["ask"]) if pd.notna(c["ask"]) else float(c["ltp"])
    put_mid = (put_bid + put_ask) / 2.0
    call_mid = (call_bid + call_ask) / 2.0
    planned_put_sell = max(0.0, put_bid - float(slippage_points))
    planned_call_sell = max(0.0, call_bid - float(slippage_points))
    credit_points = planned_put_sell + planned_call_sell

    lot = nifty_lot_size(expiry)
    multiplier = int(cfg.lots) * lot
    credit_rupees = credit_points * multiplier
    row.update({
        "signal_status": "READY" if credit_points >= cfg.min_entry_credit_points else "NO_TRADE",
        "signal_reason": "READY" if credit_points >= cfg.min_entry_credit_points else "NET_CREDIT_BELOW_MIN",
        "entry_spot": spot0,
        "forward_entry": forward,
        "atm_iv": atm_iv,
        "rv20": float(rv20) if pd.notna(rv20) else np.nan,
        "iv_rv_spread": float(atm_iv - rv20) if pd.notna(rv20) else np.nan,
        "sd_lower": lower,
        "sd_upper": upper,
        "put_strike": pk,
        "call_strike": ck,
        "put_delta": float(put["delta"]),
        "call_delta": float(call["delta"]),
        "put_bid": put_bid,
        "put_ask": put_ask,
        "put_mid": put_mid,
        "call_bid": call_bid,
        "call_ask": call_ask,
        "call_mid": call_mid,
        "planned_put_sell_price": planned_put_sell,
        "planned_call_sell_price": planned_call_sell,
        "planned_credit_points": credit_points,
        "lot_size": lot,
        "planned_credit_rupees": credit_rupees,
        "profit_target_debit_points": credit_points * (1.0 - cfg.profit_capture),
        "stop_debit_points": credit_points * cfg.stop_multiple,
        "scheduled_time_exit_timestamp": _planned_time_exit(expiry, cfg),
        "planned_entry_cost_rupees": trade_cost(
            notional_premium=credit_rupees,
            sell_premium=credit_rupees,
            buy_premium=0.0,
            orders=2,
            costs=costs,
            trade_date=as_of,
        ),
        "execution_slippage_assumption_points_per_leg": float(slippage_points),
    })
    return pd.DataFrame([row])


def initialize_ledger(signals: pd.DataFrame) -> pd.DataFrame:
    x = signals.copy()
    if x.empty:
        return x
    x["ledger_status"] = x["signal_status"].map({"READY": "SIGNAL"}).fillna("NO_TRADE")
    for c in [
        "actual_entry_timestamp",
        "actual_put_entry_fill",
        "actual_call_entry_fill",
        "actual_entry_credit_points",
        "actual_entry_credit_rupees",
        "actual_entry_slippage_vs_bid_points_per_leg",
        "actual_exit_timestamp",
        "actual_put_exit_fill",
        "actual_call_exit_fill",
        "actual_exit_debit_points",
        "actual_exit_reason",
        "gross_pnl",
        "transaction_cost",
        "net_pnl",
        "rule_deviation_flag",
    ]:
        if c not in x.columns:
            x[c] = np.nan
    return x


def record_entry_fill(
    ledger: pd.DataFrame,
    signal_id: str,
    fill_timestamp: pd.Timestamp,
    put_fill: float,
    call_fill: float,
) -> pd.DataFrame:
    x = ledger.copy()
    _check_signal_id(x, signal_id)
    idx = x.index[x["strategy_spec_hash"].eq(FROZEN_SPEC_HASH) & x["signal_status"].eq("READY") & x["strategy_id"].eq(FROZEN_STRATEGY_ID)]
    idx = idx[x.loc[idx, "signal_id"].eq(signal_id)] if "signal_id" in x.columns else idx
    if len(idx) != 1:
        raise ValueError(f"Expected one READY signal with signal_id={signal_id}, found {len(idx)}")
    i = idx[0]
    if x.at[i, "ledger_status"] not in {"SIGNAL", "OPEN"}:
        raise ValueError(f"Signal {signal_id} is not open for entry; status={x.at[i, 'ledger_status']}")

    put_fill = float(put_fill)
    call_fill = float(call_fill)
    if min(put_fill, call_fill) < 0:
        raise ValueError("Entry fills cannot be negative")

    x.at[i, "actual_entry_timestamp"] = pd.Timestamp(fill_timestamp)
    x.at[i, "actual_put_entry_fill"] = put_fill
    x.at[i, "actual_call_entry_fill"] = call_fill
    x.at[i, "actual_entry_credit_points"] = put_fill + call_fill
    x.at[i, "actual_entry_credit_rupees"] = (put_fill + call_fill) * int(x.at[i, "lot_size"])
    x.at[i, "actual_entry_slippage_vs_bid_points_per_leg"] = (
        (
            float(x.at[i, "put_bid"]) - put_fill
        )
        + (
            float(x.at[i, "call_bid"]) - call_fill
        )
    ) / 2.0
    x.at[i, "ledger_status"] = "OPEN"
    return x


def record_exit_fill(
    ledger: pd.DataFrame,
    signal_id: str,
    fill_timestamp: pd.Timestamp,
    put_fill: float,
    call_fill: float,
    exit_reason: str,
    costs,
) -> pd.DataFrame:
    x = ledger.copy()
    _check_signal_id(x, signal_id)
    idx = x.index[x["signal_id"].eq(signal_id)]
    if len(idx) != 1:
        raise ValueError(f"Expected one signal with signal_id={signal_id}, found {len(idx)}")
    i = idx[0]
    if x.at[i, "ledger_status"] != "OPEN":
        raise ValueError(f"Signal {signal_id} must be OPEN before exit")
    if pd.isna(x.at[i, "actual_entry_credit_rupees"]):
        raise ValueError("Missing actual entry fill")

    put_fill = float(put_fill)
    call_fill = float(call_fill)
    if min(put_fill, call_fill) < 0:
        raise ValueError("Exit fills cannot be negative")

    debit = put_fill + call_fill
    sell_premium = float(x.at[i, "actual_entry_credit_rupees"])
    buy_premium = debit * int(x.at[i, "lot_size"])
    gross = sell_premium - buy_premium
    transaction_cost = trade_cost(
        notional_premium=sell_premium + buy_premium,
        sell_premium=sell_premium,
        buy_premium=buy_premium,
        orders=4,
        costs=costs,
        trade_date=pd.Timestamp(x.at[i, "actual_entry_timestamp"]),
    )
    net = gross - transaction_cost

    target = float(x.at[i, "profit_target_debit_points"])
    stop = float(x.at[i, "stop_debit_points"])
    reason = str(exit_reason).lower()
    rule_flag = "RULE_REVIEW"
    if reason == "profit_target":
        rule_flag = "RULE_OK" if debit <= target + 1e-9 else "RULE_VIOLATION"
    elif reason == "stop":
        rule_flag = "RULE_OK" if debit >= stop - 1e-9 else "RULE_VIOLATION"
    elif reason == "time_exit":
        scheduled = pd.Timestamp(x.at[i, "scheduled_time_exit_timestamp"]).normalize()
        actual = pd.Timestamp(fill_timestamp).normalize()
        rule_flag = "RULE_OK" if actual == scheduled else "RULE_REVIEW"
    elif reason == "expiry":
        actual = pd.Timestamp(fill_timestamp).normalize()
        expiry = pd.Timestamp(x.at[i, "expiry"]).normalize()
        rule_flag = "RULE_OK" if actual == expiry else "RULE_REVIEW"

    x.at[i, "actual_exit_timestamp"] = pd.Timestamp(fill_timestamp)
    x.at[i, "actual_put_exit_fill"] = put_fill
    x.at[i, "actual_call_exit_fill"] = call_fill
    x.at[i, "actual_exit_debit_points"] = debit
    x.at[i, "actual_exit_reason"] = reason
    x.at[i, "gross_pnl"] = gross
    x.at[i, "transaction_cost"] = transaction_cost
    x.at[i, "net_pnl"] = net
    x.at[i, "rule_deviation_flag"] = rule_flag
    x.at[i, "ledger_status"] = "CLOSED"
    return x


def _check_signal_id(x: pd.DataFrame, signal_id: str) -> None:
    if "signal_id" not in x.columns:
        raise ValueError("Ledger has no signal_id column")


def add_signal_ids(signals: pd.DataFrame) -> pd.DataFrame:
    x = signals.copy()
    x["signal_id"] = (
        x["strategy_id"].astype(str)
        + "__"
        + pd.to_datetime(x["entry_timestamp"]).dt.strftime("%Y%m%dT%H%M")
        + "__"
        + pd.to_datetime(x["expiry"]).dt.strftime("%Y%m%d")
    )
    return x
