from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


REQUIRED_COLUMNS = {
    "timestamp", "expiry", "underlying", "spot", "strike", "option_type"
}


@dataclass(frozen=True)
class Costs:
    brokerage_per_order: float = 20.0
    stt_sell_option_pct: float = 0.0015
    exchange_txn_pct: float = 0.0003553
    sebi_turnover_pct: float = 0.000001
    stamp_duty_buy_option_pct: float = 0.00003
    gst_pct: float = 0.18
    slippage_points_per_leg: float = 0.50


def nifty_lot_size(expiry: pd.Timestamp) -> int:
    """Historical NIFTY lot size schedule relevant to the 2023+ research archive.

    50: contracts through 25-Apr-2024 / first revised weekly expiry was 02-May-2024.
    25: revised contracts through 19-Nov-2024.
    75: revised contracts from 26-Nov-2024 through 30-Dec-2025 expiry.
    65: revised contracts from the first post-30-Dec-2025 weekly expiry onward.

    Keep this function explicit so historical P&L is not distorted by today's lot size.
    """
    d = pd.Timestamp(expiry).normalize()
    if d < pd.Timestamp("2024-05-02"):
        return 50
    if d < pd.Timestamp("2024-11-26"):
        return 25
    if d < pd.Timestamp("2026-01-06"):
        return 75
    return 65


@dataclass(frozen=True)
class StrategyConfig:
    lot_size: int = 65
    risk_free_rate: float = 0.06
    calendar_days_per_year: int = 365
    trading_days_per_year: int = 252
    entry_mode: str = "weekday"       # weekday | dte
    entry_weekday: int = 2
    target_dte: int = 6
    entry_time: str = "10:00"
    sd_multiple: float = 2.0
    target_delta: float = 0.05
    strike_method: str = "hybrid"
    atm_band: int = 3
    minimum_oi: float = 0.0
    minimum_volume: float = 0.0
    profit_capture: float = 0.50
    stop_multiple: float = 2.0
    time_exit_weekday: int = 0
    time_exit_time: str = "15:00"
    settle_at_expiry: bool = True
    lots: int = 1
    min_entry_credit_points: float = 1.0
    starting_capital: float = 1_000_000.0


@dataclass
class OptionQuote:
    strike: float
    option_type: str
    bid: float
    ask: float
    ltp: float
    iv: float
    delta: float
    volume: float = 0.0
    oi: float = 0.0

    @property
    def mid(self) -> float:
        if np.isfinite(self.bid) and np.isfinite(self.ask) and self.bid >= 0 and self.ask >= self.bid:
            return float((self.bid + self.ask) / 2.0)
        return float(self.ltp)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    aliases = {
        "datetime": "timestamp", "date": "timestamp", "time": "timestamp",
        "expiry_date": "expiry", "expirydate": "expiry",
        "underlying_value": "spot", "underlyingvalue": "spot",
        "optiontype": "option_type", "type": "option_type",
        "open_interest": "oi", "openinterest": "oi",
        "close": "ltp", "last_price": "ltp", "lastprice": "ltp",
    }
    x = x.rename(columns={k: v for k, v in aliases.items() if k in x.columns})
    missing = REQUIRED_COLUMNS - set(x.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    x["timestamp"] = pd.to_datetime(x["timestamp"], errors="coerce")
    x["expiry"] = pd.to_datetime(x["expiry"], errors="coerce").dt.normalize()
    x["option_type"] = x["option_type"].astype(str).str.upper().str.strip()
    x = x[x["option_type"].isin(["CE", "PE"])]

    for c in ["spot", "future", "strike", "bid", "ask", "ltp", "entry_price", "volume", "oi", "iv", "delta"]:
        if c not in x.columns:
            x[c] = np.nan
        x[c] = pd.to_numeric(x[c], errors="coerce")

    x = x.dropna(subset=["timestamp", "expiry", "spot", "strike", "option_type"])
    return x.sort_values(["timestamp", "expiry", "strike", "option_type"]).reset_index(drop=True)


def business_year_fraction(start: pd.Timestamp, end: pd.Timestamp) -> float:
    days = max((end.normalize() - start.normalize()).days, 0)
    return max(days / 365.0, 1e-8)


def black76_price(forward: float, strike: float, vol: float, t: float, r: float, call: bool) -> float:
    if t <= 0:
        intrinsic = max(forward - strike, 0.0) if call else max(strike - forward, 0.0)
        return exp(-r * t) * intrinsic
    vol = max(float(vol), 1e-8)
    d1 = (log(forward / strike) + 0.5 * vol * vol * t) / (vol * sqrt(t))
    d2 = d1 - vol * sqrt(t)
    disc = exp(-r * t)
    if call:
        return disc * (forward * norm.cdf(d1) - strike * norm.cdf(d2))
    return disc * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))


def black76_delta(forward: float, strike: float, vol: float, t: float, r: float, call: bool) -> float:
    if t <= 0:
        if call:
            return 1.0 if forward > strike else 0.0
        return -1.0 if forward < strike else 0.0
    vol = max(float(vol), 1e-8)
    d1 = (log(forward / strike) + 0.5 * vol * vol * t) / (vol * sqrt(t))
    return exp(-r * t) * (norm.cdf(d1) if call else norm.cdf(d1) - 1.0)


def implied_vol_black76(price: float, forward: float, strike: float, t: float, r: float, call: bool) -> float:
    if any(v <= 0 for v in [price, forward, strike, t]):
        return np.nan
    intrinsic = black76_price(forward, strike, 1e-8, t, r, call)
    if price <= intrinsic + 1e-8:
        return np.nan
    upper = exp(-r * t) * (forward if call else strike)
    if price >= upper:
        return np.nan
    try:
        return float(brentq(lambda v: black76_price(forward, strike, v, t, r, call) - price, 1e-6, 5.0))
    except ValueError:
        return np.nan


def ensure_iv_delta(x: pd.DataFrame, r: float, allow_iv_calc: bool = True, price_column: str = "mid_price") -> pd.DataFrame:
    x = x.copy()
    existing_forward = x["forward"] if "forward" in x.columns else pd.Series(np.nan, index=x.index)
    fallback_forward = x["future"].where(x["future"].notna() & (x["future"] > 0), x["spot"])
    x["forward"] = existing_forward.where(existing_forward.notna() & (existing_forward > 0), fallback_forward)
    expiry_ts = x["expiry"] + pd.Timedelta(hours=15, minutes=30)
    x["t_years"] = (expiry_ts - x["timestamp"]).dt.total_seconds().clip(lower=1) / (365.0 * 24 * 3600)

    # Prefer supplied IV, otherwise infer from executable/observed price.
    if "iv" not in x.columns:
        x["iv"] = np.nan
    if allow_iv_calc:
        missing_iv = x["iv"].isna() | (x["iv"] <= 0)
        if missing_iv.any():
            # Vector-free loop keyed by original index for correctness.
            for idx, row in x.loc[missing_iv].iterrows():
                price = float(x.at[idx, price_column])
                iv = implied_vol_black76(
                    price, float(row["forward"]), float(row["strike"]), float(row["t_years"]), r,
                    row["option_type"] == "CE"
                )
                x.at[idx, "iv"] = iv

    if "delta" not in x.columns:
        x["delta"] = np.nan
    missing_delta = x["delta"].isna() & x["iv"].notna()
    for idx, row in x.loc[missing_delta].iterrows():
        x.at[idx, "delta"] = black76_delta(
            float(row["forward"]), float(row["strike"]), float(row["iv"]), float(row["t_years"]), r,
            row["option_type"] == "CE"
        )
    return x


def add_executable_prices(x: pd.DataFrame, slippage_points: float = 0.0) -> pd.DataFrame:
    x = x.copy()
    if "ltp" not in x.columns:
        x["ltp"] = np.nan
    if "entry_price" not in x.columns:
        x["entry_price"] = x["ltp"]
    if "bid" not in x.columns:
        x["bid"] = np.nan
    if "ask" not in x.columns:
        x["ask"] = np.nan

    x["mid_price"] = np.where(
        x["bid"].notna() & x["ask"].notna() & (x["ask"] >= x["bid"]),
        (x["bid"] + x["ask"]) / 2.0,
        x["ltp"],
    )
    x["sell_exec"] = np.where(x["bid"].notna(), x["bid"] - slippage_points, x["ltp"] - slippage_points)
    x["buy_exec"] = np.where(x["ask"].notna(), x["ask"] + slippage_points, x["ltp"] + slippage_points)
    x["entry_sell_exec"] = np.where(
        x["bid"].notna(), x["bid"] - slippage_points, x["entry_price"] - slippage_points
    )
    x["sell_exec"] = x["sell_exec"].clip(lower=0)
    x["buy_exec"] = x["buy_exec"].clip(lower=0)
    x["entry_sell_exec"] = x["entry_sell_exec"].clip(lower=0)
    return x


def realized_vol(spot: pd.Series, window: int = 20, annualization: int = 252) -> pd.Series:
    """Annualized volatility from a daily closing-price series."""
    ret = np.log(spot).diff()
    return ret.rolling(window).std() * np.sqrt(annualization)


def estimate_forward_from_parity(
    chain: pd.DataFrame,
    spot: float,
    r: float,
    price_column: str = "entry_price",
) -> float:
    """Estimate a short-dated forward using put-call parity, then fall back to spot.

    For matched CE/PE strikes: F ~= K + (C-P)*exp(rT). Because historical datasets
    often have no bid/ask, the observed/open price is used and the median across
    near-ATM matched strikes is taken for robustness.
    """
    if "timestamp" not in chain or "expiry" not in chain or chain.empty:
        return float(spot)
    ts = pd.Timestamp(chain["timestamp"].iloc[0])
    expiry = pd.Timestamp(chain["expiry"].iloc[0])
    expiry_ts = expiry + pd.Timedelta(hours=15, minutes=30)
    t = max((expiry_ts - ts).total_seconds() / (365.0 * 24 * 3600), 1e-8)
    px = chain.copy()
    if price_column not in px.columns:
        return float(spot)
    pivot = px.pivot_table(index="strike", columns="option_type", values=price_column, aggfunc="first")
    needed = [c for c in ["CE", "PE"] if c in pivot.columns]
    if len(needed) < 2:
        return float(spot)
    y = pivot.dropna(subset=["CE", "PE"]).copy()
    if y.empty:
        return float(spot)
    y["dist"] = (y.index.to_numpy(dtype=float) - float(spot)).__abs__()
    y = y.sort_values("dist").head(9)
    fwd = y.index.to_numpy(dtype=float) + (y["CE"].to_numpy() - y["PE"].to_numpy()) * np.exp(r * t)
    good = fwd[np.isfinite(fwd) & (fwd > 0)]
    return float(np.median(good)) if len(good) else float(spot)


def expiry_close_timestamp(expiry: pd.Timestamp) -> pd.Timestamp:
    """NIFTY settlement/reference timestamp used for option expiry calculations."""
    d = pd.Timestamp(expiry).normalize()
    return d + pd.Timedelta(hours=15, minutes=30)


def pick_atm_iv(chain: pd.DataFrame, forward: float, atm_band: int = 3) -> float:
    y = chain.copy()
    y["dist"] = (y["strike"] - forward).abs()
    candidates = y.sort_values("dist").head(max(2, atm_band * 2))
    vals = candidates.loc[candidates["iv"].notna() & (candidates["iv"] > 0), "iv"]
    if len(vals) == 0:
        raise ValueError("No valid ATM IV available at entry snapshot")
    return float(vals.median())


def sd_band(forward: float, iv: float, expiry: pd.Timestamp, ts: pd.Timestamp, sd: float) -> Tuple[float, float, float]:
    expiry_ts = expiry_close_timestamp(expiry)
    t = max((expiry_ts - ts).total_seconds() / (365.0 * 24 * 3600), 1e-8)
    move = forward * iv * sqrt(t) * sd
    return forward - move, forward + move, move


def select_strikes(chain: pd.DataFrame, forward: float, iv: float, expiry: pd.Timestamp, ts: pd.Timestamp, cfg: StrategyConfig) -> Tuple[pd.Series, pd.Series]:
    y = chain.copy()
    y = y[(y["volume"].fillna(0) >= cfg.minimum_volume) & (y["oi"].fillna(0) >= cfg.minimum_oi)]
    lower, upper, _ = sd_band(forward, iv, expiry, ts, cfg.sd_multiple)
    put = y[y["option_type"] == "PE"].copy()
    call = y[y["option_type"] == "CE"].copy()
    if put.empty or call.empty:
        raise ValueError("Insufficient PE/CE chain data")

    if cfg.strike_method == "pure_sd":
        put["dist"] = (put["strike"] - lower).abs()
        call["dist"] = (call["strike"] - upper).abs()
    elif cfg.strike_method == "delta":
        put["dist"] = (put["delta"] + cfg.target_delta).abs()
        call["dist"] = (call["delta"] - cfg.target_delta).abs()
    elif cfg.strike_method == "hybrid":
        # First locate strikes nearest the SD boundary, then among a small neighborhood
        # choose the one closest to target absolute delta.
        put["sd_dist"] = (put["strike"] - lower).abs()
        call["sd_dist"] = (call["strike"] - upper).abs()
        ppool = put.sort_values("sd_dist").head(7).copy()
        cpool = call.sort_values("sd_dist").head(7).copy()
        ppool["dist"] = (ppool["delta"] + cfg.target_delta).abs()
        cpool["dist"] = (cpool["delta"] - cfg.target_delta).abs()
        put, call = ppool, cpool
    else:
        raise ValueError(f"Unknown strike_method: {cfg.strike_method}")

    p = put.sort_values(["dist", "strike"]).iloc[0]
    c = call.sort_values(["dist", "strike"]).iloc[0]
    return p, c


def option_value_at_expiry(strike: float, spot: float, option_type: str) -> float:
    if option_type == "CE":
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


def trade_cost(
    notional_premium: float,
    sell_premium: float,
    buy_premium: float,
    orders: int,
    costs: Costs,
    trade_date: pd.Timestamp | None = None,
) -> float:
    """Estimate trade costs. Tax rates that changed in 2026 are date-aware.

    NSE currently reports STT of 0.15% on option-sale premium from 1-Apr-2026 and
    equity-option transaction outflow of Rs 3,553/crore of premium per side. Before
    1-Apr-2026, STT on option sale was 0.10%. The exchange total remained Rs 3,553/crore
    before/after the March 2026 reclassification.
    """
    turnover = max(sell_premium + buy_premium, 0.0)
    brokerage = costs.brokerage_per_order * orders
    date = pd.Timestamp(trade_date).normalize() if trade_date is not None else pd.Timestamp("2026-04-01")
    stt_rate = 0.0015 if date >= pd.Timestamp("2026-04-01") else 0.0010
    exchange_rate = costs.exchange_txn_pct
    stt = sell_premium * stt_rate
    exchange = turnover * exchange_rate
    sebi = turnover * costs.sebi_turnover_pct
    stamp = buy_premium * costs.stamp_duty_buy_option_pct
    gst_base = brokerage + exchange + sebi
    gst = gst_base * costs.gst_pct
    return brokerage + stt + exchange + sebi + stamp + gst
