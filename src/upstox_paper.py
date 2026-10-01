from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
from zoneinfo import ZoneInfo

from src.core import Costs, option_value_at_expiry, trade_cost
from src.paper_trading import (
    FROZEN_STRATEGY_ID,
    add_signal_ids,
    build_paper_signal,
    initialize_ledger,
    record_entry_fill,
    record_exit_fill,
)
from src.strangle_backtest import load_config


IST = ZoneInfo("Asia/Kolkata")
NIFTY_INDEX_KEY = "NSE_INDEX|Nifty 50"
UPSTOX_BASE_URL = "https://api.upstox.com"


def load_env_file(path: str | Path | None) -> None:
    if not path:
        return
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


class UpstoxPaperClient:
    def __init__(
        self,
        token: str,
        base_url: str = UPSTOX_BASE_URL,
        timeout_seconds: float = 15.0,
        session: requests.Session | None = None,
    ) -> None:
        token = str(token).strip()
        if not token:
            raise ValueError("Upstox token is required")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            }
        )
        self._nfo_holiday_dates: set[pd.Timestamp] | None = None
        self._holiday_cache_year: int | None = None
        self._holiday_cache_loaded_at: datetime | None = None

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = self.session.get(
            f"{self.base_url}{path}",
            params=params or {},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") not in {None, "success"}:
            raise RuntimeError(f"Upstox API returned status={payload.get('status')}")
        return payload

    def option_contracts(self, expiry_ref: str) -> pd.DataFrame:
        payload = self._get(
            "/v2/option/contract",
            {"instrument_key": NIFTY_INDEX_KEY, "expiry_date": expiry_ref},
        )
        rows = payload.get("data") or []
        return pd.DataFrame(rows)

    def market_holidays(self) -> pd.DataFrame:
        payload = self._get("/v2/market/holidays")
        rows = payload.get("data") or []
        return pd.DataFrame(rows)

    def nfo_trading_holiday_dates(self, now: datetime) -> set[pd.Timestamp]:
        year = int(now.year)
        stale = self._nfo_holiday_dates is not None and self._holiday_cache_year == year
        refresh = (
            self._nfo_holiday_dates is None
            or self._holiday_cache_year != year
            or self._holiday_cache_loaded_at is None
            or (now - self._holiday_cache_loaded_at).total_seconds() >= 21600
        )
        if refresh:
            try:
                holidays = self.market_holidays()
                dates: set[pd.Timestamp] = set()
                if not holidays.empty and "date" in holidays.columns:
                    for _, row in holidays.iterrows():
                        holiday_type = str(row.get("holiday_type", "")).upper()
                        closed = row.get("closed_exchanges")
                        closed_exchanges = (
                            {str(x).upper() for x in closed}
                            if isinstance(closed, (list, tuple, set))
                            else set()
                        )
                        if (
                            holiday_type == "TRADING_HOLIDAY"
                            and "NFO" in closed_exchanges
                        ):
                            parsed = pd.to_datetime(row["date"], errors="coerce")
                            if pd.notna(parsed):
                                dates.add(pd.Timestamp(parsed).normalize())
                self._nfo_holiday_dates = dates
                self._holiday_cache_year = year
                self._holiday_cache_loaded_at = now
            except Exception:
                # A previously fetched calendar is preferable to falling back to
                # weekday-only logic. If there is no known calendar, fail closed
                # rather than risking an entry on an exchange holiday.
                if not stale:
                    raise
        return self._nfo_holiday_dates or set()

    def is_nfo_trading_day(self, now: datetime) -> bool:
        if now.weekday() >= 5:
            return False
        return pd.Timestamp(now.date()) not in self.nfo_trading_holiday_dates(now)

    def find_dte6_expiry(self, now: datetime) -> pd.Timestamp | None:
        # The frozen research rule is an exact six-calendar-day expiry.
        # Query the exact target date instead of relying on relative-week labels.
        target = pd.Timestamp(now.date() + timedelta(days=6))
        contracts = self.option_contracts(target.strftime("%Y-%m-%d"))
        if contracts.empty or "expiry" not in contracts.columns:
            return None

        weekly = contracts
        if "weekly" in contracts.columns:
            weekly = contracts[contracts["weekly"].fillna(False).astype(bool)]
        if weekly.empty:
            return None

        expiries = pd.to_datetime(weekly["expiry"], errors="coerce").dropna().dt.normalize()
        matches = sorted(set(expiries[expiries.eq(target.normalize())]))
        return matches[0] if matches else None

    def option_chain(self, expiry: pd.Timestamp) -> list[dict[str, Any]]:
        payload = self._get(
            "/v2/option/chain",
            {
                "instrument_key": NIFTY_INDEX_KEY,
                "expiry_date": pd.Timestamp(expiry).strftime("%Y-%m-%d"),
            },
        )
        return list(payload.get("data") or [])

    @staticmethod
    def chain_to_dataframe(rows: Iterable[dict[str, Any]], as_of: datetime) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        for row in rows:
            expiry = pd.Timestamp(row["expiry"]).normalize()
            strike = float(row["strike_price"])
            spot = float(row["underlying_spot_price"])
            for option_type, key in (("CE", "call_options"), ("PE", "put_options")):
                side = row.get(key) or {}
                md = side.get("market_data") or {}
                greeks = side.get("option_greeks") or {}
                records.append(
                    {
                        "timestamp": pd.Timestamp(as_of),
                        "expiry": expiry,
                        "underlying": "NIFTY",
                        "spot": spot,
                        "future": float("nan"),
                        "strike": strike,
                        "option_type": option_type,
                        "open": pd.to_numeric(md.get("ltp"), errors="coerce"),
                        "ltp": pd.to_numeric(md.get("ltp"), errors="coerce"),
                        "bid": pd.to_numeric(md.get("bid_price"), errors="coerce"),
                        "ask": pd.to_numeric(md.get("ask_price"), errors="coerce"),
                        "volume": pd.to_numeric(md.get("volume"), errors="coerce"),
                        "oi": pd.to_numeric(md.get("oi"), errors="coerce"),
                        # Upstox option-chain IV is expressed as percent; the strategy
                        # engine expects decimal volatility.
                        "iv": pd.to_numeric(greeks.get("iv"), errors="coerce") / 100.0,
                        "delta": pd.to_numeric(greeks.get("delta"), errors="coerce"),
                    }
                )
        return pd.DataFrame(records)

    def spot_daily_history(self, end_date: pd.Timestamp | None = None, lookback_days: int = 60) -> pd.DataFrame:
        end = pd.Timestamp(end_date or datetime.now(IST).date()).normalize()
        if end > pd.Timestamp(datetime.now(IST).date()):
            end = pd.Timestamp(datetime.now(IST).date())
        to_date = end - pd.Timedelta(days=1)
        from_date = to_date - pd.Timedelta(days=int(lookback_days))
        path = (
            f"/v3/historical-candle/{requests.utils.quote(NIFTY_INDEX_KEY, safe='')}"
            f"/days/1/{to_date:%Y-%m-%d}/{from_date:%Y-%m-%d}"
        )
        payload = self._get(path)
        candles = payload.get("data", {}).get("candles") or []
        rows = []
        for candle in candles:
            if len(candle) < 5:
                continue
            rows.append(
                {
                    "timestamp": pd.Timestamp(candle[0]),
                    "spot": float(candle[4]),
                }
            )
        return pd.DataFrame(rows)


def _load_paper_state(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _save_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _append_row(df: pd.DataFrame, row: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return row.copy()
    return pd.concat([df, row], ignore_index=True)


def _send_alert(message: str) -> None:
    webhook = os.getenv("PAPER_ALERT_WEBHOOK_URL")
    if not webhook:
        return
    try:
        requests.post(
            webhook,
            json={"text": message},
            timeout=10,
        ).raise_for_status()
    except Exception:
        # Alerts must never terminate the paper engine.
        pass


def _market_open_window(now: datetime) -> bool:
    if now.weekday() >= 5:
        return False
    return dtime(9, 15) <= now.time() <= dtime(15, 35)


def _round_to_minute(now: datetime) -> datetime:
    return now.replace(second=0, microsecond=0)


def _find_open_trade(ledger: pd.DataFrame) -> pd.Series | None:
    if ledger.empty or "ledger_status" not in ledger:
        return None
    x = ledger[ledger["ledger_status"].eq("OPEN")]
    if x.empty:
        return None
    return x.iloc[-1]


def _has_signal_for_date(ledger: pd.DataFrame, entry_date: pd.Timestamp) -> bool:
    if ledger.empty or "entry_date" not in ledger:
        return False
    dates = pd.to_datetime(ledger["entry_date"], errors="coerce").dt.normalize()
    return bool((dates == entry_date.normalize()).any())


def _quote_for_strike(rows: list[dict[str, Any]], strike: float, option_type: str) -> dict[str, float] | None:
    for row in rows:
        if float(row.get("strike_price", float("nan"))) != float(strike):
            continue
        side = row.get("call_options" if option_type == "CE" else "put_options") or {}
        md = side.get("market_data") or {}
        return {
            "bid": float(md.get("bid_price")) if md.get("bid_price") is not None else float("nan"),
            "ask": float(md.get("ask_price")) if md.get("ask_price") is not None else float("nan"),
            "ltp": float(md.get("ltp")) if md.get("ltp") is not None else float("nan"),
        }
    return None


def _executable_sell(quote: dict[str, float], slippage: float) -> float:
    base = quote["bid"] if pd.notna(quote["bid"]) else quote["ltp"]
    return max(0.0, float(base) - slippage)


def _executable_buy(quote: dict[str, float], slippage: float) -> float:
    base = quote["ask"] if pd.notna(quote["ask"]) else quote["ltp"]
    return max(0.0, float(base) + slippage)


def run_paper_daemon(
    *,
    token: str,
    config_path: str | Path,
    ledger_path: str | Path = "results/paper/upstox_paper_ledger.csv",
    quote_log_path: str | Path = "results/paper/upstox_quote_log.csv",
    poll_seconds: int = 30,
    once: bool = False,
) -> None:
    cfg, costs, _ = load_config(config_path)
    client = UpstoxPaperClient(token)
    ledger_path = Path(ledger_path)
    quote_log_path = Path(quote_log_path)
    slippage = float(costs.slippage_points_per_leg)

    ledger = _load_paper_state(ledger_path)
    last_entry_attempt_date: pd.Timestamp | None = None

    print(
        f"{datetime.now(IST).isoformat()} "
        f"{FROZEN_STRATEGY_ID} PAPER_DAEMON_STARTED "
        f"config={config_path} poll_seconds={poll_seconds}",
        flush=True,
    )

    while True:
        now = datetime.now(IST)
        try:
            is_trading_day = client.is_nfo_trading_day(now)
            open_trade = _find_open_trade(ledger)

            if open_trade is None:
                if (
                    is_trading_day
                    and dtime(9, 59) <= now.time() <= dtime(10, 5)
                    and last_entry_attempt_date != now.date()
                    and not _has_signal_for_date(ledger, pd.Timestamp(now.date()))
                ):
                    expiry = client.find_dte6_expiry(now)
                    last_entry_attempt_date = now.date()
                    if expiry is None:
                        print(f"{now.isoformat()} NO_TRADE: no weekly expiry exactly 6 calendar days ahead")
                    else:
                        as_of = _round_to_minute(now)
                        chain_rows = client.option_chain(expiry)
                        options = client.chain_to_dataframe(chain_rows, as_of)
                        spot = client.spot_daily_history(end_date=pd.Timestamp(now.date()))
                        signal = build_paper_signal(
                            options=options,
                            spot_history=spot,
                            expiry=expiry,
                            as_of=pd.Timestamp(as_of),
                            cfg=cfg,
                            costs=costs,
                            slippage_points=slippage,
                        )
                        signal = add_signal_ids(signal)
                        ledger = initialize_ledger(signal)
                        _save_atomic(ledger, ledger_path)
                        status = str(signal.iloc[0]["signal_status"])
                        reason = str(signal.iloc[0]["signal_reason"])
                        print(
                            f"{as_of.isoformat()} signal={status} reason={reason} "
                            f"expiry={expiry.date()}"
                        )
                        if status == "READY":
                            row = signal.iloc[0]
                            put_fill = _executable_sell(
                                {
                                    "bid": float(row["put_bid"]),
                                    "ltp": float(row["put_mid"]),
                                },
                                slippage,
                            )
                            call_fill = _executable_sell(
                                {
                                    "bid": float(row["call_bid"]),
                                    "ltp": float(row["call_mid"]),
                                },
                                slippage,
                            )
                            ledger = record_entry_fill(
                                ledger,
                                signal_id=str(row["signal_id"]),
                                fill_timestamp=pd.Timestamp(as_of),
                                put_fill=put_fill,
                                call_fill=call_fill,
                            )
                            _save_atomic(ledger, ledger_path)
                            msg = (
                                f"{FROZEN_STRATEGY_ID} PAPER ENTRY "
                                f"{expiry.date()} PE {row['put_strike']:.0f} / CE {row['call_strike']:.0f} "
                                f"credit={put_fill + call_fill:.2f}"
                            )
                            print(msg)
                            _send_alert(msg)

            open_trade = _find_open_trade(ledger)
            if open_trade is not None and is_trading_day and _market_open_window(now):
                expiry = pd.Timestamp(open_trade["expiry"]).normalize()
                put_k = float(open_trade["put_strike"])
                call_k = float(open_trade["call_strike"])
                chain_rows = client.option_chain(expiry)
                chain_df = client.chain_to_dataframe(chain_rows, _round_to_minute(now))
                if chain_df.empty:
                    raise RuntimeError("Empty Upstox option chain for open paper trade")

                spot = float(chain_df["spot"].dropna().iloc[0])
                put_q = _quote_for_strike(chain_rows, put_k, "PE")
                call_q = _quote_for_strike(chain_rows, call_k, "CE")
                if put_q is None or call_q is None:
                    raise RuntimeError("Missing held-leg quote")
                put_buy = _executable_buy(put_q, slippage)
                call_buy = _executable_buy(call_q, slippage)
                debit = put_buy + call_buy

                target = float(open_trade["profit_target_debit_points"])
                stop = float(open_trade["stop_debit_points"])
                scheduled = pd.Timestamp(open_trade["scheduled_time_exit_timestamp"])
                exit_reason = None
                fill_ts = _round_to_minute(now)
                put_exit = put_buy
                call_exit = call_buy

                if debit <= target:
                    exit_reason = "profit_target"
                elif debit >= stop:
                    exit_reason = "stop"
                elif fill_ts >= scheduled:
                    exit_reason = "time_exit"
                elif pd.Timestamp(now.date()) >= expiry and now.time() >= dtime(15, 25):
                    put_exit = option_value_at_expiry(put_k, spot, "PE")
                    call_exit = option_value_at_expiry(call_k, spot, "CE")
                    exit_reason = "expiry"

                quote_row = {
                    "timestamp": fill_ts,
                    "signal_id": open_trade["signal_id"],
                    "expiry": expiry,
                    "spot": spot,
                    "put_strike": put_k,
                    "call_strike": call_k,
                    "put_bid": put_q["bid"],
                    "put_ask": put_q["ask"],
                    "call_bid": call_q["bid"],
                    "call_ask": call_q["ask"],
                    "buy_debit_points": debit,
                    "profit_target_debit_points": target,
                    "stop_debit_points": stop,
                    "exit_reason_candidate": exit_reason,
                }
                qdf = pd.DataFrame([quote_row])
                existing_q = pd.read_csv(quote_log_path) if quote_log_path.exists() else pd.DataFrame()
                _save_atomic(_append_row(existing_q, qdf), quote_log_path)

                if exit_reason is not None:
                    ledger = record_exit_fill(
                        ledger,
                        signal_id=str(open_trade["signal_id"]),
                        fill_timestamp=fill_ts,
                        put_fill=put_exit,
                        call_fill=call_exit,
                        exit_reason=exit_reason,
                        costs=costs,
                    )
                    _save_atomic(ledger, ledger_path)
                    closed = ledger[ledger["signal_id"].eq(str(open_trade["signal_id"]))]
                    net_pnl = float(closed.iloc[-1]["net_pnl"])
                    msg = (
                        f"{FROZEN_STRATEGY_ID} PAPER EXIT "
                        f"reason={exit_reason} expiry={expiry.date()} "
                        f"debit={put_exit + call_exit:.2f} "
                        f"net_pnl={net_pnl:.2f}"
                    )
                    print(msg)
                    _send_alert(msg)

        except Exception as exc:
            print(f"{now.isoformat()} PAPER_ENGINE_ERROR {type(exc).__name__}: {exc}", flush=True)

        if once:
            break
        time.sleep(max(5, int(poll_seconds)))
