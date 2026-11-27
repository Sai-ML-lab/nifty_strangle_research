from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from src.level_trading.levels import build_level_frame
from src.level_trading.storage import LevelStore
from src.level_trading.upstox_market_data import UpstoxMarketDataClient, load_candles, save_candles

IST = ZoneInfo("Asia/Kolkata")
DATA_DIR = Path("data/level_trading")
st.set_page_config(page_title="NIFTY Level Trading", layout="wide")
st.title("NIFTY Level Trading — Phase 1")
st.caption("Read-only market data, automatic levels and manual level management. No order placement.")


def refresh_market_data():
    token = os.getenv("UPSTOX_ANALYTICS_TOKEN", "").strip()
    if not token:
        raise RuntimeError("UPSTOX_ANALYTICS_TOKEN is not set")
    client = UpstoxMarketDataClient(token)
    now = datetime.now(IST)
    intraday = client.fetch_intraday_candles(interval_minutes=5, as_of=now)
    daily = client.fetch_daily_candles(to_date=now.date(), lookback_days=45)
    save_candles(intraday, str(DATA_DIR / "nifty_5m.parquet"))
    save_candles(daily, str(DATA_DIR / "nifty_daily.parquet"))
    return intraday, daily


store = LevelStore(DATA_DIR / "levels.sqlite3")
today = datetime.now(IST).date()

with st.sidebar:
    st.subheader("Market data")
    if st.button("Refresh from Upstox", type="primary", use_container_width=True):
        try:
            refresh_market_data()
            st.success("Market data refreshed.")
        except Exception as exc:
            st.error(f"Refresh failed: {type(exc).__name__}: {exc}")
    st.caption("Token: UPSTOX_ANALYTICS_TOKEN (local environment)")
    st.caption(f"Session: {today.isoformat()}")

intraday = load_candles(str(DATA_DIR / "nifty_5m.parquet"))
daily = load_candles(str(DATA_DIR / "nifty_daily.parquet"))
manual = store.list_manual_levels(today)
manual_rows = [{"name": r["name"], "price": r["price"], "side": r["side"],
                "source": "MANUAL", "notes": r["notes"], "id": r["id"]} for r in manual]
levels = build_level_frame(daily_candles=daily, intraday_candles=intraday,
                           session_date=today, manual_levels=manual_rows)

left, right = st.columns([2, 1])
with left:
    st.subheader("NIFTY 5-minute candles")
    if intraday.empty:
        st.info("No candles saved. Set your token and click Refresh from Upstox.")
    else:
        latest = intraday.iloc[-1]
        a, b, c = st.columns(3)
        a.metric("Latest completed close", f"{latest['close']:,.2f}")
        b.metric("Latest completed candle", str(latest["timestamp"]))
        c.metric("Completed bars", len(intraday))
        st.line_chart(intraday.set_index("timestamp")[["close"]], height=280)
        st.dataframe(intraday.tail(12), use_container_width=True, hide_index=True)

with right:
    st.subheader("Add manual level")
    with st.form("manual_level_form", clear_on_submit=True):
        name = st.text_input("Label", value="Manual S/R")
        price = st.number_input("Price", min_value=1.0, value=26000.0, step=10.0)
        side = st.selectbox("Type", ["SUPPORT", "RESISTANCE", "REFERENCE"])
        notes = st.text_input("Reason / notes")
        if st.form_submit_button("Save level", use_container_width=True):
            store.add_manual_level(session_date=today, name=name, price=price, side=side, notes=notes)
            st.success("Level saved. Refresh the page to display it.")

st.subheader("Detected levels")
if levels.empty:
    st.info("Levels appear after daily and intraday candles are available.")
else:
    cols = [c for c in ["name", "price", "side", "source", "timestamp", "confirmed_at", "notes"] if c in levels]
    st.dataframe(levels[cols].sort_values("price", ascending=False), use_container_width=True, hide_index=True)

st.subheader("Manage today's manual levels")
if manual:
    choices = {f"{r['id']} · {r['name']} · {r['price']:,.2f} · {r['side']}": r["id"] for r in manual}
    selected = st.selectbox("Select level to deactivate", list(choices))
    if st.button("Deactivate selected level"):
        store.deactivate(choices[selected])
        st.success("Level deactivated. Refresh the page.")
else:
    st.caption("No manual levels added for today.")
