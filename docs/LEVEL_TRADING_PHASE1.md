# NIFTY Level Trading — Phase 1 Engineering Specification

## Frozen decisions

- Underlying: NIFTY 50, Upstox instrument key `NSE_INDEX|Nifty 50`.
- Primary timeframe: completed 5-minute candles.
- Levels: previous-day high/low/close, first-15-minute opening range, confirmed swing highs/lows and manual support/resistance.
- Moving averages: SMA(5) and SMA(10).
- FVG: newly formed, directionally aligned 5-minute three-candle FVG.
- RSI: RSI(14) and regular divergence on NIFTY spot, 5-minute timeframe.
- Option chain: OI, change in OI and PCR; expiry selected by liquidity, spread and premium.
- Entry modes: immediate after confirmation close and retest, independently configurable.
- Execution: paper trading first, then approval-based live execution after validation.
- Dashboard: Streamlit.
- Existing short-strangle research and execution remain unchanged and separate.

## Architecture boundary

The new code lives in `src/level_trading/`. It reuses the existing Upstox read-only credential convention (`UPSTOX_ANALYTICS_TOKEN`) but does not import or modify the short-strangle signal, position-sizing, ledger or execution logic.

## Phase 1 scope

1. Read-only Upstox History V3 market-data client.
2. Fetch current-session NIFTY 5-minute candles.
3. Exclude the currently forming candle. Upstox candle timestamps represent candle start; a candle is complete only when timestamp + interval is less than or equal to the local as-of time.
4. Fetch daily candles and derive previous-session PDH, PDL and PDC.
5. Derive ORH/ORL after all three 5-minute bars from 09:15 through 09:30 have completed.
6. Derive swing highs/lows only after right-side confirmation bars exist.
7. Persist manually entered levels in local SQLite, scoped by session date.
8. Persist normalized candles in Parquet under `data/level_trading/`.
9. Provide a Streamlit dashboard for refreshing data, viewing candles/levels, adding levels and deactivating levels.

## Phase 1 non-goals

No option-chain polling, SMA/RSI/FVG calculation, breakout signal generation, options contract selection, paper order placement or live order placement. Phase 1 is read-only.

## Data contracts

Normalized candle fields: `timestamp` (Asia/Kolkata-aware), `open`, `high`, `low`, `close`, `volume`, `oi`.

Level fields: `name`, `price`, `side` (SUPPORT/RESISTANCE/REFERENCE), `source`, optional `timestamp`, optional `confirmed_at`, optional `notes`.

Manual-level table: id, session_date, name, price, side, notes, active, created_at.

## Local setup

Install requirements:

```bash
pip install -r requirements.txt
```

Set the read-only Upstox Analytics Token locally; never commit it:

```bash
export UPSTOX_ANALYTICS_TOKEN="YOUR_TOKEN"
```

Collect data once:

```bash
python run_level_market_data.py --once
```

Poll data every 60 seconds:

```bash
python run_level_market_data.py --poll-seconds 60
```

Launch dashboard:

```bash
streamlit run streamlit_level_dashboard.py
```

Run tests:

```bash
pytest -q tests/test_level_trading_phase1.py
```

## Upstox endpoints

- Intraday: `GET /v3/historical-candle/intraday/{instrument_key}/minutes/5`
- Daily: `GET /v3/historical-candle/{instrument_key}/days/1/{to_date}/{from_date}`

## Acceptance criteria

- Candle timestamps are timezone-aware and normalized to Asia/Kolkata.
- Duplicate timestamps are removed deterministically.
- Incomplete 5-minute candles are excluded from calculations.
- Previous-day levels use the last daily candle strictly before the current session.
- Opening range is unavailable until all three opening 5-minute candles have completed.
- Swing pivots are not exposed until right-side confirmation bars have completed.
- Manual levels persist across restarts and can be deactivated without deleting history.
- Missing credentials, API errors and empty data are surfaced; no synthetic data is generated.
- No order-placement code exists in Phase 1.

## Follow-on phases

- Phase 2: SMA(5/10), RSI(14) divergence, 5-minute FVG and breakout/breakdown state machine.
- Phase 3: timestamped option-chain snapshots, OI/change-in-OI/PCR and expiry/liquidity selector.
- Phase 4: immediate/retest entry simulations, risk engine and realistic options-premium backtester.
- Phase 5: prospective paper trading and dashboard trade journal.
- Phase 6: approval-based live execution with order reconciliation, idempotency, kill switch and audit trail.
