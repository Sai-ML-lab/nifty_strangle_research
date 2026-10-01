# NIFTY Weekly 2-SD Short-Strangle Research

A reproducible research framework for testing a systematic weekly NIFTY short-strangle idea and comparing it with a defined-risk iron condor.

## Research question

> Does selling OTM NIFTY weekly options around an implied-volatility-derived 2-standard-deviation boundary have positive expectancy after realistic execution costs and tail losses?

This repository is intentionally a **research/backtesting framework**, not a trading recommendation.

## Current / historical NSE assumptions encoded in the project

- NIFTY weekly options moved from Thursday to Tuesday for contracts with revised expiries from September 2025; the historical engine uses the expiry date present in the data rather than hard-coding a weekday.
- Historical NIFTY lot size is date-aware for the 2023+ archive: 50 before the first revised weekly expiry on 02-May-2024, 25 through 19-Nov-2024, 75 through 30-Dec-2025 expiry, and 65 from the first post-revision weekly expiry onward.
- Current NIFTY index options have 4 weekly expiry contracts plus monthly/longer-dated contracts.

Verify exchange specifications before live use because contract rules and charges can change.

## Data schema

Long-form option-chain snapshots:

```text
timestamp, expiry, underlying, spot, future, strike, option_type,
bid, ask, ltp, volume, oi, iv, delta
```

`bid`, `ask`, `iv`, and `delta` may be omitted in some historical datasets. The engine can fall back to `ltp`, and it can calculate IV/delta with Black-76 when enough inputs are available.

## Baseline strategy

- Underlying: NIFTY
- Entry: Wednesday 10:00 IST (configurable)
- Expiry: nearest future NIFTY Tuesday expiry present in the dataset
- Volatility: ATM implied volatility from the chain
- Strike placement: approximately 2.0 SD from the forward/spot proxy
- Position: short PE + short CE
- Default exit: 50% of premium captured, or 2x premium stop, or configured time exit, otherwise expiry settlement
- Quantity: 1 lot by default for the naked-strangle baseline

## Defined-risk comparison

An iron-condor module is planned as the next comparison. The current implementation focuses on the clean naked-strangle baseline so we can validate the original hypothesis before adding structural changes.

## Important limitations

1. A 2-SD boundary is **not** a guaranteed 95% probability boundary in real markets.
2. Historical option data without bid/ask is subject to execution-model uncertainty.
3. For datasets without intraday quotes, exits are necessarily approximations.
4. Naked short-strangle loss is theoretically unbounded; fixed-risk sizing is therefore intentionally not inferred from a broker margin number.
5. Backtest results are not evidence of future profitability.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.strangle_backtest --help
python -m src.smoke_test
```

Open the notebook in `notebooks/01_nifty_2sd_research.ipynb` after installing dependencies.

## Data acquisition

For the primary free research path, use the TradeMarkk Hugging Face dataset documented in `SOURCES.md`. It publishes NIFTY 1-minute spot in `index/NIFTY.parquet` and NIFTY option chains in `options/NIFTY/{EXPIRY}.parquet`. The downloader below pulls only the expiry files you request, rather than cloning the full 4 GB repository.

For 1-minute OHLCV files, the pipeline uses the bar **open** for entry calculations to avoid look-ahead and the bar close for later marks/exits. If bid/ask is available, the execution model uses those quotes instead. When futures are not supplied, the backtester estimates a short-dated forward from put-call parity and falls back to spot.

## Free-data download

Create/activate an environment and install dependencies first:

```bash
pip install -r requirements.txt
```

Then download a multi-year NIFTY history. A sensible first pass is 2023 through the latest available 2026 expiry in the dataset:

```bash
python tools/download_trademarkk.py \
  --start 2023-01-01 \
  --end 2026-08-04 \
  --out data/trademarkk
```

The script uses the Hugging Face API to list the repository and downloads only matching `options/NIFTY/*.parquet` files plus `index/NIFTY.parquet`. The source currently publishes the dataset under CC-BY-NC-4.0 and notes that far/illiquid strikes can be sparse or absent. See `SOURCES.md`.

Normalize each expiry into the project's compact per-expiry research format:

```bash
python tools/build_trademarkk_long.py \
  --options-dir data/trademarkk/options/NIFTY \
  --spot data/trademarkk/index/NIFTY.parquet \
  --out data/trademarkk/processed
```

Then run the memory-bounded expiry-by-expiry baseline:

```bash
python run_batch_research.py \
  --options-dir data/trademarkk/processed \
  --spot data/trademarkk/index/NIFTY.parquet \
  --config config.yaml \
  --out-dir results/trademarkk_baseline
```

This path does not require loading the entire multi-year option chain into RAM.

## Research command

```bash
python run_research.py --data data/sample_chain.csv
```

The sample output is only a software-validation fixture. **Do not interpret its P&L as a market result.**

## Real-data workflow

1. Download the free multi-year TradeMarkk history with `tools/download_trademarkk.py`.
2. Normalize it with `tools/build_trademarkk_long.py`.
3. Run the baseline with `run_batch_research.py`.
4. Inspect skipped/low-coverage expiries and verify a sample of individual trades against an independent source.
5. Only then run the parameter sweep and walk-forward selection.

The first serious research target is **2023-2026 walk-forward validation after realistic transaction costs**, with the 2-SD strategy kept as the untouched control.

## Current cost assumptions

The default config uses a brokerage placeholder of Rs 20/order plus current NSE option transaction outflow of Rs 3,553 per crore of premium (0.03553%) and current stamp/SEBI/GST settings. STT is date-aware: 0.10% on option-sale premium before 1-Apr-2026 and 0.15% from 1-Apr-2026. Broker-specific charges and slippage remain configurable.

For historical research, the lot-size engine is also date-aware; do not replace it with today's lot size when evaluating older trades.
