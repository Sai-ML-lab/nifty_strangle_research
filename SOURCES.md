# Free historical-data sources evaluated for this project

## Primary: TradeMarkk / Hugging Face

Dataset: https://huggingface.co/datasets/thetrademarkk/india-index-options-1m

As of the 2026-10-01 research check, the dataset is a 377M-row / ~4 GB collection of 1-minute OHLCV(+OI) bars for NIFTY, BANKNIFTY and SENSEX. It has `index/NIFTY.parquet` for NIFTY 1-minute spot and `options/NIFTY/{EXPIRY}.parquet` for option-chain history. The NIFTY directory is approximately 934 MB and begins in 2021, with files continuing into 2026. The published license is CC-BY-NC-4.0 and the author warns that far/illiquid option strikes can be sparse or absent.

## Cross-check: rissin / Hugging Face

Dataset: https://huggingface.co/datasets/rissin/nse-options-intraday

The dataset combines NSE F&O bhavcopy daily data with Upstox 1-minute intraday option candles. Its published coverage says NIFTY 1-minute intraday data runs from October 2024 into 2026; the daily history extends much further back. Intraday OI is NaN because the Upstox endpoint did not return it for those rows.

## Cross-check: artist-23 / Hugging Face

Dataset: https://huggingface.co/datasets/artist-23/nifty-options-data

Published coverage is December 2020 through December 2025, with timestamped option OHLC, IV, OI, strike and spot. The dataset is organized around ATM-relative strike buckets, so it is better suited to validation than to the primary full-chain backtest.

## Official validation source: NSE

NSE publishes F&O reports/bhavcopy and the exchange contract specifications used to verify expiry and tax/fee assumptions: https://www.nseindia.com/all-reports-derivatives

## Research policy

Do not mix sources into one price series without validation. Use TradeMarkk as the primary research dataset, then use the secondary datasets/NSE outputs to spot-check individual dates, expiries, prices and settlement behavior.
