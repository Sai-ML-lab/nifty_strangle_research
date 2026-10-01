# NIFTY Weekly 2-SD Short-Strangle Research

A reproducible research framework for testing a systematic weekly NIFTY short-strangle idea and comparing defined-risk alternatives.

## Research question

> Does selling OTM NIFTY weekly options around an implied-volatility-derived 2-standard-deviation boundary have positive expectancy after realistic execution costs and tail losses?

This is a research/backtesting project, not a trading recommendation.

## Quick start

```bash
git pull
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.smoke_test
python tools/download_trademarkk.py --dry-run --start 2023-01-01 --end 2026-08-04 --out data/trademarkk
python tools/download_trademarkk.py --start 2023-01-01 --end 2026-08-04 --out data/trademarkk
python tools/build_trademarkk_long.py --options-dir data/trademarkk/options/NIFTY --spot data/trademarkk/index/NIFTY.parquet --out data/trademarkk/processed
python run_batch_research.py --options-dir data/trademarkk/processed --spot data/trademarkk/index/NIFTY.parquet --config config.yaml --out-dir results/trademarkk_baseline
```

The downloader has `--dry-run` so you can inspect the expiry list before downloading.

## Frozen baseline

- Underlying: NIFTY
- Entry: Wednesday at 10:00 IST
- Expiry: next expiry represented by the historical contract data
- Strike placement: approximately +/- 2.0 implied-volatility standard deviations from the entry forward
- Position: short OTM PE + short OTM CE
- Exit: 50% of initial credit captured, or 2x credit stop, or Monday 15:00 time exit, otherwise expiry settlement
- Quantity: 1 historical lot
- Historical lot size: date-aware
- Costs: configurable, with date-aware STT

Do not optimize these parameters before running the baseline. The baseline is the control for all later experiments.

## Research sequence

1. Validate the free historical data.
2. Run the frozen 2-SD control over the full archive.
3. Inspect skipped expiries and strike/quote coverage.
4. Cross-check a sample of trades against an independent source.
5. Only then run the exploratory sweep: SD distance, delta selection, exits, entry day and IV-vs-RV filters.
6. Run walk-forward selection using only historical training windows.
7. Add and compare a defined-risk iron condor.
8. Paper trade before considering live deployment.

## Data

Primary free source: TradeMarkk / Hugging Face
https://huggingface.co/datasets/thetrademarkk/india-index-options-1m

The published dataset contains NIFTY 1-minute spot and option-chain history. It is licensed CC-BY-NC-4.0; use it for research/education and do not redistribute it commercially without permission. Far/illiquid strikes may be sparse.

Secondary validation sources are documented in `SOURCES.md`.

## Execution model

The pipeline prefers bid/ask if available. For OHLC-only historical data:

- entry uses the option bar OPEN
- exits use the option bar CLOSE
- configurable per-leg slippage is applied

This is an execution model, not an order-book reconstruction.

## Volatility model

At entry, ATM IV uses supplied IV where available; otherwise it is inferred with Black-76 from the observed option price. The short-strike boundary is:

`forward +/- SD * forward * IV * sqrt(T)`

The frozen control uses SD = 2.0.

Realized volatility uses completed daily NIFTY closes and is shifted so entry-day prices cannot leak into the RV20 filter.

## Important limitations

A 2-SD boundary is not a guaranteed 95% probability boundary in real markets. Short-volatility losses can be highly asymmetric and tail events can dominate long-run results. Backtests are sensitive to historical data quality, execution assumptions, costs, contract changes and model choices.

Do not interpret synthetic smoke-test P&L as a market result.

## Files

- `config.yaml` — frozen baseline parameters
- `src/core.py` — volatility, Black-76, strike selection, costs, historical lot sizes
- `src/data_ingest.py` — normalization and validation
- `src/strangle_backtest.py` — single-dataset backtester
- `run_batch_research.py` — memory-bounded expiry-by-expiry runner
- `src/research.py` — parameter sweeps and regime analysis
- `src/walk_forward.py` — walk-forward framework
- `tools/download_trademarkk.py` — free data downloader
- `tools/build_trademarkk_long.py` — per-expiry normalizer
- `tests/` — automated tests
## Baseline audit note

An October 1, 2026 full-data run produced 167 rows, but audit showed 7 Wednesdays contained two simultaneous expiries. The old batch runner could select both the current and following expiry for the same entry date. Those overlapping rows must not be used as the baseline result.

The runner now keeps only the earliest eligible expiry per entry date. A ledger audit command is also included:

python tools/audit_trades.py --trades results/trademarkk_baseline/baseline_trades.csv

## DTE-normalized experiment

The original control is a fixed Wednesday 10:00 IST entry. It is not a like-for-like horizon across the historical expiry-day regime change: NSE moved NIFTY weekly expiry from Thursday to Tuesday for revised contracts introduced in September 2025.

Use config_dte6.yaml to run a separate 6-calendar-day-before-expiry experiment. This is intended to compare similar DTE exposure across the historical expiry-day regimes; it does not replace the original control.

NSE references: https://nsearchives.nseindia.com/content/circulars/FAOP68685.pdf and https://nsearchives.nseindia.com/content/circulars/FAOP68747.pdf

## Experiment 3: DTE6 + IV-RV filter + execution sensitivity

Use the clean DTE6 trade ledger as the input to the IV-RV research. The threshold sweep is cheap because the filter is entry-only and does not alter the selected strikes for an already-generated trade:

```bash
python run_ivrv_research.py \
  --trades results/trademarkk_dte6/baseline_trades.csv \
  --out-dir results/ivrv
```

This produces:

- `ivrv_threshold_sweep.csv` with a no-filter baseline plus candidate IV-RV thresholds.
- `ivrv_walk_forward.csv` where the threshold is selected only from the preceding training window and then evaluated on the following test window.

The default walk-forward setup is 12 months train / 3 months test / 3 months rebalance, with a minimum of 20 training trades for a threshold candidate.

For the fixed 2-vol-point exploratory filter, use:

```bash
python run_batch_research.py \
  --options-dir data/trademarkk/processed \
  --spot data/trademarkk/index/NIFTY.parquet \
  --config config_dte6_ivrv2.yaml \
  --out-dir results/trademarkk_dte6_ivrv2
```

Run exact slippage sensitivity on the same fixed rule:

```bash
python run_slippage_sensitivity.py \
  --options-dir data/trademarkk/processed \
  --spot data/trademarkk/index/NIFTY.parquet \
  --config config_dte6_ivrv2.yaml \
  --out-dir results/slippage_sensitivity
```

The sensitivity runner defaults to 0, 0.10, 0.25, 0.50, 0.75 and 1.00 option points per leg. It reruns the actual backtest rather than simply subtracting a linear slippage estimate.

## Data-quality handling

The batch runner records `data_quality_flag` and `exit_intrinsic_gap_points`. An early-exit quote is flagged when its combined buy debit is more than 1 point below the intrinsic value implied by the recorded NIFTY spot.

Set:

```yaml
data_quality:
  exclude_warnings: true
```

to exclude those trades from the result set. Always inspect the warnings first; do not assume every anomaly is a bad trade.

## Interpreting Experiment 3

Do not select the threshold with the largest full-sample P&L. The relevant evidence is the out-of-sample walk-forward result and its sensitivity to execution assumptions.

The key outputs are:

- out-of-sample net P&L
- out-of-sample expectancy
- positive test-fold count
- selected threshold by fold
- baseline test P&L versus filtered test P&L
- worst test trade
- slippage level at which the edge disappears

Only after those checks should we proceed to SD/delta/exit optimization and the iron-condor comparison.
