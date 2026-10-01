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
8. Freeze a candidate and validate it independently.
9. Diagnose regimes/tails without selecting a new rule.
10. Run a genuinely prospective paper-trading ledger before considering any live deployment.

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
- `config_dte6_frozen_75_25.yaml` — frozen 75% capture / 2.5x stop candidate
- `src/core.py` — volatility, Black-76, strike selection, costs, historical lot sizes
- `src/data_ingest.py` — normalization and validation
- `src/strangle_backtest.py` — single-dataset backtester
- `run_batch_research.py` — memory-bounded expiry-by-expiry runner
- `src/research.py` — parameter sweeps and regime analysis
- `src/walk_forward.py` — walk-forward framework
- `tools/download_trademarkk.py` — free data downloader
- `tools/build_trademarkk_long.py` — per-expiry normalizer
- `src/slippage_replay.py` — frozen-decision execution replay
- `src/regime_gate.py` — train-only IV-RV regime gating
- `run_slippage_sensitivity.py` — frozen slippage sensitivity runner
- `run_regime_gate.py` — IV-RV regime gate runner
- `src/regime_attribution.py` — fixed-bucket regime/event diagnostics
- `run_regime_attribution.py` — attribution runner
- `src/paper_trading.py` — frozen signal and paper-ledger engine
- `run_paper_signal.py` — one as-of paper signal generator
- `run_paper_ledger.py` — append actual paper entry/exit fills
- `tests/` — automated tests

## Baseline audit note

An October 1, 2026 full-data run produced 167 rows, but audit showed 7 Wednesdays contained two simultaneous expiries. The old batch runner could select both the current and following expiry for the same entry date. Those overlapping rows must not be used as the baseline result.

The runner now keeps only the earliest eligible expiry per entry date. A ledger audit command is also included:

```bash
python tools/audit_trades.py --trades results/trademarkk_baseline/baseline_trades.csv
```

## DTE-normalized experiment

The original control is a fixed Wednesday 10:00 IST entry. It is not a like-for-like horizon across the historical expiry-day regime change: NSE moved NIFTY weekly expiry from Thursday to Tuesday for revised contracts introduced in September 2025.

Use `config_dte6.yaml` to run a separate 6-calendar-day-before-expiry experiment. This is intended to compare similar DTE exposure across the historical expiry-day regimes; it does not replace the original control.

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

Replay slippage on a **frozen trade ledger**:

```bash
python run_slippage_sensitivity.py \
  --trades results/trademarkk_dte6_ivrv2/baseline_trades.csv \
  --options-dir data/trademarkk/processed \
  --config config_dte6_ivrv2.yaml \
  --out-dir results/slippage_replay
```

The replay keeps each trade's entry date, strikes, exit timestamp and exit reason fixed. Only execution prices and resulting transaction costs change across 0, 0.10, 0.25, 0.50, 0.75 and 1.00 option points per leg. This isolates execution sensitivity; it does not re-run target/stop decisions at each slippage level.

For the train-only IV-RV regime gate:

```bash
python run_regime_gate.py \
  --trades results/trademarkk_dte6_v2/baseline_trades.csv \
  --out-dir results/ivrv_regime_gate
```

The gate uses the same 12-month train / 3-month test / 3-month rebalance structure as the IV-RV walk-forward. A fold is enabled only when the training-selected threshold has at least the minimum number of trades, training expectancy at or above 0, and training profit factor at or above 1.0. No test-period information is used to activate the regime.

## Experiment 4: frozen execution sensitivity and regime gating

Experiment 4 separates two questions. Slippage sensitivity holds the realized trade decisions fixed and replays execution under different assumptions. Regime gating selects the IV-RV threshold in the training window and activates trading only when that training-selected rule clears a pre-declared profitability gate. These diagnostics are not full-sample parameter optimizers.

## Experiment 5: SD × IV-RV walk-forward + portfolio risk

The next research stage jointly selects the implied-volatility SD distance and IV-RV entry threshold using only the preceding training window. The default grid is SD = 1.50, 1.75, 2.00, 2.50, 3.00 and IV-RV threshold = no filter, 0, 1, 2 vol points. The test window is untouched until selection is complete.

Run:

```bash
python run_sd_ivrv_research.py \
  --options-dir data/trademarkk/processed \
  --spot data/trademarkk/index/NIFTY.parquet \
  --config config_dte6.yaml \
  --out-dir results/sd_ivrv
```

The runner builds each SD ledger in one expiry-by-expiry pass, then performs the walk-forward selection. Outputs include per-SD trade ledgers, sd_ivrv_walk_forward.csv, selected_oos_trades.csv, baseline_oos_trades.csv, and oos_risk_report.csv.

The portfolio report includes total return, CAGR, max drawdown, weekly Sharpe/Sortino, worst week, weekly 5% CVaR, profit factor and losing streak. Weekly risk statistics include zero-trade weeks so a regime filter cannot look better merely by omitting inactive weeks from volatility calculations.

Do not choose the final SD/IV-RV pair from full-sample results. Use the walk-forward-selected sequence and then validate the selected OOS ledger under frozen-decision slippage replay.

## Experiment 6: stability-constrained entry selection

Experiment 6 treats parameter instability as a first-class risk. A candidate SD × IV-RV rule must meet the total training minimums and show non-negative expectancy in both calendar halves of the training window. If no candidate qualifies, the model chooses NO TRADE.

Three models are evaluated on the same walk-forward folds:
- adaptive SD + adaptive IV-RV threshold
- fixed 2-SD + adaptive IV-RV threshold
- fixed 2-SD + no IV-RV filter control

Run:

```bash
python run_stable_entry_research.py \
  --options-dir data/trademarkk/processed \
  --spot data/trademarkk/index/NIFTY.parquet \
  --config config_dte6.yaml \
  --out-dir results/stable_entry
```

The selector is deliberately not allowed to force a trade. This makes regime avoidance an explicit model outcome rather than a post-hoc interpretation.

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

## Experiment 7: stability-constrained exit matrix

After the entry experiments, exit mechanics are tested while holding the DTE6 and 2-SD entry rule fixed. The first matrix varies profit capture = 25%, 50%, 75% and stop multiple = 1.5x, 2.0x, 2.5x; the 1-DTE time exit remains fixed. Training selection requires non-negative expectancy in both calendar halves, so an unstable exit rule can result in NO TRADE.

Run:

```bash
python run_exit_research.py \
  --options-dir data/trademarkk/processed \
  --spot data/trademarkk/index/NIFTY.parquet \
  --config config_dte6.yaml \
  --out-dir results/exit_matrix
```

Outputs include the full matrix, the walk-forward-selected exit sequence, the selected OOS trade ledger, the fixed 50%-capture/2x-stop control ledger, and portfolio-level risk metrics.

## Experiment 8: frozen iron-condor comparison

The exit-matrix result is now treated as a frozen candidate rather than an adaptive rule. The next structural comparison keeps the same DTE6 + 2-SD entry and realistic costs, then compares defined-risk iron condors using only pre-declared wing widths and exit candidates.

Run:

```bash
python run_iron_condor_research.py \
  --options-dir data/trademarkk/processed \
  --spot data/trademarkk/index/NIFTY.parquet \
  --config config_dte6_iron_condor.yaml \
  --wing-width 500 1000 \
  --profit-capture 0.50 0.75 \
  --stop-multiple 2.0 2.5 \
  --holdout-start 2026-06-01 \
  --out-dir results/iron_condor
```

The default suite is intentionally small and pre-declared: 500/1000-point wings × 50%/75% capture × 2.0x/2.5x stop. It is a robustness comparison, not a new optimizer.

Each condor ledger records the actual wing widths, maximum theoretical loss, credit-to-max-loss ratio, execution costs and data-quality checks. Condor transaction costs use 8 orders for a full entry+exit round trip.

## Experiment 9: frozen post-research validation

Once a candidate is frozen, use `run_frozen_validation.py` to split its trade ledger at a declared holdout date:

```bash
python run_frozen_validation.py \
  --trades results/exit_matrix/pc_0.75_stop_2.5_trades.csv \
           results/exit_matrix/pc_0.50_stop_2_trades.csv \
  --labels strangle_75_2_5 strangle_50_2 \
  --holdout-start 2026-06-01 \
  --out results/frozen_validation_summary.csv
```

This reports development-period versus holdout-period risk metrics without re-selecting parameters.

Important: the 2026-06-01 split is a **frozen post-research diagnostic**, not a pristine independent validation, because earlier exploratory work inspected the full archive through that period. A statistically clean independent validation should use a genuinely untouched future period or an independent data source.

The first-choice frozen research set is now:
- DTE6 + 2-SD entry
- fixed 50% capture / 2x stop control
- fixed 75% capture / 2.5x stop candidate
- iron condor with 500- and 1000-point pre-declared wings

Avoid further broad parameter searches until the structural comparison and independent data validation are complete.

## Experiment 10: accelerated robustness stage

The research now avoids broad parameter optimization. Use a small frozen suite:

1. Tight-wing iron condors: 100/200/300-point requested wings with 50% or 75% profit capture and 2.0x or 2.5x stops.
2. Frozen 75% capture / 2.5x stop strangle: replay execution across 0, 0.25, 0.50, 0.75, 1.00 and 1.50 option points per leg.
3. Tail attribution: classify losses by exit reason and loss-to-entry-credit multiple, then inspect the largest losses.
4. Independent validation: build the same frozen strategy ledger from a secondary data source and compare matched dates/strikes, entry credits, exits and P&L.

Tight-wing condors also write `condor_coverage.csv`, which distinguishes unavailable wings from insufficient net credit or missing entry timestamps. This is required before interpreting an empty holdout as a strategy result.

Recommended commands:

```bash
python run_iron_condor_research.py \
  --options-dir data/trademarkk/processed \
  --spot data/trademarkk/index/NIFTY.parquet \
  --config config_dte6_iron_condor.yaml \
  --wing-width 100 200 300 \
  --profit-capture 0.50 0.75 \
  --stop-multiple 2.0 2.5 \
  --out-dir results/iron_condor_tight

python run_slippage_sensitivity.py \
  --trades results/exit_matrix/pc_0.75_stop_2.5_trades.csv \
  --options-dir data/trademarkk/processed \
  --config config_dte6_frozen_75_25.yaml \
  --slippages 0 0.25 0.50 0.75 1.00 1.50 \
  --out-dir results/slippage_frozen_75_25

python tools_tail_analysis.py \
  --trades results/exit_matrix/pc_0.75_stop_2.5_trades.csv \
  --out-dir results/tail_75_25
```

For independent validation, first produce an equivalent frozen trade ledger from the secondary source, then run:

```bash
python tools_independent_validation.py \
  --primary results/exit_matrix/pc_0.75_stop_2.5_trades.csv \
  --secondary <secondary-source-ledger.csv> \
  --out-dir results/independent_validation
```

Do not use holdout performance from the previously inspected archive as independent confirmation. The secondary-source comparison is specifically intended to detect vendor-dependent prices, missing quotes and settlement inconsistencies.

### Secondary source build: Rissin

The Rissin Hugging Face dataset is a secondary option-price source with NIFTY/intraday fields that can be normalized into the research format. The repository includes `tools/build_rissin_long.py` to filter NIFTY minute rows and emit per-expiry Parquets compatible with the frozen backtester.

Dataset: https://huggingface.co/datasets/rissin/nse-options-intraday

```bash
python tools/build_rissin_long.py \
  --source <rissin-options-file.parquet> \
  --spot data/trademarkk/index/NIFTY.parquet \
  --out data/rissin/processed \
  --start-date 2024-10-01
```

For the first cross-source test, keep the same frozen DTE6 + 2-SD + 75% capture / 2.5x stop strategy and compare the resulting ledger with `tools_independent_validation.py`.

### Independent-validation interpretation

The comparison report now distinguishes overall coverage from the actual common date window. Because the Rissin intraday series starts in October 2024, do not use the raw primary match rate across the entire 2022+ archive as the validation statistic. Use the overlap-period match rates and the distribution of P&L/credit differences. Large mean differences with small medians should trigger inspection of the detailed matched ledger for a few outlier trades.

## Experiment 11: frozen regime attribution + prospective paper ledger

This stage is deliberately **diagnostic and prospective, not an optimizer**.

The frozen candidate is:

- DTE6 entry
- 10:00 IST
- pure 2-SD strike placement
- 75% premium capture
- 2.5x initial-credit stop
- 1-DTE, 15:00 time exit
- no IV-RV filter
- 1 historical/current lot

The code rejects a paper-signal configuration if any of these strategy parameters are changed. Each signal also stores a strategy specification hash so later ledger rows can be audited back to the frozen contract.

### 11A. Regime/event attribution

Run diagnostic buckets without selecting any bucket as a trading rule:

```bash
python run_regime_attribution.py \
  --trades results/exit_matrix/pc_0.75_stop_2.5_trades.csv \
  --spot data/trademarkk/index/NIFTY.parquet \
  --out-dir results/regime_attribution
```

Outputs:

- `attribution_trades.csv` — one enriched row per historical trade.
- `regime_summary.csv` — fixed, pre-declared diagnostic buckets for IV-RV spread, RV20, entry credit, entry gap, maximum spot move, exit reason, loss size, year and quarter.

The buckets are intentionally fixed. Do not pick thresholds after seeing which bucket made the most money.

### 11B. Generate one frozen paper signal

The signal generator accepts an **as-of snapshot**. It does not inspect option prices after the supplied timestamp.

```bash
python run_paper_signal.py \
  --options <asof-options-snapshot.parquet> \
  --spot <nifty-spot-history.parquet> \
  --expiry YYYY-MM-DD \
  --as-of "YYYY-MM-DD 10:00" \
  --config config_dte6_frozen_75_25.yaml \
  --out results/paper/signal.csv
```

A READY signal records:

- entry spot and forward
- ATM IV and pre-entry RV20
- the 2-SD lower/upper boundaries
- selected PE/CE strikes and deltas
- bid/ask/mid prices
- planned executable entry prices
- initial credit
- profit-target and stop debit thresholds
- historical lot size
- scheduled time exit
- strategy ID and specification hash

It does **not** record a future P&L.

### 11C. Initialize and update the paper ledger

Initialize:

```bash
python run_paper_ledger.py \
  --action init \
  --signals results/paper/signal.csv \
  --ledger results/paper/paper_ledger.csv
```

Record actual entry fills:

```bash
python run_paper_ledger.py \
  --action entry \
  --ledger results/paper/paper_ledger.csv \
  --signal-id <signal-id> \
  --fill-timestamp "YYYY-MM-DD 10:01" \
  --put-fill <price> \
  --call-fill <price>
```

Record actual exit fills:

```bash
python run_paper_ledger.py \
  --action exit \
  --ledger results/paper/paper_ledger.csv \
  --signal-id <signal-id> \
  --fill-timestamp "YYYY-MM-DD HH:MM" \
  --put-fill <price> \
  --call-fill <price> \
  --exit-reason profit_target|stop|time_exit|expiry|manual
```

The ledger calculates gross P&L and the same transaction-cost model used by the backtest. It also flags whether the recorded fill/reason is consistent with the frozen exit rules.

### Prospective validation rule

From the point at which this stage begins, do not change the frozen strategy parameters in response to paper-trading outcomes. Record misses, data failures and execution slippage as observations.

The research question is now:

> Does the frozen historical edge survive unchanged when future signals and fills are recorded prospectively?

Only after enough prospective observations should we reconsider the model itself.


### 11D. Automated Upstox paper engine

The repository also includes an always-on paper-trading daemon backed by Upstox market data. It is deliberately **paper-only**: it never calls any order-placement API.

Upstox currently documents an Analytics Token with one-year validity and read-only access to Market Quote, Historical Data, Option Chain and WebSocket APIs, so it is the recommended credential for this workflow rather than a daily OAuth token.

The daemon uses the Upstox Option Chain endpoint for the frozen entry and held-leg quotes. The endpoint exposes underlying spot, bid/ask, volume, OI, IV and delta for each strike.

Create a local secrets file from the example:

```bash
cp .env.paper.example .env.paper
```

Put the Analytics Token only in that local file:

```text
UPSTOX_ANALYTICS_TOKEN=<your token>
```

Do not commit the real token.

Run one polling cycle first:

```bash
python run_upstox_paper.py --once
```

Run continuously during market hours:

```bash
python run_upstox_paper.py --poll-seconds 30
```

The daemon automatically:

1. Checks the Upstox current-year Market Holidays API and only treats an NFO date as tradable when it is not a weekend or NFO trading holiday. This prevents entry and quote polling on NSE derivatives holidays.
2. Detects the weekly expiry exactly 6 calendar days ahead by querying the explicit target expiry date.
3. Creates the frozen 10:00 signal.
4. Opens a paper position using bid minus the frozen slippage assumption.
5. Polls the held legs using executable ask plus slippage only during the market session.
6. Closes at the frozen 75% target, 2.5x stop, 1-DTE 15:00 time exit, or expiry fallback.
7. Persists the paper ledger and a quote-by-quote monitoring log.
8. Optionally sends a webhook alert via `PAPER_ALERT_WEBHOOK_URL`.

The holiday calendar is cached in memory for up to six hours to avoid repeated API calls. If a refresh fails, a previously fetched calendar is reused; with no known calendar, the daemon fails closed rather than risking an entry on an unknown exchange holiday.

Outputs:

```text
results/paper/
├── upstox_paper_ledger.csv
└── upstox_quote_log.csv
```

For macOS, install it as a LaunchAgent so it starts automatically when you log in and restarts if the process exits:

```bash
python tools/install_mac_launchd.py \
  --repo /absolute/path/to/nifty_strangle_research \
  --python /absolute/path/to/nifty_strangle_research/.venv/bin/python \
  --config config_dte6_frozen_75_25.yaml \
  --env-file .env.paper
```

The macOS LaunchAgent stores logs under:

```text
results/paper/logs/
├── paper.log
└── paper.error.log
```

This automation is intentionally separate from any live-trading integration. If an Upstox MCP connection is later available, it can be used for read-only account/order reconciliation; the paper engine still uses Upstox market-data APIs so the signal is deterministic and auditable.
