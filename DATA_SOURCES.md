# Data sources and research assumptions

## Primary historical dataset candidate

Options Data currently lists NIFTY 1-minute full-chain history from January 2023 through September 2026, with every live expiry/strike and OI. The vendor states that the paid option files contain OHLCV + OI, with an IV column on many files from July 2023 through 14-Aug-2026; the files do not contain bid/ask or Greeks. The free sample contains three 2026 trading days and is useful for validating the ingestion pipeline.

For this project, the adapter joins the options bars to matching NIFTY spot bars and optionally futures bars. Because the vendor's 1-minute timestamp identifies the start of the bar, entry selection uses the bar OPEN rather than the bar close to avoid look-ahead.

## Exchange validation

NSE's live option chain exposes LTP, OI, volume, IV and best bid/ask for the current chain. Its historical derivatives reports provide official end-of-day contract/settlement information. For final validation, compare historical expiries, settlement values, lot sizes and charges against NSE records.

## Historical NIFTY lot sizes used by the engine

- 50 before the first revised weekly expiry on 02-May-2024.
- 25 for revised contracts through 19-Nov-2024.
- 75 for revised contracts through 30-Dec-2025 expiry.
- 65 for the first post-revision weekly expiry onward.

These dates are based on NSE lot-size revision circulars for April 2024 and October 2025.

## Expiry schedule

The engine uses the actual expiry value present in the historical contract data. This avoids hard-coding Tuesday across the entire history because NIFTY weekly expiry moved from Thursday to Tuesday beginning with the revised schedule in September 2025.

## Important execution caveat

A 1-minute OHLC dataset cannot recreate historical bid/ask order-book execution. The default research mode therefore uses bar-open entry prices and bar-close exit marks with a configurable per-leg slippage assumption. Before any live deployment, validate the strategy again against true quote-level data or your broker's recorded fills.
