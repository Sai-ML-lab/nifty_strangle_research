# Data folder

Put historical NIFTY option-chain files here.

Preferred format: long-form Parquet for speed. CSV also works.

Required columns:

```text
timestamp, expiry, underlying, spot, strike, option_type
```

Recommended columns:

```text
future, bid, ask, ltp, volume, oi, iv, delta
```

The more of the recommended fields that are available, the less execution/volatility modeling the backtest has to infer.
