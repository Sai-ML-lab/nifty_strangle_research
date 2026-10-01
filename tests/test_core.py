import numpy as np
import pandas as pd
from src.core import black76_price, black76_delta, implied_vol_black76, sd_band, nifty_lot_size, trade_cost, Costs
from src.data_ingest import normalize_options_vendor_file


def test_black76_roundtrip():
    fwd, k, vol, t, r = 24000.0, 25000.0, 0.18, 6/365, 0.06
    p = black76_price(fwd, k, vol, t, r, True)
    iv = implied_vol_black76(p, fwd, k, t, r, True)
    assert np.isfinite(iv)
    assert abs(iv - vol) < 1e-5


def test_put_call_delta_signs():
    dc = black76_delta(24000, 25000, 0.18, 6/365, 0.06, True)
    dp = black76_delta(24000, 23000, 0.18, 6/365, 0.06, False)
    assert 0 < dc < 0.5
    assert -0.5 < dp < 0


def test_sd_band():
    lo, hi, move = sd_band(24000, 0.15, __import__('pandas').Timestamp('2026-06-09'), __import__('pandas').Timestamp('2026-06-03 10:00'), 2)
    assert lo < 24000 < hi
    assert move > 0


def test_historical_nifty_lot_size():
    import pandas as pd
    assert nifty_lot_size(pd.Timestamp("2024-04-25")) == 50
    assert nifty_lot_size(pd.Timestamp("2024-05-02")) == 25
    assert nifty_lot_size(pd.Timestamp("2024-11-26")) == 75
    assert nifty_lot_size(pd.Timestamp("2026-01-06")) == 65


def test_stt_rate_is_date_aware():
    c = Costs(exchange_txn_pct=0.0, sebi_turnover_pct=0.0, stamp_duty_buy_option_pct=0.0, brokerage_per_order=0.0)
    old = trade_cost(100000, 100000, 0, 1, c, pd.Timestamp("2026-03-31"))
    new = trade_cost(100000, 100000, 0, 1, c, pd.Timestamp("2026-04-01"))
    assert abs(old - 100.0) < 1e-9
    assert abs(new - 150.0) < 1e-9


def test_options_data_layout_a_normalizes():
    raw = pd.DataFrame({
        "datetime": ["2026-09-15 10:00:00"] * 2,
        "stock_code": ["NIFTY"] * 2,
        "exchange_code": ["NFO"] * 2,
        "product_type": ["Options"] * 2,
        "expiry_date": ["15-SEP-2026"] * 2,
        "strike_price": [24000, 24000],
        "right": ["Call", "Put"],
        "open": [100, 90],
        "high": [105, 95],
        "low": [95, 85],
        "close": [102, 92],
        "volume": [10, 20],
        "open_interest": [100, 200],
    })
    x = normalize_options_vendor_file(raw)
    assert list(x["option_type"]) == ["CE", "PE"]
    assert x["expiry"].iloc[0] == pd.Timestamp("2026-09-15")
    assert float(x["open"].iloc[0]) == 100.0


def test_parity_forward_falls_near_spot():
    from src.core import estimate_forward_from_parity, black76_price
    ts = pd.Timestamp("2026-06-03 10:00:00", tz="Asia/Kolkata")
    expiry = pd.Timestamp("2026-06-09", tz="Asia/Kolkata")
    fwd = 24020.0
    rows = []
    for k in [23800, 23900, 24000, 24100, 24200]:
        c = black76_price(fwd, k, 0.15, (expiry + pd.Timedelta(hours=15, minutes=30) - ts).total_seconds()/(365*24*3600), 0.06, True)
        p = black76_price(fwd, k, 0.15, (expiry + pd.Timedelta(hours=15, minutes=30) - ts).total_seconds()/(365*24*3600), 0.06, False)
        rows.extend([
            {"timestamp": ts, "expiry": expiry, "strike": k, "option_type": "CE", "entry_price": c, "spot": 24000.0},
            {"timestamp": ts, "expiry": expiry, "strike": k, "option_type": "PE", "entry_price": p, "spot": 24000.0},
        ])
    x = pd.DataFrame(rows)
    est = estimate_forward_from_parity(x, 24000.0, 0.06, "entry_price")
    assert abs(est - fwd) < 2.0


def test_baseline_next_expiry_mapping_avoids_overlap():
    from run_batch_research import entry_date_for_expiry
    from src.core import StrategyConfig

    cfg_weekday = 2  # Wednesday
    dates = [pd.Timestamp("2023-01-19"), pd.Timestamp("2023-01-25")]
    def entry_date_for(expiry):
        offset = (expiry.weekday() - cfg_weekday) % 7
        if offset == 0:
            offset = 7
        return (expiry - pd.Timedelta(days=offset)).normalize()

    mapping = {}
    for expiry in dates:
        d = entry_date_for(expiry)
        if d not in mapping:
            mapping[d] = expiry

    assert mapping[pd.Timestamp("2023-01-18")] == pd.Timestamp("2023-01-19")

    cfg = StrategyConfig(entry_mode="dte", target_dte=6)
    assert entry_date_for_expiry(pd.Timestamp("2023-01-19"), cfg) == pd.Timestamp("2023-01-13")
