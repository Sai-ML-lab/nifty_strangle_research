from __future__ import annotations
import pandas as pd

def test_rissin_builder_filters_nifty_intraday(tmp_path):
    from tools.build_rissin_long import build_rissin_long
    src = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-10-02 10:00"]*3),
        "underlying": ["NIFTY","NIFTY","BANKNIFTY"],
        "expiry": ["2024-10-08"]*3,
        "strike": [23000,25000,50000],
        "option_type": ["PE","CE","CE"],
        "open": [100,100,100], "close": [90,90,90],
        "volume": [1,1,1], "oi": [1,1,1], "granularity": ["1min"]*3,
    })
    spot = pd.DataFrame({"timestamp": pd.to_datetime(["2024-10-02 10:00"]), "close":[24000]})
    sp, pp = tmp_path/"rissin.parquet", tmp_path/"spot.parquet"
    src.to_parquet(sp,index=False); spot.to_parquet(pp,index=False)
    out=tmp_path/"out"
    build_rissin_long(sp,pp,out,"2024-10-01","2024-10-03")
    got=pd.read_parquet(out/"expiry=2024-10-08.parquet")
    assert len(got)==2 and set(got["option_type"])=={"PE","CE"}
