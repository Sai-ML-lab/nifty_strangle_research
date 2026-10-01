from __future__ import annotations

import pandas as pd


def test_upstox_chain_to_dataframe_converts_iv_percent():
    from src.upstox_paper import UpstoxPaperClient

    rows = [
        {
            "expiry": "2026-10-15",
            "strike_price": 24000,
            "underlying_spot_price": 23850,
            "call_options": {
                "instrument_key": "NSE_FO|1",
                "market_data": {
                    "ltp": 100,
                    "bid_price": 99,
                    "ask_price": 101,
                    "volume": 10,
                    "oi": 1000,
                },
                "option_greeks": {"iv": 20.0, "delta": 0.25},
            },
            "put_options": {
                "instrument_key": "NSE_FO|2",
                "market_data": {
                    "ltp": 120,
                    "bid_price": 119,
                    "ask_price": 121,
                    "volume": 12,
                    "oi": 1100,
                },
                "option_greeks": {"iv": 22.0, "delta": -0.30},
            },
        }
    ]
    x = UpstoxPaperClient.chain_to_dataframe(rows, pd.Timestamp("2026-10-09 10:00"))
    assert set(x["option_type"]) == {"CE", "PE"}
    assert float(x.loc[x["option_type"].eq("CE"), "iv"].iloc[0]) == 0.20
    assert float(x.loc[x["option_type"].eq("PE"), "iv"].iloc[0]) == 0.22


def test_find_dte6_expiry_queries_exact_target_date():
    from src.upstox_paper import UpstoxPaperClient

    class Fake(UpstoxPaperClient):
        def __init__(self):
            self.calls = []

        def option_contracts(self, expiry_ref):
            self.calls.append(expiry_ref)
            if expiry_ref == "2026-10-15":
                return pd.DataFrame(
                    [
                        {"expiry": "2026-10-15", "weekly": True},
                        {"expiry": "2026-10-15", "weekly": True},
                    ]
                )
            return pd.DataFrame()

    c = Fake()
    got = c.find_dte6_expiry(
        pd.Timestamp("2026-10-09 10:00", tz="Asia/Kolkata").to_pydatetime()
    )
    assert got == pd.Timestamp("2026-10-15")
    assert c.calls == ["2026-10-15"]


def test_find_dte6_expiry_rejects_non_weekly_target():
    from src.upstox_paper import UpstoxPaperClient

    class Fake(UpstoxPaperClient):
        def __init__(self):
            pass

        def option_contracts(self, expiry_ref):
            return pd.DataFrame([{"expiry": expiry_ref, "weekly": False}])

    c = Fake()
    got = c.find_dte6_expiry(
        pd.Timestamp("2026-10-09 10:00", tz="Asia/Kolkata").to_pydatetime()
    )
    assert got is None


def test_nfo_holiday_calendar_marks_trading_holiday():
    from src.upstox_paper import UpstoxPaperClient

    class Fake(UpstoxPaperClient):
        def __init__(self):
            pass

        def market_holidays(self):
            return pd.DataFrame(
                [
                    {
                        "date": "2026-10-02",
                        "holiday_type": "TRADING_HOLIDAY",
                        "closed_exchanges": ["NFO", "NSE"],
                    },
                    {
                        "date": "2026-10-03",
                        "holiday_type": "SETTLEMENT_HOLIDAY",
                        "closed_exchanges": ["NFO", "NSE"],
                    },
                ]
            )

    c = Fake()
    now = pd.Timestamp("2026-10-01 15:00", tz="Asia/Kolkata").to_pydatetime()
    holidays = c.nfo_trading_holiday_dates(now)
    assert holidays == {pd.Timestamp("2026-10-02")}
    assert c.is_nfo_trading_day(now) is True
    assert (
        c.is_nfo_trading_day(
            pd.Timestamp("2026-10-02 10:00", tz="Asia/Kolkata").to_pydatetime()
        )
        is False
    )


def test_nfo_calendar_uses_cached_data_for_short_interval():
    from src.upstox_paper import UpstoxPaperClient

    class Fake(UpstoxPaperClient):
        def __init__(self):
            super().__init__("token")
            self.calls = 0

        def market_holidays(self):
            self.calls += 1
            return pd.DataFrame(
                [
                    {
                        "date": "2026-10-02",
                        "holiday_type": "TRADING_HOLIDAY",
                        "closed_exchanges": ["NFO"],
                    }
                ]
            )

    c = Fake()
    t1 = pd.Timestamp("2026-10-01 09:00", tz="Asia/Kolkata").to_pydatetime()
    t2 = pd.Timestamp("2026-10-01 09:30", tz="Asia/Kolkata").to_pydatetime()
    assert c.is_nfo_trading_day(t1) is True
    assert c.is_nfo_trading_day(t2) is True
    assert c.calls == 1


def test_launchd_plist_contains_keepalive_and_unbuffered_python():
    from pathlib import Path
    from tools.install_mac_launchd import build_plist

    p = build_plist(
        label="com.test.paper",
        repo=Path("/tmp/repo"),
        python_executable="python3",
        config="config.yaml",
        env_file=".env.paper",
        log_dir=Path("/tmp/paper-logs"),
    )
    assert p["KeepAlive"] is True
    assert p["RunAtLoad"] is True
    assert p["ProgramArguments"][1] == "-u"
    assert p["ProgramArguments"][2].endswith("run_upstox_paper.py")
