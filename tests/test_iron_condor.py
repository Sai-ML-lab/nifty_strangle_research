from __future__ import annotations

import pandas as pd


def test_condor_payoff_is_bounded():
    from src.iron_condor import _condor_intrinsic

    values = [
        _condor_intrinsic(s, 23000, 23500, 24500, 25000)
        for s in [22000, 23000, 23500, 24000, 24500, 25000, 26000]
    ]
    assert all(0 <= x <= 500 for x in values)
    assert max(values) == 500


def test_condor_wing_selection_is_at_least_requested_width():
    from src.iron_condor import _pick_wing

    chain = pd.DataFrame(
        {
            "strike": [22000, 22500, 23000, 23500, 24500, 25000, 25500],
            "option_type": ["PE"] * 4 + ["CE"] * 3,
        }
    )
    put_wing, put_width = _pick_wing(chain, 23500, "PE", 500)
    call_wing, call_width = _pick_wing(chain, 24500, "CE", 500)
    assert put_wing == 23000
    assert put_width == 500
    assert call_wing == 25000
    assert call_width == 500


def test_condor_intrinsic_is_never_above_wing_width():
    from src.iron_condor import _condor_intrinsic
    assert _condor_intrinsic(22000, 23000, 23500, 24500, 25000) == 500
    assert _condor_intrinsic(26000, 23000, 23500, 24500, 25000) == 500
