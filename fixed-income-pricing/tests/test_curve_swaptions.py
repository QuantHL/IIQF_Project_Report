from pathlib import Path

import numpy as np
import pytest

from curve import SofrCurve, build_sofr_curve
from swaps import par_swap_rate
from swaptions import black_swaption, forward_swap_rate

DATA = Path(__file__).resolve().parents[1] / "data" / "sofr_2026-10-01.csv"
SPEC = dict(notional=100.0, K=0.045, T0=2.0, swap_tenor=5.0, freq=0.5, vol=0.15)


@pytest.fixture
def curve():
    return SofrCurve()


def test_reprices_input_swaps(curve):
    for T, S in curve.swap_rates.items():
        assert par_swap_rate(curve.discount, T, freq=1.0) == pytest.approx(S, abs=1e-12)


def test_reprices_term_sofr(curve):
    for T, R in curve.term_sofr.items():
        assert curve.discount(T) == pytest.approx(1 / (1 + R * 365 * T / 360))


def test_report_table_5(curve):
    expected = {2.0: 0.914162, 3.0: 0.871684, 5.0: 0.794255, 7.0: 0.722219}
    for T, P in expected.items():
        assert curve.discount(T) == pytest.approx(P, abs=1e-6)


def test_linear_interpolation(curve):
    mid = 0.5 * (curve.zero_rate(3.0) + curve.zero_rate(4.0))
    assert curve.zero_rate(3.5) == pytest.approx(mid)


def test_csv_round_trip(curve):
    c2 = SofrCurve.from_csv(DATA)
    assert np.allclose(c2.discount(np.arange(0.5, 10, 0.5)), curve.discount(np.arange(0.5, 10, 0.5)))
    assert np.isclose(build_sofr_curve()(5.0), curve.discount(5.0))


def test_swaption_report_value(curve):
    res = black_swaption(**SPEC, discount=curve.discount)
    assert res["forward_swap_rate"] == pytest.approx(0.047683, abs=2e-6)
    assert res["value"] == pytest.approx(2.1737, abs=2e-4)


def test_payer_receiver_parity(curve):
    pay = black_swaption(**SPEC, discount=curve.discount, payer=True)
    rec = black_swaption(**SPEC, discount=curve.discount, payer=False)
    fwd = SPEC["notional"] * pay["annuity"] * (pay["forward_swap_rate"] - SPEC["K"])
    assert pay["value"] - rec["value"] == pytest.approx(fwd)


def test_zero_vol_limit_is_intrinsic(curve):
    res = black_swaption(**{**SPEC, "vol": 1e-8}, discount=curve.discount)
    assert res["value"] == pytest.approx(res["intrinsic"], abs=1e-8)


def test_atm_forward_consistency(curve):
    S0 = forward_swap_rate(curve.discount, 2.0, 5.0, 0.5)
    pay = black_swaption(**{**SPEC, "K": S0}, discount=curve.discount, payer=True)
    rec = black_swaption(**{**SPEC, "K": S0}, discount=curve.discount, payer=False)
    assert pay["value"] == pytest.approx(rec["value"])
