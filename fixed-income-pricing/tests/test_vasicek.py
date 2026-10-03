import numpy as np
import pytest

from vasicek import A, B, zcb_price, zero_rate, long_rate, zcb_option_price
from swaps import swap_rate_from_zcbs, vasicek_swap_rate, vasicek_discount, swap_value


def test_report_table_2(params):
    """
    Report Table 2 (a = 0.25, b = 4%, sigma = 1.5%). Its prices correspond to
    r0 ~ 3.91%; with r0 = 4% the formula gives 0.9608, 0.9233, 0.8875, 0.8532, 0.8204.
    """
    expected = {1: 0.9616, 2: 0.9247, 3: 0.8893, 4: 0.8554, 5: 0.8229}
    for T, P in expected.items():
        assert zcb_price(0.0391, 0, T, **params) == pytest.approx(P, abs=5e-4)
    assert zcb_price(0.04, 0, 5.0, **params) == pytest.approx(0.8204, abs=1e-4)


def test_boundary_and_limits(params):
    assert zcb_price(0.04, 2.0, 2.0, **params) == pytest.approx(1.0)
    assert B(0, 1e-8, params["a"]) == pytest.approx(1e-8)
    # long end converges to b - sigma^2 / (2 a^2)
    assert zero_rate(0.04, 0, 500.0, **params) == pytest.approx(long_rate(**params), abs=2e-4)
    # sigma -> 0, r0 = b: flat curve at b
    assert zero_rate(0.03, 0, 7.0, 0.5, 0.03, 1e-10) == pytest.approx(0.03)


def test_vectorised(params):
    T = np.array([1.0, 2.0, 5.0])
    r = np.array([0.01, 0.02, 0.03])
    assert zcb_price(0.04, 0, T, **params).shape == (3,)
    assert zcb_price(r, 1.0, 2.0, **params).shape == (3,)


def test_swap_rate(params):
    res = vasicek_swap_rate(0.04, **params, maturity=5)
    assert res["swap_rate"] == pytest.approx(swap_rate_from_zcbs(res["Z"]))
    assert swap_value(vasicek_discount(0.04, **params), res["swap_rate"], 5.0) \
        == pytest.approx(0.0, abs=1e-14)
    assert 0.03 < res["swap_rate"] < 0.045


def test_option_put_call_parity(params):
    L, K, T, S = 1000.0, 900.0, 4.0, 5.0
    c = zcb_option_price(0.04, T, S, K, **params, face_value=L, call=True)
    p = zcb_option_price(0.04, T, S, K, **params, face_value=L, call=False)
    fwd = L * zcb_price(0.04, 0, S, **params) - K * zcb_price(0.04, 0, T, **params)
    assert c - p == pytest.approx(fwd)
    assert c >= max(fwd, 0.0)
