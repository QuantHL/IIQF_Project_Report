"""Analytical vs Monte Carlo agreement."""
import pytest

from monte_carlo import mc_zcb_price, mc_zcb_call, variance_reduction_factor
from vasicek import zcb_price, zcb_option_price


def test_mc_zcb_matches_analytical(params):
    exact = zcb_price(0.04, 0, 5.0, **params)
    mc = mc_zcb_price(0.04, 5.0, **params, n_paths=20_000, n_steps=500, seed=1)
    assert abs(mc["price"] - exact) < 4 * mc["std_err"] + 1e-5   # + trapezoid bias


def test_plain_mc_matches_analytical(params):
    exact = zcb_price(0.04, 0, 5.0, **params)
    mc = mc_zcb_price(0.04, 5.0, **params, n_paths=20_000, n_steps=250, seed=2,
                      antithetic=False)
    assert abs(mc["price"] - exact) < 4 * mc["std_err"]


def test_antithetic_reduces_variance(params):
    vr = variance_reduction_factor(0.04, 5.0, **params, n_paths=5_000, n_steps=100)
    assert vr["factor"] > 10


def test_mc_option_matches_closed_form(params):
    exact = zcb_option_price(0.04, 4.0, 5.0, 900.0, **params, face_value=1000.0)
    mc = mc_zcb_call(0.04, **params, n_paths=20_000, n_steps=400, seed=3)
    assert abs(mc["price"] - exact) < 4 * mc["std_err"] + 0.02
