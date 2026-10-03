import numpy as np
import pandas as pd
import pytest

from calibration import calibrate_vasicek, infer_dt, load_rates, expected_rate, rate_std


def _simulate(a, b, sigma, r0, n, dt, seed=0):
    rng = np.random.default_rng(seed)
    alpha = np.exp(-a * dt)
    s = sigma * np.sqrt((1 - alpha ** 2) / (2 * a))
    r = np.empty(n)
    r[0] = r0
    for k in range(1, n):
        r[k] = alpha * r[k - 1] + b * (1 - alpha) + s * rng.standard_normal()
    return r


def test_recovers_parameters_from_simulated_data():
    a, b, sigma = 0.8, 0.03, 0.01
    r = _simulate(a, b, sigma, 0.05, 40_000, 1 / 52)
    fit = calibrate_vasicek(r, dt=1 / 52)
    assert fit["sigma"] == pytest.approx(sigma, rel=0.02)
    assert fit["b"] == pytest.approx(b, abs=0.005)
    assert fit["a"] == pytest.approx(a, rel=0.25)       # a is the hardest to pin down


def test_no_mean_reversion_raises():
    r = 0.01 * 1.001 ** np.arange(500)                  # explosive series, alpha > 1
    with pytest.raises(ValueError):
        calibrate_vasicek(r, dt=1 / 365)


@pytest.mark.parametrize("freq, expected", [("D", 1 / 365), ("B", 1 / 252),
                                            ("W", 1 / 52), ("MS", 1 / 12)])
def test_infer_dt(freq, expected):
    idx = pd.date_range("2020-01-01", periods=200, freq=freq)
    assert infer_dt(pd.Series(0.01, index=idx)) == pytest.approx(expected)


def test_load_rates_handles_fred_missing_values(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("observation_date,DFF\n2024-01-02,5.33\n2024-01-01,.\n2024-01-03,5.31\n")
    r = load_rates(str(f))
    assert list(r.values) == pytest.approx([0.0533, 0.0531])
    assert r.index.is_monotonic_increasing


def test_moments_limits():
    assert expected_rate(0.05, 0.0, 0.3, 0.02) == pytest.approx(0.05)
    assert expected_rate(0.05, 1e3, 0.3, 0.02) == pytest.approx(0.02)
    assert rate_std(1e3, 0.3, 0.01) == pytest.approx(0.01 / np.sqrt(0.6))
