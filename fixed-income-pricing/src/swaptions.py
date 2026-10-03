"""
Part II(b): European swaption with Black's formula on the SOFR curve.

Payer swaption: right at T0 to enter a swap paying fixed K, receiving float,
with fixed payments at T_i = T0 + tau i, i = 1..n.

    Annuity       A  = sum tau_i P(0,T_i)                         (numeraire)
    Forward rate  S0 = (P(0,T0) - P(0,T_n)) / A
    Under the annuity measure S(t) is a martingale; lognormal with vol nu:
    Payer         V  = N A [S0 Phi(d+) - K Phi(d-)]
    Receiver      V  = N A [K Phi(-d-) - S0 Phi(-d+)]
    d+- = [ln(S0/K) +- nu^2 T0 / 2] / (nu sqrt(T0))

Usage:
    python src/swaptions.py
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import norm


def swap_schedule(T0: float, swap_tenor: float, freq: float) -> np.ndarray:
    n = int(round(swap_tenor / freq))
    return T0 + freq * np.arange(1, n + 1)


def annuity_factor(discount: Callable, T0: float, swap_tenor: float, freq: float) -> float:
    times = swap_schedule(T0, swap_tenor, freq)
    return float(np.sum(freq * discount(times)))


def forward_swap_rate(discount: Callable, T0: float, swap_tenor: float, freq: float) -> float:
    times = swap_schedule(T0, swap_tenor, freq)
    A = annuity_factor(discount, T0, swap_tenor, freq)
    return float((discount(T0) - discount(times[-1])) / A)


def black_swaption(notional: float, K: float, T0: float, swap_tenor: float,
                   freq: float, vol: float, discount: Callable,
                   payer: bool = True) -> dict:
    """Black (lognormal) swaption price with all intermediate quantities."""
    times = swap_schedule(T0, swap_tenor, freq)
    A = annuity_factor(discount, T0, swap_tenor, freq)
    S0 = forward_swap_rate(discount, T0, swap_tenor, freq)

    sd = vol * np.sqrt(T0)
    d_plus = (np.log(S0 / K) + 0.5 * sd ** 2) / sd
    d_minus = d_plus - sd

    if payer:
        value = notional * A * (S0 * norm.cdf(d_plus) - K * norm.cdf(d_minus))
        intrinsic = notional * A * max(S0 - K, 0.0)
    else:
        value = notional * A * (K * norm.cdf(-d_minus) - S0 * norm.cdf(-d_plus))
        intrinsic = notional * A * max(K - S0, 0.0)

    return {"value": float(value), "annuity": A, "forward_swap_rate": S0,
            "d_plus": float(d_plus), "d_minus": float(d_minus),
            "N(d_plus)": float(norm.cdf(d_plus)), "N(d_minus)": float(norm.cdf(d_minus)),
            "intrinsic": float(intrinsic), "time_value": float(value - intrinsic),
            "payment_times": times}


def european_swaption(notional, K, T0, swap_tenor, freq, vol, discount, payer=True) -> float:
    """Swaption value only (see black_swaption for the breakdown)."""
    return black_swaption(notional, K, T0, swap_tenor, freq, vol, discount, payer)["value"]


def black_vega(notional, K, T0, swap_tenor, freq, vol, discount) -> float:
    """dV/dvol (same for payer and receiver)."""
    A = annuity_factor(discount, T0, swap_tenor, freq)
    S0 = forward_swap_rate(discount, T0, swap_tenor, freq)
    sd = vol * np.sqrt(T0)
    d_plus = (np.log(S0 / K) + 0.5 * sd ** 2) / sd
    return float(notional * A * S0 * norm.pdf(d_plus) * np.sqrt(T0))


# ------------------------------------------------------------------ run
if __name__ == "__main__":
    from curve import SofrCurve

    curve = SofrCurve()
    spec = dict(notional=100.0, K=0.045, T0=2.0, swap_tenor=5.0, freq=0.5, vol=0.15)
    res = black_swaption(**spec, discount=curve.discount, payer=True)

    print("European payer swaption: 2y into 5y, K = 4.5%, semi-annual, vol 15%, N = $100\n")
    print(curve.table(np.r_[spec["T0"], res["payment_times"]]).to_string(
        formatters={"y(T)": "{:.4%}".format, "P(0,T)": "{:.6f}".format}))
    print()
    print(f"Annuity A            = {res['annuity']:.6f}")
    print(f"Forward swap rate S0 = {res['forward_swap_rate']:.4%}")
    print(f"d+ = {res['d_plus']:.6f}   d- = {res['d_minus']:.6f}")
    print(f"N(d+) = {res['N(d_plus)']:.6f}   N(d-) = {res['N(d_minus)']:.6f}")
    print(f"Payer swaption value = ${res['value']:.4f}")
    print(f"  intrinsic N A (S0-K) = ${res['intrinsic']:.4f}, time value = ${res['time_value']:.4f}")
    rec = black_swaption(**spec, discount=curve.discount, payer=False)
    print(f"Receiver swaption    = ${rec['value']:.4f}")
    print(f"Parity check payer - receiver = {res['value'] - rec['value']:.4f} "
          f"vs N A (S0 - K) = {spec['notional'] * res['annuity'] * (res['forward_swap_rate'] - spec['K']):.4f}")
    print(f"Vega (per 1.00 vol)  = {black_vega(**spec, discount=curve.discount):.4f}")
