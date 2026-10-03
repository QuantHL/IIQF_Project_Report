"""
Part I(d): par swap rate from a strip of zero-coupon bond prices.

For a swap with fixed payments at T_1..T_n (accruals tau_i) and a floating
leg on SOFR, the floating leg is worth 1 - P(0,T_n) per unit notional, so

    S = (1 - P(0,T_n)) / sum_i tau_i P(0,T_i)

With annual payments (tau_i = 1) this is the formula in the brief:
    S = (1 - Z(N)) / sum_{i=1}^N Z(i)

Usage:
    python src/swaps.py
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from vasicek import zcb_price


def payment_times(maturity: float, freq: float = 1.0, start: float = 0.0) -> np.ndarray:
    """Fixed-leg payment dates start+freq, ..., start+maturity."""
    n = int(round(maturity / freq))
    return start + freq * np.arange(1, n + 1)


def annuity(discount: Callable, times, freq: float = 1.0) -> float:
    """PV of 1 per year paid on the fixed leg: sum tau_i P(0,T_i)."""
    return float(np.sum(freq * discount(np.asarray(times))))


def par_swap_rate(discount: Callable, maturity: float, freq: float = 1.0,
                  start: float = 0.0) -> float:
    """
    Par (forward) swap rate for a swap from `start` to `start + maturity`.
    discount(T) must return P(0,T) (vectorised). For start = 0, P(0,0) = 1.
    """
    times = payment_times(maturity, freq, start)
    P_start = 1.0 if start == 0 else float(discount(start))
    return (P_start - float(discount(times[-1]))) / annuity(discount, times, freq)


def swap_rate_from_zcbs(Z) -> float:
    """S = (1 - Z(N)) / sum Z(i) for a list of annual ZCB prices Z(1..N)."""
    Z = np.asarray(Z, dtype=float)
    return (1 - Z[-1]) / Z.sum()


def swap_value(discount: Callable, fixed_rate: float, maturity: float,
               freq: float = 1.0, notional: float = 1.0, payer: bool = True) -> float:
    """Value at t=0 of a payer (pay fixed, receive float) or receiver swap."""
    times = payment_times(maturity, freq)
    float_leg = 1.0 - float(discount(times[-1]))
    fixed_leg = fixed_rate * annuity(discount, times, freq)
    v = notional * (float_leg - fixed_leg)
    return v if payer else -v


def vasicek_discount(r0: float, a: float, b: float, sigma: float) -> Callable:
    """P(0,T) as a function of T from the Vasicek closed form."""
    return lambda T: zcb_price(r0, 0.0, T, a, b, sigma)


def vasicek_swap_rate(r0: float, a: float, b: float, sigma: float,
                      maturity: float = 5.0, freq: float = 1.0) -> dict:
    """Swap rate with Vasicek ZCB prices; returns the ZCB strip as well."""
    times = payment_times(maturity, freq)
    Z = zcb_price(r0, 0.0, times, a, b, sigma)
    S = (1 - Z[-1]) / np.sum(freq * Z)
    return {"times": times, "Z": Z, "sum_Z": float(np.sum(Z)), "swap_rate": float(S)}


# ------------------------------------------------------------------ run
if __name__ == "__main__":
    from calibration import fetch_fed_rates, calibrate_vasicek

    params = calibrate_vasicek(fetch_fed_rates())
    a, b, sigma = params["a"], params["b"], params["sigma"]
    r0 = 0.04

    res = vasicek_swap_rate(r0, a, b, sigma, maturity=5.0, freq=1.0)
    print(f"a = {a:.4f}, b = {b:.4%}, sigma = {sigma:.4%}, r(0) = {r0:.2%}\n")
    print(" i    Z(0,i)")
    for t, z in zip(res["times"], res["Z"]):
        print(f"{t:2.0f}  {z:.6f}")
    print(f"sum   {res['sum_Z']:.6f}")
    print(f"\n5y annual par swap rate S = (1 - Z(5)) / sum Z(i) = {res['swap_rate']:.4%}")
    disc = vasicek_discount(r0, a, b, sigma)
    print(f"check: swap value at S = {swap_value(disc, res['swap_rate'], 5.0):.2e}")
