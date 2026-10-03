"""
Part I(b), I(e): closed-form Vasicek pricing.

Model:  dr = a(b - r) dt + sigma dW   (risk-neutral)
ZCB:    P(t,T) = A(t,T) exp(-B(t,T) r(t))

    B(t,T) = (1 - e^{-a(T-t)}) / a
    A(t,T) = exp[(b - sigma^2/(2a^2)) (B - (T-t)) - sigma^2 B^2 / (4a)]

Also the closed-form European option on a ZCB (Jamshidian 1989), used to
validate the Monte Carlo price of part I(e).

Usage:
    python src/vasicek.py          # calibrate to DFF, price the 5y ZCB
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


# ------------------------------------------------------------------ ZCB
def B(t: float, T: float, a: float) -> float:
    """Duration sensitivity B(t,T) = (1 - exp(-a(T-t))) / a."""
    tau = np.asarray(T, dtype=float) - t
    return (1 - np.exp(-a * tau)) / a


def A(t: float, T: float, a: float, b: float, sigma: float) -> float:
    """Factor A(t,T) capturing the long-run level b and volatility sigma."""
    tau = np.asarray(T, dtype=float) - t
    B_tT = B(t, T, a)
    exponent = (
        (b - sigma ** 2 / (2 * a ** 2)) * (B_tT - tau)
        - sigma ** 2 / (4 * a) * B_tT ** 2
    )
    return np.exp(exponent)


def zcb_price(r_t: float, t: float, T: float, a: float, b: float, sigma: float,
              face_value: float = 1.0) -> float | np.ndarray:
    """Analytical zero-coupon bond price P(t,T) under the Vasicek model.

    r_t may be an array (e.g. simulated short rates at time t)."""
    return face_value * A(t, T, a, b, sigma) * np.exp(-B(t, T, a) * np.asarray(r_t))


def zero_rate(r_t: float, t: float, T: float, a: float, b: float,
              sigma: float) -> float | np.ndarray:
    """Continuously compounded zero rate y(t,T) = -ln P(t,T) / (T-t)."""
    tau = np.asarray(T, dtype=float) - t
    return -np.log(zcb_price(r_t, t, T, a, b, sigma)) / tau


def long_rate(a: float, b: float, sigma: float) -> float:
    """Limit of y(0,T) as T -> infinity: b - sigma^2 / (2a^2)."""
    return b - sigma ** 2 / (2 * a ** 2)


def term_structure(r0: float, maturities, a: float, b: float, sigma: float):
    """Table of P(0,T) and y(0,T) for a list of maturities (pandas DataFrame)."""
    import pandas as pd
    T = np.asarray(maturities, dtype=float)
    return pd.DataFrame({"T": T,
                         "P(0,T)": zcb_price(r0, 0.0, T, a, b, sigma),
                         "y(0,T)": zero_rate(r0, 0.0, T, a, b, sigma)}).set_index("T")


# ------------------------------------------------------------------ ZCB option
def zcb_option_price(r0: float, T_opt: float, T_bond: float, K: float,
                     a: float, b: float, sigma: float, face_value: float = 1.0,
                     call: bool = True) -> float:
    """
    Closed-form price at t=0 of a European option expiring at T_opt on a ZCB
    maturing at T_bond (> T_opt) with the given face value, strike K (in $).

        sigma_P = sigma/a (1 - e^{-a(T_bond-T_opt)}) sqrt((1 - e^{-2a T_opt}) / 2a)
        h       = ln(L P(0,T_bond) / (K P(0,T_opt))) / sigma_P + sigma_P / 2
        Call    = L P(0,T_bond) N(h) - K P(0,T_opt) N(h - sigma_P)
    """
    P_S = zcb_price(r0, 0.0, T_bond, a, b, sigma)
    P_T = zcb_price(r0, 0.0, T_opt, a, b, sigma)
    sig_p = sigma / a * (1 - np.exp(-a * (T_bond - T_opt))) \
        * np.sqrt((1 - np.exp(-2 * a * T_opt)) / (2 * a))
    h = np.log(face_value * P_S / (K * P_T)) / sig_p + 0.5 * sig_p
    if call:
        return face_value * P_S * norm.cdf(h) - K * P_T * norm.cdf(h - sig_p)
    return K * P_T * norm.cdf(-h + sig_p) - face_value * P_S * norm.cdf(-h)


# ------------------------------------------------------------------ run
if __name__ == "__main__":
    from calibration import fetch_fed_rates, calibrate_vasicek

    # Part (a): fit a, b, sigma to Fed rate data
    r = fetch_fed_rates("DFF")
    fit = calibrate_vasicek(r)
    a, b, sigma = fit["a"], fit["b"], fit["sigma"]
    r0 = r.iloc[-1]

    # Part (b): 5-year zero-coupon bond, face value $1
    t, T = 0.0, 5.0
    print(f"Data up to {r.index[-1].date()}")
    print(f"a = {a:.4f}, b = {b:.4%}, sigma = {sigma:.4%}, r(0) = {r0:.4%}")
    print(f"B(0,5) = {B(t, T, a):.6f}")
    print(f"A(0,5) = {A(t, T, a, b, sigma):.6f}")
    print(f"P(0,5) = {zcb_price(r0, t, T, a, b, sigma):.6f}   (r0 = latest DFF)")
    print(f"P(0,5) = {zcb_price(0.04, t, T, a, b, sigma):.6f}   (r0 = 4%)")
    print()
    print(term_structure(0.04, [1, 2, 3, 4, 5, 7, 10], a, b, sigma).to_string(
        formatters={"P(0,T)": "{:.6f}".format, "y(0,T)": "{:.4%}".format}))
