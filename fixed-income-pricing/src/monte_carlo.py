"""
Part I(c), I(e): Monte Carlo under the Vasicek model.

Paths use the exact discretisation (no time-step bias at grid points):

    r_{k+1} = alpha r_k + beta + s Z_k,     Z_k ~ N(0,1)
    alpha = e^{-a dt},  beta = b(1 - alpha),  s^2 = sigma^2 (1 - alpha^2) / (2a)

ZCB:   P(0,T) = E[exp(-int_0^T r ds)], integral by the trapezoidal rule.
Option on ZCB: simulate r(T_opt), price the bond at T_opt with the
exponential-affine formula, take max(P - K, 0) and discount.

Variance reduction: antithetic variates (Z and -Z paths, averaged per pair).

Usage:
    python src/monte_carlo.py
"""
from __future__ import annotations

import numpy as np

from vasicek import zcb_price, zcb_option_price


def _step_coeffs(a: float, b: float, sigma: float, dt: float):
    """alpha, beta, s of the exact AR(1) transition over dt."""
    alpha = np.exp(-a * dt)
    beta = b * (1 - alpha)
    s = sigma * np.sqrt((1 - np.exp(-2 * a * dt)) / (2 * a))
    return alpha, beta, s


def _summary(estimates: np.ndarray, n_paths_total: int) -> dict:
    price = estimates.mean()
    se = estimates.std(ddof=1) / np.sqrt(len(estimates))
    return {"price": price, "std_err": se,
            "ci_lower": price - 1.96 * se, "ci_upper": price + 1.96 * se,
            "n_paths": n_paths_total}


# ------------------------------------------------------------------ paths
def simulate_paths(r0: float, T: float, a: float, b: float, sigma: float,
                   n_paths: int = 20, n_steps: int = 1260,
                   seed: int | None = 42) -> tuple[np.ndarray, np.ndarray]:
    """Full short-rate paths (for plotting). Returns times (n_steps+1,), r (n_paths, n_steps+1)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    alpha, beta, s = _step_coeffs(a, b, sigma, dt)
    r = np.empty((n_paths, n_steps + 1))
    r[:, 0] = r0
    Z = rng.standard_normal((n_paths, n_steps))
    for k in range(n_steps):
        r[:, k + 1] = alpha * r[:, k] + beta + s * Z[:, k]
    return np.linspace(0.0, T, n_steps + 1), r


def simulate_rate_and_integral(r0: float, T: float, a: float, b: float, sigma: float,
                               n_paths: int, n_steps: int, rng: np.random.Generator,
                               antithetic: bool = True):
    """
    Step all paths to T keeping only the current rate and the running integral
    of r (memory O(n_paths)). With antithetic=True returns (r_T, I) for the +Z
    and -Z paths; otherwise for n_paths independent paths.
    """
    dt = T / n_steps
    alpha, beta, s = _step_coeffs(a, b, sigma, dt)
    signs = (1.0, -1.0) if antithetic else (1.0,)
    r = [np.full(n_paths, r0) for _ in signs]
    integral = [np.zeros(n_paths) for _ in signs]
    for _ in range(n_steps):
        Z = rng.standard_normal(n_paths)
        for j, sgn in enumerate(signs):
            r_new = alpha * r[j] + beta + sgn * s * Z
            integral[j] += 0.5 * (r[j] + r_new) * dt          # trapezoidal rule
            r[j] = r_new
    return r, integral


# ------------------------------------------------------------------ ZCB
def mc_zcb_price(r0: float, T: float, a: float, b: float, sigma: float,
                 n_paths: int = 50_000, n_steps: int = 1260,
                 seed: int | None = 42, antithetic: bool = True) -> dict:
    """
    Monte Carlo ZCB price P(0,T) under Vasicek.

    antithetic=True : n_paths antithetic pairs (2 * n_paths paths in total);
                      each estimate is the pair average.
    antithetic=False: n_paths independent paths (plain MC).
    Returns dict: price, std_err, ci_lower, ci_upper, n_paths.
    """
    rng = np.random.default_rng(seed)
    _, integral = simulate_rate_and_integral(r0, T, a, b, sigma, n_paths, n_steps,
                                             rng, antithetic)
    if antithetic:
        estimates = 0.5 * (np.exp(-integral[0]) + np.exp(-integral[1]))
        return _summary(estimates, 2 * n_paths)
    return _summary(np.exp(-integral[0]), n_paths)


def variance_reduction_factor(r0: float, T: float, a: float, b: float, sigma: float,
                              n_paths: int = 20_000, n_steps: int = 252,
                              seed: int = 7) -> dict:
    """
    Compare plain MC and antithetic MC at equal total number of paths.
    Factor = Var(plain estimator) / Var(antithetic estimator).
    """
    plain = mc_zcb_price(r0, T, a, b, sigma, 2 * n_paths, n_steps, seed, antithetic=False)
    anti = mc_zcb_price(r0, T, a, b, sigma, n_paths, n_steps, seed, antithetic=True)
    return {"plain": plain, "antithetic": anti,
            "factor": (plain["std_err"] / anti["std_err"]) ** 2}


# ------------------------------------------------------------------ option on ZCB
def mc_zcb_call(r0: float, a: float, b: float, sigma: float,
                T_opt: float = 4.0, T_bond: float = 5.0,
                face_value: float = 1000.0, K: float = 900.0,
                n_paths: int = 50_000, n_steps: int = 1008,
                seed: int | None = 42, antithetic: bool = True,
                discounting: str = "pathwise") -> dict:
    """
    European call on a ZCB:  payoff max(P(T_opt, T_bond) - K, 0) at T_opt.

    discounting:
      "pathwise" - multiply each payoff by exp(-int_0^T_opt r ds) on its own
                   path (exact risk-neutral price).
      "zcb"      - average payoff times P(0, T_opt) (the hint in the brief).
                   Ignores the covariance between the discount factor and
                   the payoff, so it is an approximation (bias of either sign).
      "none"     - undiscounted average payoff.
    Returns dict: price, std_err, ci_lower, ci_upper, mean_payoff, P0T, n_paths.
    """
    rng = np.random.default_rng(seed)
    r_T, integral = simulate_rate_and_integral(r0, T_opt, a, b, sigma, n_paths,
                                               n_steps, rng, antithetic)
    P0T = float(zcb_price(r0, 0.0, T_opt, a, b, sigma))

    def value(rT, I):
        payoff = np.maximum(zcb_price(rT, T_opt, T_bond, a, b, sigma, face_value) - K, 0.0)
        if discounting == "pathwise":
            return payoff, payoff * np.exp(-I)
        if discounting == "zcb":
            return payoff, payoff * P0T
        if discounting == "none":
            return payoff, payoff
        raise ValueError(f"unknown discounting '{discounting}'")

    vals = [value(rT, I) for rT, I in zip(r_T, integral)]
    payoffs = np.mean([v[0] for v in vals], axis=0)
    estimates = np.mean([v[1] for v in vals], axis=0)
    out = _summary(estimates, len(vals) * n_paths)
    out.update(mean_payoff=payoffs.mean(), P0T=P0T,
               prob_itm=np.mean([np.mean(v[0] > 0) for v in vals]))
    return out


# ------------------------------------------------------------------ run
if __name__ == "__main__":
    from calibration import fetch_fed_rates, calibrate_vasicek

    params = calibrate_vasicek(fetch_fed_rates())                       # part (a)
    a, b, sigma = params["a"], params["b"], params["sigma"]
    print(f"a = {a:.4f}, b = {b:.4%}, sigma = {sigma:.4%}\n")

    # Part (c): 5y ZCB with r(0) = 4%
    r0, T = 0.04, 5.0
    mc = mc_zcb_price(r0, T, a, b, sigma, n_paths=50_000, n_steps=1260)
    exact = zcb_price(r0, 0.0, T, a, b, sigma)
    print("Part I(c) - 5y ZCB, face $1, r(0) = 4%")
    print(f"  Analytical price : {exact:.6f}")
    print(f"  Monte Carlo price: {mc['price']:.6f} (s.e. {mc['std_err']:.2e}, "
          f"95% CI [{mc['ci_lower']:.6f}, {mc['ci_upper']:.6f}], {mc['n_paths']:,} paths)")
    vr = variance_reduction_factor(r0, T, a, b, sigma)
    print(f"  Antithetic variance reduction factor: {vr['factor']:.0f}x\n")

    # Part (e): call on 5y ZCB, face $1000, expiry 4y, strike $900
    print("Part I(e) - call on 5y ZCB, face $1000, expiry 4y, K = $900")
    for disc in ("pathwise", "zcb", "none"):
        c = mc_zcb_call(r0, a, b, sigma, discounting=disc)
        print(f"  MC ({disc:<8}): {c['price']:9.4f}  (s.e. {c['std_err']:.4f})")
    c = mc_zcb_call(r0, a, b, sigma, discounting="zcb")
    print(f"  mean payoff at 4y = {c['mean_payoff']:.4f}, P(0,4) = {c['P0T']:.6f}, "
          f"P(ITM) = {c['prob_itm']:.2%}")
    print(f"  Closed form      : {zcb_option_price(r0, 4.0, 5.0, 900.0, a, b, sigma, 1000.0):9.4f}")
