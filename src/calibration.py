"""
Part I(a): calibrate the Vasicek short-rate model to Federal Reserve rate data.

Model:  dr = a(b - r) dt + sigma dW
Exact discretisation over a step dt is an AR(1) process

    r_{t+dt} = alpha * r_t + beta + eps,   eps ~ N(0, s^2)

with alpha = exp(-a dt), beta = b (1 - alpha), s^2 = sigma^2 (1 - alpha^2) / (2a).
OLS on (r_t, r_{t+dt}) gives alpha, beta, s; inverting the map gives a, b, sigma.

The loader is generic: any FRED series id, any CSV URL or local file with
(date, rate) columns, and any [start, end] window.

Usage:
    python src/calibration.py                       # DFF, 2000-01-01 to today
    python src/calibration.py --series DGS3MO --start 2015-01-01
    python src/calibration.py --source data/my_rates.csv --end 2024-12-31
"""
from __future__ import annotations

import argparse
import datetime as dt_
from pathlib import Path

import numpy as np
import pandas as pd

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


# ------------------------------------------------------------------ data
def _to_series(df: pd.DataFrame, start=None, end=None, percent=True) -> pd.Series:
    """First two columns (date, rate) -> sorted decimal rate series."""
    df = df.iloc[:, :2].copy()
    df.columns = ["date", "rate"]
    df["date"] = pd.to_datetime(df["date"])
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")    # FRED uses '.'
    r = df.dropna().set_index("date").sort_index()["rate"]
    if start:
        r = r.loc[start:]
    if end:
        r = r.loc[:end]
    return r / 100.0 if percent else r


def load_rates(source: str, start: str | None = None, end: str | None = None,
               percent: bool = True) -> pd.Series:
    """Read a 2-column CSV (date, rate) from a URL or file. Returns decimals."""
    return _to_series(pd.read_csv(source), start, end, percent)


def fetch_fed_rates(series: str = "DFF", start: str | None = "2000-01-01",
                    end: str | None = None, cache: bool = True) -> pd.Series:
    """
    Download a FRED series and return it as a decimal rate.

    With cache=True a timestamped copy is written to data/. If the download
    fails (no network), the most recent cached pull of the series is used.
    """
    try:
        raw = pd.read_csv(FRED_URL.format(series=series))
        if cache:
            DATA_DIR.mkdir(exist_ok=True)
            raw.to_csv(DATA_DIR / f"{series}_{dt_.date.today().isoformat()}.csv", index=False)
    except Exception as exc:                                     # offline
        cached = sorted(DATA_DIR.glob(f"{series}_*.csv"))
        if not cached:
            raise RuntimeError(f"Cannot download {series} and no cached copy found") from exc
        print(f"[fetch_fed_rates] download failed, using cached {cached[-1].name}")
        raw = pd.read_csv(cached[-1])
    return _to_series(raw, start, end)


def infer_dt(r: pd.Series) -> float:
    """Time step in years, from the spacing of the dates."""
    gaps = np.diff(r.index.values).astype("timedelta64[D]").astype(float)
    med = np.median(gaps)
    if med <= 1.5:                                  # daily
        # business-day series (weekend gaps) vs calendar-day series (DFF)
        return 1 / 252 if np.mean(gaps > 1.5) > 0.1 else 1 / 365
    if med <= 8:
        return 1 / 52                               # weekly
    if med <= 32:
        return 1 / 12                               # monthly
    return 1 / 4                                    # quarterly


# ------------------------------------------------------------------ fit
def calibrate_vasicek(rates: pd.Series | np.ndarray, dt: float | None = None) -> dict:
    """
    OLS calibration of Vasicek parameters using the exact AR(1) form
        r_{t+dt} = alpha * r_t + beta + eps,   eps ~ N(0, s^2).

    Returns a dict with keys: a, b, sigma, alpha, beta, s, dt, half_life,
    r_last, n_obs.
    """
    if dt is None:
        dt = infer_dt(rates) if isinstance(rates, pd.Series) else 1 / 365
    r = np.asarray(rates, dtype=float)
    x, y = r[:-1], r[1:]                                        # r_t and r_{t+dt}

    X = np.column_stack([x, np.ones_like(x)])
    (alpha, beta), *_ = np.linalg.lstsq(X, y, rcond=None)       # regression
    s2 = (y - (alpha * x + beta)).var(ddof=2)                   # residual variance

    if not 0 < alpha < 1:
        raise ValueError(f"alpha = {alpha:.6f}: no mean reversion in this window.")

    # Map AR(1) coefficients back to Vasicek parameters
    a = -np.log(alpha) / dt
    b = beta / (1 - alpha)
    sigma = np.sqrt(2 * a * s2 / (1 - alpha ** 2))
    return dict(a=a, b=b, sigma=sigma, alpha=alpha, beta=beta, s=np.sqrt(s2),
                dt=dt, half_life=np.log(2) / a, r_last=r[-1], n_obs=len(r))


def rolling_calibration(rates: pd.Series, window_years: float = 10,
                        step: str = "YE") -> pd.DataFrame:
    """Re-calibrate on a trailing window at each period end (parameter stability)."""
    dt = infer_dt(rates)
    rows = []
    for end in rates.resample(step).last().index:
        win = rates.loc[end - pd.DateOffset(days=int(365.25 * window_years)):end]
        if len(win) < 50:
            continue
        try:
            p = calibrate_vasicek(win, dt)
            rows.append(dict(end=end, a=p["a"], b=p["b"], sigma=p["sigma"]))
        except ValueError:
            rows.append(dict(end=end, a=np.nan, b=np.nan, sigma=np.nan))
    return pd.DataFrame(rows).set_index("end")


# ------------------------------------------------------------------ "values at any point"
def expected_rate(r0: float, t: float, a: float, b: float) -> float | np.ndarray:
    """E[r(t) | r(0)] = r0 e^{-at} + b (1 - e^{-at})."""
    t = np.asarray(t, dtype=float)
    return r0 * np.exp(-a * t) + b * (1 - np.exp(-a * t))


def rate_std(t: float, a: float, sigma: float) -> float | np.ndarray:
    """Std[r(t) | r(0)] = sigma sqrt((1 - e^{-2at}) / 2a)."""
    t = np.asarray(t, dtype=float)
    return sigma * np.sqrt((1 - np.exp(-2 * a * t)) / (2 * a))


def rate_at(rates: pd.Series, date) -> float:
    """Observed rate on any calendar date (last available value on or before it)."""
    return float(rates.asof(pd.Timestamp(date)))


# ------------------------------------------------------------------ run
def main(argv=None):
    p = argparse.ArgumentParser(description="Calibrate Vasicek to a FRED / CSV rate series")
    p.add_argument("--series", default="DFF", help="FRED series id (default DFF)")
    p.add_argument("--source", default=None, help="CSV URL or file instead of FRED")
    p.add_argument("--start", default="2000-01-01")
    p.add_argument("--end", default=None)
    args = p.parse_args(argv)

    if args.source:
        r = load_rates(args.source, args.start, args.end)
    else:
        r = fetch_fed_rates(args.series, args.start, args.end)
    fit = calibrate_vasicek(r)

    print(f"Series   : {args.source or args.series}  "
          f"({r.index[0].date()} to {r.index[-1].date()}, {fit['n_obs']} obs, dt = 1/{1/fit['dt']:.0f})")
    print(f"alpha    = {fit['alpha']:.6f}   beta = {fit['beta']:.3e}   s = {fit['s']:.3e}")
    print(f"a        = {fit['a']:.4f}  (half-life {fit['half_life']:.2f} y)")
    print(f"b        = {fit['b']:.4%}")
    print(f"sigma    = {fit['sigma']:.4%}")
    print(f"r(last)  = {fit['r_last']:.4%}")
    for h in (1, 2, 5, 10):
        print(f"E[r({h:>2}y)] = {expected_rate(fit['r_last'], h, fit['a'], fit['b']):.4%}"
              f"  +/- {rate_std(h, fit['a'], fit['sigma']):.4%}")
    return fit


if __name__ == "__main__":
    main()
