"""
Part II(a): SOFR discount curve with linear interpolation of zero rates.

Short end (T <= 1y): CME Term SOFR, simple ACT/360 rates
    y(T) = ln(1 + R T 365/360) / T
Long end (T >= 2y): SOFR OIS par swap rates (annual fixed leg), bootstrapped
    P(0,n) = (1 - S_n sum_{i<n} P(0,i)) / (1 + S_n),   y(n) = -ln P(0,n) / n
Unquoted annual swap tenors (4y, 6y, 8y, 9y) are linearly interpolated in S,
and P(0,1) comes from the 1y Term SOFR rate.
Between nodes, zero rates are linearly interpolated (flat extrapolation), and
    P(0,T) = exp(-y(T) T).

Usage:
    python src/curve.py
    python src/curve.py --csv data/sofr_2026-10-01.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Market data, 1 Oct 2026 (report Table 4)
TERM_SOFR = {1 / 12: 0.0391628, 3 / 12: 0.0404730,
             6 / 12: 0.0422578, 1.0: 0.0454462}                # CME Term SOFR
SWAP_RATES = {2.0: 0.0459, 3.0: 0.0468, 5.0: 0.0471,
              7.0: 0.0475, 10.0: 0.0483}                        # SOFR OIS swap rates


class SofrCurve:
    """Zero curve on bootstrapped nodes; linear interpolation in zero rates."""

    def __init__(self, term_sofr: dict = TERM_SOFR, swap_rates: dict = SWAP_RATES):
        self.term_sofr = dict(sorted(term_sofr.items()))
        self.swap_rates = dict(sorted(swap_rates.items()))
        zero = {}

        # Short end: simple ACT/360 rate -> continuously compounded zero rate
        for T, R in self.term_sofr.items():
            zero[T] = np.log(1 + R * 365 * T / 360) / T

        # Long end: bootstrap annual par swap rates
        years = np.arange(1, int(max(self.swap_rates)) + 1)
        quoted = list(self.swap_rates)
        S = np.interp(years, quoted, [self.swap_rates[t] for t in quoted])
        P = {1: np.exp(-zero[1.0])}
        for n, S_n in zip(years[1:], S[1:]):
            annuity = sum(P[i] for i in range(1, n))
            P[n] = (1 - S_n * annuity) / (1 + S_n)
            zero[float(n)] = -np.log(P[n]) / n

        self.par_swap_rates = dict(zip(years.astype(float), S))  # incl. interpolated
        self.tenors = np.array(sorted(zero))
        self.zeros = np.array([zero[t] for t in self.tenors])

    # -------------------------------------------------------------- queries
    def zero_rate(self, T):
        """Continuously compounded zero rate y(T), linear interpolation."""
        return np.interp(T, self.tenors, self.zeros)

    def discount(self, T):
        """Discount factor P(0,T) = exp(-y(T) T)."""
        T = np.asarray(T, dtype=float)
        return np.exp(-self.zero_rate(T) * T)

    __call__ = discount

    def forward_rate(self, T1, T2):
        """Continuously compounded forward rate between T1 and T2."""
        return np.log(self.discount(T1) / self.discount(T2)) / (np.asarray(T2) - T1)

    def nodes(self) -> pd.DataFrame:
        """Bootstrapped nodes: tenor, source, zero rate, discount factor."""
        src = ["Term SOFR" if t in self.term_sofr
               else "OIS swap" if t in self.swap_rates
               else "OIS swap (interp.)" for t in self.tenors]
        return pd.DataFrame({"source": src, "zero_rate": self.zeros,
                             "discount": self.discount(self.tenors)},
                            index=pd.Index(self.tenors, name="T"))

    def table(self, times) -> pd.DataFrame:
        times = np.asarray(times, dtype=float)
        return pd.DataFrame({"y(T)": self.zero_rate(times), "P(0,T)": self.discount(times)},
                            index=pd.Index(times, name="T"))

    # -------------------------------------------------------------- I/O
    @classmethod
    def from_csv(cls, path) -> "SofrCurve":
        """CSV with columns tenor_years, rate_pct, instrument ('term_sofr' or 'ois')."""
        df = pd.read_csv(path)
        ts = df[df.instrument == "term_sofr"]
        ois = df[df.instrument == "ois"]
        return cls(dict(zip(ts.tenor_years, ts.rate_pct / 100)),
                   dict(zip(ois.tenor_years, ois.rate_pct / 100)))

    def to_csv(self, path):
        rows = [(t, r * 100, "term_sofr") for t, r in self.term_sofr.items()] + \
               [(t, r * 100, "ois") for t, r in self.swap_rates.items()]
        pd.DataFrame(rows, columns=["tenor_years", "rate_pct", "instrument"]).to_csv(path, index=False)


def build_sofr_curve(term_sofr: dict = TERM_SOFR, swap_rates: dict = SWAP_RATES):
    """Returns: function discount(T) -> P(0,T)."""
    return SofrCurve(term_sofr, swap_rates).discount


# ------------------------------------------------------------------ run
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build the SOFR discount curve")
    p.add_argument("--csv", default=None, help="market data CSV (default: built-in 1 Oct 2026)")
    args = p.parse_args()

    curve = SofrCurve.from_csv(args.csv) if args.csv else SofrCurve()
    fmt = {"zero_rate": "{:.4%}".format, "discount": "{:.6f}".format,
           "y(T)": "{:.4%}".format, "P(0,T)": "{:.6f}".format}
    print("Curve nodes")
    print(curve.nodes().to_string(formatters=fmt))
    print("\nSwaption payment grid")
    print(curve.table(np.arange(2.0, 7.01, 0.5)).to_string(formatters=fmt))
