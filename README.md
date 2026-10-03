# Vasicek Short-Rate Model and Swaption Valuation

IIQF capstone project (CPFE). Vasicek calibration to Fed data, ZCB pricing (closed form and
Monte Carlo), swap rate, ZCB option, SOFR curve and European swaption with Black's formula.

## Layout

```
fixed-income-pricing/
|-- data/                     # Raw FRED / SOFR CSV pulls (timestamped)
|   |-- DFF_<date>.csv        #   written by fetch_fed_rates(); offline fallback
|   `-- sofr_2026-10-01.csv   #   SOFR market data (report Table 4)
|-- src/
|   |-- vasicek.py            # ZCB pricing: B, A, zcb_price(), closed-form ZCB option
|   |-- calibration.py        # OLS calibration from FRED data (generic URL / series / window)
|   |-- curve.py              # SOFR yield curve: bootstrap + linear interpolation
|   |-- monte_carlo.py        # MC simulation + antithetic variates, ZCB and ZCB call
|   |-- swaps.py              # Swap rate from ZCB strip
|   `-- swaptions.py          # Black's formula for swaptions
|-- notebooks/
|   |-- 01_calibration.ipynb             # Part I(a)
|   |-- 02_term_structure.ipynb          # Parts I(b), I(d)
|   |-- 03_monte_carlo_validation.ipynb  # Parts I(c), I(e)
|   `-- 04_swaption_pricing.ipynb        # Part II(a)-(d)
|-- tests/                    # Unit tests (analytical vs MC agreement, curve repricing, parity)
|-- requirements.txt
`-- README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

Each module runs as a script (from the project root):

```bash
python src/calibration.py                                   # Part I(a), DFF 2000 -> latest
python src/calibration.py --series DGS3MO --start 2015-01-01  # any FRED series / window
python src/calibration.py --source path/or/url.csv --end 2024-12-31
python src/vasicek.py        # Part I(b)
python src/monte_carlo.py    # Parts I(c), I(e)
python src/swaps.py          # Part I(d)
python src/curve.py          # Part II(a)
python src/swaptions.py      # Part II(b)
pytest -q                    # tests
```

The notebooks add `../src` to `sys.path`; open them with `notebooks/` as the working directory.

## Results (DFF 2000-01-01 to 2026-10-01, pulled 3 Oct 2026)

| Item | Value |
|---|---|
| a, b, sigma | 0.1900, 2.032%, 1.240% (half-life 3.65 y) |
| I(b) Z(0,5), r(0) = latest DFF 3.88% | 0.852497 |
| I(b)/(c) Z(0,5), r(0) = 4%, analytical | 0.849201 |
| I(c) Monte Carlo, 100k antithetic paths | 0.849210 (s.e. 9.0e-6); variance reduction ~290x |
| I(d) 5y annual swap rate, r(0) = 4% | 3.3405% |
| I(e) call on ZCB, face $1000, T = 4, K = $900 | MC pathwise 63.22, closed form 63.22, Z(0,4) x avg payoff 62.75 |
| II(b) payer swaption 2y x 5y, K = 4.5%, vol 15% | A = 4.0255, S0 = 4.7682%, **V = $2.1736** |

Calibrated parameters change with the window and with each new FRED pull
(e.g. 2010 -> latest gives a ~ 0.02, b ~ 10.7%: almost no mean reversion). See notebook 01.

## Modelling notes

* Exact AR(1) discretisation is used both for calibration and simulation (no Euler bias).
* Statistical parameters are used as risk-neutral (zero market price of risk).
* Vasicek allows negative rates (about 7% of simulated paths have r(5) < 0).
* SOFR curve: Term SOFR (ACT/360 simple) at the short end, annual OIS par swaps bootstrapped at
  the long end, linear interpolation of continuously compounded zero rates, single curve for
  discounting and projection.
