# Vasicek Short-Rate Model and Swaption Valuation

Vasicek calibration to Fed data, ZCB pricing (closed form and Monte Carlo), swap rate, ZCB option, SOFR curve and European swaption with Black's formula.

## Layout

```
IIQF_Project_Report/
|-- data/                     # Raw FRED / SOFR CSV pulls (timestamped)
|   |-- DFF_<date>.csv        #   written by fetch_fed_rates(); offline fallback
|   |-- sofr_2026-10-01.csv   #   SOFR market data (report Table 4)
|-- src/
|   |-- vasicek.py            # ZCB pricing: B, A, zcb_price(), closed-form ZCB option
|   |-- calibration.py        # OLS calibration from FRED data (generic URL / series / window)
|   |-- curve.py              # SOFR yield curve: bootstrap + linear interpolation
|   |-- monte_carlo.py        # MC simulation + antithetic variates, ZCB and ZCB call
|   |-- swaps.py              # Swap rate from ZCB strip
|   |-- swaptions.py          # Black's formula for swaptions
|-- notebooks/
|   |-- 01_calibration.ipynb             
|   |-- 02_term_structure.ipynb          
|   |-- 03_monte_carlo_validation.ipynb  
|   |-- 04_swaption_pricing.ipynb        
|-- tests/                    # Unit tests (analytical vs MC agreement)
|-- requirements.txt
|-- README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

Each module runs as a script (from the project root):

```bash
python src/calibration.py                                   
python src/calibration.py --series DGS3MO --start 2015-01-01  # any FRED series / window
python src/calibration.py --source path/or/url.csv --end 2024-12-31
python src/vasicek.py       
python src/monte_carlo.py    
python src/swaps.py          
python src/curve.py          
python src/swaptions.py      
pytest -q                    # tests
```

The notebooks add `../src` to `sys.path`; open them with `notebooks/` as the working directory.

