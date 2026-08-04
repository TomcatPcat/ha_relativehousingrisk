# portfolio_olg — Canadian Housing + Financial Portfolio (OLG)

Primary research path: **illiquid housing in utility** + **liquid financial wealth**, both with **persistent, correlated Canadian return risk**, life-cycle / OLG panels for inequality.

## Economic object

See [`docs/MODEL.md`](docs/MODEL.md).

- Housing stock \(h\) enters \(u(c,h)\); costly keep/adjust
- Liquid financial wealth earns \(R^f\) (Canada financial market placeholders)
- Housing faces \(R^h\) (Canada housing market placeholders)
- Joint Markov returns from bivariate AR(1) (`src/returns_ca.py`)
- Optional simulation layer: wealth-dependent financial volatility
- Cohort and mixed-age **panel** sims with drawn or provided initials
- Configurable **`DecisionSimulation`**: $\theta$ → solve → sims → placeholder moments (pre-SMM)

**Out of scope here:** mortgages (see `../housing_mortgage/`), GE prices.

## Stack

- Python 3.9+
- Numba (+ optional EconModel/ConSav utilities)
- Legacy demos still use HARK (`econ-ark`)

## Setup

```bash
cd portfolio_olg
python -m pip install -r requirements.txt
```

## Layout

```text
portfolio_olg/
  docs/MODEL.md
  src/
    age_grid.py             # chronological age ↔ period mapping
    returns_ca.py           # bivariate Canada returns
    housing_finance/        # DP, initials, cohort/panel sim, DecisionSimulation
      initial.py
      simulate.py
      decision_simulation.py
      model.py
    lifecycle_profiles.py   # age means + quantiles
    inequality.py
    agents.py               # LEGACY HARK builders
    ar1_returns.py          # shared AR(1) / Rouwenhorst helpers
  scripts/
    run_returns_ca.py
    run_housing_finance_baseline.py
    run_lifecycle_profiles.py
    run_housing_finance_olg.py
    run_wealth_vol.py
    run_baseline.py         # LEGACY HARK
    run_olg_inequality.py   # LEGACY HARK
    run_ar1_returns.py      # LEGACY HARK
```

## Quick start (primary)

Interactive checks (recommended):

```bash
jupyter notebook notebooks/02_run_and_check_housing_finance.ipynb
```

Or scripts:

```bash
python scripts/run_returns_ca.py
python scripts/run_housing_finance_baseline.py
python scripts/run_lifecycle_profiles.py   # cohort means + quantiles by age
python scripts/run_housing_finance_olg.py
python scripts/run_wealth_vol.py
```

## Legacy HARK demos

Safe vs single risky share (no housing in utility):

```bash
python scripts/run_baseline.py
python scripts/run_olg_inequality.py
python scripts/run_ar1_returns.py
```
