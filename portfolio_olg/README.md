# portfolio_olg — Safe/Risky Portfolio + OLG Life-Cycle (HARK)

Partial-equilibrium life-cycle / overlapping-generations households choose a **risk-free** and **risky** asset share. Large panel simulations track **wealth inequality over calendar time**.

## Stack

- Python 3.10+
- [HARK](https://github.com/econ-ark/HARK) via `pip install econ-ark`
- Classes: `RiskyAssetConsumerType` / `PortfolioConsumerType`

Upstream: MIT License (Econ-ARK).

## Setup

```bash
cd portfolio_olg
python -m pip install -r requirements.txt
```

## Layout

```text
portfolio_olg/
  src/           # builders, inequality metrics, AR(1) returns helpers
  notebooks/     # exploratory notebooks
  scripts/       # runnable demos (baseline, OLG inequality, AR1)
  data/          # placeholders for future calibration
  output/        # simulation figures / tables
```

## Milestones

1. Baseline infinite-horizon and life-cycle portfolio solve/simulate
2. OLG-style overlapping cohorts + inequality time series (PE prices)
3. Persistent idiosyncratic AR(1) risky returns
4. Calibration to external moments (deferred)

## Quick start

```bash
python scripts/run_baseline.py
python scripts/run_olg_inequality.py
python scripts/run_ar1_returns.py
```
