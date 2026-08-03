# housing_mortgage — Durable / Housing + Liquid (+ Mortgage)

Life-cycle (finite horizon) consumption-saving with a **durable / housing-like stock** and **liquid resources**, based on Jeppe Druedahl's DurableConsumptionModel (ConSav). A **mortgage** state is designed and scaffolded on top; full LTV / amortization dynamics are incremental.

Not a full OLG demographic model in v1 — finite life-cycle only.

## Stack

- Python 3.9+ (3.10+ recommended)
- [`EconModel`](https://github.com/NumEconCopenhagen/EconModel), [`ConSav`](https://github.com/NumEconCopenhagen/ConsumptionSaving)
- Vendored example from [ConsumptionSavingNotebooks / DurableConsumptionModel](https://github.com/NumEconCopenhagen/ConsumptionSavingNotebooks/tree/master/02.%20DurableConsumptionModel)

Upstream: MIT License (NumEconCopenhagen / Druedahl).

## Setup

```bash
cd housing_mortgage
python -m pip install -r requirements.txt
```

Numba is required. Optional C++ acceleration in the upstream example is **not** required for the Python/Numba path used here.

## Layout

```text
housing_mortgage/
  vendor/DurableConsumptionModel/   # upstream example (attribution preserved)
  src/                              # wrappers, housing map, mortgage design/scaffold
  notebooks/
  scripts/
  data/
  output/
  docs/MORTGAGE_DESIGN.md
```

## Milestones

1. Run original durable + liquid life-cycle solve/simulate
2. Document mapping durable → housing PE (limits: no LTV/mortgage yet)
3. Design + scaffold mortgage state on durable+liquid
4. Optional house-price / return AR(1) later
5. Calibration (deferred)

## Quick start

```bash
python scripts/run_durable_baseline.py
python scripts/run_mortgage_scaffold.py
```
