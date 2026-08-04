# HAmodel

Two related heterogeneous-agent (HA) research projects in one repo.

| Folder | Focus | Stack | Agent |
|--------|--------|--------|--------|
| [`portfolio_olg/`](portfolio_olg/) | Canadian **housing stock in utility** + **liquid financial wealth**, correlated persistent return risk, OLG inequality panels (HARK safe/risky demos are legacy) | Python + Numba (+ optional HARK) | Agent A |
| [`housing_mortgage/`](housing_mortgage/) | Life-cycle liquid savings + durable/housing, then mortgage extension | Python + [EconModel](https://github.com/NumEconCopenhagen/EconModel) / [ConSav](https://github.com/NumEconCopenhagen/ConsumptionSaving) + Druedahl DurableConsumptionModel | Agent B |

## Setup

Shared conda env (optional):

```bash
conda env create -f environment.yml
conda activate hamodel
```

Or install each project's `requirements.txt` into separate environments if dependencies conflict.

**Note:** On Python 3.9, pin `econ-ark==0.14.1` for legacy HARK demos. Newer HARK releases may require Python 3.10+.

## Conventions

- Each subproject is self-contained (`README.md`, `requirements.txt`, `src/`, `notebooks/`, `data/`).
- Calibration datasets are deferred; keep raw data out of git.
- Upstream licenses (MIT) are attributed in each subproject README.

## Status

See each subfolder README for milestones.
