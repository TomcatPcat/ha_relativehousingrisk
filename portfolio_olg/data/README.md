# Calibration / survey moments

Do **not** commit confidential StatCan microdata. Store absolute paths in
`local_paths.yaml` (gitignored). Start from `local_paths.example.yaml`.

## Paths

| Key | Content |
|-----|---------|
| `sfs_panel` | SFS EFAM panel CSV (1999, 2005, 2012, 2016, 2019, 2023) |
| `shs_ry2023` | SHS 2023 PUMF root (`Data/TXT/PUMF_SHS_2023.txt`) |
| `shs_edm` | SHS EDM root (`2017/data/RAW.zip`, `2019/shs_2019.csv`) |
| `shs_consolidated` | Pre-redesign SHS 1997–2009 consolidated CSV |

Environment overrides: `SFS_PANEL_PATH`, `SHS_RY2023_PATH`, `SHS_EDM_PATH`,
`SHS_CONSOLIDATED_PATH`.

## SFS wealth identity (ex-auto)

```
nw = housing_gross - housing_debt + financial_gross - financial_debt
   = housing_net + financial_net
```

Autos (`auto_assets`, `auto_debt`, `auto_net`) are stored for robustness but
excluded from primary `nw`. Moments by year:

- homeownership
- housing share of NW (gross and net)
- financial share of NW (gross and net)
- aggregate NW / net income (`pefatinc > 0`)

Run: `python scripts/run_survey_moments.py` → `output/survey/*.csv`.

## SHS consumption / income

- Tier A (2017, 2019, 2023): `TC001 / (HHTOTINC - TX010)`
- Tier B (1997–2009): `consumption / net_income`
- Do **not** splice Tier A and B into one trend (SHS redesign break).
- Note: the HRAM2 consolidated Tier B file has near-zero `consumption` in
  **2001–2002** (data artifact); treat those years with caution.

## Planned later

- `targets.yaml` filled from the CSVs above
- `estimate.py` SMM objective + optimizer
