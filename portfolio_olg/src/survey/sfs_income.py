"""SFS income scaling and MIE age-band harmonization for price2income analyses."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .weighted import wmean, wquantile

# Harmonized MIE age bands (consistent across 1999–2023 waves)
AGE_BANDS: tuple[str, ...] = ("<25", "25-34", "35-44", "45-54", "55-64", "65+")

INCOME_DEFS: tuple[str, ...] = ("market", "net", "market_transfer")
DENOMS: tuple[str, ...] = ("hh", "per_earner", "per_capita")

# Analysis column names: {def}_{denom}
INCOME_MEASURES: tuple[str, ...] = tuple(
    f"inc_{d}_{k}" for d in INCOME_DEFS for k in DENOMS
)

INCOME_LABELS: dict[str, str] = {
    "inc_market_hh": "Market income (household)",
    "inc_market_per_earner": "Market income (per earner)",
    "inc_market_per_capita": "Market income (per capita)",
    "inc_net_hh": "Net income (household)",
    "inc_net_per_earner": "Net income (per earner)",
    "inc_net_per_capita": "Net income (per capita)",
    "inc_market_transfer_hh": "Market+transfers (household)",
    "inc_market_transfer_per_earner": "Market+transfers (per earner)",
    "inc_market_transfer_per_capita": "Market+transfers (per capita)",
}


def _age_band_from_continuous(age: pd.Series) -> pd.Series:
    a = pd.to_numeric(age, errors="coerce")
    out = pd.Series(pd.NA, index=age.index, dtype="object")
    out = out.mask(a < 25, "<25")
    out = out.mask((a >= 25) & (a < 35), "25-34")
    out = out.mask((a >= 35) & (a < 45), "35-44")
    out = out.mask((a >= 45) & (a < 55), "45-54")
    out = out.mask((a >= 55) & (a < 65), "55-64")
    out = out.mask(a >= 65, "65+")
    return out


def _age_band_from_agegrp(agegrp: pd.Series, scheme: str) -> pd.Series:
    """
    Map StatCan ``pagemieg`` codes to six bands.

    g14 (1999–2019): 1=16–19 … 14=80+ (five-year bands).
    g7 (2023): 1=<25, 2=25–34, …, 6=65–79, 7=80+.
    """
    g = pd.to_numeric(agegrp, errors="coerce")
    out = pd.Series(pd.NA, index=agegrp.index, dtype="object")
    if scheme == "g7":
        mapping = {
            1: "<25",
            2: "25-34",
            3: "35-44",
            4: "45-54",
            5: "55-64",
            6: "65+",
            7: "65+",
        }
    else:  # g14
        mapping = {
            1: "<25",
            2: "<25",
            3: "25-34",
            4: "25-34",
            5: "35-44",
            6: "35-44",
            7: "45-54",
            8: "45-54",
            9: "55-64",
            10: "55-64",
            11: "65+",
            12: "65+",
            13: "65+",
            14: "65+",
        }
    for code, label in mapping.items():
        out = out.mask(g == code, label)
    return out


def harmonize_mie_age_band(df: pd.DataFrame) -> pd.Series:
    """
    Harmonized MIE age band: ``<25``, ``25-34``, …, ``65+``.

    Prefer continuous ``age`` when available; otherwise ``agegrp`` with a
    year-level scheme detected by ``max(agegrp)`` (g14 vs g7).
    """
    if "year" not in df.columns:
        raise KeyError("harmonize_mie_age_band requires 'year'")

    out = pd.Series(pd.NA, index=df.index, dtype="object")
    has_age = "age" in df.columns
    has_grp = "agegrp" in df.columns

    for year, idx in df.groupby("year", sort=False).groups.items():
        sub = df.loc[idx]
        band = pd.Series(pd.NA, index=idx, dtype="object")

        if has_age and sub["age"].notna().any():
            band = _age_band_from_continuous(sub["age"])
            # Fill remaining with agegrp if needed
            need = band.isna()
            if need.any() and has_grp:
                gmax = pd.to_numeric(sub.loc[need, "agegrp"], errors="coerce").max()
                scheme = "g7" if (np.isfinite(gmax) and gmax <= 7) else "g14"
                fill = _age_band_from_agegrp(sub.loc[need, "agegrp"], scheme)
                band = band.where(~need, fill)
        elif has_grp:
            gmax = pd.to_numeric(sub["agegrp"], errors="coerce").max()
            scheme = "g7" if (np.isfinite(gmax) and gmax <= 7) else "g14"
            band = _age_band_from_agegrp(sub["agegrp"], scheme)
        out.loc[idx] = band.to_numpy()

    return out.astype("object")


def add_income_measures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add market / net / market+transfer incomes and hh / per-earner / per-capita.

    - ``inc_market`` = loader ``inc_gross`` (StatCan market income)
    - ``inc_net`` = after-tax
    - ``inc_market_transfer`` = market + government transfers
    - Earners: ``dem_num_earners`` with 3+ top-code kept as 3; code 9 / missing → NA
    - Capita: ``dem_fam_size``; code 9 / missing → NA
    - Per-earner series are NA when ``n_earners < 1``
    """
    out = df.copy()

    market = pd.to_numeric(out.get("inc_gross", np.nan), errors="coerce")
    net = pd.to_numeric(out.get("inc_net", np.nan), errors="coerce")
    transfers = pd.to_numeric(out.get("inc_transfers", np.nan), errors="coerce")
    # Residual codes already scrubbed in load_sfs; keep NaN (do not fill).
    out["inc_market"] = market
    out["inc_net"] = net
    out["inc_market_transfer"] = market + transfers.fillna(0.0)
    # If market missing, market+transfer is missing even if transfers known
    out.loc[market.isna(), "inc_market_transfer"] = np.nan

    earners = pd.to_numeric(out.get("dem_num_earners", np.nan), errors="coerce")
    # StatCan: 9 = not stated
    earners = earners.where((earners >= 0) & (earners <= 3))
    out["n_earners"] = earners.clip(upper=3)

    fam = pd.to_numeric(out.get("dem_fam_size", np.nan), errors="coerce")
    fam = fam.where((fam >= 1) & (fam < 9))  # drop not-stated (9)
    out["n_capita"] = fam

    out["age_band"] = harmonize_mie_age_band(out)
    out["age_2534"] = out["age_band"] == "25-34"

    for d in INCOME_DEFS:
        y = out[f"inc_{d}"]
        out[f"inc_{d}_hh"] = y
        pe = y / out["n_earners"]
        pe = pe.where(out["n_earners"] >= 1)
        out[f"inc_{d}_per_earner"] = pe
        pc = y / out["n_capita"]
        pc = pc.where(out["n_capita"] >= 1)
        out[f"inc_{d}_per_capita"] = pc

    return out


def income_moments(
    df: pd.DataFrame,
    *,
    by: Union[str, Sequence[str], None] = "year",
    measures: Optional[Sequence[str]] = None,
    age_band: Optional[str] = None,
    age_2534_only: bool = False,
) -> pd.DataFrame:
    """
    Weighted mean and median of income measures by group.

    Parameters
    ----------
    by :
        Grouping column(s), default ``year``. Use ``(\"year\", \"age_band\")``
        for age-colored time series.
    age_band :
        If set, keep only that harmonized band (e.g. ``\"25-34\"``).
    age_2534_only :
        Shortcut for ``age_band=\"25-34\"``.
    """
    measures = list(measures) if measures is not None else list(INCOME_MEASURES)
    work = df
    if age_2534_only:
        age_band = "25-34"
    if age_band is not None:
        if "age_band" not in work.columns:
            raise KeyError("income_moments: run add_income_measures first")
        work = work.loc[work["age_band"] == age_band].copy()

    missing = [c for c in measures if c not in work.columns]
    if missing:
        raise KeyError(f"income_moments missing columns: {missing}")
    if "weight" not in work.columns:
        raise KeyError("income_moments missing weight")

    if by is None:
        group_keys: list[str] = []
        groups: Iterable[tuple] = [(None, work)]
    else:
        keys = [by] if isinstance(by, str) else list(by)
        group_keys = keys
        for k in keys:
            if k not in work.columns:
                raise KeyError(f"income_moments missing group column {k}")
        groups = work.groupby(keys, sort=True, dropna=False)

    rows: list[dict] = []
    for key, g in groups:
        w = g["weight"].to_numpy(dtype=float)
        if group_keys:
            if len(group_keys) == 1:
                gvals = {group_keys[0]: key[0] if isinstance(key, tuple) else key}
            else:
                vals = key if isinstance(key, tuple) else (key,)
                gvals = dict(zip(group_keys, vals))
        else:
            gvals = {}

        for col in measures:
            x = g[col].to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
            n_ok = int(ok.sum())
            base = {
                **gvals,
                "variable": col,
                "label": INCOME_LABELS.get(col, col),
                "n": n_ok,
            }
            if n_ok == 0:
                rows.append({**base, "stat": "mean", "value": float("nan")})
                rows.append({**base, "stat": "median", "value": float("nan")})
                continue
            rows.append({**base, "stat": "mean", "value": wmean(x[ok], w[ok])})
            med = float(np.atleast_1d(wquantile(x[ok], w[ok], [0.5]))[0])
            rows.append({**base, "stat": "median", "value": med})

    cols = group_keys + ["variable", "label", "stat", "value", "n"]
    return pd.DataFrame(rows)[cols]


def moments_wide(long: pd.DataFrame, *, index: Sequence[str] = ("year",)) -> pd.DataFrame:
    """Pivot long income moments to wide: one column per variable_stat."""
    idx = list(index)
    wide = long.pivot_table(
        index=idx,
        columns=["variable", "stat"],
        values="value",
        aggfunc="first",
    )
    wide.columns = [f"{v}_{s}" for v, s in wide.columns]
    return wide.reset_index()


def load_crea_spreadsheet_price_income_ratios(
    path: Optional[Union[str, Path]] = None,
    *,
    including_zeros: bool = False,
) -> pd.DataFrame:
    """
    Pre-computed CREA price / StatCan average income ratios from the workbook.

    Sheet ``Average Resale Price, CREA`` stores age-group ratios for
    15–24, 25–34, 35–44, 45–54 (excluding or including zero incomes).

    Returns a wide DataFrame indexed by year with columns named by age label.
    """
    if path is None:
        path = Path(__file__).resolve().parents[2] / "data" / "HousePriceIndices.xlsx"
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CREA house price workbook not found: {path}")

    raw = pd.read_excel(path, sheet_name="Average Resale Price, CREA", header=None)
    # Header row with age labels sits under "Excluding zeroes" / "Including zeroes"
    label_row = None
    for i, row in raw.iterrows():
        v = row.iloc[3]
        if isinstance(v, str) and "15 to 24" in v:
            label_row = int(i)
            break
    if label_row is None:
        label_row = 40

    if including_zeros:
        cols = [9, 10, 11, 12]
    else:
        cols = [3, 4, 5, 6]
    labels = [str(raw.iloc[label_row, c]).strip() for c in cols]

    records: list[dict] = []
    for i, row in raw.iterrows():
        if i <= label_row:
            continue
        try:
            y = int(float(row.iloc[0]))
        except (TypeError, ValueError):
            continue
        if y < 1900 or y > 2100:
            continue
        rec = {"year": y}
        ok = False
        for c, lab in zip(cols, labels):
            val = pd.to_numeric(row.iloc[c], errors="coerce")
            rec[lab] = float(val) if pd.notna(val) else float("nan")
            if pd.notna(val):
                ok = True
        if ok:
            records.append(rec)
    out = pd.DataFrame(records).set_index("year").sort_index()
    out.attrs["including_zeros"] = including_zeros
    return out


def load_crea_avg_resale_price(
    path: Optional[Union[str, Path]] = None,
) -> pd.Series:
    """
    CREA Canada average residential sale price (C$), annual.

    Default path: ``portfolio_olg/data/HousePriceIndices.xlsx``, sheet
    ``Average Resale Price, CREA``.
    """
    if path is None:
        path = Path(__file__).resolve().parents[2] / "data" / "HousePriceIndices.xlsx"
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CREA house price workbook not found: {path}")

    raw = pd.read_excel(path, sheet_name="Average Resale Price, CREA", header=None)
    years: list[int] = []
    prices: list[float] = []
    for _, row in raw.iterrows():
        try:
            y = int(float(row.iloc[0]))
        except (TypeError, ValueError):
            continue
        if y < 1900 or y > 2100:
            continue
        p = pd.to_numeric(row.iloc[1], errors="coerce")
        if pd.isna(p):
            continue
        years.append(y)
        prices.append(float(p))
    if not years:
        raise ValueError(f"No CREA average prices parsed from {path}")
    s = pd.Series(prices, index=pd.Index(years, name="year"), name="crea_avg_price")
    return s.sort_index()


def price_to_income_ratios(
    income_means: pd.DataFrame,
    crea_price: Optional[pd.Series] = None,
    *,
    measures: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    CREA average house price ÷ weighted mean income by year (and universe).

    Parameters
    ----------
    income_means :
        Long moments with columns ``year``, ``variable``, ``stat``, ``value``,
        and optional ``universe``. Only ``stat == \"mean\"`` rows are used.
    crea_price :
        Annual CREA average sale price; loaded from the default workbook if None.
    """
    if crea_price is None:
        crea_price = load_crea_avg_resale_price()
    measures = list(measures) if measures is not None else list(INCOME_MEASURES)

    means = income_means[income_means["stat"] == "mean"].copy()
    means = means[means["variable"].isin(measures)]
    if means.empty:
        raise ValueError("price_to_income_ratios: no mean income rows")

    rows: list[dict] = []
    has_uni = "universe" in means.columns
    group_cols = ["year"] + (["universe"] if has_uni else [])
    for key, g in means.groupby(group_cols, sort=True, dropna=False):
        if has_uni:
            year, uni = key if isinstance(key, tuple) else (key, None)
        else:
            year, uni = (key if not isinstance(key, tuple) else key[0]), None
        year_i = int(year)
        price = float(crea_price.get(year_i, np.nan)) if year_i in crea_price.index else float("nan")
        for _, row in g.iterrows():
            y_mean = float(row["value"])
            ratio = price / y_mean if np.isfinite(price) and np.isfinite(y_mean) and y_mean != 0 else float("nan")
            out_row = {
                "year": year_i,
                "variable": row["variable"],
                "label": row.get("label", INCOME_LABELS.get(row["variable"], row["variable"])),
                "crea_avg_price": price,
                "mean_income": y_mean,
                "price_to_income": ratio,
            }
            if has_uni:
                out_row["universe"] = uni
            rows.append(out_row)
    cols = (["year", "universe"] if has_uni else ["year"]) + [
        "variable",
        "label",
        "crea_avg_price",
        "mean_income",
        "price_to_income",
    ]
    return pd.DataFrame(rows)[cols]
