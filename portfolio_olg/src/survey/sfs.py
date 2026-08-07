"""SFS (Survey of Financial Security) loaders and aggregate moments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .paths import load_local_paths, require_file
from .weighted import wmean, wquantile, wshare

# StatCan EFAM PUMF → analysis names
_RENAME = {
    "pefamid": "famid",
    "pweight": "weight",
    "pagemieg": "agegrp",
    "pagemie": "age",
    "peducmie": "dem_educ",
    "pefatinc": "inc_net",
    "pefgtr": "inc_transfers",
    "pefmjsif": "inc_source",
    "pefmtinc": "inc_gross",
    "pfsz": "dem_fam_size",
    "pftenur": "dem_fam_tenure",
    "ppvres": "dem_prov",
    "pregion": "dem_region",
    "pretirme": "dem_retired",
    "psexmie": "dem_sex",
    "pnbearg": "dem_num_earners",
    "pwaotpen": "w_oth_ret_funds",
    "pwaprval": "w_hous_princ_res_val",
    "pwarppg": "w_pens_employer_val_gc",
    "pwarppt": "w_pens_employer_val_tm",
    "pwarrif": "w_rrif",
    "pwarrspl": "w_rrsp",
    "pwastbnd": "w_bonds",
    "pwastdep": "w_bank_acc",
    "pwastmui": "w_mut_funds",
    "pwastoin": "w_oth_inv",
    "pwastonf": "w_oth_nonfin",
    "pwastrst": "w_hous_oth_real_est",
    "pwaststk": "w_stocks",
    "pwastvhe": "w_vehicles",
    "pwatfs": "w_tfsa",
    "pwatotpg": "w_tot_incl_pens_gc",
    "pwatotpt": "w_tot_incl_pens_tm",
    "pwbuseq": "w_bus_val",
    "pwdprmor": "debt_mort_princ_res_val",
    "pwdsloan": "debt_stud_loans_val",
    "pwdstcrd": "debt_cc_installment_val",
    "pwdstloc": "debt_loc_val",
    "pwdstodb": "debt_oth_val",
    "pwdstomr": "debt_mort_oth_real_est_val",
    "pwdstvhn": "debt_vehicles_val",
    "pwdtotal": "debt_tot_val",
    "pwnetwpg": "w_networth_gc",
    "pwnetwpt": "w_networth_tm",
}

_ZERO_FILL = (
    "w_bank_acc",
    "w_bonds",
    "w_mut_funds",
    "w_oth_inv",
    "w_oth_nonfin",
    "w_hous_princ_res_val",
    "w_hous_oth_real_est",
    "w_stocks",
    "w_vehicles",
    "w_tfsa",
    "w_rrsp",
    "w_rrif",
    "w_oth_ret_funds",
    "w_bus_val",
    "w_pens_employer_val_gc",
    "w_pens_employer_val_tm",
    "debt_mort_princ_res_val",
    "debt_mort_oth_real_est_val",
    "debt_stud_loans_val",
    "debt_cc_installment_val",
    "debt_loc_val",
    "debt_oth_val",
    "debt_vehicles_val",
    "debt_tot_val",
)

# Income stays as NaN when residual / not stated (do not zero-fill).
_INCOME_COLS = ("inc_gross", "inc_net", "inc_transfers")

# StatCan PUMF residual / not-stated codes for dollar amounts (8- and 9-digit).
_RESIDUAL_DOLLAR_CODES = frozenset(
    {
        99999996,
        99999997,
        99999998,
        99999999,
        999999996,
        999999997,
        999999998,
        999999999,
    }
)

SFS_YEARS: tuple[int, ...] = (1999, 2005, 2012, 2016, 2019, 2023)


def _scrub_residual_codes(s: pd.Series) -> pd.Series:
    """
    Convert StatCan residual dollar codes to NaN.

    Codes such as ``99999999`` (not stated) must not enter means as income.
    Also flags any absolute value ``>= 99_999_996`` as residual for safety.
    """
    x = pd.to_numeric(s, errors="coerce")
    residual = x.isin(_RESIDUAL_DOLLAR_CODES) | (x.abs() >= 99_999_996)
    return x.mask(residual)


def _na2zero(s: pd.Series) -> pd.Series:
    return _scrub_residual_codes(s).fillna(0.0)


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    return _na2zero(df[name])


def build_wealth_identity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct housing / financial gross, debt, net and primary NW (ex-auto).

    Identity
    --------
    ``nw = housing_assets_gross - housing_debt + financial_assets_gross - financial_debt``
    ``     = housing_net + financial_net``.

    Autos are stored separately and are **not** in ``nw``.
    Employer pensions use the terminal-value series ``w_pens_employer_val_tm``
    when present (else GC), matching StatCan ``pwnetwpt`` spirit.
    """
    out = df.copy()

    out["housing_assets_gross"] = _col(out, "w_hous_princ_res_val") + _col(
        out, "w_hous_oth_real_est"
    )
    out["housing_debt"] = _col(out, "debt_mort_princ_res_val") + _col(
        out, "debt_mort_oth_real_est_val"
    )
    out["housing_net"] = out["housing_assets_gross"] - out["housing_debt"]

    pens = _col(out, "w_pens_employer_val_tm")
    pens_gc = _col(out, "w_pens_employer_val_gc")
    # Prefer TM; fall back to GC where TM is all zero / missing year
    if float(pens.abs().sum()) == 0.0 and float(pens_gc.abs().sum()) > 0.0:
        pens = pens_gc
    out["w_pens_employer"] = pens

    out["financial_assets_gross"] = (
        _col(out, "w_bank_acc")
        + _col(out, "w_bonds")
        + _col(out, "w_stocks")
        + _col(out, "w_mut_funds")
        + _col(out, "w_oth_inv")
        + _col(out, "w_oth_nonfin")
        + _col(out, "w_tfsa")
        + _col(out, "w_rrsp")
        + _col(out, "w_rrif")
        + _col(out, "w_oth_ret_funds")
        + _col(out, "w_bus_val")
        + pens
    )
    out["financial_debt"] = (
        _col(out, "debt_stud_loans_val")
        + _col(out, "debt_cc_installment_val")
        + _col(out, "debt_loc_val")
        + _col(out, "debt_oth_val")
    )
    out["financial_net"] = out["financial_assets_gross"] - out["financial_debt"]

    out["nw"] = out["housing_net"] + out["financial_net"]

    out["auto_assets"] = _col(out, "w_vehicles")
    out["auto_debt"] = _col(out, "debt_vehicles_val")
    out["auto_net"] = out["auto_assets"] - out["auto_debt"]
    out["nw_with_auto"] = out["nw"] + out["auto_net"]

    tenure = out["dem_fam_tenure"] if "dem_fam_tenure" in out.columns else None
    if tenure is not None:
        t_num = pd.to_numeric(tenure, errors="coerce")
        owner = (t_num == 1) | tenure.astype(str).str.lower().isin(
            {"1", "owned", "owner"}
        )
    else:
        owner = pd.Series(False, index=out.index)
    owner = owner | (out["w_hous_princ_res_val"] > 0)
    out["owner"] = owner.astype(bool)

    return out


def load_sfs(
    path: Optional[Union[str, Path]] = None,
    *,
    years: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """
    Load the SFS 1999–2023 panel and build the ex-auto wealth identity.

    Parameters
    ----------
    path :
        CSV path. Default from :func:`load_local_paths`.
    years :
        Subset of survey years (default all SFS waves in the panel).
    """
    if path is None:
        path = load_local_paths()["sfs_panel"]
    path = require_file(Path(path), "SFS panel")

    raw = pd.read_csv(path, low_memory=False)
    raw.columns = [str(c).strip().lower() for c in raw.columns]

    out = raw.copy()
    for old, new in _RENAME.items():
        if old in out.columns and new not in out.columns:
            out[new] = out[old]

    if "year" not in out.columns:
        raise ValueError("SFS panel missing 'year' column")
    if "weight" not in out.columns:
        raise ValueError("SFS panel missing weight (pweight)")

    for c in _ZERO_FILL:
        if c in out.columns:
            out[c] = _na2zero(out[c])
        else:
            out[c] = 0.0

    # Income: scrub residual codes → NaN (exclude from weighted moments)
    for c in _INCOME_COLS:
        if c in out.columns:
            out[c] = _scrub_residual_codes(out[c])
        else:
            out[c] = np.nan

    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out = out[out["weight"].notna() & (out["weight"] > 0)].copy()

    keep_years = tuple(years) if years is not None else SFS_YEARS
    out = out[out["year"].isin(keep_years)].copy()
    out = build_wealth_identity(out)
    return out.reset_index(drop=True)


def sfs_aggregate_moments(
    df: pd.DataFrame,
    by: Union[str, Sequence[str], None] = "year",
) -> pd.DataFrame:
    """
    Weighted SFS aggregates by group (default survey year).

    Moments
    -------
    - ``homeownership``: weighted owner rate
    - ``housing_share_gross``, ``housing_share_net``: sum(w H)/sum(w NW)
    - ``financial_share_gross``, ``financial_share_net``
    - ``nw_to_netincome``: sum(w NW)/sum(w Y) for Y = inc_net > 0
      (households with non-positive net income are excluded from both
      numerator and denominator of this ratio only)

    Shares use primary ``nw`` (ex-auto).
    """
    required = (
        "weight",
        "owner",
        "housing_assets_gross",
        "housing_net",
        "financial_assets_gross",
        "financial_net",
        "nw",
        "inc_net",
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"sfs_aggregate_moments missing columns: {missing}")

    if by is None:
        groups: Iterable[tuple] = [(None, df)]
        group_keys: list[str] = []
    else:
        keys = [by] if isinstance(by, str) else list(by)
        group_keys = keys
        groups = df.groupby(keys, sort=True, dropna=False)

    rows = []
    for key, g in groups:
        w = g["weight"].to_numpy(dtype=float)
        nw = g["nw"].to_numpy(dtype=float)
        y = g["inc_net"].to_numpy(dtype=float)
        pos = np.isfinite(y) & (y > 0) & np.isfinite(nw) & np.isfinite(w) & (w > 0)

        row = {
            "n": int(len(g)),
            "homeownership": wmean(g["owner"].astype(float).to_numpy(), w),
            "housing_share_gross": wshare(g["housing_assets_gross"], nw, w),
            "housing_share_net": wshare(g["housing_net"], nw, w),
            "financial_share_gross": wshare(g["financial_assets_gross"], nw, w),
            "financial_share_net": wshare(g["financial_net"], nw, w),
            "nw_to_netincome": wshare(nw[pos], y[pos], w[pos]),
            "mean_nw": wmean(nw, w),
            "mean_inc_net": wmean(y[pos], w[pos]) if pos.any() else float("nan"),
        }
        if group_keys:
            if len(group_keys) == 1:
                row[group_keys[0]] = key[0] if isinstance(key, tuple) else key
            else:
                vals = key if isinstance(key, tuple) else (key,)
                for k, v in zip(group_keys, vals):
                    row[k] = v
        rows.append(row)

    cols = group_keys + [
        "n",
        "homeownership",
        "housing_share_gross",
        "housing_share_net",
        "financial_share_gross",
        "financial_share_net",
        "nw_to_netincome",
        "mean_nw",
        "mean_inc_net",
    ]
    return pd.DataFrame(rows)[cols]


# Default household-level variables for distribution charts
SFS_DIST_VARS: Dict[str, str] = {
    "nw": "Net worth (ex-auto)",
    "inc_net": "Net income",
    "housing_net": "Net housing wealth",
    "financial_net": "Net financial wealth",
    "housing_share_hh": "Housing portfolio share (H_net / NW)",
}

DEFAULT_QUANTILES: Tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)


def sfs_distribution_moments(
    df: pd.DataFrame,
    *,
    by: str = "year",
    variables: Optional[Mapping[str, str]] = None,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    nw_share_eps: float = 1.0,
) -> pd.DataFrame:
    """
    Weighted mean and quantiles of household-level SFS variables by group.

    Adds ``housing_share_hh = housing_net / nw`` for rows with ``|nw| >= nw_share_eps``.

    Returns a long DataFrame with columns ``by``, ``variable``, ``label``,
    ``stat`` (``mean`` / ``p10`` / …), ``value``, and ``n``.
    """
    variables = dict(variables) if variables is not None else dict(SFS_DIST_VARS)
    work = df.copy()
    if "housing_share_hh" in variables:
        nw = work["nw"].to_numpy(dtype=float)
        hn = work["housing_net"].to_numpy(dtype=float)
        share = np.full(nw.shape, np.nan, dtype=float)
        ok = np.isfinite(nw) & np.isfinite(hn) & (np.abs(nw) >= float(nw_share_eps))
        share[ok] = hn[ok] / nw[ok]
        work["housing_share_hh"] = share

    missing = [c for c in variables if c not in work.columns]
    if missing:
        raise KeyError(f"sfs_distribution_moments missing columns: {missing}")
    if "weight" not in work.columns:
        raise KeyError("sfs_distribution_moments missing weight")
    if by not in work.columns:
        raise KeyError(f"sfs_distribution_moments missing group column {by}")

    q_labels = {p: f"p{int(round(100 * p))}" for p in quantiles}
    rows: list[dict] = []
    for key, g in work.groupby(by, sort=True, dropna=False):
        w = g["weight"].to_numpy(dtype=float)
        for col, label in variables.items():
            x = g[col].to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
            n_ok = int(ok.sum())
            base = {by: key, "variable": col, "label": label, "n": n_ok}
            if n_ok == 0:
                rows.append({**base, "stat": "mean", "value": float("nan")})
                for p, lab in q_labels.items():
                    rows.append({**base, "stat": lab, "value": float("nan")})
                continue
            rows.append({**base, "stat": "mean", "value": wmean(x[ok], w[ok])})
            qs = wquantile(x[ok], w[ok], list(quantiles))
            qs = np.atleast_1d(qs)
            for p, lab, val in zip(quantiles, q_labels.values(), qs):
                rows.append({**base, "stat": lab, "value": float(val)})

    return pd.DataFrame(rows)
