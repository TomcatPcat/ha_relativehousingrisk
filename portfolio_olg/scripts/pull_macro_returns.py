"""
Pull Canadian asset price / rate series, deflate by CPI, plot, and estimate
a diagonal-persistence bivariate AR(1) for (equity, housing) real returns.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "macro_returns"
OUT.mkdir(parents=True, exist_ok=True)

UA = "HAmodel-portfolio_olg/0.1 (research; local)"


def _get(url: str, timeout: int = 120) -> bytes:
    req = Request(url, headers={"User-Agent": UA})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _statcan_csv(table: str) -> pd.DataFrame:
    """Download StatCan table ZIP and return the English data CSV."""
    url = f"https://www150.statcan.gc.ca/n1/tbl/csv/{table}-eng.zip"
    raw = _get(url)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        # Prefer the large data file (not MetaData)
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        data_name = next(
            (n for n in names if "MetaData" not in n and "metadata" not in n.lower()),
            names[0],
        )
        with zf.open(data_name) as fh:
            return pd.read_csv(fh, low_memory=False)


def _parse_ref_date(s: pd.Series) -> pd.Series:
    # Formats like 1981-01, 1981-01-01, Jan-1981
    return pd.to_datetime(s, errors="coerce")


def load_cpi() -> pd.Series:
    """Canada all-items CPI, monthly (table 18-10-0004)."""
    df = _statcan_csv("18100004")
    # Filter Canada, All-items
    geo = df["GEO"].astype(str) if "GEO" in df.columns else df.iloc[:, 1].astype(str)
    products = None
    for c in df.columns:
        if "Products" in c or "product" in c.lower() or "Type of" in c:
            products = df[c].astype(str)
            break
    if products is None:
        # fallback: VALUE column with All-items in some dim
        products = df.filter(like="Products").iloc[:, 0].astype(str)
    mask = geo.str.contains("Canada", case=False) & products.str.contains(
        "All-items", case=False, na=False
    )
    # Prefer not seasonally adjusted if a column exists
    sub = df.loc[mask].copy()
    if "UOM" in sub.columns:
        pass
    date_col = "REF_DATE" if "REF_DATE" in sub.columns else sub.columns[0]
    val_col = "VALUE" if "VALUE" in sub.columns else "VALUE"
    s = sub[[date_col, val_col]].dropna()
    s[date_col] = _parse_ref_date(s[date_col])
    s = s.dropna().set_index(date_col)[val_col].astype(float).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s.name = "cpi"
    return s


def load_nhpi() -> pd.Series:
    """Canada New Housing Price Index — total (house and land), monthly."""
    df = _statcan_csv("18100205")
    geo = df["GEO"].astype(str)
    # New housing price indexes dimension
    nhpi_dim = None
    for c in df.columns:
        if "New housing price" in c or "Housing" in c:
            nhpi_dim = df[c].astype(str)
            break
    if nhpi_dim is None:
        # Often "New housing price indexes"
        cand = [c for c in df.columns if "index" in c.lower() or "Housing" in c]
        nhpi_dim = df[cand[0]].astype(str) if cand else df.iloc[:, 3].astype(str)

    mask = geo.str.fullmatch("Canada", case=False) | (
        geo.str.contains("^Canada$", case=False, regex=True)
    )
    # Prefer "House and land" / total
    labels = nhpi_dim.str.lower()
    mask &= labels.str.contains("house and land") | labels.str.contains("total")
    sub = df.loc[mask].copy()
    if sub.empty:
        # Canada + first available series
        sub = df.loc[geo.str.contains("Canada", case=False)].copy()
        # pick House and land if present in any row
        if nhpi_dim is not None:
            prefer = nhpi_dim.loc[sub.index].str.contains("House and land", case=False, na=False)
            if prefer.any():
                sub = sub.loc[prefer]

    date_col = "REF_DATE"
    s = sub[[date_col, "VALUE"]].dropna()
    s[date_col] = _parse_ref_date(s[date_col])
    s = s.dropna().set_index(date_col)["VALUE"].astype(float).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s.name = "nhpi"
    return s


def load_tsx() -> pd.Series:
    """S&P/TSX Composite close index from StatCan table 10-10-0125."""
    df = _statcan_csv("10100125")
    # Find series label column
    label_col = None
    for c in df.columns:
        if "Toronto Stock" in c or "Stock Exchange" in c:
            label_col = c
            break
    if label_col is None:
        label_col = [c for c in df.columns if c not in ("REF_DATE", "GEO", "VALUE", "UOM", "SCALAR_FACTOR", "VECTOR", "COORDINATE", "STATUS", "SYMBOL", "TERMINATED", "DECIMALS")][0]

    lab = df[label_col].astype(str)
    mask = lab.str.contains("Composite", case=False) & lab.str.contains("close", case=False)
    if not mask.any():
        mask = lab.str.contains("S&P/TSX Composite", case=False) & lab.str.contains("close", case=False)
    sub = df.loc[mask, ["REF_DATE", "VALUE"]].dropna()
    sub["REF_DATE"] = _parse_ref_date(sub["REF_DATE"])
    s = sub.dropna().set_index("REF_DATE")["VALUE"].astype(float).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    # Daily → month-end
    s = s.resample("ME").last()
    s.name = "tsx"
    return s


def load_boc_series(series_name: str) -> pd.Series:
    """Bank of Canada Valet observations for one series."""
    url = f"https://www.bankofcanada.ca/valet/observations/{series_name}/json?start_date=1950-01-01"
    data = json.loads(_get(url).decode("utf-8"))
    rows = []
    for obs in data.get("observations", []):
        d = obs.get("d")
        cell = obs.get(series_name, {})
        v = cell.get("v") if isinstance(cell, dict) else cell
        if v in (None, ""):
            continue
        rows.append((d, float(v)))
    if not rows:
        raise RuntimeError(f"No observations for BoC series {series_name}")
    s = pd.Series({pd.Timestamp(d): v for d, v in rows}).sort_index()
    s.name = series_name
    return s


def to_month_end(s: pd.Series) -> pd.Series:
    if s.empty:
        return s
    # Weekly/daily → month-end last observation
    freq = pd.infer_freq(s.index)
    out = s.resample("ME").last()
    out.name = s.name
    return out


def load_bis_house_nominal() -> pd.Series:
    """BIS/FRED nominal residential property price index for Canada (quarterly)."""
    import ssl
    from urllib.request import urlopen

    ctx = ssl._create_unverified_context()
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=QCAN628BIS"
    raw = urlopen(url, context=ctx, timeout=60).read()
    df = pd.read_csv(io.BytesIO(raw))
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    s = df.set_index("observation_date")["QCAN628BIS"].astype(float).sort_index()
    s.name = "house_bis"
    s.index = s.index + pd.offsets.QuarterEnd(0)
    return s


def load_bis_house_real() -> pd.Series:
    """BIS/FRED real residential property price index for Canada (quarterly)."""
    import ssl
    from urllib.request import urlopen

    ctx = ssl._create_unverified_context()
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=QCAR628BIS"
    raw = urlopen(url, context=ctx, timeout=60).read()
    df = pd.read_csv(io.BytesIO(raw))
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    s = df.set_index("observation_date")["QCAR628BIS"].astype(float).sort_index()
    s.name = "house_bis_real"
    s.index = s.index + pd.offsets.QuarterEnd(0)
    return s


def load_crea_mls_hpi() -> pd.Series:
    """CREA MLS Home Price Index (Canada), via Bank of Canada FVI feed."""
    s = load_boc_series("FVI_CREA_MLS_HPI_CANADA")
    s = to_month_end(s)
    s.name = "crea_mls_hpi"
    return s


def load_statcan_rppi() -> pd.Series:
    """
    StatCan Residential Property Price Index (table 18-10-0169, inactive).

    Six-CMA composite, house, resale. Quarterly from 2017.
    """
    df = _statcan_csv("18100169")
    geo = df["GEO"].astype(str)
    btype = df["Building type"].astype(str)
    ctype = df["Construction type"].astype(str)
    mask = (
        geo.str.contains("Six census metropolitan area composite", case=False, na=False)
        & btype.str.fullmatch("House", case=False)
        & ctype.str.fullmatch("Resale", case=False)
    )
    sub = df.loc[mask, ["REF_DATE", "VALUE"]].dropna()
    if sub.empty:
        # fallback: total building / total construction
        mask = geo.str.contains("Six census metropolitan area composite", case=False, na=False)
        mask &= btype.str.contains("Total", case=False, na=False)
        mask &= ctype.str.contains("Total", case=False, na=False)
        sub = df.loc[mask, ["REF_DATE", "VALUE"]].dropna()
    sub["REF_DATE"] = _parse_ref_date(sub["REF_DATE"])
    s = sub.dropna().set_index("REF_DATE")["VALUE"].astype(float).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    # quarterly → month-end of quarter
    s.index = pd.to_datetime(s.index) + pd.offsets.MonthEnd(0)
    s.name = "rppi_house_resale"
    return s


def load_nhpi_components() -> Dict[str, pd.Series]:
    """NHPI Canada: house+land total, house only, land only if available."""
    df = _statcan_csv("18100205")
    geo = df["GEO"].astype(str)
    dim_col = next(
        (c for c in df.columns if "New housing price" in c or "Housing" in c),
        None,
    )
    if dim_col is None:
        return {"nhpi": to_month_end(load_nhpi())}
    dim = df[dim_col].astype(str)
    out: Dict[str, pd.Series] = {}
    mapping = {
        "nhpi_house_land": "house and land",
        "nhpi_house": "house only",
        "nhpi_land": "land only",
    }
    for name, needle in mapping.items():
        mask = geo.str.contains("^Canada$", case=False, regex=True) & dim.str.contains(
            needle, case=False, na=False
        )
        sub = df.loc[mask, ["REF_DATE", "VALUE"]].dropna()
        if sub.empty:
            continue
        sub["REF_DATE"] = _parse_ref_date(sub["REF_DATE"])
        s = sub.dropna().set_index("REF_DATE")["VALUE"].astype(float).sort_index()
        s = to_month_end(s[~s.index.duplicated(keep="last")])
        s.name = name
        out[name] = s
    if "nhpi_house_land" in out:
        out["nhpi"] = out["nhpi_house_land"].rename("nhpi")
    elif not out:
        out["nhpi"] = to_month_end(load_nhpi())
    return out


def _rebase(s: pd.Series, base_date: pd.Timestamp) -> pd.Series:
    """Rebase series to 100 at base_date (or first valid on/after)."""
    s = s.dropna()
    if s.empty:
        return s
    if base_date in s.index and pd.notna(s.loc[base_date]):
        base_val = float(s.loc[base_date])
    else:
        later = s.loc[s.index >= base_date]
        if later.empty:
            base_val = float(s.iloc[0])
        else:
            base_val = float(later.iloc[0])
    if base_val == 0 or not np.isfinite(base_val):
        return s * np.nan
    return 100.0 * s / base_val


def plot_housing_index_comparison(df: pd.DataFrame) -> None:
    """Compare available Canadian housing price indices (nominal and real)."""
    nominal_cols = [
        ("house_bis", "BIS existing homes (national)"),
        ("nhpi_house_land", "NHPI house+land (new)"),
        ("nhpi_house", "NHPI house only (new)"),
        ("crea_mls_hpi", "CREA MLS HPI (BoC FVI)"),
        ("rppi_house_resale", "StatCan RPPI house resale (6 CMAs)"),
    ]
    # Prefer dedicated nhpi_house_land; fall back to nhpi
    if "nhpi_house_land" not in df.columns and "nhpi" in df.columns:
        nominal_cols = [
            ("house_bis", "BIS existing homes (national)"),
            ("nhpi", "NHPI (new homes)"),
            ("nhpi_house", "NHPI house only (new)"),
            ("crea_mls_hpi", "CREA MLS HPI (BoC FVI)"),
            ("rppi_house_resale", "StatCan RPPI house resale (6 CMAs)"),
        ]

    # Common rebase date: first month where at least BIS+NHPI exist; also show
    # a short-window rebase at 2017 for RPPI/MLS overlap.
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)

    # Panel A: long sample, rebase ~1981 (or first NHPI)
    base_long = pd.Timestamp("1981-01-31")
    for col, label in nominal_cols:
        if col not in df.columns or not df[col].notna().any():
            continue
        s = _rebase(df[col], base_long)
        axes[0].plot(s.index, s, label=label, lw=1.5)
    axes[0].set_yscale("log")
    axes[0].set_title("Canadian housing price indices — nominal, rebased (log)")
    axes[0].set_ylabel(f"Index (100 ≈ {base_long.date()})")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Panel B: real (CPI-deflated), same rebase
    for col, label in nominal_cols:
        real_col = f"{col}_real"
        if real_col not in df.columns or not df[real_col].notna().any():
            continue
        s = _rebase(df[real_col], base_long)
        axes[1].plot(s.index, s, label=label, lw=1.5)
    axes[1].set_yscale("log")
    axes[1].set_title("Same indices — real (÷ CPI), rebased (log)")
    axes[1].set_ylabel(f"Real index (100 ≈ {base_long.date()})")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "housing_indices_comparison.png", dpi=140)
    plt.close(fig)

    # Short-window overlap chart (2017+) for MLS / RPPI / others
    fig, ax = plt.subplots(figsize=(11, 5))
    base_short = pd.Timestamp("2017-01-31")
    for col, label in nominal_cols:
        real_col = f"{col}_real"
        if real_col not in df.columns or not df[real_col].notna().any():
            continue
        s = _rebase(df[real_col].loc["2017":], base_short)
        if s.notna().sum() < 4:
            continue
        ax.plot(s.index, s, label=label, lw=1.6)
    ax.set_title("Housing indices since 2017 — real, rebased to 2017=100")
    ax.set_ylabel("Real index")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "housing_indices_comparison_since2017.png", dpi=140)
    plt.close(fig)
    print(f"Wrote {OUT / 'housing_indices_comparison.png'}")
    print(f"Wrote {OUT / 'housing_indices_comparison_since2017.png'}")


def build_panel() -> pd.DataFrame:
    print("Downloading CPI...")
    cpi = to_month_end(load_cpi())
    print("Downloading NHPI components...")
    nhpi_parts = load_nhpi_components()
    print("Downloading BIS house prices (existing stock)...")
    house_bis = load_bis_house_nominal()
    house_bis_real = load_bis_house_real()
    print("Downloading CREA MLS HPI (BoC)...")
    try:
        crea = load_crea_mls_hpi()
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: CREA MLS HPI failed: {exc}")
        crea = pd.Series(dtype=float, name="crea_mls_hpi")
    print("Downloading StatCan RPPI (house resale)...")
    try:
        rppi = load_statcan_rppi()
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: RPPI failed: {exc}")
        rppi = pd.Series(dtype=float, name="rppi_house_resale")
    print("Downloading TSX...")
    tsx = to_month_end(load_tsx())

    boc = {
        "mortgage_5y": "V80691335",
        "gic_1y": "V80691339",
        "savings_deposit": "V80691338",
        "bond_5y": "BD.CDN.5YR.DQ.YLD",
        "bond_10y": "BD.CDN.10YR.DQ.YLD",
    }
    rates: Dict[str, pd.Series] = {}
    for name, code in boc.items():
        print(f"Downloading BoC {name} ({code})...")
        try:
            rates[name] = to_month_end(load_boc_series(code))
        except Exception as exc:  # noqa: BLE001
            print(f"  WARN: {name} failed: {exc}")

    df = pd.DataFrame(
        {
            "cpi": cpi,
            **nhpi_parts,
            "house_bis": house_bis,
            "house_bis_real_pub": house_bis_real,
            "crea_mls_hpi": crea,
            "rppi_house_resale": rppi,
            "tsx": tsx,
            **rates,
        }
    ).sort_index()

    # Forward-fill lower-frequency house series for plotting continuity
    for col in (
        "house_bis",
        "house_bis_real_pub",
        "rppi_house_resale",
        "crea_mls_hpi",
    ):
        if col in df.columns:
            df[col] = df[col].ffill()

    # Real (CPI-deflated) levels for each housing index
    housing_nominal = [
        c
        for c in (
            "house_bis",
            "nhpi",
            "nhpi_house_land",
            "nhpi_house",
            "nhpi_land",
            "crea_mls_hpi",
            "rppi_house_resale",
        )
        if c in df.columns
    ]
    for col in housing_nominal:
        df[f"{col}_real"] = df[col] / df["cpi"]

    df["tsx_real"] = df["tsx"] / df["cpi"]
    # Alias used elsewhere
    if "house_bis_real" in df.columns:
        pass
    df["house_real"] = df.get("house_bis_real", df["house_bis"] / df["cpi"])

    base = df[["house_real", "tsx_real", "cpi"]].dropna().index.min()
    if pd.notna(base):
        rebase_cols = [f"{c}_real" for c in housing_nominal] + [
            "tsx_real",
            "house_real",
            "house_bis_real_pub",
        ]
        for col in rebase_cols:
            if col in df.columns and pd.notna(df.loc[base, col]):
                df[col] = 100.0 * df[col] / df.loc[base, col]

    df["pi_m"] = df["cpi"].pct_change(fill_method=None)

    # Primary housing = BIS existing-home; NHPI kept for comparison
    df["r_house_m"] = (1.0 + df["house_bis"].pct_change(fill_method=None)) / (
        1.0 + df["pi_m"]
    ) - 1.0
    nhpi_src = "nhpi_house_land" if "nhpi_house_land" in df.columns else "nhpi"
    if nhpi_src in df.columns:
        df["r_nhpi_m"] = (1.0 + df[nhpi_src].pct_change(fill_method=None)) / (
            1.0 + df["pi_m"]
        ) - 1.0
    if "crea_mls_hpi" in df.columns:
        df["r_mls_m"] = (1.0 + df["crea_mls_hpi"].pct_change(fill_method=None)) / (
            1.0 + df["pi_m"]
        ) - 1.0
    if "rppi_house_resale" in df.columns:
        df["r_rppi_m"] = (1.0 + df["rppi_house_resale"].pct_change(fill_method=None)) / (
            1.0 + df["pi_m"]
        ) - 1.0
    df["r_equity_m"] = (1.0 + df["tsx"].pct_change(fill_method=None)) / (1.0 + df["pi_m"]) - 1.0

    for col in rates:
        if col not in df.columns:
            continue
        i_a = df[col] / 100.0
        i_m = (1.0 + i_a) ** (1.0 / 12.0) - 1.0
        df[f"{col}_real_m"] = (1.0 + i_m) / (1.0 + df["pi_m"]) - 1.0
        pi_y = df["cpi"].pct_change(12, fill_method=None)
        df[f"{col}_real_a"] = (1.0 + i_a) / (1.0 + pi_y) - 1.0

    for col, src in [
        ("cum_house_real", "r_house_m"),
        ("cum_nhpi_real", "r_nhpi_m"),
        ("cum_mls_real", "r_mls_m"),
        ("cum_rppi_real", "r_rppi_m"),
        ("cum_equity_real", "r_equity_m"),
        ("cum_mortgage_real", "mortgage_5y_real_m"),
        ("cum_gic_real", "gic_1y_real_m"),
        ("cum_bond5_real", "bond_5y_real_m"),
        ("cum_deposit_real", "savings_deposit_real_m"),
    ]:
        if src not in df.columns:
            continue
        start = df[src].first_valid_index()
        cum = pd.Series(np.nan, index=df.index)
        if start is not None:
            path = (1.0 + df.loc[start:, src].fillna(0.0)).cumprod()
            cum.loc[start:] = 100.0 * path / path.iloc[0]
        df[col] = cum

    return df


def annualize_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Annual real returns for equity and housing (calendar-year compound)."""
    cols = [c for c in ("r_equity_m", "r_house_m", "r_nhpi_m") if c in df.columns]
    ann = df[cols].copy()
    g = ann.groupby(ann.index.year)

    def _ann(x: pd.Series) -> float:
        x = x.dropna()
        if x.size < 4:  # need enough months (quarterly house → ~4 obs/yr after ffill)
            # still allow if at least 1 monthly after ffill
            if x.size == 0:
                return float("nan")
        return float(np.prod(1.0 + x) - 1.0)

    out = pd.DataFrame({c: g[c].apply(_ann) for c in cols})
    out = out.rename(
        columns={
            "r_equity_m": "r_equity",
            "r_house_m": "r_house",
            "r_nhpi_m": "r_nhpi",
        }
    )
    out.index = pd.to_datetime(out.index.astype(str) + "-12-31")
    return out.dropna(subset=["r_equity", "r_house"])


def estimate_diag_ar1(
    r_f: np.ndarray,
    r_h: np.ndarray,
) -> Dict[str, float]:
    """
    Estimate
        r_{t+1} - mu = Phi (r_t - mu) + eps_{t+1},
    with Phi = diag(rho_f, rho_h) and Corr(eps) free.

    OLS equation-by-equation on demeaned series.
    """
    rf = np.asarray(r_f, dtype=float)
    rh = np.asarray(r_h, dtype=float)
    ok = np.isfinite(rf) & np.isfinite(rh)
    rf, rh = rf[ok], rh[ok]
    mu_f, mu_h = float(rf.mean()), float(rh.mean())
    xf, xh = rf - mu_f, rh - mu_h
    # y_t = x_{t+1}, x_t = lag
    yf, yh = xf[1:], xh[1:]
    xf_l, xh_l = xf[:-1], xh[:-1]
    rho_f = float(np.dot(xf_l, yf) / np.dot(xf_l, xf_l))
    rho_h = float(np.dot(xh_l, yh) / np.dot(xh_l, xh_l))
    eps_f = yf - rho_f * xf_l
    eps_h = yh - rho_h * xh_l
    sig_f = float(eps_f.std(ddof=1))
    sig_h = float(eps_h.std(ddof=1))
    corr = float(np.corrcoef(eps_f, eps_h)[0, 1])
    # Unconditional corr of returns (for comparison)
    unc_corr = float(np.corrcoef(rf, rh)[0, 1])
    return {
        "mu_f": mu_f,
        "mu_h": mu_h,
        "rho_f": rho_f,
        "rho_h": rho_h,
        "sigma_f": sig_f,
        "sigma_h": sig_h,
        "corr_innov": corr,
        "corr_unconditional": unc_corr,
        "n_obs": float(len(yf)),
        "mean_Rf_gross": 1.0 + mu_f,
        "mean_Rh_gross": 1.0 + mu_h,
    }


def make_plots(df: pd.DataFrame, ann: pd.DataFrame, est: Dict[str, float]) -> None:
    # 1) Cumulative real indices
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for col, label in [
        ("cum_equity_real", "Equity TSX real (price only)"),
        ("cum_house_real", "Housing BIS real (existing)"),
        ("cum_nhpi_real", "Housing NHPI real (new homes)"),
        ("cum_bond5_real", "5y GoC yield compounded real"),
        ("cum_gic_real", "1y GIC compounded real"),
        ("cum_deposit_real", "Savings deposit compounded real"),
        ("cum_mortgage_real", "5y mortgage rate compounded real"),
    ]:
        if col in df.columns and df[col].notna().any():
            ax.plot(df.index, df[col], label=label, lw=1.6)
    ax.set_yscale("log")
    ax.set_title("Cumulative real indices (log scale, start=100)")
    ax.set_ylabel("Index (real)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "cumulative_real_indices.png", dpi=140)
    plt.close(fig)

    # 2) Monthly real growth rates (smoothed 12m for readability)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for col, label in [
        ("r_equity_m", "Equity real monthly"),
        ("r_house_m", "Housing BIS real monthly"),
        ("r_nhpi_m", "NHPI real monthly"),
    ]:
        if col not in df.columns:
            continue
        s = df[col]
        axes[0].plot(s.index, s.rolling(12, min_periods=6).mean(), label=label + " (12m MA)")
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_title("Real asset price growth (12-month MA of monthly rates)")
    axes[0].set_ylabel("Monthly rate")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for col, label in [
        ("bond_5y_real_a", "5y bond real yield"),
        ("bond_10y_real_a", "10y bond real yield"),
        ("gic_1y_real_a", "1y GIC real"),
        ("savings_deposit_real_a", "Savings deposit real"),
        ("mortgage_5y_real_a", "5y mortgage real"),
    ]:
        if col in df.columns:
            axes[1].plot(df.index, df[col], label=label, lw=1.2)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_title("Real interest rates (annual Fisher, using y/y CPI)")
    axes[1].set_ylabel("Annual real rate")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "real_growth_and_rates.png", dpi=140)
    plt.close(fig)

    # 3) Annual equity vs housing scatter
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(ann["r_house"], ann["r_equity"], alpha=0.75)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("Housing BIS real annual return")
    ax.set_ylabel("Equity (TSX) real annual return")
    ax.set_title(
        f"Annual real returns\n"
        f"uncorr={est['corr_unconditional']:.2f}, innov corr={est['corr_innov']:.2f}\n"
        f"rho_f={est['rho_f']:.2f}, rho_h={est['rho_h']:.2f}"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "equity_housing_annual_scatter.png", dpi=140)
    plt.close(fig)


def main() -> None:
    df = build_panel()
    df.to_csv(OUT / "macro_returns_monthly.csv")
    print(f"Wrote {OUT / 'macro_returns_monthly.csv'} shape={df.shape}")

    ann = annualize_returns(df)
    ann.to_csv(OUT / "macro_returns_annual.csv")
    est = estimate_diag_ar1(ann["r_equity"].to_numpy(), ann["r_house"].to_numpy())
    pd.Series(est).to_csv(OUT / "bivar_ar1_equity_housing.csv", header=["value"])
    print("Diagonal AR(1) estimates (annual real returns, equity vs BIS housing):")
    for k, v in est.items():
        print(f"  {k}: {v:.6g}" if isinstance(v, float) else f"  {k}: {v}")

    make_plots(df, ann, est)
    plot_housing_index_comparison(df)
    print(f"Figures in {OUT}")

    for c in [
        "cpi",
        "nhpi",
        "nhpi_house",
        "house_bis",
        "crea_mls_hpi",
        "rppi_house_resale",
        "tsx",
        "mortgage_5y",
        "gic_1y",
        "bond_5y",
        "savings_deposit",
    ]:
        if c in df.columns and df[c].notna().any():
            s = df[c].dropna()
            print(
                f"  {c}: {s.index.min().date()} -> {s.index.max().date()} (n={len(s)})"
            )


if __name__ == "__main__":
    main()
