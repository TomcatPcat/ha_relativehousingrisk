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

# Allow running as a script from portfolio_olg/
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.returns_estimate import (  # noqa: E402
    annual_real_indices,
    estimate_bivar_ar1_from_indices,
    estimate_diag_ar1_simple,
    save_return_estimates,
)

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


def load_tsx60() -> pd.Series:
    """S&P/TSX 60 index from StatCan table 10-10-0125 (monthly; ~1982–2023)."""
    df = _statcan_csv("10100125")
    label_col = next(
        (c for c in df.columns if "Toronto Stock" in c or "Stock Exchange" in c),
        None,
    )
    if label_col is None:
        raise RuntimeError("Could not find TSX label column in 10-10-0125")
    lab = df[label_col].astype(str)
    mask = lab.str.contains(r"60\s+Index", case=False, regex=True)
    sub = df.loc[mask, ["REF_DATE", "VALUE"]].dropna()
    if sub.empty:
        raise RuntimeError("S&P/TSX 60 Index not found in StatCan 10-10-0125")
    sub["REF_DATE"] = _parse_ref_date(sub["REF_DATE"])
    s = sub.dropna().set_index("REF_DATE")["VALUE"].astype(float).sort_index()
    s = to_month_end(s[~s.index.duplicated(keep="last")])
    s.name = "tsx60"
    return s


def load_fred(series_id: str) -> pd.Series:
    """Download a FRED series as a pandas Series indexed by observation date."""
    import ssl

    ctx = ssl._create_unverified_context()
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    raw = urlopen(url, context=ctx, timeout=60).read()
    df = pd.read_csv(io.BytesIO(raw))
    date_col = "observation_date" if "observation_date" in df.columns else df.columns[0]
    val_col = series_id if series_id in df.columns else df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col])
    s = df.set_index(date_col)[val_col].astype(float).sort_index()
    s = s[np.isfinite(s)]
    s.name = series_id
    return s


def load_sp500() -> pd.Series:
    """
    S&P 500 monthly **price** index (USD).

    Prefer long monthly history from datahub.io (Shiller/S&P); fall back to FRED
    SP500 (shorter daily sample) if needed.
    """
    try:
        raw = _get("https://datahub.io/core/s-and-p-500/r/data.csv", timeout=90)
        df = pd.read_csv(io.BytesIO(raw))
        df["Date"] = pd.to_datetime(df["Date"])
        s = df.set_index("Date")["SP500"].astype(float).sort_index()
        s = s[np.isfinite(s) & (s > 0)]
        s = to_month_end(s[~s.index.duplicated(keep="last")])
        s.name = "sp500"
        if s.size >= 120:
            return s
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: datahub S&P500 failed ({exc}); trying FRED SP500")
    s = to_month_end(load_fred("SP500"))
    s.name = "sp500"
    return s


def load_sp500_shiller_table() -> pd.DataFrame:
    """Shiller/S&P monthly table with ``SP500`` price and annualized ``Dividend``."""
    raw = _get("https://datahub.io/core/s-and-p-500/r/data.csv", timeout=90)
    df = pd.read_csv(io.BytesIO(raw))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")[["SP500", "Dividend"]].astype(float).sort_index()
    df = df[np.isfinite(df["SP500"]) & (df["SP500"] > 0)]
    # Recent rows often have Dividend=0 before updates; carry last positive
    div = df["Dividend"].where(df["Dividend"] > 0)
    df["Dividend"] = div.ffill()
    df.index = df.index + pd.offsets.MonthEnd(0)
    df = df[~df.index.duplicated(keep="last")]
    return df


def load_yahoo_monthly_adj(symbol: str) -> pd.Series:
    """Yahoo Finance monthly adjusted close (dividend-adjusted ≈ total return)."""
    import ssl

    ctx = ssl._create_unverified_context()
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?interval=1mo&range=max"
    )
    raw = urlopen(Request(url, headers={"User-Agent": UA}), context=ctx, timeout=60).read()
    payload = json.loads(raw.decode("utf-8"))
    result = (payload.get("chart") or {}).get("result") or []
    if not result:
        raise RuntimeError(f"Yahoo chart empty for {symbol}: {payload.get('chart', {}).get('error')}")
    res = result[0]
    ts = res.get("timestamp") or []
    adj = ((res.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose")
    if not ts or not adj:
        raise RuntimeError(f"Yahoo adjclose missing for {symbol}")
    s = pd.Series(adj, index=pd.to_datetime(ts, unit="s"), dtype=float).sort_index()
    s = s[np.isfinite(s) & (s > 0)]
    s = to_month_end(s[~s.index.duplicated(keep="last")])
    s.name = symbol.replace("^", "").replace(".", "_")
    return s


def returns_to_index(r: pd.Series, *, name: str, base: float = 100.0) -> pd.Series:
    """Cumulative index from simple returns, rebased to ``base`` at first valid."""
    r = r.astype(float)
    start = r.first_valid_index()
    out = pd.Series(np.nan, index=r.index, dtype=float)
    if start is None:
        out.name = name
        return out
    path = (1.0 + r.loc[start:].fillna(0.0)).cumprod()
    out.loc[path.index] = base * path / float(path.iloc[0])
    out.name = name
    return out


def total_return_from_price_and_annual_div(
    price: pd.Series,
    annual_div: pd.Series,
    *,
    name: str = "tr",
) -> pd.Series:
    """
    Monthly total-return index from price and **annualized** dividend level.

    Uses Shiller-style income: ``R_t = (P_t + D_t/12) / P_{t-1} - 1``.
    """
    p = price.astype(float)
    d = annual_div.reindex(p.index).astype(float).ffill()
    r = (p + d / 12.0) / p.shift(1) - 1.0
    return returns_to_index(r, name=name)


def total_return_from_price_and_yield(
    price: pd.Series,
    yield_pct: pd.Series,
    *,
    name: str = "tr",
) -> pd.Series:
    """
    Monthly total-return index from price and annual dividend **yield percent**.

    ``R_t = ΔP/P_{t-1} + y_t/(100·12)`` with yield forward-filled.
    """
    p = price.astype(float)
    y = yield_pct.reindex(p.index).astype(float).ffill()
    r = p.pct_change(fill_method=None) + y / (100.0 * 12.0)
    return returns_to_index(r, name=name)


def splice_return_series(
    primary_r: pd.Series,
    filler_r: pd.Series,
) -> pd.Series:
    """Prefer ``primary_r``; fill remaining months from ``filler_r``."""
    out = primary_r.astype(float).copy()
    out = out.combine_first(filler_r.astype(float))
    return out


def load_tsx_dividend_yield() -> pd.Series:
    """TSX Composite dividend yield (%), StatCan 10-10-0125 (ends ~2016)."""
    df = _statcan_csv("10100125")
    label_col = next(
        (c for c in df.columns if "Toronto Stock" in c or "Stock Exchange" in c),
        None,
    )
    if label_col is None:
        raise RuntimeError("Could not find TSX label column in 10-10-0125")
    lab = df[label_col].astype(str)
    mask = lab.str.contains("dividend yields", case=False, na=False)
    sub = df.loc[mask, ["REF_DATE", "VALUE"]].dropna()
    if sub.empty:
        raise RuntimeError("TSX dividend yield not found in StatCan 10-10-0125")
    sub["REF_DATE"] = _parse_ref_date(sub["REF_DATE"])
    s = sub.dropna().set_index("REF_DATE")["VALUE"].astype(float).sort_index()
    s = to_month_end(s[~s.index.duplicated(keep="last")])
    s.name = "tsx_div_yield"
    return s


def load_sp500_tr_usd() -> pd.Series:
    """
    S&P 500 monthly **total return** index (USD).

    Long sample from Shiller price + dividend; after Shiller dividend coverage
    ends, splice Yahoo ``^SP500TR`` growth when available.
    """
    shiller = load_sp500_shiller_table()
    tr = total_return_from_price_and_annual_div(
        shiller["SP500"], shiller["Dividend"], name="sp500_tr_usd"
    )
    try:
        yahoo_tr = load_yahoo_monthly_adj("%5ESP500TR")
        # Splice Yahoo growth after last Shiller month with positive dividend path
        last = tr.dropna().index.max()
        y_r = yahoo_tr.pct_change(fill_method=None)
        t_r = tr.pct_change(fill_method=None)
        # Prefer Shiller through last; Yahoo thereafter
        y_r_after = y_r.loc[y_r.index > last]
        r = t_r.combine_first(y_r)  # fill any holes
        if not y_r_after.empty:
            r.loc[y_r_after.index] = y_r_after
        tr = returns_to_index(r, name="sp500_tr_usd")
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: Yahoo ^SP500TR splice skipped: {exc}")
    return tr


def load_cad_per_usd() -> pd.Series:
    """CAD per USD monthly average (FRED EXCAUS)."""
    s = to_month_end(load_fred("EXCAUS"))
    s.name = "cad_per_usd"
    return s


def equal_weight_combined_index(
    leg_a: pd.Series,
    leg_b: pd.Series,
    *,
    weight_a: float = 0.5,
) -> pd.Series:
    """
    Build a cumulative index from equal-weight (default 50/50) simple returns.

    Uses overlapping months only. Output rebased to 100 at first valid date.
    """
    w_a = float(weight_a)
    w_b = 1.0 - w_a
    a = leg_a.astype(float)
    b = leg_b.astype(float)
    ra = a.pct_change(fill_method=None)
    rb = b.pct_change(fill_method=None)
    both = ra.notna() & rb.notna()
    r = (w_a * ra + w_b * rb).where(both)
    start = r.first_valid_index()
    if start is None:
        out = pd.Series(np.nan, index=a.index.union(b.index), name="equity_comb")
        return out
    path = (1.0 + r.loc[start:].fillna(0.0)).cumprod()
    out = pd.Series(np.nan, index=a.index.union(b.index), dtype=float)
    out.loc[path.index] = 100.0 * path / path.iloc[0]
    out.name = "equity_comb"
    return out.sort_index()


def splice_with_filler(primary: pd.Series, filler: pd.Series) -> pd.Series:
    """
    Use ``primary`` over its sample; outside that window use ``filler`` levels,
    spliced so the series is continuous at the join dates.

    Typical use: S&P/TSX 60 (1982–2023 StatCan) with Composite fill outside.
    """
    p = primary.dropna().astype(float).sort_index()
    f = filler.dropna().astype(float).sort_index()
    if p.empty:
        return f.rename(getattr(primary, "name", None))
    if f.empty:
        return p
    idx = p.index.union(f.index).sort_values()
    out = pd.Series(np.nan, index=idx, dtype=float)

    # Align primary onto filler level at first overlapping date
    overlap = p.index.intersection(f.index)
    if overlap.empty:
        out.loc[p.index] = p
        return out.rename(p.name)

    join0 = overlap.min()
    scale = float(f.loc[join0] / p.loc[join0])
    p_aln = p * scale
    out.loc[p_aln.index] = p_aln

    # Before join: filler
    before = f.loc[f.index < join0]
    out.loc[before.index] = before

    # After last primary: continue with filler growth from last spliced level
    last = p_aln.index.max()
    if last in f.index:
        after = f.loc[f.index > last]
        if not after.empty:
            out.loc[after.index] = float(out.loc[last]) * (after / float(f.loc[last]))
    out.name = p.name
    return out.sort_index()


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
    print("Downloading TSX Composite...")
    tsx = to_month_end(load_tsx())
    print("Downloading S&P/TSX 60...")
    try:
        tsx60 = to_month_end(load_tsx60())
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: TSX60 failed: {exc}")
        tsx60 = pd.Series(dtype=float, name="tsx60")
    print("Downloading TSX dividend yield...")
    try:
        tsx_yield = to_month_end(load_tsx_dividend_yield())
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: TSX yield failed: {exc}")
        tsx_yield = pd.Series(dtype=float, name="tsx_div_yield")
    print("Downloading S&P 500 price (USD)...")
    try:
        sp500 = to_month_end(load_sp500())
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: S&P500 price failed: {exc}")
        sp500 = pd.Series(dtype=float, name="sp500")
    print("Downloading S&P 500 total return (USD)...")
    try:
        sp500_tr_usd = to_month_end(load_sp500_tr_usd())
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: S&P500 TR failed: {exc}")
        sp500_tr_usd = pd.Series(dtype=float, name="sp500_tr_usd")
    print("Downloading CAD per USD (EXCAUS)...")
    try:
        cadusd = to_month_end(load_cad_per_usd())
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: EXCAUS failed: {exc}")
        cadusd = pd.Series(dtype=float, name="cad_per_usd")
    print("Downloading Yahoo ETF adj closes (TR splice)...")
    try:
        xiu_adj = load_yahoo_monthly_adj("XIU.TO")
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: XIU.TO failed: {exc}")
        xiu_adj = pd.Series(dtype=float, name="XIU_TO")
    try:
        xic_adj = load_yahoo_monthly_adj("XIC.TO")
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: XIC.TO failed: {exc}")
        xic_adj = pd.Series(dtype=float, name="XIC_TO")

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
            "tsx60": tsx60,
            "tsx_div_yield": tsx_yield,
            "sp500": sp500,
            "sp500_tr_usd": sp500_tr_usd,
            "cad_per_usd": cadusd,
            "xiu_adj": xiu_adj,
            "xic_adj": xic_adj,
            **rates,
        }
    ).sort_index()

    # Forward-fill lower-frequency house series for plotting continuity
    for col in (
        "house_bis",
        "house_bis_real_pub",
        "rppi_house_resale",
        "crea_mls_hpi",
        "tsx_div_yield",
    ):
        if col in df.columns:
            df[col] = df[col].ffill()

    # --- Total-return equity legs (primary) ---
    # S&P500 TR in CAD: USD total-return wealth × CAD per USD
    if "sp500_tr_usd" in df.columns and "cad_per_usd" in df.columns:
        df["sp500_tr_cad"] = df["sp500_tr_usd"] * df["cad_per_usd"]
    # Price-only CAD (kept for comparison)
    if "sp500" in df.columns and "cad_per_usd" in df.columns:
        df["sp500_cad"] = df["sp500"] * df["cad_per_usd"]

    # TSX Composite TR: price + StatCan yield; after yield ends splice XIC.TO adj
    if "tsx" in df.columns and "tsx_div_yield" in df.columns and df["tsx_div_yield"].notna().any():
        tsx_tr_yield = total_return_from_price_and_yield(
            df["tsx"], df["tsx_div_yield"], name="tsx_tr"
        )
        r_tsx_tr = tsx_tr_yield.pct_change(fill_method=None)
        if "xic_adj" in df.columns and df["xic_adj"].notna().any():
            r_xic = df["xic_adj"].pct_change(fill_method=None)
            yield_end = df["tsx_div_yield"].dropna().index.max()
            r_tsx = r_tsx_tr.copy()
            after_idx = r_tsx.index[r_tsx.index > yield_end]
            r_tsx.loc[after_idx] = r_xic.reindex(after_idx)
            df["tsx_tr"] = returns_to_index(r_tsx, name="tsx_tr")
        else:
            df["tsx_tr"] = tsx_tr_yield
    elif "tsx" in df.columns:
        df["tsx_tr"] = df["tsx"]

    # TSX60 TR: apply Composite yield to TSX60 price; splice XIU.TO adj when available
    if "tsx60" in df.columns and df["tsx60"].notna().any():
        if "tsx_div_yield" in df.columns and df["tsx_div_yield"].notna().any():
            tsx60_tr_core = total_return_from_price_and_yield(
                df["tsx60"], df["tsx_div_yield"], name="tsx60_tr"
            )
            r60 = tsx60_tr_core.pct_change(fill_method=None)
        else:
            r60 = df["tsx60"].pct_change(fill_method=None)
        if "xiu_adj" in df.columns and df["xiu_adj"].notna().any():
            r_xiu = df["xiu_adj"].pct_change(fill_method=None)
            # Prefer XIU adj over the ETF sample (true TR); yield construction before
            start_xiu = df["xiu_adj"].dropna().index.min()
            if start_xiu is not None:
                idx = r60.index[r60.index >= start_xiu]
                r60.loc[idx] = r_xiu.reindex(idx)
        df["tsx60_tr"] = returns_to_index(r60, name="tsx60_tr")
    else:
        df["tsx60_tr"] = pd.Series(dtype=float, name="tsx60_tr")

    # Splice TSX60 TR with Composite TR outside TSX60 window
    if df["tsx60_tr"].notna().any() and df["tsx_tr"].notna().any():
        df["tsx60_tr_spliced"] = splice_with_filler(
            df["tsx60_tr"].dropna(), df["tsx_tr"].dropna()
        )
    elif df["tsx_tr"].notna().any():
        df["tsx60_tr_spliced"] = df["tsx_tr"]

    # Price-only spliced TSX60 (comparison)
    if "tsx60" in df.columns and df["tsx60"].notna().any() and "tsx" in df.columns:
        df["tsx60_spliced"] = splice_with_filler(df["tsx60"].dropna(), df["tsx"].dropna())
    elif "tsx" in df.columns:
        df["tsx60_spliced"] = df["tsx"]

    # Primary combined equity: 50/50 S&P500 TR (CAD) + TSX60 TR
    if "sp500_tr_cad" in df.columns and "tsx60_tr_spliced" in df.columns:
        df["equity_comb"] = equal_weight_combined_index(
            df["sp500_tr_cad"], df["tsx60_tr_spliced"], weight_a=0.5
        )
        df["equity_comb"].name = "equity_comb"
    # Price-only combined for comparison
    if "sp500_cad" in df.columns and "tsx60_spliced" in df.columns:
        df["equity_comb_price"] = equal_weight_combined_index(
            df["sp500_cad"], df["tsx60_spliced"], weight_a=0.5
        )

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

    equity_nominal = [
        c
        for c in (
            "tsx",
            "tsx_tr",
            "tsx60",
            "tsx60_tr",
            "tsx60_tr_spliced",
            "tsx60_spliced",
            "sp500_cad",
            "sp500_tr_cad",
            "equity_comb",
            "equity_comb_price",
        )
        if c in df.columns and df[c].notna().any()
    ]
    for col in equity_nominal:
        df[f"{col}_real"] = df[col] / df["cpi"]

    df["tsx_real"] = df["tsx"] / df["cpi"]
    # Alias used elsewhere
    if "house_bis_real" in df.columns:
        pass
    df["house_real"] = df.get("house_bis_real", df["house_bis"] / df["cpi"])

    # Preferred equity for estimation: 50/50 total-return combined
    if "equity_comb_real" in df.columns and df["equity_comb_real"].notna().any():
        df["equity_real"] = df["equity_comb_real"]
    elif "tsx_tr_real" in df.columns:
        df["equity_real"] = df["tsx_tr_real"]
    else:
        df["equity_real"] = df["tsx_real"]

    base = df[["house_real", "equity_real", "cpi"]].dropna().index.min()
    if pd.notna(base):
        rebase_cols = [f"{c}_real" for c in housing_nominal] + [
            f"{c}_real" for c in equity_nominal
        ] + [
            "tsx_real",
            "equity_real",
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

    # Equity monthly real returns (primary = combined TR; keep legs for comparison)
    equity_ret_src = {
        "r_equity_m": "equity_comb" if "equity_comb" in df.columns else "tsx_tr",
        "r_tsx_m": "tsx",
        "r_tsx_tr_m": "tsx_tr",
        "r_tsx60_m": "tsx60",
        "r_tsx60_tr_m": "tsx60_tr",
        "r_sp500_cad_m": "sp500_cad",
        "r_sp500_tr_cad_m": "sp500_tr_cad",
        "r_equity_comb_m": "equity_comb",
        "r_equity_comb_price_m": "equity_comb_price",
    }
    for rcol, src in equity_ret_src.items():
        if src in df.columns and df[src].notna().any():
            df[rcol] = (1.0 + df[src].pct_change(fill_method=None)) / (
                1.0 + df["pi_m"]
            ) - 1.0

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
        ("cum_tsx_real", "r_tsx_m"),
        ("cum_tsx_tr_real", "r_tsx_tr_m"),
        ("cum_tsx60_real", "r_tsx60_m"),
        ("cum_tsx60_tr_real", "r_tsx60_tr_m"),
        ("cum_sp500_cad_real", "r_sp500_cad_m"),
        ("cum_sp500_tr_cad_real", "r_sp500_tr_cad_m"),
        ("cum_equity_comb_real", "r_equity_comb_m"),
        ("cum_equity_comb_price_real", "r_equity_comb_price_m"),
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
    """Annual real returns for equity legs and housing (calendar-year compound)."""
    cols = [
        c
        for c in (
            "r_equity_m",
            "r_equity_comb_m",
            "r_equity_comb_price_m",
            "r_tsx_m",
            "r_tsx_tr_m",
            "r_tsx60_m",
            "r_tsx60_tr_m",
            "r_sp500_cad_m",
            "r_sp500_tr_cad_m",
            "r_house_m",
            "r_nhpi_m",
        )
        if c in df.columns
    ]
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
            "r_equity_comb_m": "r_equity_comb",
            "r_equity_comb_price_m": "r_equity_comb_price",
            "r_tsx_m": "r_tsx",
            "r_tsx_tr_m": "r_tsx_tr",
            "r_tsx60_m": "r_tsx60",
            "r_tsx60_tr_m": "r_tsx60_tr",
            "r_sp500_cad_m": "r_sp500_cad",
            "r_sp500_tr_cad_m": "r_sp500_tr_cad",
            "r_house_m": "r_house",
            "r_nhpi_m": "r_nhpi",
        }
    )
    out.index = pd.to_datetime(out.index.astype(str) + "-12-31")
    # Primary equity for bivariate AR(1) is the combined index when available
    if "r_equity" not in out.columns and "r_tsx" in out.columns:
        out["r_equity"] = out["r_tsx"]
    return out.dropna(subset=["r_equity", "r_house"])


def make_plots(df: pd.DataFrame, ann: pd.DataFrame, est: Dict[str, float]) -> None:
    # 1) Cumulative real indices
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for col, label in [
        ("cum_equity_comb_real", "Equity 50/50 TR S&P500(CAD)+TSX60 real"),
        ("cum_sp500_tr_cad_real", "S&P500 CAD total return real"),
        ("cum_tsx_tr_real", "TSX Composite total return real"),
        ("cum_equity_comb_price_real", "50/50 price-only (comparison)"),
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
        ("r_equity_comb_m", "50/50 TR combined real monthly"),
        ("r_sp500_tr_cad_m", "S&P500 CAD TR real monthly"),
        ("r_tsx_tr_m", "TSX Composite TR real monthly"),
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
    idx = annual_real_indices(df)
    idx.to_csv(OUT / "annual_real_price_indices.csv")
    est = estimate_bivar_ar1_from_indices(idx["P_equity"], idx["P_house"])
    est_simple = estimate_diag_ar1_simple(
        ann["r_equity"].to_numpy(), ann["r_house"].to_numpy()
    )
    pd.Series(est).to_csv(OUT / "bivar_ar1_equity_housing.csv", header=["value"])
    out_json = save_return_estimates(
        est,
        OUT / "bivar_ar1_canada_return_params.json",
        comparison=est_simple,
        meta={
            "equity": "equity_comb_real year-end (50/50 S&P500 CAD TR + TSX60 TR)",
            "housing": "house_real year-end (BIS/CPI)",
            "frequency": "annual",
            "returns": "Δlog real total-return index; AR(1) on demeaned log growth",
            "Rf_bar_mapping": "mean(exp(Δlog P))",
            "dividends": "S&P500: Shiller D/12 (+ Yahoo ^SP500TR splice); Canada: StatCan yield (+ XIU/XIC adj splice)",
        },
    )
    print(f"Wrote {out_json}")
    print("Diagonal AR(1) on demeaned annual Δlog real indices (primary equity = combined):")
    for k, v in est.items():
        print(f"  {k}: {v:.6g}" if isinstance(v, float) else f"  {k}: {v}")

    # Scatter still uses simple annual returns for readability
    make_plots(df, ann, est_simple)
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
        "tsx60",
        "sp500",
        "sp500_cad",
        "equity_comb",
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
