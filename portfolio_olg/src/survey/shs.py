"""SHS (Survey of Household Spending) loaders and C/Y moments."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .paths import load_local_paths, require_dir, require_file
from .weighted import wmean, wshare

# 1-indexed inclusive positions from StatCan reading cards / CA_costsinflation.
_COLS_2023 = [
    ("CASEID", 1, 6),
    ("WEIGHTD", 7, 24),
    ("PROV", 25, 26),
    ("RP_AGEGP", 36, 37),
    ("TENURE", 47, 47),
    ("HHTOTINC", 66, 76),
    ("TC001", 3664, 3674),
    ("TX010", 3994, 4004),
]

# 2017 diary: clothing=CL001→CL030 slot; taxes=TX001→TX010 slot.
_COLS_2017 = [
    ("CASEID", 1, 6),
    ("WEIGHTD", 7, 24),
    ("PROV", 25, 26),
    ("RP_AGEGP", 37, 38),
    ("TENURE", 49, 49),
    ("HHTOTINC", 236, 246),
    ("TC001", 3988, 3998),
    ("TX010", 4362, 4372),
]

_MISSING_NUMERIC = 99_999_999.96

_TIER_A_YEARS = (2017, 2019, 2023)


def _read_fwf(path: Path, cols: Sequence[tuple]) -> pd.DataFrame:
    names = [c[0] for c in cols]
    # pandas colspecs are 0-indexed half-open [start, end)
    colspecs = [(c[1] - 1, c[2]) for c in cols]
    df = pd.read_fwf(
        path,
        colspecs=colspecs,
        names=names,
        encoding="latin-1",
        dtype=str,
    )
    return df


def _to_num(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s.astype(str).str.strip(), errors="coerce")
    v = v.where(v < _MISSING_NUMERIC, np.nan)
    return v


def _harmonize_tier_a(df: pd.DataFrame, year: int) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "year": int(year),
            "tier": "A",
            "caseid": df["CASEID"].astype(str).str.strip(),
            "weight": _to_num(df["WEIGHTD"]),
            "hhtotinc": _to_num(df["HHTOTINC"]),
            "consumption": _to_num(df["TC001"]),
            "taxes": _to_num(df["TX010"]) if "TX010" in df.columns else np.nan,
        }
    )
    if "PROV" in df.columns:
        out["prov"] = df["PROV"].astype(str).str.strip()
    if "TENURE" in df.columns:
        out["tenure"] = df["TENURE"].astype(str).str.strip()
    out["income_at"] = out["hhtotinc"] - out["taxes"]
    # Prefer after-tax; fall back to total income if taxes missing
    out["inc_net"] = out["income_at"].where(out["income_at"].notna(), out["hhtotinc"])
    out = out[out["weight"].notna() & (out["weight"] > 0)].copy()
    return out.reset_index(drop=True)


def load_shs_2023(ry2023: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    root = Path(ry2023) if ry2023 is not None else load_local_paths()["shs_ry2023"]
    root = require_dir(Path(root), "SHS RY2023")
    path = require_file(root / "Data" / "TXT" / "PUMF_SHS_2023.txt", "SHS 2023 PUMF")
    return _harmonize_tier_a(_read_fwf(path, _COLS_2023), 2023)


def load_shs_2019(edm: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    root = Path(edm) if edm is not None else load_local_paths()["shs_edm"]
    root = require_dir(Path(root), "SHS EDM")
    path = require_file(root / "2019" / "shs_2019.csv", "SHS 2019 CSV")
    raw = pd.read_csv(path, low_memory=False)
    upper = {str(c).strip().upper(): c for c in raw.columns}

    def pick(*names: str) -> Optional[str]:
        for n in names:
            if n in raw.columns:
                return n
            if n.upper() in upper:
                return upper[n.upper()]
        return None

    frame = pd.DataFrame()
    for std, cands in [
        ("CASEID", ("CASEID", "CaseID")),
        ("WEIGHTD", ("WEIGHTD", "WeightD")),
        ("PROV", ("PROV", "Prov")),
        ("TENURE", ("TENURE", "Tenure")),
        ("HHTOTINC", ("HHTOTINC", "HH_TotInc", "HHTotInc")),
        ("TC001", ("TC001",)),
        ("TX010", ("TX010", "TX001")),
    ]:
        src = pick(*cands)
        frame[std] = raw[src] if src is not None else np.nan
    return _harmonize_tier_a(frame, 2019)


def load_shs_2017(edm: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    root = Path(edm) if edm is not None else load_local_paths()["shs_edm"]
    root = require_dir(Path(root), "SHS EDM")
    zip_path = root / "2017" / "data" / "RAW.zip"
    member = "RAW/pumf_shs2017_diary_flatfile.txt"
    if not zip_path.is_file():
        raise FileNotFoundError(f"SHS 2017 RAW.zip not found at {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as fh:
            # Temporary extract to bytes via pandas from a NamedTemporaryFile
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
                tmp.write(fh.read())
                tmp_path = Path(tmp.name)
    try:
        df = _read_fwf(tmp_path, _COLS_2017)
    finally:
        tmp_path.unlink(missing_ok=True)
    return _harmonize_tier_a(df, 2017)


def load_shs_tier_a(
    *,
    ry2023: Optional[Union[str, Path]] = None,
    edm: Optional[Union[str, Path]] = None,
    years: Sequence[int] = _TIER_A_YEARS,
) -> pd.DataFrame:
    """Load post-redesign SHS years (2017, 2019, 2023). Skips missing sources."""
    paths = load_local_paths()
    ry = ry2023 if ry2023 is not None else paths["shs_ry2023"]
    ed = edm if edm is not None else paths["shs_edm"]
    pieces = []
    loaders = {
        2017: lambda: load_shs_2017(ed),
        2019: lambda: load_shs_2019(ed),
        2023: lambda: load_shs_2023(ry),
    }
    for y in years:
        if y not in loaders:
            continue
        try:
            pieces.append(loaders[y]())
        except FileNotFoundError:
            continue
    if not pieces:
        raise FileNotFoundError("No SHS Tier A years could be loaded")
    return pd.concat(pieces, ignore_index=True)


def load_shs_tier_b(
    path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load pre-redesign consolidated SHS (1997–2009)."""
    if path is None:
        path = load_local_paths()["shs_consolidated"]
    path = require_file(Path(path), "SHS consolidated")
    raw = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame(
        {
            "year": pd.to_numeric(raw["survey_year"], errors="coerce"),
            "tier": "B",
            "caseid": raw["case_id"].astype(str) if "case_id" in raw.columns else np.arange(len(raw)),
            "weight": pd.to_numeric(raw["weights"], errors="coerce"),
            "consumption": pd.to_numeric(raw["consumption"], errors="coerce"),
            "inc_net": pd.to_numeric(raw["net_income"], errors="coerce"),
            "hhtotinc": pd.to_numeric(raw["gross_income"], errors="coerce")
            if "gross_income" in raw.columns
            else np.nan,
            "taxes": pd.to_numeric(raw["personal_taxes"], errors="coerce")
            if "personal_taxes" in raw.columns
            else np.nan,
        }
    )
    out["income_at"] = out["inc_net"]
    out = out[out["weight"].notna() & (out["weight"] > 0)].copy()
    out = out[out["year"].notna()].copy()
    out["year"] = out["year"].astype(int)
    return out.reset_index(drop=True)


def load_shs_all(
    *,
    include_tier_a: bool = True,
    include_tier_b: bool = True,
) -> pd.DataFrame:
    """Concatenate Tier A and Tier B with a ``tier`` column (do not splice trends)."""
    frames = []
    if include_tier_a:
        try:
            frames.append(load_shs_tier_a())
        except FileNotFoundError:
            pass
    if include_tier_b:
        try:
            frames.append(load_shs_tier_b())
        except FileNotFoundError:
            pass
    if not frames:
        raise FileNotFoundError("No SHS data could be loaded")
    return pd.concat(frames, ignore_index=True)


def shs_cy_moments(
    df: pd.DataFrame,
    by: Union[str, Sequence[str], None] = "year",
) -> pd.DataFrame:
    """
    Aggregate consumption-to-net-income ratio: ``sum(w C) / sum(w Y)`` for Y > 0.

    Uses ``consumption`` and ``inc_net`` (after-tax + transfers).
    """
    for c in ("weight", "consumption", "inc_net"):
        if c not in df.columns:
            raise KeyError(f"shs_cy_moments missing {c}")

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
        c = g["consumption"].to_numpy(dtype=float)
        y = g["inc_net"].to_numpy(dtype=float)
        pos = np.isfinite(y) & (y > 0) & np.isfinite(c) & np.isfinite(w) & (w > 0)
        row = {
            "n": int(pos.sum()),
            "n_raw": int(len(g)),
            "cy_ratio": wshare(c[pos], y[pos], w[pos]),
            "mean_consumption": wmean(c[pos], w[pos]) if pos.any() else float("nan"),
            "mean_inc_net": wmean(y[pos], w[pos]) if pos.any() else float("nan"),
        }
        if "tier" in g.columns:
            tiers = sorted({str(t) for t in g["tier"].dropna().unique()})
            row["tier"] = ",".join(tiers) if tiers else None
        if group_keys:
            if len(group_keys) == 1:
                row[group_keys[0]] = key[0] if isinstance(key, tuple) else key
            else:
                vals = key if isinstance(key, tuple) else (key,)
                for k, v in zip(group_keys, vals):
                    row[k] = v
        rows.append(row)

    front = group_keys + (["tier"] if any("tier" in r for r in rows) else [])
    cols = front + ["n", "n_raw", "cy_ratio", "mean_consumption", "mean_inc_net"]
    # de-dupe while preserving order
    seen: set[str] = set()
    cols = [c for c in cols if not (c in seen or seen.add(c))]
    frame = pd.DataFrame(rows)
    return frame[[c for c in cols if c in frame.columns]]
