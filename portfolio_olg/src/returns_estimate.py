"""Estimate and load diagonal bivariate AR(1) return parameters."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .returns_ca import CanadaReturnParams

PathLike = Union[str, Path]

_ROOT = Path(__file__).resolve().parents[1]

# Tracked calibration artifact (committed); estimation also writes under output/.
TRACKED_ESTIMATE_PATH = _ROOT / "params" / "bivar_ar1_canada_return_params.json"
OUTPUT_ESTIMATE_PATH = (
    _ROOT / "output" / "macro_returns" / "bivar_ar1_canada_return_params.json"
)
# Prefer tracked params so demos work without regenerating output/
DEFAULT_ESTIMATE_PATH = TRACKED_ESTIMATE_PATH


def annual_real_indices(
    monthly: pd.DataFrame,
    *,
    equity_col: str = "equity_comb_real",
    housing_col: str = "house_real",
) -> pd.DataFrame:
    """
    Calendar year-end levels of real equity and housing price indices.

    Default equity is the 50/50 S&P500(CAD)+TSX60 combined **total-return**
    real index.
    """
    need = [equity_col, housing_col]
    missing = [c for c in need if c not in monthly.columns]
    if missing:
        raise KeyError(f"monthly panel missing columns: {missing}")
    sub = monthly[need].copy()
    sub = sub.dropna(how="any")
    ann = sub.groupby(sub.index.year).last()
    ann.index = pd.to_datetime(ann.index.astype(str) + "-12-31")
    return ann.rename(columns={equity_col: "P_equity", housing_col: "P_house"})


def log_diff_growth(level: np.ndarray | pd.Series) -> np.ndarray:
    """First difference of log levels: g_t = log P_t - log P_{t-1}."""
    x = np.asarray(level, dtype=float)
    if x.size < 2:
        raise ValueError("need at least 2 index levels for log differences")
    if np.any(~np.isfinite(x)) or np.any(x <= 0):
        raise ValueError("index levels must be positive and finite")
    return np.diff(np.log(x))


def _diag_ar1_on_demeaned(xf: np.ndarray, xh: np.ndarray) -> Dict[str, float]:
    """OLS diagonal AR(1) on already-demeaned series (zero mean by construction)."""
    if xf.size < 3:
        raise ValueError("need at least 3 overlapping observations for AR(1)")
    yf, yh = xf[1:], xh[1:]
    xf_l, xh_l = xf[:-1], xh[:-1]
    rho_f = float(np.dot(xf_l, yf) / np.dot(xf_l, xf_l))
    rho_h = float(np.dot(xh_l, yh) / np.dot(xh_l, xh_l))
    eps_f = yf - rho_f * xf_l
    eps_h = yh - rho_h * xh_l
    return {
        "rho_f": rho_f,
        "rho_h": rho_h,
        "sigma_f": float(eps_f.std(ddof=1)),
        "sigma_h": float(eps_h.std(ddof=1)),
        "corr_innov": float(np.corrcoef(eps_f, eps_h)[0, 1]),
        "n_obs_ar1": float(len(yf)),
    }


def estimate_diag_ar1_simple(r_f: np.ndarray, r_h: np.ndarray) -> Dict[str, float]:
    """
    Diagonal bivariate AR(1) on overlapping **simple** returns (comparison path).

        (r_{t+1} - mu) = Phi (r_t - mu) + eps_{t+1},  Phi = diag(rho_f, rho_h).
    """
    rf = np.asarray(r_f, dtype=float)
    rh = np.asarray(r_h, dtype=float)
    ok = np.isfinite(rf) & np.isfinite(rh)
    rf, rh = rf[ok], rh[ok]
    if rf.size < 3:
        raise ValueError("need at least 3 overlapping observations for AR(1)")
    mu_f, mu_h = float(rf.mean()), float(rh.mean())
    ar = _diag_ar1_on_demeaned(rf - mu_f, rh - mu_h)
    return {
        "mu_f": mu_f,
        "mu_h": mu_h,
        "rho_f": ar["rho_f"],
        "rho_h": ar["rho_h"],
        "sigma_f": ar["sigma_f"],
        "sigma_h": ar["sigma_h"],
        "corr_innov": ar["corr_innov"],
        "corr_unconditional": float(np.corrcoef(rf, rh)[0, 1]),
        "n_obs": float(rf.size),
        "n_obs_ar1": ar["n_obs_ar1"],
        "mean_Rf_gross": 1.0 + mu_f,
        "mean_Rh_gross": 1.0 + mu_h,
    }


# Back-compat alias used by older call sites
estimate_diag_ar1 = estimate_diag_ar1_simple


def estimate_bivar_ar1_from_indices(
    P_f: np.ndarray | pd.Series,
    P_h: np.ndarray | pd.Series,
) -> Dict[str, float]:
    """
    Log-difference real price indices, save mean log growth, demean, AR(1).

    Steps
    -----
    1. ``g^i_t = Δ log P^i_t``
    2. ``μ_i = mean(g^i)``  (saved as ``mean_log_f`` / ``mean_log_h``)
    3. ``ε^i_t = g^i_t - μ_i``
    4. Diagonal AR(1) on ``(ε^f, ε^h)``

    Also reports ``mean_Rf_gross = mean(exp(g^f))`` (sample arithmetic gross),
    which matches the model's ``Rf_bar = E[R]`` target more closely than
    ``exp(μ)`` alone.
    """
    # Align finite positive levels
    pf = np.asarray(P_f, dtype=float)
    ph = np.asarray(P_h, dtype=float)
    n = min(pf.size, ph.size)
    pf, ph = pf[-n:], ph[-n:]
    ok = np.isfinite(pf) & np.isfinite(ph) & (pf > 0) & (ph > 0)
    pf, ph = pf[ok], ph[ok]

    g_f = log_diff_growth(pf)
    g_h = log_diff_growth(ph)
    # log_diff drops first obs; levels already paired so growths stay paired
    mean_log_f = float(g_f.mean())
    mean_log_h = float(g_h.mean())
    eps_f = g_f - mean_log_f
    eps_h = g_h - mean_log_h
    ar = _diag_ar1_on_demeaned(eps_f, eps_h)
    gross_f = np.exp(g_f)
    gross_h = np.exp(g_h)
    return {
        "mean_log_f": mean_log_f,
        "mean_log_h": mean_log_h,
        "exp_mean_log_f": float(np.exp(mean_log_f)),
        "exp_mean_log_h": float(np.exp(mean_log_h)),
        "mean_Rf_gross": float(gross_f.mean()),
        "mean_Rh_gross": float(gross_h.mean()),
        "rho_f": ar["rho_f"],
        "rho_h": ar["rho_h"],
        "sigma_f": ar["sigma_f"],
        "sigma_h": ar["sigma_h"],
        "corr_innov": ar["corr_innov"],
        "corr_unconditional": float(np.corrcoef(g_f, g_h)[0, 1]),
        "n_obs": float(g_f.size),
        "n_obs_ar1": ar["n_obs_ar1"],
        # aliases used by mapping / comparison with simple path
        "mu_f": mean_log_f,
        "mu_h": mean_log_h,
    }


def estimates_to_canada_return_params(
    est: Mapping[str, float],
    *,
    n_f: int = 3,
    n_h: int = 3,
    use_exp_mean_log: bool = False,
) -> CanadaReturnParams:
    """
    Map estimates into ``CanadaReturnParams``.

    - ``Rf_bar`` / ``Rh_bar`` ← sample arithmetic gross ``mean_R*_gross``
      (or ``exp_mean_log_*`` if ``use_exp_mean_log=True``)
    - ``rho_*``, ``sigma_*``, ``corr`` ← demeaned log AR(1)
    """
    if use_exp_mean_log:
        rf_bar = float(est["exp_mean_log_f"])
        rh_bar = float(est["exp_mean_log_h"])
    else:
        rf_bar = float(est["mean_Rf_gross"])
        rh_bar = float(est["mean_Rh_gross"])
    return CanadaReturnParams(
        Rf_bar=rf_bar,
        Rh_bar=rh_bar,
        rho_f=float(est["rho_f"]),
        rho_h=float(est["rho_h"]),
        sigma_f=float(est["sigma_f"]),
        sigma_h=float(est["sigma_h"]),
        corr=float(est["corr_innov"]),
        n_f=int(n_f),
        n_h=int(n_h),
    )


def compare_estimate_dicts(
    log_est: Mapping[str, float],
    simple_est: Mapping[str, float],
) -> pd.DataFrame:
    """Side-by-side comparison of log-index vs simple-return AR(1) estimates."""
    rows = [
        ("mean growth (log mu vs simple mu)", "mean_log_f", "mu_f", "mean_log_h", "mu_h"),
        ("exp(mean log) vs 1+mu", "exp_mean_log_f", "mean_Rf_gross", "exp_mean_log_h", "mean_Rh_gross"),
        ("arith. gross E[R]", "mean_Rf_gross", "mean_Rf_gross", "mean_Rh_gross", "mean_Rh_gross"),
        ("rho", "rho_f", "rho_f", "rho_h", "rho_h"),
        ("sigma innov", "sigma_f", "sigma_f", "sigma_h", "sigma_h"),
        ("corr innov", "corr_innov", "corr_innov", "corr_innov", "corr_innov"),
        ("corr unc.", "corr_unconditional", "corr_unconditional", "corr_unconditional", "corr_unconditional"),
    ]
    out = []
    for label, lf, sf, lh, sh in rows:
        out.append(
            {
                "quantity": label,
                "log_f": float(log_est[lf]),
                "simple_f": float(simple_est[sf]),
                "log_h": float(log_est[lh]),
                "simple_h": float(simple_est[sh]),
            }
        )
    return pd.DataFrame(out).set_index("quantity")


def save_return_estimates(
    est: Mapping[str, float],
    path: Optional[PathLike] = None,
    *,
    n_f: int = 3,
    n_h: int = 3,
    meta: Optional[Mapping[str, Any]] = None,
    comparison: Optional[Mapping[str, float]] = None,
) -> Path:
    """Write raw estimates plus mapped ``CanadaReturnParams`` fields to JSON.

    Always also mirrors to ``TRACKED_ESTIMATE_PATH`` when ``path`` is the output
    artifact (or omitted), so the demo notebook can load a committed file.
    """
    path = Path(path) if path is not None else OUTPUT_ESTIMATE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    params = estimates_to_canada_return_params(est, n_f=n_f, n_h=n_h)
    payload: Dict[str, Any] = {
        "estimates": {k: float(v) for k, v in est.items()},
        "canada_return_params": asdict(params),
        "mapping_notes": {
            "mean_log_f": "saved mean log growth (equity); not Rf_bar directly",
            "mean_log_h": "saved mean log growth (housing); not Rh_bar directly",
            "mean_Rf_gross": "Rf_bar  (sample mean of exp(Δlog P))",
            "mean_Rh_gross": "Rh_bar  (sample mean of exp(Δlog P))",
            "rho_f": "rho_f",
            "rho_h": "rho_h",
            "sigma_f": "sigma_f",
            "sigma_h": "sigma_h",
            "corr_innov": "corr",
            "equity": "total return (dividends included) for financial asset",
            "housing": "price-only (no rent) for housing asset",
        },
        "meta": dict(meta or {}),
    }
    if comparison is not None:
        payload["simple_return_comparison"] = {k: float(v) for k, v in comparison.items()}
    text = json.dumps(payload, indent=2)
    path.write_text(text, encoding="utf-8")
    # Keep committed params in sync when writing the usual output path
    if path.resolve() != TRACKED_ESTIMATE_PATH.resolve():
        TRACKED_ESTIMATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        TRACKED_ESTIMATE_PATH.write_text(text, encoding="utf-8")
    return path


def load_return_estimates(path: Optional[PathLike] = None) -> Dict[str, Any]:
    """Load JSON written by :func:`save_return_estimates`."""
    if path is not None:
        path = Path(path)
        candidates = [path]
    else:
        candidates = [TRACKED_ESTIMATE_PATH, OUTPUT_ESTIMATE_PATH]
    for cand in candidates:
        if cand.is_file():
            return json.loads(cand.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        "Return estimates not found at "
        + " or ".join(str(c) for c in candidates)
        + ". Run notebooks/04_estimate_bivariate_returns.ipynb first."
    )


def load_canada_return_params(
    path: Optional[PathLike] = None,
    *,
    n_f: Optional[int] = None,
    n_h: Optional[int] = None,
) -> CanadaReturnParams:
    """
    Load mapped ``CanadaReturnParams`` from the saved JSON artifact.

    Optional ``n_f`` / ``n_h`` override grid sizes without re-estimating.
    """
    payload = load_return_estimates(path)
    raw = dict(payload["canada_return_params"])
    if n_f is not None:
        raw["n_f"] = int(n_f)
    if n_h is not None:
        raw["n_h"] = int(n_h)
    return CanadaReturnParams(**raw)


def load_mean_log_growth(path: Optional[PathLike] = None) -> Tuple[float, float]:
    """Return ``(mean_log_f, mean_log_h)`` from the saved artifact."""
    est = load_return_estimates(path)["estimates"]
    return float(est["mean_log_f"]), float(est["mean_log_h"])
