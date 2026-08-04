"""Inequality metrics for simulated wealth panels."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .survey.weighted import wmean, wquantile


def gini(x: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """
    Gini coefficient of a nonnegative 1-d array (NaNs dropped).

    With ``weights``, uses the Brown weighted formula
    ``G = 1 - 2 * sum_i w_i (cumsum(w x) - w x / 2) / (sum(w) * sum(w x))``.
    """
    v = np.asarray(x, dtype=float).ravel()
    if weights is None:
        v = v[np.isfinite(v)]
        v = v[v >= 0]
        if v.size == 0:
            return float("nan")
        if np.allclose(v, 0):
            return 0.0
        v = np.sort(v)
        n = v.size
        idx = np.arange(1, n + 1)
        return float((2.0 * np.sum(idx * v) / (n * np.sum(v))) - (n + 1) / n)

    w = np.asarray(weights, dtype=float).ravel()
    if w.size != v.size:
        raise ValueError("weights length mismatch")
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0) & (v >= 0)
    v, w = v[ok], w[ok]
    if v.size == 0:
        return float("nan")
    if np.allclose(v, 0):
        return 0.0
    order = np.argsort(v)
    v, w = v[order], w[order]
    sw = float(w.sum())
    wx = float(np.dot(w, v))
    if sw <= 0 or wx == 0:
        return float("nan")
    return float(1.0 - 2.0 * np.sum(w * (np.cumsum(w * v) - w * v / 2.0)) / (sw * wx))


def top_share(
    x: np.ndarray,
    frac: float = 0.1,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Share of total wealth held by the top ``frac`` of (weighted) agents."""
    v = np.asarray(x, dtype=float).ravel()
    if weights is None:
        v = v[np.isfinite(v)]
        total = v.sum()
        if total <= 0 or v.size == 0:
            return float("nan")
        k = max(1, int(np.ceil(frac * v.size)))
        return float(np.sort(v)[-k:].sum() / total)

    w = np.asarray(weights, dtype=float).ravel()
    if w.size != v.size:
        raise ValueError("weights length mismatch")
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[ok], w[ok]
    if v.size == 0:
        return float("nan")
    order = np.argsort(-v)
    v, w = v[order], w[order]
    total = float(np.dot(w, v))
    if total <= 0:
        return float("nan")
    sw = float(w.sum())
    target = frac * sw
    cum_w = np.cumsum(w)
    k = int(np.searchsorted(cum_w, target, side="left")) + 1
    k = min(max(k, 1), v.size)
    return float(np.dot(w[:k], v[:k]) / total)


def wealth_inequality_path(
    wealth: np.ndarray,
    alive_mask: Optional[np.ndarray] = None,
    top_fracs: Iterable[float] = (0.01, 0.1),
    weights: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Calendar-time inequality path from a (T, N) wealth array.

    Parameters
    ----------
    wealth :
        Absolute wealth by period and agent.
    alive_mask :
        Optional boolean (T, N); if given, dead agents are excluded.
    weights :
        Optional length-N or (T, N) agent weights.
    """
    w = np.asarray(wealth, dtype=float)
    t_sim, n = w.shape
    if weights is None:
        wt_arr = np.ones((t_sim, n), dtype=float)
    else:
        ww = np.asarray(weights, dtype=float)
        if ww.ndim == 1:
            wt_arr = np.broadcast_to(ww, (t_sim, n))
        else:
            wt_arr = ww
    rows = []
    for t in range(t_sim):
        wt = w[t]
        ww = wt_arr[t]
        if alive_mask is not None:
            mask = np.asarray(alive_mask[t], dtype=bool)
            wt = wt[mask]
            ww = ww[mask]
        row = {
            "t": t,
            "mean_wealth": float(wmean(wt, ww)) if wt.size else float("nan"),
            "median_wealth": float(wquantile(wt, ww, 0.5)) if wt.size else float("nan"),
            "gini": gini(wt, ww),
            "n_alive": int(wt.size),
        }
        for f in top_fracs:
            row[f"top{int(100 * f)}_share"] = top_share(wt, f, ww)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_path(df: pd.DataFrame) -> Dict[str, float]:
    """Compact summary of an inequality path DataFrame."""
    return {
        "gini_final": float(df["gini"].iloc[-1]),
        "gini_mean": float(df["gini"].mean()),
        "top10_final": float(df.get("top10_share", df.filter(like="top10").iloc[:, 0]).iloc[-1]),
        "mean_wealth_final": float(df["mean_wealth"].iloc[-1]),
    }
