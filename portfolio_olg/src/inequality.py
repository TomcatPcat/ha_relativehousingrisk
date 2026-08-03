"""Inequality metrics for simulated wealth panels."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def gini(x: np.ndarray) -> float:
    """Gini coefficient of a nonnegative 1-d array (NaNs dropped)."""
    v = np.asarray(x, dtype=float).ravel()
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


def top_share(x: np.ndarray, frac: float = 0.1) -> float:
    """Share of total wealth held by the top ``frac`` of agents."""
    v = np.asarray(x, dtype=float).ravel()
    v = v[np.isfinite(v)]
    total = v.sum()
    if total <= 0 or v.size == 0:
        return float("nan")
    k = max(1, int(np.ceil(frac * v.size)))
    return float(np.sort(v)[-k:].sum() / total)


def wealth_inequality_path(
    wealth: np.ndarray,
    alive_mask: Optional[np.ndarray] = None,
    top_fracs: Iterable[float] = (0.01, 0.1),
) -> pd.DataFrame:
    """
    Calendar-time inequality path from a (T, N) wealth array.

    Parameters
    ----------
    wealth :
        Absolute wealth by period and agent.
    alive_mask :
        Optional boolean (T, N); if given, dead agents are excluded.
    """
    w = np.asarray(wealth, dtype=float)
    t_sim, _ = w.shape
    rows = []
    for t in range(t_sim):
        wt = w[t]
        if alive_mask is not None:
            wt = wt[np.asarray(alive_mask[t], dtype=bool)]
        row = {
            "t": t,
            "mean_wealth": float(np.nanmean(wt)) if wt.size else float("nan"),
            "median_wealth": float(np.nanmedian(wt)) if wt.size else float("nan"),
            "gini": gini(wt),
            "n_alive": int(wt.size),
        }
        for f in top_fracs:
            row[f"top{int(100 * f)}_share"] = top_share(wt, f)
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
