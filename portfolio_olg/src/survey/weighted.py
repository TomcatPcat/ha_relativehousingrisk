"""Fast weighted means, quantiles, and sum-ratio shares (NumPy only)."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float]]


def _as_1d(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=float).ravel()


def _mask(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.isfinite(x) & np.isfinite(w) & (w > 0)


def wsum(x: ArrayLike, w: Optional[ArrayLike] = None) -> float:
    """Weighted sum ``sum(w * x)`` (unweighted if ``w`` is None)."""
    xv = _as_1d(x)
    if w is None:
        ok = np.isfinite(xv)
        return float(xv[ok].sum()) if ok.any() else float("nan")
    wv = _as_1d(w)
    if xv.size != wv.size:
        raise ValueError(f"x length {xv.size} != w length {wv.size}")
    ok = _mask(xv, wv)
    if not ok.any():
        return float("nan")
    return float(np.dot(xv[ok], wv[ok]))


def wmean(x: ArrayLike, w: Optional[ArrayLike] = None) -> float:
    """Weighted mean."""
    xv = _as_1d(x)
    if w is None:
        ok = np.isfinite(xv)
        return float(xv[ok].mean()) if ok.any() else float("nan")
    wv = _as_1d(w)
    if xv.size != wv.size:
        raise ValueError(f"x length {xv.size} != w length {wv.size}")
    ok = _mask(xv, wv)
    if not ok.any():
        return float("nan")
    sw = float(wv[ok].sum())
    if sw <= 0:
        return float("nan")
    return float(np.dot(xv[ok], wv[ok]) / sw)


def wshare(
    numer: ArrayLike,
    denom: ArrayLike,
    w: Optional[ArrayLike] = None,
) -> float:
    """
    Aggregate share ``sum(w * numer) / sum(w * denom)``.

    Rows with non-finite numer/denom or non-positive weight are dropped.
    """
    n = _as_1d(numer)
    d = _as_1d(denom)
    if n.size != d.size:
        raise ValueError("numer and denom length mismatch")
    if w is None:
        wv = np.ones(n.size, dtype=float)
    else:
        wv = _as_1d(w)
        if wv.size != n.size:
            raise ValueError("w length mismatch")
    ok = np.isfinite(n) & np.isfinite(d) & np.isfinite(wv) & (wv > 0)
    if not ok.any():
        return float("nan")
    den = float(np.dot(d[ok], wv[ok]))
    if den == 0.0:
        return float("nan")
    return float(np.dot(n[ok], wv[ok]) / den)


def wquantile(
    x: ArrayLike,
    w: Optional[ArrayLike] = None,
    probs: Union[float, Sequence[float]] = 0.5,
) -> Union[float, np.ndarray]:
    """
    Weighted sample quantiles via cumulative weight (type-1 / left-continuous).

    Sort ``x``, normalize positive finite weights, and return the smallest
    value whose cumulative weight is at least ``p``.
    """
    xv = _as_1d(x)
    single = isinstance(probs, (float, int, np.floating, np.integer))
    ps = np.atleast_1d(np.asarray(probs, dtype=float)).ravel()
    if np.any((ps < 0) | (ps > 1)):
        raise ValueError("probs must be in [0, 1]")

    if w is None:
        wv = np.ones(xv.size, dtype=float)
    else:
        wv = _as_1d(w)
        if wv.size != xv.size:
            raise ValueError("w length mismatch")

    ok = _mask(xv, wv)
    if not ok.any():
        out = np.full(ps.shape, np.nan)
        return float(out[0]) if single else out

    xs = xv[ok]
    ws = wv[ok]
    order = np.argsort(xs, kind="mergesort")
    xs = xs[order]
    ws = ws[order]
    ws = ws / ws.sum()
    cdf = np.cumsum(ws)

    out = np.empty(ps.shape, dtype=float)
    for i, p in enumerate(ps):
        if p <= 0:
            out[i] = float(xs[0])
        elif p >= 1:
            out[i] = float(xs[-1])
        else:
            out[i] = float(xs[int(np.searchsorted(cdf, p, side="left"))])
    return float(out[0]) if single else out


def wsum_product(x: ArrayLike, w: Optional[ArrayLike] = None) -> float:
    """Alias for :func:`wsum`."""
    return wsum(x, w)
