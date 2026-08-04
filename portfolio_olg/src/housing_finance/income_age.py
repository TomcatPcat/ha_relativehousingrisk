"""Deterministic age component of labor income (placeholder for data)."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np


def inc_age_deterministic(age: float) -> float:
    """
    Deterministic log-income age component g(age).

    Currently identically zero. Replace / override via
    ``HousingFinanceParams.age_income_profile`` when data are available.

    Total log income: log y = g(age) + log z.
    """
    return 0.0


def build_age_income_log_profile(
    ages: np.ndarray,
    profile: Optional[Union[np.ndarray, Sequence[float]]] = None,
) -> np.ndarray:
    """
    Log age factors g(age) of length T.

    If ``profile`` is given, it must have length T and is interpreted as
    **log** income age effects g_t (not levels).
    """
    ages = np.asarray(ages, dtype=float).ravel()
    T = ages.size
    if profile is None:
        return np.array([inc_age_deterministic(float(a)) for a in ages], dtype=float)
    g = np.asarray(profile, dtype=float).ravel()
    if g.size != T:
        raise ValueError(f"age_income_profile length {g.size} != T={T}")
    return g


def build_age_income_factors(
    ages: np.ndarray,
    profile: Optional[Union[np.ndarray, Sequence[float]]] = None,
) -> np.ndarray:
    """Multiplicative factors exp(g(age)) so that y = factor * z."""
    g = build_age_income_log_profile(ages, profile=profile)
    return np.exp(g)
