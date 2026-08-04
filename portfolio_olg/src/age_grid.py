"""Map model periods to chronological ages over a fixed life span."""

from __future__ import annotations

from dataclasses import replace
from typing import Tuple

import numpy as np

from .returns_ca import CanadaReturnParams


def period_years(T: int, age_min: float, age_max: float) -> float:
    """
    Years spanned by one model step ``t -> t+1``.

    Ages are equally spaced on ``[age_min, age_max]`` with ``T`` nodes, so

        period_years = (age_max - age_min) / (T - 1).
    """
    if T < 2:
        raise ValueError("T must be >= 2 to define period length over [age_min, age_max]")
    if age_max <= age_min:
        raise ValueError("age_max must exceed age_min")
    return float(age_max - age_min) / float(T - 1)


def chronological_ages(T: int, age_min: float, age_max: float) -> np.ndarray:
    """
    Chronological ages at each model period index ``t = 0,...,T-1``.

    Always starts at ``age_min`` and ends at ``age_max``:

        age(t) = age_min + t * period_years.
    """
    if T < 1:
        raise ValueError("T must be >= 1")
    if age_max < age_min:
        raise ValueError("age_max must be >= age_min")
    if T == 1:
        return np.array([float(age_min)], dtype=float)
    return np.linspace(float(age_min), float(age_max), int(T))


def age_at(t: int, T: int, age_min: float, age_max: float) -> float:
    """Chronological age at period index ``t``."""
    ages = chronological_ages(T, age_min, age_max)
    if t < 0 or t >= T:
        raise IndexError(f"t={t} outside 0..{T-1}")
    return float(ages[t])


def compound_ar1_annual_to_period(
    rho: float, sigma: float, dy: float
) -> Tuple[float, float]:
    """
    Map an annual AR(1) ``(rho, sigma)`` to a ``dy``-year step.

    Keeps the same unconditional variance and sets ``rho_period = rho**dy``.
    """
    rho = float(rho)
    sigma = float(sigma)
    dy = float(dy)
    rho_p = float(rho**dy)
    if abs(1.0 - rho**2) < 1e-14:
        sigma_p = sigma * np.sqrt(max(dy, 0.0))
    else:
        unc = sigma**2 / (1.0 - rho**2)
        sigma_p = float(np.sqrt(max(unc * (1.0 - rho_p**2), 0.0)))
    return rho_p, sigma_p


def scale_return_params_to_period(
    annual: CanadaReturnParams, dy: float
) -> CanadaReturnParams:
    """Compound annual Canada return placeholders to a multi-year model period."""
    dy = float(dy)
    rho_f, sigma_f = compound_ar1_annual_to_period(annual.rho_f, annual.sigma_f, dy)
    rho_h, sigma_h = compound_ar1_annual_to_period(annual.rho_h, annual.sigma_h, dy)
    return replace(
        annual,
        Rf_bar=float(annual.Rf_bar**dy),
        Rh_bar=float(annual.Rh_bar**dy),
        rho_f=rho_f,
        rho_h=rho_h,
        sigma_f=sigma_f,
        sigma_h=sigma_h,
    )


def scale_beta_annual_to_period(beta_annual: float, dy: float) -> float:
    return float(beta_annual**dy)


def scale_delta_annual_to_period(delta_annual: float, dy: float) -> float:
    """One-period depreciation rate over ``dy`` years."""
    return float(1.0 - (1.0 - delta_annual) ** dy)
