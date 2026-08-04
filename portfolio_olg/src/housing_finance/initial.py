"""Initial conditions for cohort and panel simulations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np


def ages_to_period_index(
    ages: np.ndarray,
    age_grid: np.ndarray,
) -> np.ndarray:
    """Map chronological ages onto nearest life-cycle period indices."""
    ages = np.asarray(ages, dtype=float)
    grid = np.asarray(age_grid, dtype=float)
    idx = np.empty(ages.shape, dtype=int)
    for i, a in enumerate(ages.ravel()):
        idx.ravel()[i] = int(np.argmin(np.abs(grid - a)))
    return idx.reshape(ages.shape)


@dataclass
class InitialSample:
    """
    Cross-section of N agents at a simulation start.

    Required: liquid cash-on-hand ``m``, housing ``h``, income index ``i_z``,
    return index ``i_r``. Optional ``t_age`` is the life-cycle period index
    (needed for mixed-age panels; ignored / forced to 0 for young cohorts).
    Optional ``weight`` is a per-agent sample weight (default ones).
    """

    m: np.ndarray
    h: np.ndarray
    i_z: np.ndarray
    i_r: np.ndarray
    t_age: Optional[np.ndarray] = None
    weight: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.m = np.asarray(self.m, dtype=float).ravel()
        self.h = np.asarray(self.h, dtype=float).ravel()
        self.i_z = np.asarray(self.i_z, dtype=int).ravel()
        self.i_r = np.asarray(self.i_r, dtype=int).ravel()
        n = self.m.size
        for name, arr in (("h", self.h), ("i_z", self.i_z), ("i_r", self.i_r)):
            if arr.size != n:
                raise ValueError(f"{name} length {arr.size} != m length {n}")
        if self.t_age is not None:
            self.t_age = np.asarray(self.t_age, dtype=int).ravel()
            if self.t_age.size != n:
                raise ValueError(f"t_age length {self.t_age.size} != m length {n}")
        if self.weight is None:
            self.weight = np.ones(n, dtype=float)
        else:
            self.weight = np.asarray(self.weight, dtype=float).ravel()
            if self.weight.size != n:
                raise ValueError(f"weight length {self.weight.size} != m length {n}")
            if np.any(~np.isfinite(self.weight)) or np.any(self.weight < 0):
                raise ValueError("weight must be finite and nonnegative")

    @property
    def N(self) -> int:
        return int(self.m.size)

    def validate_bounds(
        self,
        *,
        Nz: int,
        Nr: int,
        T: Optional[int] = None,
    ) -> None:
        if np.any(self.i_z < 0) or np.any(self.i_z >= Nz):
            raise ValueError(f"i_z out of bounds for Nz={Nz}")
        if np.any(self.i_r < 0) or np.any(self.i_r >= Nr):
            raise ValueError(f"i_r out of bounds for Nr={Nr}")
        if self.t_age is not None and T is not None:
            if np.any(self.t_age < 0) or np.any(self.t_age >= T):
                raise ValueError(f"t_age out of bounds for T={T}")

    def copy(self) -> "InitialSample":
        return InitialSample(
            m=self.m.copy(),
            h=self.h.copy(),
            i_z=self.i_z.copy(),
            i_r=self.i_r.copy(),
            t_age=None if self.t_age is None else self.t_age.copy(),
            weight=None if self.weight is None else self.weight.copy(),
        )

    def with_t_age(self, t_age: Union[int, np.ndarray]) -> "InitialSample":
        out = self.copy()
        if isinstance(t_age, (int, np.integer)):
            out.t_age = np.full(out.N, int(t_age), dtype=int)
        else:
            out.t_age = np.asarray(t_age, dtype=int).ravel()
            if out.t_age.size != out.N:
                raise ValueError("t_age length mismatch")
        return out

    @classmethod
    def from_arrays(
        cls,
        m: Sequence[float],
        h: Sequence[float],
        i_z: Sequence[int],
        i_r: Sequence[int],
        t_age: Optional[Sequence[int]] = None,
        weight: Optional[Sequence[float]] = None,
    ) -> "InitialSample":
        return cls(m=m, h=h, i_z=i_z, i_r=i_r, t_age=t_age, weight=weight)

    @classmethod
    def from_placeholder_csv(cls, path: Union[str, Path]) -> "InitialSample":
        """
        Load a placeholder initial sample from CSV.

        Expected columns: ``m``, ``h``, ``i_z``, ``i_r``, and optionally
        ``t_age`` and ``weight``.
        Real Canadian microdata mapping is left for the calibration stage.
        """
        path = Path(path)
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pandas is required to load InitialSample CSV") from exc
        df = pd.read_csv(path)
        required = ("m", "h", "i_z", "i_r")
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"CSV missing columns {missing}; need {required}")
        t_age = df["t_age"].to_numpy() if "t_age" in df.columns else None
        weight = df["weight"].to_numpy() if "weight" in df.columns else None
        return cls(
            m=df["m"].to_numpy(),
            h=df["h"].to_numpy(),
            i_z=df["i_z"].to_numpy(),
            i_r=df["i_r"].to_numpy(),
            t_age=t_age,
            weight=weight,
        )


@dataclass
class InitialDistribution:
    """
    Placeholder draw distribution for initial states.

    Mirrors the previous hard-coded cohort uniforms; replace later with
    estimated Canadian cross-section distributions.
    """

    m_low: float = 0.2
    m_high: float = 1.5
    h_low: float = 0.0
    h_high: float = 1.0
    # If True and T given, draw t_age uniform on 0..T-1 (mixed-age panel).
    # If False, all agents start at t_age=0 (young cohort).
    mixed_ages: bool = False

    def draw(
        self,
        n: int,
        rng: np.random.Generator,
        *,
        Nz: int,
        Nr: int,
        T: Optional[int] = None,
        t_age: Optional[np.ndarray] = None,
    ) -> InitialSample:
        m = rng.uniform(self.m_low, self.m_high, size=n)
        h = rng.uniform(self.h_low, self.h_high, size=n)
        i_z = rng.integers(0, Nz, size=n)
        i_r = rng.integers(0, Nr, size=n)
        if t_age is not None:
            ages = np.asarray(t_age, dtype=int).ravel()
            if ages.size != n:
                raise ValueError("t_age length must equal n")
        elif self.mixed_ages:
            if T is None:
                raise ValueError("T required when mixed_ages=True")
            ages = rng.integers(0, T, size=n)
        else:
            ages = np.zeros(n, dtype=int)
        return InitialSample(
            m=m, h=h, i_z=i_z, i_r=i_r, t_age=ages, weight=np.ones(n, dtype=float)
        )
