"""Lifecycle cohort means and quantiles from simulated paths."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Major quantiles for lifecycle distribution plots / tables
DEFAULT_QUANTILES: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)

# Default variables of interest for a housing-finance cohort
DEFAULT_VARIABLES: tuple[str, ...] = (
    "net_worth",
    "m",
    "h",
    "a",
    "c",
    "y",
)

VARIABLE_LABELS: Dict[str, str] = {
    "net_worth": "Net worth (m + h)",
    "m": "Liquid cash-on-hand (m)",
    "h": "Housing stock (h)",
    "a": "End-of-period liquid saving (a)",
    "c": "Consumption (c)",
    "y": "Labor income (y)",
    "adj": "Housing adjustment indicator",
}


def _qcol(q: float) -> str:
    return f"q{int(round(100 * q)):02d}"


def age_profile(
    x: np.ndarray,
    ages: Optional[np.ndarray] = None,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> pd.DataFrame:
    """
    Mean and quantiles by chronological age from a (T, N) simulated path.

    Parameters
    ----------
    x :
        Simulated path of shape (T, N).
    ages :
        Chronological ages of length T (e.g. ``model.ages``). If omitted,
        uses period indices ``0,...,T-1``.

    Returns
    -------
    DataFrame with columns ``t`` (period index), ``age`` (chronological),
    ``mean``, and quantile columns ``q10``, ``q25``, ...
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"expected (T, N) array, got shape {arr.shape}")
    T = arr.shape[0]
    if ages is None:
        ages_arr = np.arange(T, dtype=float)
    else:
        ages_arr = np.asarray(ages, dtype=float)
        if ages_arr.shape != (T,):
            raise ValueError(f"ages must have shape ({T},), got {ages_arr.shape}")
    qs = tuple(float(q) for q in quantiles)
    rows = []
    for t in range(T):
        xt = arr[t]
        xt = xt[np.isfinite(xt)]
        row = {
            "t": t,
            "age": float(ages_arr[t]),
            "mean": float(np.mean(xt)) if xt.size else float("nan"),
        }
        if xt.size:
            qvals = np.quantile(xt, qs)
            for q, v in zip(qs, qvals):
                row[_qcol(q)] = float(v)
        else:
            for q in qs:
                row[_qcol(q)] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def cohort_lifecycle_profiles(
    paths: Mapping[str, np.ndarray],
    variables: Optional[Iterable[str]] = None,
    ages: Optional[np.ndarray] = None,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> pd.DataFrame:
    """
    Stack age profiles for several simulated series into a long table.

    Parameters
    ----------
    paths :
        Mapping of name -> (T, N) arrays (e.g. ``model.sim.__dict__`` or the
        dict returned by ``simulate_cohort``).
    variables :
        Which keys to include. Defaults to net worth, wealth components,
        consumption, and income.
    ages :
        Chronological ages length T. Prefer ``model.ages``.
    """
    vars_ = tuple(variables) if variables is not None else DEFAULT_VARIABLES
    frames = []
    for name in vars_:
        if name not in paths:
            raise KeyError(f"simulated path '{name}' not found; have {sorted(paths)}")
        df = age_profile(np.asarray(paths[name]), ages=ages, quantiles=quantiles)
        df.insert(0, "variable", name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def profiles_from_sim(
    sim: Union[Mapping[str, np.ndarray], object],
    variables: Optional[Iterable[str]] = None,
    ages: Optional[np.ndarray] = None,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> pd.DataFrame:
    """
    Convenience wrapper accepting a dict or ``model.sim`` namespace.

    If ``ages`` is omitted, uses ``sim.ages`` when present.
    """
    if hasattr(sim, "__dict__") and not isinstance(sim, Mapping):
        paths = {k: v for k, v in vars(sim).items() if isinstance(v, np.ndarray)}
        if ages is None and hasattr(sim, "ages"):
            ages = np.asarray(sim.ages, dtype=float)
    else:
        paths = dict(sim)  # type: ignore[arg-type]
        if ages is None and "ages" in paths and np.asarray(paths["ages"]).ndim == 1:
            ages = np.asarray(paths["ages"], dtype=float)
    # Do not treat the ages vector as a simulated path variable
    paths = {k: v for k, v in paths.items() if k != "ages" and np.asarray(v).ndim == 2}
    return cohort_lifecycle_profiles(
        paths, variables=variables, ages=ages, quantiles=quantiles
    )


def pivot_profile(profiles: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Wide age × statistic table for one variable (index = chronological age)."""
    sub = profiles.loc[profiles["variable"] == variable].drop(columns=["variable"])
    return sub.set_index("age")


def plot_lifecycle_profiles(
    profiles: pd.DataFrame,
    variables: Optional[Iterable[str]] = None,
    *,
    show_iqr: bool = True,
    show_p10_p90: bool = True,
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """
    Plot mean lifecycle paths vs chronological age with optional bands.

    Expects the long table from ``cohort_lifecycle_profiles``.
    """
    vars_ = list(variables) if variables is not None else list(
        dict.fromkeys(profiles["variable"].tolist())
    )
    n = len(vars_)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (5.5 * ncols, 3.4 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat = axes.ravel()

    xcol = "age" if "age" in profiles.columns else "t"

    for i, name in enumerate(vars_):
        ax = flat[i]
        sub = profiles.loc[profiles["variable"] == name].sort_values(xcol)
        x = sub[xcol].to_numpy()
        mean = sub["mean"].to_numpy()
        if show_p10_p90 and "q10" in sub.columns and "q90" in sub.columns:
            ax.fill_between(
                x,
                sub["q10"],
                sub["q90"],
                color="C0",
                alpha=0.12,
                label="p10–p90",
            )
        if show_iqr and "q25" in sub.columns and "q75" in sub.columns:
            ax.fill_between(
                x,
                sub["q25"],
                sub["q75"],
                color="C0",
                alpha=0.25,
                label="p25–p75",
            )
        ax.plot(x, mean, color="C0", lw=2, label="mean")
        if "q50" in sub.columns:
            ax.plot(x, sub["q50"], color="C0", lw=1.5, ls="--", label="median")
        ax.set_xlabel("Age")
        ax.set_title(VARIABLE_LABELS.get(name, name))
        ax.legend(loc="best", fontsize=8)

    for j in range(n, len(flat)):
        flat[j].set_visible(False)

    fig.tight_layout()
    return fig
