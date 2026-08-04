"""Simulate housing-finance cohorts and mixed-age panels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Literal, Optional, Union

import numpy as np

from .initial import InitialDistribution, InitialSample

if TYPE_CHECKING:
    from .model import HousingFinanceModel

OnTerminal = Literal["replace_young", "exit"]


def _nearest_index(grid: np.ndarray, x: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(grid, x)
    idx = np.clip(idx, 1, grid.size - 1)
    left = grid[idx - 1]
    right = grid[idx]
    pick_left = np.abs(x - left) <= np.abs(x - right)
    return np.where(pick_left, idx - 1, idx)


def _resolve_initial(
    model: "HousingFinanceModel",
    *,
    n: int,
    initial: Optional[InitialSample],
    distribution: Optional[InitialDistribution],
    rng: np.random.Generator,
    force_young: bool,
    mixed_ages: bool,
) -> InitialSample:
    par = model.par
    Nz = par.Nz
    Nr = int(model.Rf.size)
    if initial is not None:
        sample = initial.copy()
        if sample.N != n:
            raise ValueError(f"initial sample N={sample.N} != requested n={n}")
        if force_young or sample.t_age is None:
            sample.t_age = np.zeros(n, dtype=int)
        sample.validate_bounds(Nz=Nz, Nr=Nr, T=par.T)
        return sample
    dist = distribution or InitialDistribution(mixed_ages=mixed_ages)
    if force_young:
        dist = InitialDistribution(
            m_low=dist.m_low,
            m_high=dist.m_high,
            h_low=dist.h_low,
            h_high=dist.h_high,
            mixed_ages=False,
        )
    sample = dist.draw(n, rng, Nz=Nz, Nr=Nr, T=par.T)
    sample.validate_bounds(Nz=Nz, Nr=Nr, T=par.T)
    return sample


def _draw_next_states(
    i_z: np.ndarray,
    i_r: np.ndarray,
    Pi_z_cdf: np.ndarray,
    Pi_r_cdf: np.ndarray,
    rng: np.random.Generator,
) -> tuple:
    N = i_z.size
    u_z = rng.random(N)
    u_r = rng.random(N)
    i_zp = np.empty(N, dtype=int)
    i_rp = np.empty(N, dtype=int)
    for i in range(N):
        i_zp[i] = int(np.searchsorted(Pi_z_cdf[i_z[i]], u_z[i]))
        i_rp[i] = int(np.searchsorted(Pi_r_cdf[i_r[i]], u_r[i]))
    return i_zp, i_rp


def _policies_at_ages(
    model: "HousingFinanceModel",
    t_age: np.ndarray,
    m: np.ndarray,
    h: np.ndarray,
    i_z: np.ndarray,
    i_r: np.ndarray,
    alive: np.ndarray,
) -> tuple:
    """Look up c, h_end, adj for heterogeneous life-cycle ages."""
    N = m.size
    i_m = _nearest_index(model.grid_m, m)
    i_h = _nearest_index(model.grid_h, h)
    c = np.zeros(N)
    h_end = np.zeros(N)
    adj = np.zeros(N)
    for i in range(N):
        if not alive[i]:
            continue
        t = int(t_age[i])
        c[i] = model.sol.c[t, i_m[i], i_h[i], i_z[i], i_r[i]]
        h_end[i] = model.sol.h_end[t, i_m[i], i_h[i], i_z[i], i_r[i]]
        adj[i] = model.sol.adj[t, i_m[i], i_h[i], i_z[i], i_r[i]]
    return c, h_end, adj


def _liquid_saving(
    m: np.ndarray,
    h: np.ndarray,
    c: np.ndarray,
    h_end: np.ndarray,
    adj: np.ndarray,
    tau: float,
    alive: np.ndarray,
) -> np.ndarray:
    N = m.size
    a = np.zeros(N)
    for i in range(N):
        if not alive[i]:
            continue
        if adj[i] > 0.5:
            x = m[i] + (1.0 - tau) * h[i]
            a[i] = max(x - h_end[i] - c[i], 0.0)
        else:
            a[i] = max(m[i] - c[i], 0.0)
    return a


def simulate_cohort(
    model: "HousingFinanceModel",
    *,
    initial: Optional[InitialSample] = None,
    distribution: Optional[InitialDistribution] = None,
    seed: Optional[int] = None,
    n: Optional[int] = None,
    wealth_vol_kappa: float = 0.0,
    wealth_vol_norm: float = 5.0,
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Simulate one birth cohort through the full life cycle ``t=0,...,T-1``.

    Initial states come from ``initial``, else ``distribution.draw``, else the
    default young-cohort ``InitialDistribution``.
    """
    par = model.par
    rng = np.random.default_rng(par.sim_seed if seed is None else seed)
    N = int(n if n is not None else (initial.N if initial is not None else par.simN))
    T = par.T

    sample = _resolve_initial(
        model,
        n=N,
        initial=initial,
        distribution=distribution,
        rng=rng,
        force_young=True,
        mixed_ages=False,
    )
    m = sample.m.copy()
    h = sample.h.copy()
    i_z = sample.i_z.copy()
    i_r = sample.i_r.copy()
    alive = np.ones(N, dtype=bool)

    c_path = np.zeros((T, N))
    m_path = np.zeros((T, N))
    h_path = np.zeros((T, N))
    a_path = np.zeros((T, N))
    y_path = np.zeros((T, N))
    adj_path = np.zeros((T, N))
    nw_path = np.zeros((T, N))
    Rf_path = np.zeros((T, N))
    Rh_path = np.zeros((T, N))

    Pi_z_cdf = np.cumsum(model.Pi_z, axis=1)
    Pi_r_cdf = np.cumsum(model.Pi_r, axis=1)
    delta = getattr(model, "delta_period", par.delta)

    for t in range(T):
        t_age = np.full(N, t, dtype=int)
        y = model.z_grid[i_z]
        m_path[t] = m
        h_path[t] = h
        y_path[t] = y
        nw_path[t] = m + h
        Rf_path[t] = model.Rf[i_r]
        Rh_path[t] = model.Rh[i_r]

        c, h_end, adj = _policies_at_ages(model, t_age, m, h, i_z, i_r, alive)
        a = _liquid_saving(m, h, c, h_end, adj, par.tau, alive)
        c_path[t] = c
        a_path[t] = a
        adj_path[t] = adj

        if t == T - 1:
            break

        i_zp, i_rp = _draw_next_states(i_z, i_r, Pi_z_cdf, Pi_r_cdf, rng)
        Rf = model.Rf[i_rp].copy()
        Rh = model.Rh[i_rp]
        if wealth_vol_kappa > 0:
            W = np.maximum(m + h, 1e-8)
            mult = 1.0 + wealth_vol_kappa * (W / wealth_vol_norm)
            mean_f = float(np.mean(model.Rf))
            Rf = mean_f + mult * (Rf - mean_f)
            Rf = np.maximum(Rf, 0.01)

        m = Rf * a + model.z_grid[i_zp]
        h = Rh * (1.0 - delta) * h_end
        m = np.clip(m, model.grid_m[0], model.grid_m[-1] * 1.5)
        h = np.clip(h, model.grid_h[0], model.grid_h[-1] * 1.5)
        i_z = i_zp
        i_r = i_rp

    return {
        "c": c_path,
        "m": m_path,
        "h": h_path,
        "a": a_path,
        "y": y_path,
        "adj": adj_path,
        "net_worth": nw_path,
        "Rf_realized": Rf_path,
        "Rh_realized": Rh_path,
        "alive": np.ones((T, N), dtype=bool),
        "t_age": np.tile(np.arange(T)[:, None], (1, N)),
        "chronological_age": np.tile(model.ages[:, None], (1, N)),
    }


def simulate_panel(
    model: "HousingFinanceModel",
    *,
    n_periods: int,
    initial: Optional[InitialSample] = None,
    distribution: Optional[InitialDistribution] = None,
    on_terminal: OnTerminal = "replace_young",
    seed: Optional[int] = None,
    n: Optional[int] = None,
    wealth_vol_kappa: float = 0.0,
    wealth_vol_norm: float = 5.0,
) -> Dict[str, Union[np.ndarray, float, str]]:
    """
    Simulate a mixed-age cross-section for ``n_periods`` calendar steps.

    Each agent carries a life-cycle index ``t_age`` and uses the corresponding
    policy. When an agent reaches the terminal age:

    - ``replace_young``: redraw a young agent (default; keeps N fixed)
    - ``exit``: mark dead (paths become NaN afterward)
    """
    if on_terminal not in ("replace_young", "exit"):
        raise ValueError("on_terminal must be 'replace_young' or 'exit'")
    if n_periods < 1:
        raise ValueError("n_periods must be >= 1")

    par = model.par
    rng = np.random.default_rng(par.sim_seed if seed is None else seed)
    N = int(n if n is not None else (initial.N if initial is not None else par.simN))
    T = par.T

    sample = _resolve_initial(
        model,
        n=N,
        initial=initial,
        distribution=distribution or InitialDistribution(mixed_ages=True),
        rng=rng,
        force_young=False,
        mixed_ages=True,
    )
    if sample.t_age is None:
        sample.t_age = rng.integers(0, T, size=N)

    m = sample.m.copy()
    h = sample.h.copy()
    i_z = sample.i_z.copy()
    i_r = sample.i_r.copy()
    t_age = sample.t_age.copy()
    alive = np.ones(N, dtype=bool)

    c_path = np.full((n_periods, N), np.nan)
    m_path = np.full((n_periods, N), np.nan)
    h_path = np.full((n_periods, N), np.nan)
    a_path = np.full((n_periods, N), np.nan)
    y_path = np.full((n_periods, N), np.nan)
    adj_path = np.full((n_periods, N), np.nan)
    nw_path = np.full((n_periods, N), np.nan)
    Rf_path = np.full((n_periods, N), np.nan)
    Rh_path = np.full((n_periods, N), np.nan)
    t_age_path = np.full((n_periods, N), -1, dtype=int)
    chron_path = np.full((n_periods, N), np.nan)
    alive_path = np.zeros((n_periods, N), dtype=bool)

    Pi_z_cdf = np.cumsum(model.Pi_z, axis=1)
    Pi_r_cdf = np.cumsum(model.Pi_r, axis=1)
    delta = getattr(model, "delta_period", par.delta)
    young_dist = InitialDistribution(mixed_ages=False)

    for k in range(n_periods):
        alive_path[k] = alive
        t_age_path[k] = t_age
        chron_path[k] = np.where(alive, model.ages[np.clip(t_age, 0, T - 1)], np.nan)

        m_path[k] = np.where(alive, m, np.nan)
        h_path[k] = np.where(alive, h, np.nan)
        y = model.z_grid[i_z]
        y_path[k] = np.where(alive, y, np.nan)
        nw_path[k] = np.where(alive, m + h, np.nan)
        Rf_path[k] = np.where(alive, model.Rf[i_r], np.nan)
        Rh_path[k] = np.where(alive, model.Rh[i_r], np.nan)

        c, h_end, adj = _policies_at_ages(model, t_age, m, h, i_z, i_r, alive)
        a = _liquid_saving(m, h, c, h_end, adj, par.tau, alive)
        c_path[k] = np.where(alive, c, np.nan)
        a_path[k] = np.where(alive, a, np.nan)
        adj_path[k] = np.where(alive, adj, np.nan)

        if k == n_periods - 1:
            break

        # Transition only living agents that are not at terminal age yet
        continue_mask = alive & (t_age < T - 1)
        terminal_mask = alive & (t_age >= T - 1)

        i_zp, i_rp = _draw_next_states(i_z, i_r, Pi_z_cdf, Pi_r_cdf, rng)
        Rf = model.Rf[i_rp].copy()
        Rh = model.Rh[i_rp]
        if wealth_vol_kappa > 0:
            W = np.maximum(m + h, 1e-8)
            mult = 1.0 + wealth_vol_kappa * (W / wealth_vol_norm)
            mean_f = float(np.mean(model.Rf))
            Rf = mean_f + mult * (Rf - mean_f)
            Rf = np.maximum(Rf, 0.01)

        m_next = Rf * a + model.z_grid[i_zp]
        h_next = Rh * (1.0 - delta) * h_end
        m_next = np.clip(m_next, model.grid_m[0], model.grid_m[-1] * 1.5)
        h_next = np.clip(h_next, model.grid_h[0], model.grid_h[-1] * 1.5)

        # Age continuing agents
        m = np.where(continue_mask, m_next, m)
        h = np.where(continue_mask, h_next, h)
        i_z = np.where(continue_mask, i_zp, i_z)
        i_r = np.where(continue_mask, i_rp, i_r)
        t_age = np.where(continue_mask, t_age + 1, t_age)

        # Handle terminal agents
        if np.any(terminal_mask):
            if on_terminal == "exit":
                alive = alive & ~terminal_mask
            else:
                n_rep = int(terminal_mask.sum())
                repl = young_dist.draw(
                    n_rep, rng, Nz=par.Nz, Nr=int(model.Rf.size), T=T
                )
                idxs = np.where(terminal_mask)[0]
                m[idxs] = repl.m
                h[idxs] = repl.h
                i_z[idxs] = repl.i_z
                i_r[idxs] = repl.i_r
                t_age[idxs] = 0

    return {
        "c": c_path,
        "m": m_path,
        "h": h_path,
        "a": a_path,
        "y": y_path,
        "adj": adj_path,
        "net_worth": nw_path,
        "Rf_realized": Rf_path,
        "Rh_realized": Rh_path,
        "alive": alive_path,
        "t_age": t_age_path,
        "chronological_age": chron_path,
        "on_terminal": on_terminal,
        "n_periods": float(n_periods),
    }
