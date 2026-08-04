"""Simulate housing-finance life-cycle cohorts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import numpy as np

if TYPE_CHECKING:
    from .model import HousingFinanceModel


def _nearest_index(grid: np.ndarray, x: np.ndarray) -> np.ndarray:
    # vectorized nearest
    idx = np.searchsorted(grid, x)
    idx = np.clip(idx, 1, grid.size - 1)
    left = grid[idx - 1]
    right = grid[idx]
    pick_left = np.abs(x - left) <= np.abs(x - right)
    return np.where(pick_left, idx - 1, idx)


def simulate_cohort(
    model: "HousingFinanceModel",
    wealth_vol_kappa: float = 0.0,
    wealth_vol_norm: float = 5.0,
) -> Dict[str, np.ndarray]:
    par = model.par
    rng = np.random.default_rng(par.sim_seed)
    N = par.simN
    T = par.T

    # initial states
    m = rng.uniform(0.2, 1.5, size=N)
    h = rng.uniform(0.0, 1.0, size=N)
    # stationary income / returns
    i_z = rng.integers(0, par.Nz, size=N)
    i_r = rng.integers(0, model.Rf.size, size=N)

    c_path = np.zeros((T, N))
    m_path = np.zeros((T, N))
    h_path = np.zeros((T, N))
    a_path = np.zeros((T, N))
    adj_path = np.zeros((T, N))
    nw_path = np.zeros((T, N))
    Rf_path = np.zeros((T, N))

    Pi_z_cdf = np.cumsum(model.Pi_z, axis=1)
    Pi_r_cdf = np.cumsum(model.Pi_r, axis=1)

    for t in range(T):
        m_path[t] = m
        h_path[t] = h
        nw_path[t] = m + h

        i_m = _nearest_index(model.grid_m, m)
        i_h = _nearest_index(model.grid_h, h)

        c = model.sol.c[t, i_m, i_h, i_z, i_r]
        h_end = model.sol.h_end[t, i_m, i_h, i_z, i_r]
        adj = model.sol.adj[t, i_m, i_h, i_z, i_r]

        # reconstruct liquid saving from budget
        a = np.empty(N)
        for i in range(N):
            if adj[i] > 0.5:
                x = m[i] + (1.0 - par.tau) * h[i]
                a[i] = max(x - h_end[i] - c[i], 0.0)
            else:
                a[i] = max(m[i] - c[i], 0.0)

        c_path[t] = c
        h_path[t] = h  # beginning-of-period housing
        a_path[t] = a
        adj_path[t] = adj

        if t == T - 1:
            Rf_path[t] = model.Rf[i_r]
            break

        # draw next income / return states
        u_z = rng.random(N)
        u_r = rng.random(N)
        i_zp = np.empty(N, dtype=int)
        i_rp = np.empty(N, dtype=int)
        for i in range(N):
            i_zp[i] = int(np.searchsorted(Pi_z_cdf[i_z[i]], u_z[i]))
            i_rp[i] = int(np.searchsorted(Pi_r_cdf[i_r[i]], u_r[i]))

        Rf = model.Rf[i_rp].copy()
        Rh = model.Rh[i_rp]
        if wealth_vol_kappa > 0:
            W = np.maximum(m + h, 1e-8)
            mult = 1.0 + wealth_vol_kappa * (W / wealth_vol_norm)
            mean_f = float(np.mean(model.Rf))
            Rf = mean_f + mult * (Rf - mean_f)
            Rf = np.maximum(Rf, 0.01)

        Rf_path[t] = Rf
        y = model.z_grid[i_zp]
        m = Rf * a + y
        h = Rh * (1.0 - par.delta) * h_end
        # keep on grid support
        m = np.clip(m, model.grid_m[0], model.grid_m[-1] * 1.5)
        h = np.clip(h, model.grid_h[0], model.grid_h[-1] * 1.5)
        i_z = i_zp
        i_r = i_rp

    return {
        "c": c_path,
        "m": m_path,
        "h": h_path,
        "a": a_path,
        "adj": adj_path,
        "net_worth": nw_path,
        "Rf_realized": Rf_path,
    }
