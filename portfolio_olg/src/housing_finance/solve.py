"""Numba solvers for keep / adjust housing-finance problem."""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from .utility import u


@njit(fastmath=True)
def _interp2(grid_m, grid_h, V, m, h):
    """Bilinear interpolation of V[i_m, i_h] (other states fixed)."""
    nm = grid_m.size
    nh = grid_h.size
    # m index
    if m <= grid_m[0]:
        i_m = 0
        w_m = 0.0
    elif m >= grid_m[nm - 1]:
        i_m = nm - 2
        w_m = 1.0
    else:
        i_m = np.searchsorted(grid_m, m) - 1
        i_m = max(0, min(i_m, nm - 2))
        w_m = (m - grid_m[i_m]) / (grid_m[i_m + 1] - grid_m[i_m])
    # h index
    if h <= grid_h[0]:
        i_h = 0
        w_h = 0.0
    elif h >= grid_h[nh - 1]:
        i_h = nh - 2
        w_h = 1.0
    else:
        i_h = np.searchsorted(grid_h, h) - 1
        i_h = max(0, min(i_h, nh - 2))
        w_h = (h - grid_h[i_h]) / (grid_h[i_h + 1] - grid_h[i_h])

    v00 = V[i_m, i_h]
    v10 = V[i_m + 1, i_h]
    v01 = V[i_m, i_h + 1]
    v11 = V[i_m + 1, i_h + 1]
    return (
        (1 - w_m) * (1 - w_h) * v00
        + w_m * (1 - w_h) * v10
        + (1 - w_m) * w_h * v01
        + w_m * w_h * v11
    )


@njit(fastmath=True)
def expected_continuation(
    a,
    h_end,
    i_z,
    i_r,
    V_next,
    grid_m,
    grid_h,
    z_grid,
    Pi_z,
    Rf,
    Rh,
    Pi_r,
    delta,
):
    """E[V(m',h',z',r') | a, h_end, z, r]."""
    nz = z_grid.size
    nr = Rf.size
    ev = 0.0
    for i_zp in range(nz):
        pz = Pi_z[i_z, i_zp]
        if pz <= 0.0:
            continue
        y = z_grid[i_zp]
        for i_rp in range(nr):
            pr = Pi_r[i_r, i_rp]
            if pr <= 0.0:
                continue
            m_p = Rf[i_rp] * a + y
            h_p = Rh[i_rp] * (1.0 - delta) * h_end
            # clamp into grid for interpolation stability
            m_p = min(max(m_p, grid_m[0]), grid_m[grid_m.size - 1])
            h_p = min(max(h_p, grid_h[0]), grid_h[grid_h.size - 1])
            ev += pz * pr * _interp2(grid_m, grid_h, V_next[:, :, i_zp, i_rp], m_p, h_p)
    return ev


@njit(fastmath=True)
def solve_period(
    V_next,
    grid_m,
    grid_h,
    z_grid,
    Pi_z,
    Rf,
    Rh,
    Pi_r,
    beta,
    alpha,
    rho,
    h_ubar,
    tau,
    delta,
    n_c,
):
    """
    Solve one age given continuation V_next[m,h,z,r].

    Returns V, c_pol, h_pol, adj_pol with same shape as state grids.
    h_pol is end-of-period housing; adj_pol=1 if adjust.
    """
    nm = grid_m.size
    nh = grid_h.size
    nz = z_grid.size
    nr = Rf.size

    V = np.empty((nm, nh, nz, nr))
    c_pol = np.empty((nm, nh, nz, nr))
    h_pol = np.empty((nm, nh, nz, nr))
    adj_pol = np.zeros((nm, nh, nz, nr))

    for i_m in prange(nm):
        m = grid_m[i_m]
        for i_h in range(nh):
            h = grid_h[i_h]
            for i_z in range(nz):
                for i_r in range(nr):
                    # ---- keep ----
                    best_v = -1e300
                    best_c = 0.0
                    best_h_end = h
                    best_adj = 0.0

                    if m > 1e-10:
                        for ic in range(n_c):
                            # c on (0, m]
                            c = (ic + 1) / n_c * m
                            a = m - c
                            if a < -1e-12:
                                continue
                            a = max(a, 0.0)
                            cont = expected_continuation(
                                a,
                                h,
                                i_z,
                                i_r,
                                V_next,
                                grid_m,
                                grid_h,
                                z_grid,
                                Pi_z,
                                Rf,
                                Rh,
                                Pi_r,
                                delta,
                            )
                            val = u(c, h, alpha, rho, h_ubar) + beta * cont
                            if val > best_v:
                                best_v = val
                                best_c = c
                                best_h_end = h
                                best_adj = 0.0
                    else:
                        cont = expected_continuation(
                            0.0,
                            h,
                            i_z,
                            i_r,
                            V_next,
                            grid_m,
                            grid_h,
                            z_grid,
                            Pi_z,
                            Rf,
                            Rh,
                            Pi_r,
                            delta,
                        )
                        best_v = u(1e-12, h, alpha, rho, h_ubar) + beta * cont
                        best_c = 1e-12
                        best_h_end = h

                    # ---- adjust ----
                    x = m + (1.0 - tau) * h
                    if x > 1e-10:
                        for i_hn in range(nh):
                            h_new = grid_h[i_hn]
                            if h_new > x:
                                continue
                            rest = x - h_new
                            for ic in range(n_c):
                                c = (ic + 1) / n_c * rest
                                a = rest - c
                                if a < -1e-12:
                                    continue
                                a = max(a, 0.0)
                                cont = expected_continuation(
                                    a,
                                    h_new,
                                    i_z,
                                    i_r,
                                    V_next,
                                    grid_m,
                                    grid_h,
                                    z_grid,
                                    Pi_z,
                                    Rf,
                                    Rh,
                                    Pi_r,
                                    delta,
                                )
                                val = u(c, h_new, alpha, rho, h_ubar) + beta * cont
                                if val > best_v:
                                    best_v = val
                                    best_c = c
                                    best_h_end = h_new
                                    best_adj = 1.0

                    V[i_m, i_h, i_z, i_r] = best_v
                    c_pol[i_m, i_h, i_z, i_r] = best_c
                    h_pol[i_m, i_h, i_z, i_r] = best_h_end
                    adj_pol[i_m, i_h, i_z, i_r] = best_adj

    return V, c_pol, h_pol, adj_pol


@njit(fastmath=True)
def solve_terminal(
    grid_m,
    grid_h,
    z_grid,
    Rf,
    Rh,
    alpha,
    rho,
    h_ubar,
    tau,
    n_c,
):
    """Last period: sell housing (pay tau) and consume."""
    nm = grid_m.size
    nh = grid_h.size
    nz = z_grid.size
    nr = Rf.size
    V = np.empty((nm, nh, nz, nr))
    c_pol = np.empty((nm, nh, nz, nr))
    h_pol = np.zeros((nm, nh, nz, nr))
    adj_pol = np.ones((nm, nh, nz, nr))
    for i_m in range(nm):
        for i_h in range(nh):
            x = grid_m[i_m] + (1.0 - tau) * grid_h[i_h]
            c = max(x, 1e-12)
            # utility from consuming resources; housing services from pre-sale h
            val = u(c, grid_h[i_h], alpha, rho, h_ubar)
            for i_z in range(nz):
                for i_r in range(nr):
                    V[i_m, i_h, i_z, i_r] = val
                    c_pol[i_m, i_h, i_z, i_r] = c
    return V, c_pol, h_pol, adj_pol
