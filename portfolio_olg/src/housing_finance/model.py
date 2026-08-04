"""Housing + financial life-cycle model (ConSav-style namespaces, Numba VFI)."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

import numpy as np

from ..returns_ca import CanadaReturnParams, JointReturnChain, discretize_canada_returns
from . import solve as solve_mod


def _rouwenhorst_income(rho: float, sigma: float, n: int) -> tuple:
    from ..ar1_returns import RouwenhorstAR1, _stationary_dist

    ar = RouwenhorstAR1(rho=rho, sigma=sigma, n=n)
    x, Pi = ar.discretize()
    z = np.exp(x)
    dist = _stationary_dist(Pi)
    z = z / float(dist @ z)
    return z, Pi


@dataclass
class HousingFinanceParams:
    T: int = 8
    beta: float = 0.96
    alpha: float = 0.7
    rho: float = 2.0
    h_ubar: float = 0.05
    tau: float = 0.06
    delta: float = 0.02
    Nm: int = 18
    Nh: int = 12
    m_max: float = 8.0
    h_max: float = 5.0
    Nz: int = 3
    rho_z: float = 0.9
    sigma_z: float = 0.15
    returns: CanadaReturnParams = field(default_factory=CanadaReturnParams)
    n_c: int = 12
    simN: int = 2_000
    sim_seed: int = 1


class HousingFinanceModel:
    """Life-cycle PE model with liquid financial wealth and housing in utility."""

    def __init__(
        self,
        params: Optional[HousingFinanceParams] = None,
        return_chain: Optional[JointReturnChain] = None,
    ):
        self.par = params or HousingFinanceParams()
        self.return_chain = return_chain or discretize_canada_returns(self.par.returns)
        self.sol = SimpleNamespace()
        self.sim = SimpleNamespace()
        self._build_grids()

    def _build_grids(self) -> None:
        par = self.par
        self.grid_m = par.m_max * (np.linspace(0, 1, par.Nm) ** 1.5)
        self.grid_h = par.h_max * (np.linspace(0, 1, par.Nh) ** 1.5)
        self.z_grid, self.Pi_z = _rouwenhorst_income(par.rho_z, par.sigma_z, par.Nz)
        self.Rf = self.return_chain.Rf
        self.Rh = self.return_chain.Rh
        self.Pi_r = self.return_chain.Pi

    def solve(self) -> None:
        par = self.par
        T = par.T
        shape = (T, par.Nm, par.Nh, par.Nz, self.Rf.size)
        V = np.zeros(shape)
        c_pol = np.zeros(shape)
        h_pol = np.zeros(shape)
        adj_pol = np.zeros(shape)

        Vt, ct, ht, at = solve_mod.solve_terminal(
            self.grid_m,
            self.grid_h,
            self.z_grid,
            self.Rf,
            self.Rh,
            par.alpha,
            par.rho,
            par.h_ubar,
            par.tau,
            par.n_c,
        )
        V[T - 1] = Vt
        c_pol[T - 1] = ct
        h_pol[T - 1] = ht
        adj_pol[T - 1] = at

        for t in range(T - 2, -1, -1):
            Vt, ct, ht, at = solve_mod.solve_period(
                V[t + 1],
                self.grid_m,
                self.grid_h,
                self.z_grid,
                self.Pi_z,
                self.Rf,
                self.Rh,
                self.Pi_r,
                par.beta,
                par.alpha,
                par.rho,
                par.h_ubar,
                par.tau,
                par.delta,
                par.n_c,
            )
            V[t] = Vt
            c_pol[t] = ct
            h_pol[t] = ht
            adj_pol[t] = at

        self.sol.V = V
        self.sol.c = c_pol
        self.sol.h_end = h_pol
        self.sol.adj = adj_pol

    def simulate(self, wealth_vol_kappa: float = 0.0, wealth_vol_norm: float = 5.0) -> None:
        from .simulate import simulate_cohort

        if not hasattr(self.sol, "c"):
            raise RuntimeError("Call solve() before simulate()")
        out = simulate_cohort(
            self,
            wealth_vol_kappa=wealth_vol_kappa,
            wealth_vol_norm=wealth_vol_norm,
        )
        for k, v in out.items():
            setattr(self.sim, k, v)
