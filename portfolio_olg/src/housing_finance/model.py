"""Housing + financial life-cycle model (ConSav-style namespaces, Numba VFI)."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

import numpy as np

from ..age_grid import (
    chronological_ages,
    compound_ar1_annual_to_period,
    period_years,
    scale_beta_annual_to_period,
    scale_delta_annual_to_period,
    scale_return_params_to_period,
)
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
    """
    Life-cycle parameters.

    ``T`` periods always span chronological ages ``[age_min, age_max]`` with
    equal spacing. Annual placeholders (``beta``, ``delta``, income AR(1),
    ``returns``) are compounded to the model period length
    ``(age_max - age_min) / (T - 1)`` years inside ``HousingFinanceModel``.
    """

    T: int = 8
    age_min: float = 25.0
    age_max: float = 80.0
    # Annual placeholders (compounded to period frequency in the model)
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
        *,
        scale_annual_to_period: bool = True,
    ):
        self.par = params or HousingFinanceParams()
        par = self.par
        self.ages = chronological_ages(par.T, par.age_min, par.age_max)
        self.period_years = (
            period_years(par.T, par.age_min, par.age_max) if par.T >= 2 else 0.0
        )
        self.scale_annual_to_period = scale_annual_to_period

        dy = self.period_years if (scale_annual_to_period and par.T >= 2) else 1.0
        self.beta_period = (
            scale_beta_annual_to_period(par.beta, dy) if scale_annual_to_period else par.beta
        )
        self.delta_period = (
            scale_delta_annual_to_period(par.delta, dy)
            if scale_annual_to_period
            else par.delta
        )
        if scale_annual_to_period and par.T >= 2:
            self.rho_z_period, self.sigma_z_period = compound_ar1_annual_to_period(
                par.rho_z, par.sigma_z, dy
            )
            period_returns = scale_return_params_to_period(par.returns, dy)
        else:
            self.rho_z_period, self.sigma_z_period = par.rho_z, par.sigma_z
            period_returns = par.returns

        self.returns_period = period_returns
        self.return_chain = return_chain or discretize_canada_returns(period_returns)
        self.sol = SimpleNamespace()
        self.sim = SimpleNamespace()
        self._build_grids()

    def _build_grids(self) -> None:
        par = self.par
        self.grid_m = par.m_max * (np.linspace(0, 1, par.Nm) ** 1.5)
        self.grid_h = par.h_max * (np.linspace(0, 1, par.Nh) ** 1.5)
        self.z_grid, self.Pi_z = _rouwenhorst_income(
            self.rho_z_period, self.sigma_z_period, par.Nz
        )
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
                self.beta_period,
                par.alpha,
                par.rho,
                par.h_ubar,
                par.tau,
                self.delta_period,
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

    def simulate(
        self,
        wealth_vol_kappa: float = 0.0,
        wealth_vol_norm: float = 5.0,
        *,
        initial=None,
        distribution=None,
        seed=None,
        n=None,
    ) -> None:
        from .simulate import simulate_cohort

        if not hasattr(self.sol, "c"):
            raise RuntimeError("Call solve() before simulate()")
        out = simulate_cohort(
            self,
            initial=initial,
            distribution=distribution,
            seed=seed,
            n=n,
            wealth_vol_kappa=wealth_vol_kappa,
            wealth_vol_norm=wealth_vol_norm,
        )
        for k, v in out.items():
            setattr(self.sim, k, v)
        self.sim.ages = self.ages.copy()
        self.sim.period_years = self.period_years

    def simulate_panel(
        self,
        n_periods: int,
        *,
        initial=None,
        distribution=None,
        on_terminal: str = "replace_young",
        seed=None,
        n=None,
        wealth_vol_kappa: float = 0.0,
        wealth_vol_norm: float = 5.0,
    ) -> None:
        from .simulate import simulate_panel

        if not hasattr(self.sol, "c"):
            raise RuntimeError("Call solve() before simulate_panel()")
        out = simulate_panel(
            self,
            n_periods=n_periods,
            initial=initial,
            distribution=distribution,
            on_terminal=on_terminal,  # type: ignore[arg-type]
            seed=seed,
            n=n,
            wealth_vol_kappa=wealth_vol_kappa,
            wealth_vol_norm=wealth_vol_norm,
        )
        self.sim.panel = SimpleNamespace(
            **{k: v for k, v in out.items() if not isinstance(v, str)}
        )
        self.sim.panel_on_terminal = out.get("on_terminal")
