"""Bivariate Canada housing + financial return discretization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .ar1_returns import RouwenhorstAR1, _stationary_dist


@dataclass
class CanadaReturnParams:
    """
    Canadian equity/housing return process parameters.

    Defaults are stylized placeholders. Prefer loading estimates via
    ``returns_estimate.load_canada_return_params`` (see notebook 04).
    """

    # Gross return means (annual)
    Rf_bar: float = 1.06
    Rh_bar: float = 1.04
    # Persistence of demeaned returns (docs: log; estimates currently simple)
    rho_f: float = 0.55
    rho_h: float = 0.70
    # Innovation vols (docs: log; estimates currently simple)
    sigma_f: float = 0.16
    sigma_h: float = 0.10
    # Correlation of innovations
    corr: float = 0.35
    # Grid size per margin (joint size = n_f * n_h)
    n_f: int = 3
    n_h: int = 3


@dataclass
class JointReturnChain:
    """Discrete joint Markov chain for (Rf, Rh)."""

    Rf: np.ndarray  # (N,)
    Rh: np.ndarray  # (N,)
    Pi: np.ndarray  # (N, N) row-stochastic
    params: CanadaReturnParams

    @property
    def N(self) -> int:
        return int(self.Rf.size)

    def stationary(self) -> np.ndarray:
        return _stationary_dist(self.Pi)

    def summary(self) -> dict:
        dist = self.stationary()
        Rf, Rh = self.Rf, self.Rh
        mean_f = float(dist @ Rf)
        mean_h = float(dist @ Rh)
        std_f = float(np.sqrt(dist @ (Rf - mean_f) ** 2))
        std_h = float(np.sqrt(dist @ (Rh - mean_h) ** 2))
        cov = float(dist @ ((Rf - mean_f) * (Rh - mean_h)))
        corr = cov / (std_f * std_h) if std_f > 0 and std_h > 0 else float("nan")
        return {
            "N": self.N,
            "mean_Rf": mean_f,
            "mean_Rh": mean_h,
            "std_Rf": std_f,
            "std_Rh": std_h,
            "corr_Rf_Rh": corr,
            "rho_f": self.params.rho_f,
            "rho_h": self.params.rho_h,
            "sigma_f": self.params.sigma_f,
            "sigma_h": self.params.sigma_h,
            "corr_innov": self.params.corr,
        }


def _gaussian_copula_joint_transitions(
    Pi_f: np.ndarray,
    Pi_h: np.ndarray,
    corr: float,
) -> np.ndarray:
    """
    Build a joint transition on the product grid.

    Independent Kronecker product when corr≈0. For nonzero corr, tilt the
    independent joint toward the diagonal of innovation ranks (simple, stable).
    """
    nf, nh = Pi_f.shape[0], Pi_h.shape[0]
    N = nf * nh
    # Independent kernel: P((i,j)->(i',j')) = Pf(i,i') Ph(j,j')
    Pi_ind = np.zeros((N, N))
    for i in range(nf):
        for j in range(nh):
            s = i * nh + j
            for i2 in range(nf):
                for j2 in range(nh):
                    s2 = i2 * nh + j2
                    Pi_ind[s, s2] = Pi_f[i, i2] * Pi_h[j, j2]

    if abs(corr) < 1e-12:
        return Pi_ind

    # Destination tilt: prefer co-movement of financial and housing ranks
    rank_f = np.repeat(np.linspace(-1.0, 1.0, nf), nh)
    rank_h = np.tile(np.linspace(-1.0, 1.0, nh), nf)
    dest_align = np.exp(3.0 * corr * rank_f * rank_h)
    Pi = Pi_ind * dest_align[None, :]
    Pi = Pi / Pi.sum(axis=1, keepdims=True)
    return Pi


def discretize_canada_returns(params: Optional[CanadaReturnParams] = None) -> JointReturnChain:
    """
    Discretize bivariate persistent returns for Canada financial + housing.

    Each margin uses Rouwenhorst; joint transitions combine margins with a
    correlation tilt (see MODEL.md).
    """
    params = params or CanadaReturnParams()
    ar_f = RouwenhorstAR1(rho=params.rho_f, sigma=params.sigma_f, n=params.n_f)
    ar_h = RouwenhorstAR1(rho=params.rho_h, sigma=params.sigma_h, n=params.n_h)
    xf, Pf = ar_f.discretize()
    xh, Ph = ar_h.discretize()

    # Product grid of log states
    Rf_list = []
    Rh_list = []
    for i in range(params.n_f):
        for j in range(params.n_h):
            Rf_list.append(np.exp(xf[i]))
            Rh_list.append(np.exp(xh[j]))
    Rf_raw = np.asarray(Rf_list, dtype=float)
    Rh_raw = np.asarray(Rh_list, dtype=float)

    Pi = _gaussian_copula_joint_transitions(Pf, Ph, params.corr)
    dist = _stationary_dist(Pi)

    # Normalize so E[R] matches bars
    Rf = params.Rf_bar * Rf_raw / float(dist @ Rf_raw)
    Rh = params.Rh_bar * Rh_raw / float(dist @ Rh_raw)
    return JointReturnChain(Rf=Rf, Rh=Rh, Pi=Pi, params=params)


def scale_financial_vol(
    base: JointReturnChain,
    vol_mult: float,
) -> JointReturnChain:
    """
    Scale financial return dispersion around its mean by ``vol_mult``.

    Used for wealth-dependent volatility approximations (re-center on Rf_bar).
    """
    dist = base.stationary()
    mean_f = float(dist @ base.Rf)
    Rf = mean_f + vol_mult * (base.Rf - mean_f)
    Rf = np.maximum(Rf, 0.01)
    # Renormalize mean
    Rf = base.params.Rf_bar * Rf / float(dist @ Rf)
    return JointReturnChain(Rf=Rf, Rh=base.Rh.copy(), Pi=base.Pi.copy(), params=base.params)
