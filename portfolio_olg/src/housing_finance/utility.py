"""Utility: Cobb-Douglas over consumption and housing; De Nardi bequest."""

from numba import njit
import numpy as np


@njit(fastmath=True)
def u(c, h, alpha, rho, h_ubar):
    c = max(c, 1e-12)
    ht = h + h_ubar
    bundle = (c ** alpha) * (ht ** (1.0 - alpha))
    if abs(rho - 1.0) < 1e-8:
        return np.log(bundle)
    return (bundle ** (1.0 - rho) - 1.0) / (1.0 - rho)


@njit(fastmath=True)
def bequest_wealth(m, h, tau):
    """
    Resources available for terminal consumption / bequest.

    Default: liquidate housing at proportional cost tau.
    Hook for later housing/financial aggregators.
    """
    return m + (1.0 - tau) * h


@njit(fastmath=True)
def bequest_u(W, phi, kappa, rho):
    """
    De Nardi (2004)-style warm-glow bequest utility with CRRA curvature rho.

    upsilon(W) = phi * ((W + kappa)^(1-rho) - 1) / (1 - rho)
    (log form when rho ≈ 1). Same functional class as De Nardi–Yang.
    """
    if phi <= 0.0:
        return 0.0
    x = max(W + kappa, 1e-12)
    if abs(rho - 1.0) < 1e-8:
        return phi * np.log(x)
    return phi * (x ** (1.0 - rho) - 1.0) / (1.0 - rho)
