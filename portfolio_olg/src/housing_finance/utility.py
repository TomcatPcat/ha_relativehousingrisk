"""Utility: Cobb-Douglas over consumption and housing."""

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
