"""AR(1) idiosyncratic return risk helpers (Rouwenhorst discretization)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _stationary_dist(Pi: np.ndarray) -> np.ndarray:
    """Row-stochastic Pi -> stationary distribution (power iteration)."""
    n = Pi.shape[0]
    dist = np.ones(n) / n
    for _ in range(10_000):
        dist_new = dist @ Pi
        if np.linalg.norm(dist_new - dist, ord=1) < 1e-14:
            break
        dist = dist_new
    dist = np.maximum(dist, 0.0)
    s = dist.sum()
    if s <= 0:
        return np.ones(n) / n
    return dist / s


@dataclass
class RouwenhorstAR1:
    """
    Discretize log-return AR(1): x' = rho * x + sigma * eps, eps ~ N(0,1).

    Risky gross return R = exp(mu + x) with E[R] approximately RiskyAvg when
    mu is set accordingly.
    """

    rho: float
    sigma: float
    n: int = 5
    mu: float = 0.0

    def discretize(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (grid_of_x, transition_matrix)."""
        if self.n < 2:
            raise ValueError("n must be >= 2")
        p = (1.0 + self.rho) / 2.0
        q = p
        psi = self.sigma * np.sqrt(self.n - 1.0)
        # Build transition recursively (Rouwenhorst)
        pii = np.array([[p, 1 - p], [1 - q, q]], dtype=float)
        for n in range(3, self.n + 1):
            pii_new = np.zeros((n, n))
            pii_new[: n - 1, : n - 1] += p * pii
            pii_new[: n - 1, 1:] += (1 - p) * pii
            pii_new[1:, : n - 1] += (1 - q) * pii
            pii_new[1:, 1:] += q * pii
            # Normalize double-counted middle rows
            pii_new[1 : n - 1, :] /= 2.0
            pii = pii_new
        grid = np.linspace(-psi, psi, self.n)
        # Row-normalize for numerical safety
        pii = pii / pii.sum(axis=1, keepdims=True)
        return grid, pii

    def gross_returns(self, risky_avg: float = 1.08) -> Tuple[np.ndarray, np.ndarray]:
        """
        Map AR(1) states to gross returns with mean close to ``risky_avg``.

        Uses R_i = risky_avg * exp(x_i) / E[exp(x)] under stationary dist.
        """
        grid, Pi = self.discretize()
        dist = _stationary_dist(Pi)
        exp_x = np.exp(grid + self.mu)
        e_exp = float(dist @ exp_x)
        R = risky_avg * exp_x / e_exp
        return R, Pi


def attach_ar1_risky_returns(
    params: Dict[str, Any],
    rho: float = 0.6,
    sigma: float = 0.15,
    n: int = 5,
    risky_avg: float = 1.08,
) -> Dict[str, Any]:
    """
    Return a copy of agent params with RiskyDstn-like objects replaced by
    a discrete AR(1) approximation for *solve-time* expectations.

    HARK's built-in solver expects an iid RiskyDstn each period. For the
    life-cycle solve we approximate the AR(1) by its unconditional discrete
    distribution (mixture). Simulation can then draw persistent paths via
    ``simulate_ar1_return_paths``.
    """
    out = dict(params)
    ar1 = RouwenhorstAR1(rho=rho, sigma=sigma, n=n)
    R, Pi = ar1.gross_returns(risky_avg=risky_avg)
    dist = _stationary_dist(Pi)
    # Match HARK naming used by constructors when possible
    out["RiskyAvg"] = float(np.sum(dist * R))
    out["RiskyStd"] = float(np.sqrt(np.sum(dist * (R - out["RiskyAvg"]) ** 2)))
    out["RiskyCount"] = n
    out["_ar1_R_grid"] = R
    out["_ar1_Pi"] = Pi
    out["_ar1_dist"] = dist
    out["_ar1_rho"] = rho
    out["_ar1_sigma"] = sigma
    return out


def simulate_ar1_return_paths(
    t_sim: int,
    n_agents: int,
    rho: float,
    sigma: float,
    risky_avg: float = 1.08,
    seed: int = 0,
    n_states: int = 7,
) -> np.ndarray:
    """
    Simulate idiosyncratic AR(1) gross returns of shape (T, N).

    Uses Rouwenhorst Markov chain for each agent independently.
    """
    rng = np.random.default_rng(seed)
    ar1 = RouwenhorstAR1(rho=rho, sigma=sigma, n=n_states)
    R, Pi = ar1.gross_returns(risky_avg=risky_avg)
    # CDF of transitions
    cdf = np.cumsum(Pi, axis=1)
    dist = _stationary_dist(Pi)
    states = rng.choice(n_states, size=n_agents, p=dist)
    out = np.empty((t_sim, n_agents), dtype=float)
    for t in range(t_sim):
        out[t] = R[states]
        u = rng.random(n_agents)
        new_states = np.empty(n_agents, dtype=int)
        for i in range(n_agents):
            new_states[i] = int(np.searchsorted(cdf[states[i]], u[i]))
        states = new_states
    return out
