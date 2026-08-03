"""Demonstrate AR(1) idiosyncratic risky returns vs iid."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents import build_lifecycle_portfolio, solve_and_simulate, wealth_levels  # noqa: E402
from src.ar1_returns import RouwenhorstAR1, attach_ar1_risky_returns, simulate_ar1_return_paths  # noqa: E402
from src.inequality import gini, top_share, wealth_inequality_path  # noqa: E402


def run_hark_with_unconditional_ar1(
    rho: float = 0.7,
    sigma: float = 0.18,
    t_cycle: int = 25,
    agent_count: int = 2000,
    seed: int = 7,
) -> np.ndarray:
    """
    Solve with unconditional discrete return distribution matching AR(1)
    moments, simulate with idiosyncratic iid draws (HARK default).
    """
    ar1_params = attach_ar1_risky_returns(
        {},
        rho=rho,
        sigma=sigma,
        n=5,
        risky_avg=1.08,
    )
    agent = build_lifecycle_portfolio(
        t_cycle=t_cycle,
        t_retire=20,
        agent_count=agent_count,
        t_sim=t_cycle,
        seed=seed,
        sim_common_rrisky=False,
        risky_avg=ar1_params["RiskyAvg"],
        risky_std=ar1_params["RiskyStd"],
        extra={"RiskyCount": ar1_params["RiskyCount"]},
    )
    agent.aXtraCount = 36
    agent.quiet = True
    agent.verbose = 0
    solve_and_simulate(agent)
    return wealth_levels(agent)


def demonstrate_persistent_paths(t_sim: int = 80, n_agents: int = 500) -> pd.DataFrame:
    """Simulate raw AR(1) return paths and summarize cross-section dispersion."""
    R = simulate_ar1_return_paths(
        t_sim=t_sim,
        n_agents=n_agents,
        rho=0.75,
        sigma=0.20,
        risky_avg=1.08,
        seed=123,
        n_states=7,
    )
    # Agent-level mean returns and their dispersion (persistence → cross-section)
    mean_R = R.mean(axis=0)
    return pd.DataFrame(
        {
            "t": np.arange(t_sim),
            "cross_std_R": R.std(axis=1),
            "cross_mean_R": R.mean(axis=1),
            "p10_meanR": np.quantile(mean_R, 0.1),
            "p90_meanR": np.quantile(mean_R, 0.9),
        }
    )


def main() -> None:
    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)

    print("=== Rouwenhorst AR(1) discretization ===")
    ar1 = RouwenhorstAR1(rho=0.7, sigma=0.18, n=5)
    R_grid, Pi = ar1.gross_returns(1.08)
    print(f"  R grid: {np.round(R_grid, 3)}")
    print(f"  Pi row sums: {Pi.sum(axis=1)}")

    print("=== HARK life-cycle with AR(1)-matched unconditional returns ===")
    w = run_hark_with_unconditional_ar1()
    path = wealth_inequality_path(w)
    path.to_csv(out_dir / "ar1_matched_inequality.csv", index=False)
    print(
        f"  final gini={gini(w[-1]):.3f}, top10={top_share(w[-1], 0.1):.3f}, "
        f"mean wealth={w[-1].mean():.3f}"
    )

    print("=== Persistent idiosyncratic return paths (simulation layer) ===")
    disp = demonstrate_persistent_paths()
    disp.to_csv(out_dir / "ar1_return_dispersion.csv", index=False)
    print(
        f"  mean cross-sectional std(R)={disp['cross_std_R'].mean():.3f}; "
        f"p10/p90 of agent mean R = {disp['p10_meanR'].iloc[0]:.3f} / {disp['p90_meanR'].iloc[0]:.3f}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(path["t"], path["gini"])
    axes[0].set_title("Wealth Gini (AR1-matched HARK sim)")
    axes[0].set_xlabel("period")
    axes[1].plot(disp["t"], disp["cross_std_R"])
    axes[1].set_title("Cross-sectional std of AR(1) R")
    axes[1].set_xlabel("period")
    fig.tight_layout()
    fig_path = out_dir / "ar1_returns_demo.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {fig_path}")
    print("AR(1) returns OK.")


if __name__ == "__main__":
    main()
