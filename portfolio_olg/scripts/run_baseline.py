"""Solve and simulate baseline HARK portfolio agents."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents import (  # noqa: E402
    build_infinite_horizon_portfolio,
    build_lifecycle_portfolio,
    solve_and_simulate,
    wealth_levels,
)
from src.inequality import gini, top_share  # noqa: E402


def main() -> None:
    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)

    print("=== Infinite-horizon portfolio ===")
    ih = build_infinite_horizon_portfolio(agent_count=2000, seed=1, sim_common_rrisky=False)
    # Smaller grids for a fast smoke test
    ih.aXtraCount = 48
    ih.constructors["aXtraGrid"](ih) if hasattr(ih, "constructors") else None
    solve_and_simulate(ih)
    w_ih = wealth_levels(ih)[-1]
    print(f"  final mean wealth={w_ih.mean():.3f}, gini={gini(w_ih):.3f}, top10={top_share(w_ih, 0.1):.3f}")

    print("=== Life-cycle portfolio ===")
    lc = build_lifecycle_portfolio(
        t_cycle=25,
        t_retire=20,
        agent_count=1500,
        t_sim=40,
        seed=2,
        sim_common_rrisky=False,
    )
    lc.aXtraCount = 40
    solve_and_simulate(lc)
    w_lc = wealth_levels(lc)
    print(
        f"  terminal mean wealth={w_lc[-1].mean():.3f}, "
        f"gini={gini(w_lc[-1]):.3f}, top10={top_share(w_lc[-1], 0.1):.3f}"
    )

    # Simple figure: mean wealth path in life-cycle sim
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(w_lc.mean(axis=1), label="mean wealth")
    ax.plot(np.median(w_lc, axis=1), label="median wealth")
    ax.set_xlabel("simulation period")
    ax.set_ylabel("wealth level")
    ax.set_title("Life-cycle portfolio: wealth path")
    ax.legend()
    fig.tight_layout()
    fig_path = out_dir / "baseline_lifecycle_wealth.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {fig_path}")
    print("Baseline OK.")


if __name__ == "__main__":
    main()
