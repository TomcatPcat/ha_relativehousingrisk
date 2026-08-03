"""OLG-style overlapping cohort panel and calendar-time inequality path."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents import build_lifecycle_portfolio, wealth_levels  # noqa: E402
from src.inequality import summarize_path, wealth_inequality_path  # noqa: E402


def simulate_olg_panel(
    t_cycle: int = 30,
    t_retire: int = 25,
    agents_per_cohort: int = 200,
    calendar_t: int = 60,
    seed: int = 10,
    risky_std: float = 0.20,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Overlapping generations via staggered cohort entry.

    Solve the life-cycle problem once (PE prices), then re-simulate each
    entering cohort with a different seed. Cohort ``k`` enters at calendar
    time ``k`` and lives ``t_cycle`` periods.
    """
    n_cohorts = calendar_t
    wealth_cal = np.full((calendar_t, n_cohorts * agents_per_cohort), np.nan)
    alive = np.zeros_like(wealth_cal, dtype=bool)

    # Solve once
    template = build_lifecycle_portfolio(
        t_cycle=t_cycle,
        t_retire=t_retire,
        agent_count=agents_per_cohort,
        t_sim=t_cycle,
        seed=seed,
        sim_common_rrisky=False,
        risky_std=risky_std,
    )
    template.aXtraCount = 32
    template.quiet = True
    template.verbose = 0
    template.solve()

    for k in range(n_cohorts):
        agent = template
        agent.seed = seed + k
        agent.AgentCount = agents_per_cohort
        agent.T_sim = t_cycle
        agent.initialize_sim()
        agent.simulate()
        w = wealth_levels(agent)  # (t_cycle, N)

        col0 = k * agents_per_cohort
        col1 = col0 + agents_per_cohort
        for age in range(t_cycle):
            t = k + age
            if t >= calendar_t:
                break
            wealth_cal[t, col0:col1] = w[age]
            alive[t, col0:col1] = True

    path = wealth_inequality_path(wealth_cal, alive_mask=alive)
    return wealth_cal, path


def main() -> None:
    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)

    print("Simulating OLG panel (baseline risk)...")
    _, path_base = simulate_olg_panel(risky_std=0.15, calendar_t=40, agents_per_cohort=80, t_cycle=20, t_retire=16)
    print("Simulating OLG panel (high return risk)...")
    _, path_high = simulate_olg_panel(risky_std=0.35, calendar_t=40, agents_per_cohort=80, t_cycle=20, t_retire=16, seed=20)

    path_base.to_csv(out_dir / "olg_inequality_baseline.csv", index=False)
    path_high.to_csv(out_dir / "olg_inequality_highrisk.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(path_base["t"], path_base["gini"], label="risky_std=0.15")
    ax.plot(path_high["t"], path_high["gini"], label="risky_std=0.35")
    ax.set_xlabel("calendar time")
    ax.set_ylabel("wealth Gini")
    ax.set_title("OLG portfolio: inequality transition (PE prices)")
    ax.legend()
    fig.tight_layout()
    fig_path = out_dir / "olg_gini_transition.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)

    print("Baseline:", summarize_path(path_base))
    print("High risk:", summarize_path(path_high))
    print(f"Wrote {fig_path}")
    print("OLG inequality OK.")


if __name__ == "__main__":
    main()
