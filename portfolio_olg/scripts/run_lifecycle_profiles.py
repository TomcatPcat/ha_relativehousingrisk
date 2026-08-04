"""Simulate one lifecycle cohort; report means and major quantiles by age."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.housing_finance import HousingFinanceModel, HousingFinanceParams  # noqa: E402
from src.lifecycle_profiles import (  # noqa: E402
    DEFAULT_VARIABLES,
    plot_lifecycle_profiles,
    profiles_from_sim,
)
from src.returns_ca import CanadaReturnParams  # noqa: E402
import numpy as np  # noqa: E402


def main() -> None:
    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)

    ret = CanadaReturnParams(n_f=3, n_h=3, sigma_f=0.16, sigma_h=0.10, corr=0.35)
    par = HousingFinanceParams(
        T=8,
        Nm=16,
        Nh=12,
        Nz=3,
        n_c=12,
        simN=4_000,
        sim_seed=1,
        returns=ret,
        m_max=7.0,
        h_max=4.5,
    )
    model = HousingFinanceModel(par)
    print(
        f"Solving T={par.T}, ages {par.age_min:.0f}–{par.age_max:.0f}, "
        f"period_years={model.period_years:.4f}, simN={par.simN}..."
    )
    print("  age grid:", np.array2string(model.ages, precision=2))
    model.solve()
    print("Simulating cohort...")
    model.simulate()

    profiles = profiles_from_sim(model.sim, variables=DEFAULT_VARIABLES, ages=model.ages)
    csv_path = out_dir / "lifecycle_cohort_profiles.csv"
    profiles.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    # Compact console view: mean and median by chronological age
    for var in DEFAULT_VARIABLES:
        sub = profiles.loc[
            profiles["variable"] == var, ["t", "age", "mean", "q10", "q50", "q90"]
        ]
        print(f"\n=== {var} ===")
        print(sub.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))

    fig = plot_lifecycle_profiles(profiles, variables=DEFAULT_VARIABLES)
    fig_path = out_dir / "lifecycle_cohort_profiles.png"
    fig.savefig(fig_path, dpi=120)
    print(f"\nWrote {fig_path}")
    print("Lifecycle cohort profiles OK.")


if __name__ == "__main__":
    main()
