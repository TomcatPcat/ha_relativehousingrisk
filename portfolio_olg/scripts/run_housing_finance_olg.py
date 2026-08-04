"""OLG panel inequality using the housing-finance simulator."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.housing_finance import HousingFinanceModel, HousingFinanceParams  # noqa: E402
from src.inequality import summarize_path, wealth_inequality_path  # noqa: E402
from src.returns_ca import CanadaReturnParams  # noqa: E402


def run_olg(
    sigma_f: float,
    calendar_t: int = 24,
    agents_per_cohort: int = 80,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Staggered cohort entry: solve once, resimulate each cohort with new seed.
    Net worth = liquid m + housing h.
    """
    ret = CanadaReturnParams(
        n_f=3,
        n_h=3,
        sigma_f=sigma_f,
        sigma_h=0.10,
        corr=0.35,
    )
    par = HousingFinanceParams(
        T=6,
        Nm=12,
        Nh=8,
        Nz=3,
        n_c=8,
        simN=agents_per_cohort,
        sim_seed=seed,
        returns=ret,
        m_max=6.0,
        h_max=4.0,
    )
    model = HousingFinanceModel(par)
    model.solve()

    n_cohorts = calendar_t
    wealth = np.full((calendar_t, n_cohorts * agents_per_cohort), np.nan)
    alive = np.zeros_like(wealth, dtype=bool)

    for k in range(n_cohorts):
        model.par.sim_seed = seed + k
        model.par.simN = agents_per_cohort
        model.simulate()
        nw = model.sim.net_worth  # (T, N)
        t_life = nw.shape[0]
        col0 = k * agents_per_cohort
        col1 = col0 + agents_per_cohort
        for age in range(t_life):
            t = k + age
            if t >= calendar_t:
                break
            wealth[t, col0:col1] = nw[age]
            alive[t, col0:col1] = True

    return wealth_inequality_path(wealth, alive_mask=alive)


def main() -> None:
    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)

    print("OLG panel: baseline financial vol...")
    path_lo = run_olg(sigma_f=0.10, seed=1)
    print("OLG panel: high financial vol...")
    path_hi = run_olg(sigma_f=0.22, seed=1)

    path_lo.to_csv(out_dir / "hf_olg_gini_lowvol.csv", index=False)
    path_hi.to_csv(out_dir / "hf_olg_gini_highvol.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(path_lo["t"], path_lo["gini"], label="sigma_f=0.10")
    ax.plot(path_hi["t"], path_hi["gini"], label="sigma_f=0.22")
    ax.set_xlabel("calendar time")
    ax.set_ylabel("net-worth Gini")
    ax.set_title("Housing-finance OLG: inequality transition (PE)")
    ax.legend()
    fig.tight_layout()
    fig_path = out_dir / "hf_olg_gini_transition.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)

    print("Low vol:", summarize_path(path_lo))
    print("High vol:", summarize_path(path_hi))
    print(f"Wrote {fig_path}")
    print("Housing-finance OLG OK.")


if __name__ == "__main__":
    main()
