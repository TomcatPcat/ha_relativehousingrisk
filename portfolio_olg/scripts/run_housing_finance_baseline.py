"""Solve and simulate the housing-finance life-cycle baseline."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.housing_finance import HousingFinanceModel, HousingFinanceParams  # noqa: E402
from src.returns_ca import CanadaReturnParams  # noqa: E402


def main() -> None:
    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)

    # Fast smoke grids
    ret = CanadaReturnParams(n_f=3, n_h=3, sigma_f=0.16, sigma_h=0.10, corr=0.35)
    par = HousingFinanceParams(
        T=6,
        Nm=14,
        Nh=10,
        Nz=3,
        n_c=10,
        simN=1500,
        returns=ret,
        m_max=6.0,
        h_max=4.0,
    )
    model = HousingFinanceModel(par)
    print("Solving housing-finance life-cycle DP...")
    model.solve()
    print(f"  V[0] mean={model.sol.V[0].mean():.4f}")
    print("Simulating...")
    model.simulate()

    print(
        f"  mean c={model.sim.c.mean():.3f}, mean h={model.sim.h.mean():.3f}, "
        f"mean m={model.sim.m.mean():.3f}, adj rate={model.sim.adj.mean():.3f}"
    )
    print(f"  mean net worth={model.sim.net_worth.mean():.3f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(model.sim.m.mean(axis=1), label="liquid m")
    ax.plot(model.sim.h.mean(axis=1), label="housing h")
    ax.plot(model.sim.c.mean(axis=1), label="consumption c")
    ax.plot(model.sim.net_worth.mean(axis=1), label="net worth")
    ax.set_xlabel("age / period")
    ax.set_title("Housing-finance baseline (Canada returns placeholders)")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "housing_finance_baseline.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Wrote {path}")
    print("Housing-finance baseline OK.")


if __name__ == "__main__":
    main()
