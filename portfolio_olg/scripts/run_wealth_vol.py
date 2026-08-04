"""Wealth-dependent financial return volatility (simulation layer)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.housing_finance import HousingFinanceModel, HousingFinanceParams  # noqa: E402
from src.inequality import gini  # noqa: E402
from src.returns_ca import CanadaReturnParams, discretize_canada_returns, scale_financial_vol  # noqa: E402


def main() -> None:
    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)

    ret = CanadaReturnParams(n_f=3, n_h=3, sigma_f=0.14, sigma_h=0.10, corr=0.35)
    par = HousingFinanceParams(
        T=6,
        Nm=12,
        Nh=8,
        Nz=3,
        n_c=8,
        simN=2000,
        returns=ret,
        m_max=6.0,
        h_max=4.0,
    )

    # Solve once with constant-vol chain
    model = HousingFinanceModel(par)
    print("Solving (constant sigma_f at solve time)...")
    model.solve()

    print("Simulating constant vol...")
    model.par.sim_seed = 11
    model.simulate(wealth_vol_kappa=0.0)
    nw_const = model.sim.net_worth.copy()
    gini_const = [gini(nw_const[t]) for t in range(nw_const.shape[0])]

    print("Simulating wealth-scaled financial vol (kappa=0.8)...")
    model.par.sim_seed = 11
    model.simulate(wealth_vol_kappa=0.8, wealth_vol_norm=4.0)
    nw_w = model.sim.net_worth.copy()
    gini_w = [gini(nw_w[t]) for t in range(nw_w.shape[0])]

    # Also show scale_financial_vol helper for solve-time approximation grids
    base = discretize_canada_returns(ret)
    scaled = scale_financial_vol(base, vol_mult=1.5)
    print("scale_financial_vol(1.5) summary:", scaled.summary())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(nw_const.mean(axis=1), label="const vol")
    axes[0].plot(nw_w.mean(axis=1), label="wealth-scaled vol")
    axes[0].set_title("Mean net worth")
    axes[0].legend()
    axes[1].plot(gini_const, label="const vol")
    axes[1].plot(gini_w, label="wealth-scaled vol")
    axes[1].set_title("Wealth Gini")
    axes[1].legend()
    fig.suptitle("Wealth-dependent financial volatility (sim approximation)")
    fig.tight_layout()
    path = out_dir / "wealth_dependent_vol.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)

    np.savetxt(
        out_dir / "wealth_vol_gini_compare.csv",
        np.column_stack([gini_const, gini_w]),
        delimiter=",",
        header="gini_const,gini_wealth_scaled",
        comments="",
    )
    print(f"Wrote {path}")
    print(
        f"Final Gini const={gini_const[-1]:.3f}, wealth-scaled={gini_w[-1]:.3f}"
    )
    print("Wealth-vol OK.")


if __name__ == "__main__":
    main()
