"""Exercise mortgage scaffold accounting (no full DP yet)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.mortgage_scaffold import (  # noqa: E402
    MortgageParams,
    MortgageState,
    demo_path,
    feasible_purchase,
    ltv,
    max_originating_mortgage,
    min_payment,
)


def main() -> None:
    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)

    params = MortgageParams(r_mortgage=0.04, maturity=30, ltv_max=0.8)
    housing, price = 2.5, 1.0
    m_max = max_originating_mortgage(housing, price, params)
    print(f"Max mortgage at LTV={params.ltv_max}: {m_max:.3f}")
    print(f"Min payment on max loan: {min_payment(m_max, params.r_mortgage, params.maturity):.4f}")
    print(f"Feasible with liquid=0.6? {feasible_purchase(0.6, housing, m_max, price, params)}")
    print(f"Feasible with liquid=0.1? {feasible_purchase(0.1, housing, m_max, price, params)}")

    state = MortgageState(liquid=0.5, housing=housing, mortgage=m_max, house_price=price)
    print(
        f"State equity={state.home_equity:.3f}, net worth={state.net_worth:.3f}, "
        f"LTV={ltv(state.housing, state.mortgage, state.house_price):.3f}"
    )

    path = demo_path(n_periods=40, housing=housing, price=price, params=params)
    np.savetxt(out_dir / "mortgage_demo_amortization.csv", path, delimiter=",", header="balance", comments="")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(path)
    ax.set_xlabel("period")
    ax.set_ylabel("mortgage balance")
    ax.set_title("Scaffold: illustrative amortization path")
    fig.tight_layout()
    fig_path = out_dir / "mortgage_amortization_demo.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {fig_path}")
    print("Mortgage scaffold OK. See docs/MORTGAGE_DESIGN.md for DP integration plan.")


if __name__ == "__main__":
    main()
