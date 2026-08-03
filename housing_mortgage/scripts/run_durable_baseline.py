"""Run a fast Python/Numba DurableConsumptionModel baseline (no C++)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "DurableConsumptionModel"
sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(ROOT))

from DurableConsumptionModel import DurableConsumptionModelClass  # noqa: E402
from src.housing_map import describe_housing_mapping  # noqa: E402


def build_fast_model() -> DurableConsumptionModelClass:
    """Coarse grids / short horizon for a reliable smoke test."""
    model = DurableConsumptionModelClass(
        name="durable_baseline_fast",
        par=dict(
            solmethod="nvfi",
            T=4,
            Np=12,
            Nn=12,
            Nm=40,
            Nx=40,
            Na=40,
            Npsi=3,
            Nxi=3,
            simN=2_000,
            do_print=True,
            do_print_period=False,
        ),
    )
    return model


def main() -> None:
    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)

    print(describe_housing_mapping())
    print("\n=== Solving DurableConsumptionModel (nvfi, coarse) ===")
    model = build_fast_model()
    model.solve()
    print("Solve done. Checksums:")
    model.checksum(simple=True)

    print("=== Simulating ===")
    model.simulate()
    sim = model.sim
    # Typical fields: c, d, a, m, n, p, ...
    mean_c = float(np.mean(sim.c))
    mean_n = float(np.mean(sim.n))
    mean_m = float(np.mean(sim.m))
    print(f"  mean c={mean_c:.4f}, mean n (durable/housing)={mean_n:.4f}, mean m (liquid)={mean_m:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.mean(sim.n, axis=1), label="mean housing/durable n")
    ax.plot(np.mean(sim.m, axis=1), label="mean liquid m")
    ax.plot(np.mean(sim.c, axis=1), label="mean consumption c")
    ax.set_xlabel("age / period")
    ax.set_title("Durable baseline life-cycle means")
    ax.legend()
    fig.tight_layout()
    fig_path = out_dir / "durable_baseline_means.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)

    map_path = out_dir / "housing_mapping.txt"
    map_path.write_text(describe_housing_mapping(), encoding="utf-8")
    print(f"Wrote {fig_path}")
    print(f"Wrote {map_path}")
    print("Durable baseline OK.")


if __name__ == "__main__":
    main()
