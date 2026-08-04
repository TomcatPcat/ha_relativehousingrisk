"""Compute SFS / SHS weighted aggregate moments; write CSV under output/survey/."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.survey import (  # noqa: E402
    load_sfs,
    load_shs_all,
    sfs_aggregate_moments,
    shs_cy_moments,
)


def main() -> None:
    out = ROOT / "output" / "survey"
    out.mkdir(parents=True, exist_ok=True)

    print("Loading SFS...")
    sfs = load_sfs()
    sfs_m = sfs_aggregate_moments(sfs, by="year")
    sfs_path = out / "sfs_aggregate_moments.csv"
    sfs_m.to_csv(sfs_path, index=False)
    print(sfs_m.to_string(index=False))
    print(f"Wrote {sfs_path}")

    print("\nLoading SHS (Tier A + B)...")
    shs = load_shs_all()
    # Report by year within tier so A/B are not spliced
    shs_m = shs_cy_moments(shs, by=["tier", "year"])
    shs_path = out / "shs_cy_moments.csv"
    shs_m.to_csv(shs_path, index=False)
    print(shs_m.to_string(index=False))
    print(f"Wrote {shs_path}")


if __name__ == "__main__":
    main()
