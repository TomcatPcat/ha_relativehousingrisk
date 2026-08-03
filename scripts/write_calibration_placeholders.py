"""Write deferred calibration placeholder files for both projects."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "portfolio_olg"))

from src.calibration import write_example_targets  # noqa: E402


def main() -> None:
    p1 = write_example_targets(ROOT / "portfolio_olg", "portfolio_olg")
    # housing uses same helper copied conceptually
    p2_root = ROOT / "housing_mortgage"
    moments = {
        "homeownership_rate": None,
        "mean_ltv": None,
        "median_liquid": None,
    }
    path = p2_root / "data" / "targets.example.yaml"
    lines = ["# Example calibration targets (fill in later; not estimated)", "moments:"]
    for k, v in moments.items():
        lines.append(f"  {k}: {v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {p1}")
    print(f"Wrote {path}")
    print("Calibration hooks deferred (placeholders only).")


if __name__ == "__main__":
    main()
