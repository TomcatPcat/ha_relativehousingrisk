"""Shared calibration placeholders (deferred for both projects)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

# Example moment names only — no estimation yet.
DEFAULT_TARGETS: Dict[str, Any] = {
    "portfolio_olg": {
        "wealth_gini": None,
        "top10_wealth_share": None,
        "mean_risky_share": None,
    },
    "housing_mortgage": {
        "homeownership_rate": None,
        "mean_ltv": None,
        "median_liquid": None,
    },
}


def targets_path(project_root: Path) -> Path:
    return project_root / "data" / "targets.example.yaml"


def write_example_targets(project_root: Path, project_key: str) -> Path:
    """Write a YAML-like example file with null targets."""
    path = targets_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    moments = DEFAULT_TARGETS[project_key]
    lines = ["# Example calibration targets (fill in later; not estimated)", "moments:"]
    for k, v in moments.items():
        lines.append(f"  {k}: {v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
