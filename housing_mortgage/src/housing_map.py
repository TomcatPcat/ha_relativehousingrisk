"""Map Druedahl DurableConsumptionModel objects onto a housing PE interpretation."""

from __future__ import annotations

from typing import Dict

HOUSING_STATE_MAP: Dict[str, str] = {
    "n": "housing / durable stock (illiquid; adjustment costs)",
    "m": "liquid cash-on-hand / market resources",
    "p": "permanent income factor",
    "x": "total resources used when adjusting durable (m + (1-tau)*n style)",
    "a": "end-of-period liquid assets (post-decision)",
    "d": "chosen durable / housing stock after adjustment",
    "c": "nondurable consumption",
    "R": "safe liquid return factor",
    "delta": "durable depreciation (housing maintenance / wear)",
    "tau": "transaction / adjustment cost parameter on durable trades",
}


def describe_housing_mapping() -> str:
    lines = [
        "DurableConsumptionModel -> housing PE mapping",
        "============================================",
        "",
        "Treat the durable good as a housing stock with nonconvex adjustment costs.",
        "Liquid resources remain the nondurable / liquid savings account.",
        "",
        "Limits of the stock model (before mortgage extension):",
        "- No separate mortgage liability state.",
        "- No LTV / LTI origination constraints.",
        "- No amortization schedule, refinance, or default.",
        "- Housing is continuous on a grid, not discrete tenure sizes.",
        "- Prices/returns are exogenous (partial equilibrium).",
        "",
        "State / parameter map:",
    ]
    for k, v in HOUSING_STATE_MAP.items():
        lines.append(f"  {k:6} -> {v}")
    return "\n".join(lines)
