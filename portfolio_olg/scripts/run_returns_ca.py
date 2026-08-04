"""Print summary of discretized Canada housing/finance returns."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.returns_ca import CanadaReturnParams, discretize_canada_returns  # noqa: E402


def main() -> None:
    chain = discretize_canada_returns(CanadaReturnParams())
    s = chain.summary()
    print("Canada joint return chain (placeholders)")
    for k, v in s.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print("  Rf grid:", np_round := __import__("numpy").round(chain.Rf, 3))
    print("  Rh grid:", __import__("numpy").round(chain.Rh, 3))
    print("Returns CA OK.")


if __name__ == "__main__":
    main()
