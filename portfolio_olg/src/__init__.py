"""Portfolio OLG project: housing-finance model + legacy HARK helpers."""

from .inequality import gini, top_share, wealth_inequality_path
from .ar1_returns import RouwenhorstAR1, attach_ar1_risky_returns
from .returns_ca import CanadaReturnParams, discretize_canada_returns
from .housing_finance import HousingFinanceModel, HousingFinanceParams

# Legacy HARK builders are optional — primary housing-finance path does not need them.
try:
    from .agents import build_lifecycle_portfolio, build_infinite_horizon_portfolio
except ModuleNotFoundError as exc:
    if "HARK" not in str(exc) and "hark" not in str(exc).lower():
        raise

    def build_lifecycle_portfolio(*_args, **_kwargs):  # type: ignore[misc]
        raise ModuleNotFoundError(
            "HARK/econ-ark is required for legacy portfolio agents. "
            "Install with: pip install 'econ-ark>=0.14.1,<0.15'"
        ) from exc

    def build_infinite_horizon_portfolio(*_args, **_kwargs):  # type: ignore[misc]
        raise ModuleNotFoundError(
            "HARK/econ-ark is required for legacy portfolio agents. "
            "Install with: pip install 'econ-ark>=0.14.1,<0.15'"
        ) from exc

__all__ = [
    "HousingFinanceModel",
    "HousingFinanceParams",
    "CanadaReturnParams",
    "discretize_canada_returns",
    "build_lifecycle_portfolio",
    "build_infinite_horizon_portfolio",
    "gini",
    "top_share",
    "wealth_inequality_path",
    "RouwenhorstAR1",
    "attach_ar1_risky_returns",
]
