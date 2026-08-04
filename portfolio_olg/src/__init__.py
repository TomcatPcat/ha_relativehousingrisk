"""Portfolio OLG project: housing-finance model + legacy HARK helpers."""

from .agents import build_lifecycle_portfolio, build_infinite_horizon_portfolio
from .inequality import gini, top_share, wealth_inequality_path
from .ar1_returns import RouwenhorstAR1, attach_ar1_risky_returns
from .returns_ca import CanadaReturnParams, discretize_canada_returns
from .housing_finance import HousingFinanceModel, HousingFinanceParams

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
