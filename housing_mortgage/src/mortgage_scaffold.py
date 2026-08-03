"""
Mortgage scaffolding for a future extension of DurableConsumptionModel.

This module does **not** yet modify the upstream Bellman solver. It defines
the economic objects and simple accounting identities we will attach as a
third account (mortgage balance) alongside liquid ``m`` and housing ``n``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MortgageParams:
    """Institutional / contract parameters (annualized stylized)."""

    r_mortgage: float = 0.04  # interest rate on outstanding balance
    maturity: int = 30  # periods for full amortization schedule
    ltv_max: float = 0.80  # max loan-to-value at purchase
    prepay_allowed: bool = True
    min_payment_frac: float = 0.0  # optional floor as fraction of balance


@dataclass
class MortgageState:
    """Household balance-sheet slice at a point in time."""

    liquid: float
    housing: float
    mortgage: float
    house_price: float = 1.0

    @property
    def home_equity(self) -> float:
        return self.housing * self.house_price - self.mortgage

    @property
    def net_worth(self) -> float:
        return self.liquid + self.home_equity


def ltv(housing: float, mortgage: float, house_price: float = 1.0) -> float:
    """Loan-to-value ratio."""
    value = housing * house_price
    if value <= 0:
        return np.inf if mortgage > 0 else 0.0
    return float(mortgage / value)


def min_payment(balance: float, r: float, maturity_left: int) -> float:
    """
    Standard annuity minimum payment for remaining maturity.

    If maturity_left <= 0, require full repayment of (1+r)*balance.
    """
    bal = (1.0 + r) * balance
    if balance <= 0:
        return 0.0
    if maturity_left <= 0:
        return float(bal)
    if abs(r) < 1e-12:
        return float(bal / maturity_left)
    ann = r * (1 + r) ** maturity_left / ((1 + r) ** maturity_left - 1.0)
    return float(ann * balance)


def max_originating_mortgage(housing: float, house_price: float, params: MortgageParams) -> float:
    """Maximum mortgage allowed at purchase given LTV cap."""
    return float(params.ltv_max * housing * house_price)


def feasible_purchase(
    liquid: float,
    housing_choice: float,
    mortgage_choice: float,
    house_price: float,
    params: MortgageParams,
    adj_cost: float = 0.0,
) -> bool:
    """Cash + mortgage must cover house purchase and adjustment costs; LTV ok."""
    if mortgage_choice < -1e-12:
        return False
    if ltv(housing_choice, mortgage_choice, house_price) > params.ltv_max + 1e-9:
        return False
    down_payment = housing_choice * house_price - mortgage_choice
    return liquid + 1e-12 >= down_payment + adj_cost


def next_mortgage_balance(
    mortgage: float,
    payment: float,
    params: MortgageParams,
) -> float:
    """Update balance after interest accrual and payment (no negative balances)."""
    due = (1.0 + params.r_mortgage) * mortgage
    return float(max(0.0, due - payment))


def demo_path(
    n_periods: int = 10,
    housing: float = 2.0,
    price: float = 1.0,
    params: Optional[MortgageParams] = None,
) -> np.ndarray:
    """Illustrative amortization path for documentation / tests."""
    params = params or MortgageParams()
    m0 = max_originating_mortgage(housing, price, params)
    bal = m0
    path = np.zeros(n_periods)
    for t in range(n_periods):
        path[t] = bal
        pay = min_payment(bal, params.r_mortgage, max(params.maturity - t, 1))
        bal = next_mortgage_balance(bal, pay, params)
    return path
