# Mortgage extension design (Durable + liquid → + mortgage)

## Goal

Extend Druedahl's DurableConsumptionModel so households have three accounts:

1. **Liquid savings** \(m\) / \(a\) (already in durable model)
2. **Housing stock** \(n\) / \(d\) (durable good)
3. **Mortgage balance** \(b\) (new liability state)

Partial equilibrium, finite life-cycle (no OLG demographics in v1).

## Recommended state / choice structure (v1)

| Object | Type | Notes |
|--------|------|--------|
| Age \(t\) | discrete | existing `par.T` |
| Permanent income \(p\) | continuous grid | existing |
| Liquid \(m\) | continuous grid | existing |
| Housing \(n\) | continuous or **discrete sizes** | start continuous (durable grid); later discrete tenure |
| Mortgage \(b\) | coarse continuous grid | new; \(b \ge 0\) |
| Keep vs adjust housing | discrete | existing keep/adjust |
| Consumption \(c\), saving \(a'\) | continuous | existing |
| Mortgage payment / new loan | continuous or rule-of-thumb | new when adjusting |

## Timing (proposed)

1. Enter period with \((p, m, n, b)\).
2. Income realizes; liquid cash-on-hand updates.
3. Mortgage interest accrues; minimum payment due (annuity formula in `src/mortgage_scaffold.py`).
4. If **keep** housing: choose \(c\), \(a'\), and payment \(\ge\) minimum (optional prepay).
5. If **adjust** housing: choose new \(n'\), new mortgage \(b'\) subject to LTV, pay adjustment costs from liquid + net equity proceeds.
6. Continue to \(t+1\).

## Constraints

- **LTV at origination:** \(b' \le \lambda\, p_h n'\)
- **Payment:** \(pay \ge \underline{pay}(b, r, T-t)\)
- **Budget:** liquid must cover consumption, payments, down payment, and \(\tau\)-style transaction costs
- Optional later: default (give up house, wipe mortgage, utility / exclusion cost)

## Solver strategy

1. Keep upstream NVFI/NEGM for durable+liquid as the baseline.
2. Add \(b\) as an extra state on a coarse grid (e.g. 15–25 points).
3. First implement **keep-only mortgage amortization** (no new borrowing except at adjust).
4. Then allow refinance / new LTV-constrained loan only in the adjust branch.
5. Validate against `demo_path` accounting and simulated LTV distribution.

## Code ownership

- Upstream (do not casually edit): `vendor/DurableConsumptionModel/`
- Project logic: `src/mortgage_scaffold.py`, future `src/mortgage_model.py`
- Design doc: this file

## Out of scope for first scaffold

- General equilibrium house prices
- Aggregate credit supply
- AR(1) house-price risk (phase after mortgage DP runs)
- Full OLG demographics
