# Canadian housing + financial portfolio model

Partial-equilibrium life-cycle / OLG household problem implemented in
`src/housing_finance/` with joint Canada return risk from `src/returns_ca.py`.

This document states the equations as coded. Parameters are placeholders until
Canadian data calibration.

---

## 1. Preferences

Period utility is Cobb–Douglas in nondurable consumption \(c\) and housing
services from stock \(h\), with CRRA over the composite:

\[
u(c,h)
=
\begin{cases}
\dfrac{\bigl(c^{\alpha}(h+\underline{h})^{1-\alpha}\bigr)^{1-\rho}-1}{1-\rho}
& \rho\neq 1,\\[0.8em]
\alpha\log c+(1-\alpha)\log(h+\underline{h})
& \rho=1.
\end{cases}
\]

| Symbol | Code | Role |
|--------|------|------|
| \(\alpha\) | `alpha` | Consumption share in the composite |
| \(\rho\) | `rho` | Relative risk aversion / EIS inverse over the composite |
| \(\underline{h}\) | `h_ubar` | Housing floor (avoids \(h=0\) singularity) |
| \(\beta\) | `beta` | **Annual** discount factor (compounded to the model period; see §2) |

Life span is always chronological ages \([\texttt{age\_min},\texttt{age\_max}]\)
(default 25 to 80) discretized into `T` periods. No bequest motive: at the
last period the agent consumes remaining resources under keep/adjust (see §5).

---

## 2. Age grid and period length

Model periods \(t=0,\ldots,T-1\) are mapped to chronological ages so the life
cycle **always** spans \([\underline{a},\overline{a}]=[\texttt{age\_min},\texttt{age\_max}]\):

\[
\mathrm{age}(t)
=
\underline{a}
+
t\cdot\Delta,
\qquad
\Delta
=
\frac{\overline{a}-\underline{a}}{T-1}
=
\texttt{period\_years}.
\]

Equivalently, \(\mathrm{age}=\texttt{np.linspace(age\_min, age\_max, T)}\).
Thus \(t\to t+1\) always advances age by the same \(\Delta\) years, and
\(\mathrm{age}(0)=\underline{a}\), \(\mathrm{age}(T-1)=\overline{a}\).

Defaults: \(\underline{a}=25\), \(\overline{a}=80\). With \(T=8\),
\(\Delta=55/7\approx 7.86\) years.

Annual placeholders (\(\beta\), housing depreciation \(\delta\), income AR(1),
Canada return means/persistence/vols) are compounded to frequency \(\Delta\)
inside `HousingFinanceModel` (`src/age_grid.py`). Cohort profile plots use
**chronological age** on the horizontal axis.

| Object | Code |
|--------|------|
| Age nodes | `model.ages` |
| Years per step | `model.period_years` |
| Helpers | `src/age_grid.py` |

---

## 3. State variables

At the start of period \(t\) (chronological age \(\mathrm{age}(t)\)), after
income and returns have realized:

| Symbol | Meaning |
|--------|---------|
| \(m_t\) | Liquid cash-on-hand |
| \(h_t\) | Illiquid housing stock (enters utility; costly to change) |
| \(z_t\) | Labor-income productivity state |
| \(s_t\) | Joint return state that indexes \((R^f_t,R^h_t)\) |

Labor income is \(y_t=z_t\) on a unit-normalized Rouwenhorst grid
(`rho_z`, `sigma_z`, `Nz`).

---

## 4. Timing within a period

1. Enter with \((m_t,h_t,z_t,s_t)\).
2. Choose **keep** or **adjust** housing (discrete margin).
3. Choose continuous controls (consumption; if adjust, also new housing).
4. End-of-period liquid saving \(a_t\ge 0\) and end-of-period housing
   \(h^{\mathrm{end}}_t\) carry forward.
5. Next period: draw \(z_{t+1}\), \(s_{t+1}\); law of motion below.

### Keep

\[
c_t\in(0,m_t],\qquad
a_t=m_t-c_t\ge 0,\qquad
h^{\mathrm{end}}_t=h_t.
\]

### Adjust

Sell current housing, pay proportional transaction cost \(\tau\), then rebuy:

\[
x_t=m_t+(1-\tau)\,h_t,
\]

\[
c_t+a_t+h^{\mathrm{new}}_t=x_t,\qquad
a_t\ge 0,\quad h^{\mathrm{new}}_t\ge 0,\quad c_t>0,
\]

\[
h^{\mathrm{end}}_t=h^{\mathrm{new}}_t.
\]

Code parameter: `tau`.

---

## 5. Laws of motion

Housing depreciates at rate \(\delta\) (`delta`) between periods, then is
rescaled by the housing gross return:

\[
\begin{aligned}
m_{t+1}
&=
R^f_{t+1}\,a_t+y_{t+1},
\\
h_{t+1}
&=
R^h_{t+1}\,(1-\delta)\,h^{\mathrm{end}}_t,
\\
y_{t+1}
&=
z_{t+1}.
\end{aligned}
\]

Income and return states are independent Markov chains:

\[
z_{t+1}\sim\Pi^z(\,\cdot\mid z_t),\qquad
s_{t+1}\sim\Pi^r(\,\cdot\mid s_t),
\]

with \(R^f_{t+1}=R^f(s_{t+1})\) and \(R^h_{t+1}=R^h(s_{t+1})\).

---

## 6. Bellman equations

Let \(V_t(m,h,z,s)\) be lifetime value at age \(t\).

\[
V_t(m,h,z,s)
=
\max\bigl\{V^{\mathrm{keep}}_t(m,h,z,s),\;V^{\mathrm{adj}}_t(m,h,z,s)\bigr\}.
\]

### Keep

\[
V^{\mathrm{keep}}_t
=
\max_{c\in(0,m]}
\Biggl\{
u(c,h)
+
\beta\,\mathbb{E}_t\bigl[
V_{t+1}\bigl(R^{f\prime}(m-c)+z',\;R^{h\prime}(1-\delta)h,\;z',s'\bigr)
\bigr]
\Biggr\}.
\]

### Adjust

\[
V^{\mathrm{adj}}_t
=
\max_{\substack{c,h^{\mathrm{new}},a\\
c+a+h^{\mathrm{new}}=x\\
a\ge 0,\;c>0,\;h^{\mathrm{new}}\ge 0}}
\Biggl\{
u(c,h^{\mathrm{new}})
+
\beta\,\mathbb{E}_t\bigl[
V_{t+1}\bigl(R^{f\prime}a+z',\;R^{h\prime}(1-\delta)h^{\mathrm{new}},\;z',s'\bigr)
\bigr]
\Biggr\},
\]

where \(x=m+(1-\tau)h\).

Expectations are over \(z'\sim\Pi^z(\cdot\mid z)\) and \(s'\sim\Pi^r(\cdot\mid s)\).

### Terminal period \(t=T-1\) (age \(\overline{a}\))

No continuation (\(\beta\) term dropped). The agent maximizes period utility only,
still choosing keep vs adjust.

---

## 7. Joint Canada return process

Logs of demeaned financial and housing returns follow a bivariate AR(1)
(placeholder Canadian moments in `CanadaReturnParams`):

\[
\begin{pmatrix}
\varepsilon^f_{t+1}\\
\varepsilon^h_{t+1}
\end{pmatrix}
=
\begin{pmatrix}
\rho_f & 0\\
0 & \rho_h
\end{pmatrix}
\begin{pmatrix}
\varepsilon^f_t\\
\varepsilon^h_t
\end{pmatrix}
+
\begin{pmatrix}
\sigma_f & 0\\
0 & \sigma_h
\end{pmatrix}
\begin{pmatrix}
u^f_{t+1}\\
u^h_{t+1}
\end{pmatrix},
\]

with \(\mathrm{Corr}(u^f,u^h)=\gamma\) (`corr`).

Gross returns are scaled to hit target means \(\bar R^f\), \(\bar R^h\):

\[
R^i_t
=
\bar R^i\cdot
\frac{\exp(\varepsilon^i_t)}{\mathbb{E}[\exp(\varepsilon^i)]},
\qquad i\in\{f,h\}.
\]

Each margin is discretized by Rouwenhorst; the joint chain is the product grid
with a correlation tilt on transitions (`discretize_canada_returns`).

Gross returns below are **annual** placeholders; the model compounds them to
\(\Delta=\texttt{period\_years}\) before discretization.

---

## 8. Wealth-dependent financial volatility (simulation layer)

The **baseline solve** uses constant \(\sigma_f\). In simulation, optional
wealth scaling (`wealth_vol_kappa`, `wealth_vol_norm`) multiplies financial
risk for richer households:

\[
\sigma_f(W)
=
\sigma_f^{\mathrm{base}}\Bigl(1+\kappa\cdot\frac{W}{\bar W}\Bigr),
\qquad
W\approx a_{\mathrm{begin}}+h.
\]

Implemented in `simulate.py` / `run_wealth_vol.py`. Solving the DP with
state-dependent \(\sigma_f\) is not yet the default.

---

## 9. Simulation modes and DecisionSimulation

### Cohort vs panel

| Mode | API | Use |
|------|-----|-----|
| Cohort lifecycle | `simulate_cohort` / `model.simulate` | One birth cohort, ages `age_min`→`age_max` |
| Mixed-age panel | `simulate_panel` / `model.simulate_panel` | Cross-section with heterogeneous ages, $K$ calendar steps |

### Initial conditions

- **`InitialSample`**: user-provided arrays (`m`, `h`, `i_z`, `i_r`, optional `t_age`); CSV stub via `from_placeholder_csv`.
- **`InitialDistribution`**: placeholder draws (default uniforms); `mixed_ages=True` for panels.

### Terminal age in panels

- `on_terminal="replace_young"` (default): replace with a new young draw (fixed $N$).
- `on_terminal="exit"`: drop the agent (paths NaN afterward).

### `DecisionSimulation` map $\theta \mapsto m(\theta)$

Configurable class in `src/housing_finance/decision_simulation.py`:

1. Merge call-time $\theta$ (params / dict overrides) into a template.
2. Solve decision rules.
3. Run cohort and/or panel sims per flags.
4. Compute **placeholder** statistics (`statistics` / `moment_vector`).

Bind flags once, then use as an SMM primitive, e.g.
`lambda theta: decision_sim(theta).moment_vector`.

The older staggered OLG inequality script remains a separate demo
(`scripts/run_housing_finance_olg.py`).

---

## 10. OLG / inequality panels

Finite-lived cohorts can also be stacked into a calendar-time panel (legacy
demo). Net worth is

\[
W_t=m_t+h_t
\]

(or liquid + housing as stored in simulation). Inequality metrics (Gini, top
shares) are computed on the cross-section each calendar date
(`src/inequality.py`, `scripts/run_housing_finance_olg.py`).

---

## 11. Out of scope (this subproject)

- Mortgages / LTV / amortization → see `../housing_mortgage/`
- General-equilibrium house prices
- Estimated Canadian SMM (targets file is a placeholder only; optimizer not wired)

---

## 12. Code map

| Piece | Location |
|-------|----------|
| Age / period mapping | `src/age_grid.py` |
| Utility | `src/housing_finance/utility.py` |
| VFI keep/adjust | `src/housing_finance/solve.py` |
| Grids / solve / simulate API | `src/housing_finance/model.py` |
| Initial conditions | `src/housing_finance/initial.py` |
| Cohort / panel sims | `src/housing_finance/simulate.py` |
| DecisionSimulation ($\theta\mapsto m$) | `src/housing_finance/decision_simulation.py` |
| Joint returns | `src/returns_ca.py` |
| Inequality helpers | `src/inequality.py` |
| Lifecycle means / quantiles | `src/lifecycle_profiles.py` |
| Check notebook | `notebooks/02_run_and_check_housing_finance.ipynb` |
