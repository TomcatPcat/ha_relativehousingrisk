# Canadian Housing + Financial Portfolio Model

Partial-equilibrium life-cycle model used in `portfolio_olg` (primary research path).
Stock HARK safe/risky scripts under `scripts/run_baseline.py` etc. are **legacy demos**.

## Objects

| Symbol | Meaning |
|--------|---------|
| \(c\) | Nondurable consumption |
| \(a\) | End-of-period liquid financial saving |
| \(m\) | Liquid cash-on-hand (beginning of period) |
| \(h\) | Illiquid housing stock (enters utility; costly to adjust) |
| \(y\) | Labor income |
| \(R^f\) | Gross return on liquid financial wealth (Canada financial market) |
| \(R^h\) | Gross return / price factor on housing (Canada housing market) |
| \(\tau\) | Proportional transaction cost when adjusting housing |
| \(\delta\) | Housing depreciation |

## Utility

\[
u(c,h)=\frac{\big(c^{\alpha}(h+\underline{h})^{1-\alpha}\big)^{1-\rho}-1}{1-\rho}
\]

(with the usual CRRA limit as \(\rho\to 1\)).

## Timing (each age \(t\))

1. Enter with liquid cash-on-hand \(m\), housing \(h\), income state, and joint return state \(s\) that pins down \((R^f,R^h)\).
2. **Keep** housing: choose \(c\in(0,m]\), set \(a=m-c\ge 0\). Continuation housing before next return draw: \((1-\delta)h\).
3. **Adjust** housing: sell stock, pay \(\tau\), resources \(x=m+(1-\tau)h\). Choose \(c\) and new housing \(h^{\mathrm{new}}\) with \(c+a+h^{\mathrm{new}}=x\), \(a\ge 0\), \(h^{\mathrm{new}}\ge 0\).
4. Next period: income realizes; return state transitions on a joint Markov chain;  
   \(m' = R^{f\prime} a + y'\),  
   \(h' = R^{h\prime}\,(1-\delta)\,h^{\mathrm{end}}\)  
   where \(h^{\mathrm{end}}=h\) if keep and \(h^{\mathrm{new}}\) if adjust.
5. Take \(\max\{\text{keep},\text{adjust}\}\) value.

## Returns (Canada placeholders)

Logs of excess / demeaned returns follow a bivariate AR(1):

\[
\begin{pmatrix} \varepsilon^f_{t+1} \\ \varepsilon^h_{t+1} \end{pmatrix}
=
\begin{pmatrix} \rho_f & 0 \\ 0 & \rho_h \end{pmatrix}
\begin{pmatrix} \varepsilon^f_t \\ \varepsilon^h_t \end{pmatrix}
+
\Sigma^{1/2}
\begin{pmatrix} u^f \\ u^h \end{pmatrix},
\quad
\mathrm{Corr}(u^f,u^h)=\gamma.
\]

Gross returns: \(R^i = \bar R^i \exp(\varepsilon^i)/\mathbb{E}[\exp(\varepsilon^i)]\) so means match \(\bar R^f,\bar R^h\).
Discretized to a joint Markov chain (see `src/returns_ca.py`).

## Wealth-dependent financial volatility (phase 2)

In simulation (and optionally at solve time via an approximation),

\[
\sigma_f(W)=\sigma_f^{\mathrm{base}}\big(1+\kappa\cdot \tfrac{W}{\bar W}\big),
\]

where \(W=a_{\mathrm{begin}}+h\) (net worth). Baseline solve may use constant \(\sigma_f\); see `run_wealth_vol.py`.

## Out of scope here

Mortgages, GE house prices, full SMM to Canadian microdata (placeholders only).
