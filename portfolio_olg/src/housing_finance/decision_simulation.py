"""Configurable solve → simulate → statistics map for later SMM."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields, replace
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from ..inequality import gini
from ..returns_ca import CanadaReturnParams
from ..survey.weighted import wmean
from .initial import InitialDistribution, InitialSample
from .model import HousingFinanceModel, HousingFinanceParams
from .simulate import OnTerminal, simulate_cohort, simulate_panel

ThetaLike = Union[HousingFinanceParams, Mapping[str, Any], None]


@dataclass
class DecisionSimulationResult:
    """Output of one DecisionSimulation call."""

    params: HousingFinanceParams
    statistics: Dict[str, float]
    moment_vector: np.ndarray
    moment_names: List[str]
    cohort: Optional[Dict[str, Any]] = None
    panel: Optional[Dict[str, Any]] = None
    model: Optional[HousingFinanceModel] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, float]:
        return dict(self.statistics)


@dataclass
class DecisionSimulation:
    """
    Bind simulation / statistics flags once; call with parameters as ``f(theta)``.

    Construction fixes settings that should not change inside an SMM loop
    (grids, which sims to run, N, panel length, initials, seeds). Each call
    merges ``theta`` into ``params_template``, solves, runs sims, and returns
    placeholder statistics suitable for a later criterion.
    """

    params_template: HousingFinanceParams = field(default_factory=HousingFinanceParams)
    run_cohort: bool = True
    run_panel: bool = True
    simN: Optional[int] = None
    panel_n_periods: int = 8
    on_terminal: OnTerminal = "replace_young"
    distribution: Optional[InitialDistribution] = None
    initial_cohort: Optional[InitialSample] = None
    initial_panel: Optional[InitialSample] = None
    seed: Optional[int] = None
    panel_seed: Optional[int] = None
    scale_annual_to_period: bool = True
    wealth_vol_kappa: float = 0.0
    wealth_vol_norm: float = 5.0
    keep_model: bool = False
    # Placeholder bag for future data moments, weights, shocks, etc.
    extra_inputs: Dict[str, Any] = field(default_factory=dict)
    # Which placeholder stats to emit (None = default set)
    statistic_names: Optional[Sequence[str]] = None

    def __call__(
        self,
        theta: ThetaLike = None,
        **extra: Any,
    ) -> DecisionSimulationResult:
        """
        Solve decision rules, run configured simulations, return statistics.

        Parameters
        ----------
        theta :
            ``HousingFinanceParams``, a dict of field overrides on
            ``params_template``, or None to use the template as-is.
            # TODO: accept a flat parameter vector once free-parameter map exists.
        **extra :
            Merged into ``extra_inputs`` for this call (placeholder hook).
        """
        merged_extra = {**self.extra_inputs, **extra}
        params = self._build_params(theta, merged_extra)
        if self.simN is not None:
            params = replace(params, simN=int(self.simN))
        if self.seed is not None:
            params = replace(params, sim_seed=int(self.seed))

        model = HousingFinanceModel(
            params, scale_annual_to_period=self.scale_annual_to_period
        )
        model.solve()

        n = int(params.simN)
        cohort_out: Optional[Dict[str, Any]] = None
        panel_out: Optional[Dict[str, Any]] = None

        if self.run_cohort:
            cohort_out = simulate_cohort(
                model,
                initial=self.initial_cohort,
                distribution=self.distribution,
                seed=self.seed if self.seed is not None else params.sim_seed,
                n=n if self.initial_cohort is None else self.initial_cohort.N,
                wealth_vol_kappa=self.wealth_vol_kappa,
                wealth_vol_norm=self.wealth_vol_norm,
            )
            for k, v in cohort_out.items():
                setattr(model.sim, k, v)
            model.sim.ages = model.ages.copy()
            model.sim.period_years = model.period_years

        if self.run_panel:
            panel_seed = (
                self.panel_seed
                if self.panel_seed is not None
                else (self.seed if self.seed is not None else params.sim_seed) + 17
            )
            panel_out = simulate_panel(
                model,
                n_periods=self.panel_n_periods,
                initial=self.initial_panel,
                distribution=self.distribution
                or InitialDistribution(mixed_ages=True),
                on_terminal=self.on_terminal,
                seed=panel_seed,
                n=n if self.initial_panel is None else self.initial_panel.N,
                wealth_vol_kappa=self.wealth_vol_kappa,
                wealth_vol_norm=self.wealth_vol_norm,
            )
            model.sim.panel = SimpleNamespace(**{
                k: v for k, v in panel_out.items() if not isinstance(v, str)
            })

        statistics = self._compute_statistics(cohort_out, panel_out, model)
        names = list(statistics.keys())
        moment_vector = np.array([statistics[k] for k in names], dtype=float)

        return DecisionSimulationResult(
            params=params,
            statistics=statistics,
            moment_vector=moment_vector,
            moment_names=names,
            cohort=cohort_out,
            panel=panel_out,
            model=model if self.keep_model else None,
            extra=merged_extra,
        )

    def _build_params(
        self,
        theta: ThetaLike,
        extra: Mapping[str, Any],
    ) -> HousingFinanceParams:
        base = deepcopy(self.params_template)
        # Placeholder: allow extra["params_overrides"] without being in theta
        overrides: Dict[str, Any] = {}
        if "params_overrides" in extra and isinstance(extra["params_overrides"], Mapping):
            overrides.update(dict(extra["params_overrides"]))

        if theta is None:
            pass
        elif isinstance(theta, HousingFinanceParams):
            return deepcopy(theta)
        elif isinstance(theta, Mapping):
            overrides.update(dict(theta))
        else:
            # TODO: theta vector ↔ params via free-parameter list
            raise TypeError(
                "theta must be HousingFinanceParams, a dict of overrides, or None "
                "(flat vector packing not implemented yet)"
            )

        if not overrides:
            return base

        # Nested CanadaReturnParams updates via returns_* keys or returns=dict
        ret_updates: Dict[str, Any] = {}
        flat: Dict[str, Any] = {}
        for k, v in overrides.items():
            if k == "returns" and isinstance(v, Mapping):
                ret_updates.update(dict(v))
            elif k == "returns" and isinstance(v, CanadaReturnParams):
                flat["returns"] = v
            elif k.startswith("returns_"):
                ret_updates[k[len("returns_") :]] = v
            else:
                flat[k] = v

        if ret_updates:
            ret = flat.get("returns", base.returns)
            if not isinstance(ret, CanadaReturnParams):
                ret = base.returns
            allowed = {f.name for f in fields(CanadaReturnParams)}
            ret_kwargs = {kk: vv for kk, vv in ret_updates.items() if kk in allowed}
            flat["returns"] = replace(ret, **ret_kwargs)

        allowed_par = {f.name for f in fields(HousingFinanceParams)}
        bad = [k for k in flat if k not in allowed_par]
        if bad:
            raise KeyError(f"Unknown HousingFinanceParams fields: {bad}")
        return replace(base, **flat)

    def _compute_statistics(
        self,
        cohort: Optional[Mapping[str, Any]],
        panel: Optional[Mapping[str, Any]],
        model: HousingFinanceModel,
    ) -> Dict[str, float]:
        """
        Placeholder model moments for SMM.

        TODO: replace with data-aligned moment list (lifecycle profiles,
        cross-section wealth shares, adjustment rates, return moments, etc.).
        """
        wanted = (
            list(self.statistic_names)
            if self.statistic_names is not None
            else None
        )
        stats: Dict[str, float] = {}

        if cohort is not None:
            nw = np.asarray(cohort["net_worth"], dtype=float)
            c = np.asarray(cohort["c"], dtype=float)
            h = np.asarray(cohort["h"], dtype=float)
            m = np.asarray(cohort["m"], dtype=float)
            adj = np.asarray(cohort["adj"], dtype=float)
            w = np.asarray(
                cohort.get("weight", np.ones(nw.shape[1], dtype=float)),
                dtype=float,
            )
            if w.ndim == 1:
                w2 = np.broadcast_to(w, nw.shape)
            else:
                w2 = w
            stats["cohort_mean_net_worth"] = float(wmean(nw.ravel(), w2.ravel()))
            stats["cohort_mean_c"] = float(wmean(c.ravel(), w2.ravel()))
            stats["cohort_mean_h"] = float(wmean(h.ravel(), w2.ravel()))
            stats["cohort_mean_m"] = float(wmean(m.ravel(), w2.ravel()))
            stats["cohort_adj_rate"] = float(wmean(adj.ravel(), w2.ravel()))
            # Age-profile anchors: first / mid / last period mean NW
            T = nw.shape[0]
            stats["cohort_nw_age_min"] = float(wmean(nw[0], w2[0]))
            stats["cohort_nw_age_mid"] = float(wmean(nw[T // 2], w2[T // 2]))
            stats["cohort_nw_age_max"] = float(wmean(nw[-1], w2[-1]))

        if panel is not None:
            nw = np.asarray(panel["net_worth"], dtype=float)
            alive = np.asarray(panel["alive"], dtype=bool)
            last = nw.shape[0] - 1
            w = np.asarray(
                panel.get("weight", np.ones(nw.shape[1], dtype=float)),
                dtype=float,
            )
            if w.ndim > 1:
                w_last = w[last]
            else:
                w_last = w
            mask = alive[last]
            nw_last = nw[last][mask]
            w_alive = w_last[mask]
            stats["panel_mean_net_worth_last"] = (
                float(wmean(nw_last, w_alive)) if nw_last.size else float("nan")
            )
            stats["panel_gini_net_worth_last"] = gini(nw_last, w_alive)
            c_panel = np.asarray(panel["c"], dtype=float)
            if w.ndim == 1:
                w_c = np.broadcast_to(w, c_panel.shape)
            else:
                w_c = w
            stats["panel_mean_c"] = float(wmean(c_panel.ravel(), w_c.ravel()))
            stats["panel_n_alive_last"] = float(alive[last].sum())
            chron = np.asarray(panel["chronological_age"], dtype=float)
            stats["panel_mean_age_last"] = float(
                wmean(chron[last][mask], w_alive) if mask.any() else np.nan
            )

        stats["period_years"] = float(model.period_years)
        stats["T"] = float(model.par.T)

        if wanted is not None:
            return {k: stats[k] for k in wanted if k in stats}
        return stats
