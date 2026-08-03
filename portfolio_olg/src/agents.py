"""Build HARK portfolio / risky-asset consumer types."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from HARK.ConsumptionSaving.ConsPortfolioModel import PortfolioConsumerType

try:
    from HARK.ConsumptionSaving.ConsPortfolioModel import RiskyAssetConsumerType
except ImportError:  # HARK 0.14.x
    try:
        from HARK.ConsumptionSaving.ConsRiskyAssetModel import (
            IndShockRiskyAssetConsumerType as RiskyAssetConsumerType,
        )
    except ImportError:  # pragma: no cover
        RiskyAssetConsumerType = PortfolioConsumerType


def _lifecycle_age_lists(
    t_cycle: int,
    t_retire: int,
    liv_prb_work: float = 0.99,
    liv_prb_retire: float = 0.95,
    perm_gro_work: float = 1.02,
    perm_gro_retire: float = 1.0,
    rfree: float = 1.02,
) -> Dict[str, Any]:
    """Age-varying survival, growth, and safe return lists of length t_cycle."""
    liv = []
    gro = []
    rfree_list = []
    for t in range(t_cycle):
        if t < t_retire:
            liv.append(liv_prb_work)
            gro.append(perm_gro_work)
        else:
            liv.append(liv_prb_retire)
            gro.append(perm_gro_retire)
        rfree_list.append(rfree)
    return {
        "LivPrb": liv,
        "PermGroFac": gro,
        "Rfree": rfree_list,
        "T_cycle": t_cycle,
        "T_retire": t_retire,
        "cycles": 1,
    }


def build_infinite_horizon_portfolio(
    agent_count: int = 10_000,
    seed: int = 0,
    sim_common_rrisky: bool = False,
    extra: Optional[Dict[str, Any]] = None,
) -> PortfolioConsumerType:
    """Infinite-horizon portfolio choice (cycles=0)."""
    params: Dict[str, Any] = {
        "cycles": 0,
        "T_cycle": 1,
        "AgentCount": agent_count,
        "T_sim": 100,
        "seed": seed,
        "CRRA": 5.0,
        "DiscFac": 0.96,
        "Rfree": [1.02],
        "RiskyAvg": 1.08,
        "RiskyStd": 0.20,
        "RiskyCount": 5,
        "LivPrb": [0.98],
        "PermGroFac": [1.01],
        "sim_common_Rrisky": sim_common_rrisky,
        "aNrmInitMean": np.log(0.5),
        "aNrmInitStd": 0.5,
        "pLvlInitMean": 0.0,
        "pLvlInitStd": 0.0,
    }
    if extra:
        params.update(extra)
    agent = PortfolioConsumerType(**params)
    agent.track_vars = ["aNrm", "mNrm", "cNrm", "pLvl", "Share"]
    return agent


def build_lifecycle_portfolio(
    t_cycle: int = 40,
    t_retire: int = 30,
    agent_count: int = 5_000,
    t_sim: Optional[int] = None,
    seed: int = 0,
    sim_common_rrisky: bool = False,
    risky_avg: float = 1.08,
    risky_std: float = 0.20,
    extra: Optional[Dict[str, Any]] = None,
) -> PortfolioConsumerType:
    """
    Finite life-cycle portfolio agent (one cycle through ages).

    Overlapping-generations panels are built by simulating a large population
    with staggered ages / re-entry (see ``run_olg_inequality``).
    """
    age = _lifecycle_age_lists(t_cycle=t_cycle, t_retire=t_retire)
    if t_sim is None:
        t_sim = t_cycle + 20

    params: Dict[str, Any] = {
        **age,
        "AgentCount": agent_count,
        "T_sim": t_sim,
        "T_age": t_cycle + 1,
        "seed": seed,
        "CRRA": 4.0,
        "DiscFac": 0.96,
        "RiskyAvg": risky_avg,
        "RiskyStd": risky_std,
        "RiskyCount": 5,
        "sim_common_Rrisky": sim_common_rrisky,
        "aNrmInitMean": np.log(0.2),
        "aNrmInitStd": 0.4,
        "pLvlInitMean": 0.0,
        "pLvlInitStd": 0.2,
        # Keep unemployment params scalar (HARK compares to shock draws)
        "UnempPrb": 0.05,
        "IncUnemp": 0.3,
        "UnempPrbRet": 0.0,
        "IncUnempRet": 0.0,
        "PermShkStd": [0.1] * t_retire + [0.0] * (t_cycle - t_retire),
        "TranShkStd": [0.1] * t_retire + [0.0] * (t_cycle - t_retire),
    }

    if extra:
        params.update(extra)

    agent = PortfolioConsumerType(**params)
    agent.track_vars = ["aNrm", "mNrm", "cNrm", "pLvl", "Share", "who_dies"]
    return agent


def solve_and_simulate(agent: PortfolioConsumerType) -> PortfolioConsumerType:
    """Solve Bellman problem then simulate tracked history."""
    agent.solve()
    agent.initialize_sim()
    agent.simulate()
    return agent


def wealth_levels(agent: PortfolioConsumerType) -> np.ndarray:
    """Absolute wealth aLvl = aNrm * pLvl from simulation history."""
    hist = agent.history
    return hist["aNrm"] * hist["pLvl"]
