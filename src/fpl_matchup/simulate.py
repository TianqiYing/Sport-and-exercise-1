"""
Joint Monte Carlo matchup simulation.

Key property: every unique player across both squads is sampled exactly
once per simulation.  This means shared players cancel correctly in the
margin distribution and reduces variance in the way real H2H FPL works.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .rules import LineupDecision, Player, apply_autosub, resolve_captain
from .sampler import HistoricalEmpiricalSampler


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class MatchupStats:
    """Summary statistics from a joint matchup simulation."""

    p_win: float
    p_draw: float
    exp_margin: float
    std_margin: float
    q10: float
    q50: float
    q90: float
    # Raw arrays (available for further analysis)
    margins: np.ndarray
    my_scores: np.ndarray
    opp_scores: np.ndarray

    def summary(self) -> str:
        return (
            f"P(win)={self.p_win:.3f}  P(draw)={self.p_draw:.3f}  "
            f"E[margin]={self.exp_margin:+.1f}  "
            f"q10/q50/q90={self.q10:+.1f}/{self.q50:+.1f}/{self.q90:+.1f}  "
            f"std={self.std_margin:.1f}"
        )


def risk_objective(stats: MatchupStats, risk: str) -> tuple:
    """
    Return a comparable tuple for a given risk profile.

    Tuples are compared lexicographically – larger is better.

    - ``conservative``: (q10, p_win)
    - ``balanced``:     (p_win, exp_margin)
    - ``optimistic``:   (q90, exp_margin)
    """
    if risk == "conservative":
        return (stats.q10, stats.p_win)
    if risk == "balanced":
        return (stats.p_win, stats.exp_margin)
    if risk == "optimistic":
        return (stats.q90, stats.exp_margin)
    raise ValueError(f"Unknown risk profile: {risk!r}.  Use conservative|balanced|optimistic.")


# ---------------------------------------------------------------------------
# Scoring helper (one simulation path)
# ---------------------------------------------------------------------------

def _score_lineup(
    decision: LineupDecision,
    squad_map: Dict[int, Player],
    minutes: Dict[int, int],
    points: Dict[int, float],
) -> float:
    """Score a lineup given a realised minutes/points draw."""
    effective_xi = apply_autosub(decision, minutes, squad_map)
    base = sum(
        points.get(pid, 0.0)
        for pid in effective_xi
        if minutes.get(pid, 0) > 0
    )
    cap = resolve_captain(decision, effective_xi, minutes)
    if cap is not None:
        base += points.get(cap, 0.0)
    return base


# ---------------------------------------------------------------------------
# Main simulation entry point
# ---------------------------------------------------------------------------

def simulate_matchup_distribution(
    gw: int,
    my_decision: LineupDecision,
    my_squad_map: Dict[int, Player],
    opp_decision: LineupDecision,
    opp_squad_map: Dict[int, Player],
    sampler: HistoricalEmpiricalSampler,
    n_sims: int = 10_000,
    seed: int = 0,
) -> MatchupStats:
    """
    Joint Monte Carlo matchup simulation.

    Each unique player across both squads is sampled **exactly once** per
    simulation, ensuring that shared players cancel correctly in the margin
    distribution.

    Parameters
    ----------
    gw : int
        Target gameweek (informational; sampler is already restricted to
        history < gw).
    my_decision : LineupDecision
        My starting XI, bench order, captain and vice-captain.
    my_squad_map : Dict[int, Player]
        My 15-player squad as a pid → Player mapping.
    opp_decision : LineupDecision
        Opponent's fixed lineup decision.
    opp_squad_map : Dict[int, Player]
        Opponent's 15-player squad.
    sampler : HistoricalEmpiricalSampler
        Pre-built sampler trained on history before *gw*.
    n_sims : int
        Number of Monte Carlo samples.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    MatchupStats
        Summary stats + raw margin/score arrays.
    """
    # Collect all unique player IDs across both squads
    all_pids: List[int] = list(set(
        my_decision.xi + my_decision.bench
        + opp_decision.xi + opp_decision.bench
    ))
    pid_to_idx = {pid: i for i, pid in enumerate(all_pids)}

    # Reset sampler RNG so results are reproducible for a given seed
    sampler.rng = np.random.default_rng(seed)

    # Vectorised batch sampling: shape (n_players, n_sims)
    mins_matrix, pts_matrix = sampler.batch_sample(all_pids, n_sims)

    # Score each simulation
    my_scores = np.empty(n_sims, dtype=float)
    opp_scores = np.empty(n_sims, dtype=float)

    for sim in range(n_sims):
        minutes = {pid: int(mins_matrix[pid_to_idx[pid], sim]) for pid in all_pids}
        points = {pid: float(pts_matrix[pid_to_idx[pid], sim]) for pid in all_pids}

        my_scores[sim] = _score_lineup(my_decision, my_squad_map, minutes, points)
        opp_scores[sim] = _score_lineup(opp_decision, opp_squad_map, minutes, points)

    margins = my_scores - opp_scores

    return MatchupStats(
        p_win=float(np.mean(margins > 0)),
        p_draw=float(np.mean(margins == 0)),
        exp_margin=float(margins.mean()),
        std_margin=float(margins.std(ddof=0)),
        q10=float(np.quantile(margins, 0.10)),
        q50=float(np.quantile(margins, 0.50)),
        q90=float(np.quantile(margins, 0.90)),
        margins=margins,
        my_scores=my_scores,
        opp_scores=opp_scores,
    )
