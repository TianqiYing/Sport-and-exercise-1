"""
Transfer planning: find the best single-GW transfers and build
anti-opponent squads from scratch.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .data import build_squad_map
from .optimize import greedy_lineup_initial, optimize_lineup_vs_opponent, order_bench
from .rules import LineupDecision, Player, POS_MIN
from .sampler import HistoricalEmpiricalSampler
from .simulate import MatchupStats, risk_objective, simulate_matchup_distribution


# ---------------------------------------------------------------------------
# Transfer result
# ---------------------------------------------------------------------------

@dataclass
class TransferResult:
    """One committed transfer and its downstream lineup/stats."""
    transfer_out: int          # element ID transferred out
    transfer_in: int           # element ID transferred in
    new_squad_ids: List[int]   # 15-player squad after this transfer
    lineup: LineupDecision     # optimised lineup for the new squad
    stats: MatchupStats        # matchup stats after full lineup optimisation


# ---------------------------------------------------------------------------
# Transfer optimisation
# ---------------------------------------------------------------------------

def optimize_transfers(
    my_squad_map: Dict[int, Player],
    opp_decision: LineupDecision,
    opp_squad_map: Dict[int, Player],
    sampler: HistoricalEmpiricalSampler,
    gw: int,
    free_transfers: int = 1,
    universe: Optional[List[Player]] = None,
    costs: Optional[Dict[int, int]] = None,
    risk: str = "balanced",
    n_out_candidates: int = 6,
    n_in_candidates: int = 25,
    n_sims_inner: int = 300,
    n_sims_final: int = 5000,
    seed: int = 42,
) -> List[TransferResult]:
    """
    Greedy sequential transfer search: find up to *free_transfers* transfers
    that maximise the risk objective vs the fixed opponent.

    Algorithm
    ---------
    For each transfer round:
    1. Consider the bottom *n_out_candidates* squad players (by predicted_mean)
       as candidates for transfer out.
    2. For each out-candidate position, gather the top *n_in_candidates*
       universe players (by predicted_mean) not already in the squad.
    3. For each (out, in) pair: build the new squad, compute a quick
       greedy lineup, simulate with *n_sims_inner* sims.
    4. Commit the best pair if it improves on the current baseline.
    5. Run full ``optimize_lineup_vs_opponent`` on the committed squad for
       the final result.

    Parameters
    ----------
    my_squad_map : Dict[int, Player]
        My current 15-player squad.
    opp_decision : LineupDecision
        Opponent's fixed starting decision.
    opp_squad_map : Dict[int, Player]
        Opponent's 15-player squad.
    sampler : HistoricalEmpiricalSampler
        Pre-built sampler (covers all players including universe candidates).
    gw : int
        Target gameweek.
    free_transfers : int
        Maximum number of transfers to make.
    universe : list of Player, optional
        Players available for transfer in.  Defaults to all players known to
        the sampler (from its ``player_positions`` dict).
    costs : Dict[int, int], optional
        Element → ``now_cost`` in tenths of millions.  If provided, transfers
        must be budget-neutral (cost_in ≤ cost_out + any saved budget).
    risk : str
        Risk profile for the objective.
    n_out_candidates : int
        How many of my worst-mean players to consider for transfer out.
    n_in_candidates : int
        How many universe players per position to consider for transfer in.
    n_sims_inner : int
        Fast simulation count for transfer candidate evaluation.
    n_sims_final : int
        Simulation count for the final lineup optimisation.
    seed : int
        Base RNG seed.

    Returns
    -------
    List[TransferResult]
        One entry per committed transfer (may be fewer than *free_transfers*
        if no further improvement is found).
    """
    if free_transfers <= 0:
        return []

    # Build universe list if not provided
    if universe is None:
        universe = [
            Player(
                player_id=pid,
                name=sampler.player_positions.get(pid, ""),  # best we can do
                position=sampler.player_positions.get(pid, "MID"),
                team="",
            )
            for pid in sampler.player_positions
        ]

    current_squad_ids = list(my_squad_map.keys())
    current_squad_map = dict(my_squad_map)  # mutable copy
    results: List[TransferResult] = []

    # Build predicted_mean for all universe players
    predicted_mean_all = {
        p.player_id: sampler.predicted_mean(p.player_id) for p in universe
    }

    def _quick_objective(squad_map: Dict[int, Player], s: int) -> tuple:
        """Fast greedy lineup + quick simulation for transfer search."""
        pred = {pid: sampler.predicted_mean(pid) for pid in squad_map}
        dec = greedy_lineup_initial(squad_map, pred)
        stats = simulate_matchup_distribution(
            gw, dec, squad_map, opp_decision, opp_squad_map,
            sampler, n_sims=n_sims_inner, seed=s,
        )
        return risk_objective(stats, risk)

    # Baseline objective (no transfers)
    current_obj = _quick_objective(current_squad_map, seed)

    for round_idx in range(free_transfers):
        pred_my = {pid: sampler.predicted_mean(pid) for pid in current_squad_ids}
        round_seed = seed + (round_idx + 1) * 9973

        # Bottom n_out_candidates from my squad
        out_candidates = sorted(
            current_squad_ids,
            key=lambda p: pred_my[p],
        )[:n_out_candidates]

        # Team counts in current squad
        team_counts = Counter(current_squad_map[p].team for p in current_squad_ids)

        best_improvement: Optional[Tuple[int, int, tuple]] = None  # (out, in, obj)

        for out_pid in out_candidates:
            out_player = current_squad_map[out_pid]
            out_pos = out_player.position
            out_cost = costs.get(out_pid, 0) if costs else 0
            budget_available = out_cost  # budget-neutral by default

            # Team counts without the outgoing player
            adjusted_counts = Counter(team_counts)
            adjusted_counts[out_player.team] -= 1

            # Top n_in_candidates universe players of same position
            in_pool = [
                p for p in universe
                if p.position == out_pos
                and p.player_id not in current_squad_ids
                and adjusted_counts.get(p.team, 0) < 3
                and (
                    costs is None
                    or costs.get(p.player_id, budget_available + 1) <= budget_available
                )
            ]
            in_pool.sort(key=lambda p: predicted_mean_all.get(p.player_id, 0.0), reverse=True)
            in_pool = in_pool[:n_in_candidates]

            for idx, in_player in enumerate(in_pool):
                trial_ids = [p for p in current_squad_ids if p != out_pid] + [in_player.player_id]
                trial_map = {**{p: current_squad_map[p] for p in trial_ids if p != in_player.player_id},
                             in_player.player_id: in_player}

                cand_seed = round_seed + idx * 31 + hash(out_pid) % 997
                obj = _quick_objective(trial_map, cand_seed)
                if best_improvement is None or obj > best_improvement[2]:
                    best_improvement = (out_pid, in_player.player_id, obj, trial_ids, trial_map)

        if best_improvement is None or best_improvement[2] <= current_obj:
            break  # no improvement found this round

        out_id, in_id, new_obj, new_ids, new_map = best_improvement

        # Full lineup optimisation on the winning transfer
        lineup, stats = optimize_lineup_vs_opponent(
            my_squad_map=new_map,
            opp_decision=opp_decision,
            opp_squad_map=opp_squad_map,
            sampler=sampler,
            gw=gw,
            risk=risk,
            n_sims_search=max(500, n_sims_inner),
            n_sims_final=n_sims_final,
            seed=round_seed,
        )

        results.append(TransferResult(
            transfer_out=out_id,
            transfer_in=in_id,
            new_squad_ids=list(new_ids),
            lineup=lineup,
            stats=stats,
        ))

        # Commit and continue
        current_squad_ids = list(new_ids)
        current_squad_map = new_map
        current_obj = risk_objective(stats, risk)

    return results


# ---------------------------------------------------------------------------
# Auto B: build a squad from scratch to beat the opponent
# ---------------------------------------------------------------------------

_SQUAD_QUOTAS: Dict[str, int] = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
_TOTAL_BUDGET = 1000  # tenths of millions (£100 m)


def build_squad_vs_opponent(
    universe: List[Player],
    sampler: HistoricalEmpiricalSampler,
    opp_decision: LineupDecision,
    opp_squad_map: Dict[int, Player],
    gw: int,
    risk: str = "balanced",
    costs: Optional[Dict[int, int]] = None,
    budget: int = _TOTAL_BUDGET,
    n_sims_final: int = 5000,
    seed: int = 42,
) -> Tuple[List[int], LineupDecision, MatchupStats]:
    """
    Greedily build a 15-player squad targeting the opponent, then optimise.

    Constraints
    -----------
    - Position quotas: 2 GK, 5 DEF, 5 MID, 3 FWD.
    - Max 3 players from the same real team.
    - Budget: total ``now_cost`` ≤ *budget* (only enforced if *costs* provided).

    Algorithm
    ---------
    Fill positions in *two passes* so that high-value outfield picks never
    exhaust the budget before GKs are allocated:

    1. **Reserve pass**: for each position, greedily pick the best player that
       fits within (remaining_budget − min_cost_reserved_for_other_positions).
    2. Because GKs are the hardest constraint (fewest candidates, smallest
       budgets) and are typically lower predicted_mean, they are filled first
       before outfield quota is exhausted.

    Parameters
    ----------
    universe : list of Player
        All available players.
    sampler : HistoricalEmpiricalSampler
        Pre-built sampler (used for predicted_mean).
    opp_decision, opp_squad_map, gw, risk, seed : as in optimize_lineup_vs_opponent.
    costs : Dict[int, int], optional
        Element → ``now_cost``.  If None, budget constraint is ignored.
    budget : int
        Maximum total squad cost in tenths of millions (default 1000 = £100 m).
    n_sims_final : int
        Simulations for the final lineup evaluation.

    Returns
    -------
    squad_ids : List[int]
    best_lineup : LineupDecision
    final_stats : MatchupStats
    """
    predicted_mean = {p.player_id: sampler.predicted_mean(p.player_id) for p in universe}

    # ------------------------------------------------------------------ #
    # Position-sequential greedy fill with static budget reservation.     #
    #                                                                      #
    # Fill positions sequentially in this order: GK → DEF → FWD → MID.  #
    # GK and FWD are filled first because they have fewer candidates and  #
    # would otherwise lose budget to high-cost midfielders.               #
    #                                                                      #
    # Budget reservation: before picking a player at position P, reserve  #
    # min_cost[Q] * remaining_slots[Q] for each *other* position Q.  The  #
    # min costs are pre-computed once from the full universe (no team      #
    # constraint applied — pessimistic but stable).                        #
    # ------------------------------------------------------------------ #

    # Pre-compute minimum cost per position across the whole universe.
    # Use 0 if costs not provided (no budget constraint).
    if costs is not None:
        min_cost_pos: Dict[str, int] = {}
        for pos in _SQUAD_QUOTAS:
            pos_costs = [costs.get(p.player_id, 0) for p in universe if p.position == pos]
            pos_costs = [c for c in pos_costs if c > 0]
            min_cost_pos[pos] = min(pos_costs) if pos_costs else 0
    else:
        min_cost_pos = {pos: 0 for pos in _SQUAD_QUOTAS}

    # Fill order: hardest-to-fill positions first
    fill_order = ["GK", "DEF", "FWD", "MID"]

    # Build per-position candidate lists sorted by predicted_mean desc
    pos_ranked: Dict[str, List[Player]] = {
        pos: sorted(
            [p for p in universe if p.position == pos],
            key=lambda p: predicted_mean.get(p.player_id, 0.0),
            reverse=True,
        )
        for pos in _SQUAD_QUOTAS
    }

    squad_ids: List[int] = []
    picked: set = set()
    team_counts: Counter = Counter()
    pos_counts: Dict[str, int] = {pos: 0 for pos in _SQUAD_QUOTAS}
    remaining_budget = budget

    for pos in fill_order:
        quota = _SQUAD_QUOTAS[pos]
        while pos_counts[pos] < quota:
            # Compute budget reservation for all still-unfilled slots of OTHER positions
            reserved = 0
            if costs is not None:
                for other_pos, other_quota in _SQUAD_QUOTAS.items():
                    if other_pos == pos:
                        continue
                    slots_left = other_quota - pos_counts[other_pos]
                    reserved += slots_left * min_cost_pos[other_pos]
                # Also reserve for remaining slots of this position (after this pick)
                slots_after = (quota - pos_counts[pos]) - 1
                reserved += slots_after * min_cost_pos[pos]

            picked_one = False
            for player in pos_ranked[pos]:
                pid = player.player_id
                if pid in picked:
                    continue
                if team_counts[player.team] >= 3:
                    continue
                if costs is not None:
                    cost = costs.get(pid, 0)
                    if cost + reserved > remaining_budget:
                        continue
                # Accept
                squad_ids.append(pid)
                picked.add(pid)
                pos_counts[pos] += 1
                team_counts[player.team] += 1
                if costs is not None:
                    remaining_budget -= costs.get(pid, 0)
                picked_one = True
                break

            if not picked_one:
                break  # cannot fill this position with remaining budget/team constraints

    # Check all quotas were met — report per-position detail on failure
    unmet = {
        pos: (pos_counts.get(pos, 0), quota)
        for pos, quota in _SQUAD_QUOTAS.items()
        if pos_counts.get(pos, 0) < quota
    }
    if unmet:
        details = ", ".join(
            f"{pos}: needed {quota} got {got}" for pos, (got, quota) in unmet.items()
        )
        avail = {pos: sum(1 for p in universe if p.position == pos) for pos in _SQUAD_QUOTAS}
        raise ValueError(
            f"Could not fill all squad positions ({details}). "
            f"Universe has {avail}. "
            "Check budget or max-3-per-team constraints."
        )

    # Build player info for squad_map using actual Player objects from universe
    pid_to_player = {p.player_id: p for p in universe}
    squad_map = {pid: pid_to_player[pid] for pid in squad_ids}

    lineup, stats = optimize_lineup_vs_opponent(
        my_squad_map=squad_map,
        opp_decision=opp_decision,
        opp_squad_map=opp_squad_map,
        sampler=sampler,
        gw=gw,
        risk=risk,
        n_sims_final=n_sims_final,
        seed=seed,
    )

    return squad_ids, lineup, stats
