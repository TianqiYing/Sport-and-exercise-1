"""
Lineup optimizer: find the best starting XI, bench order, captain and
vice-captain from my 15-player squad to maximise a matchup objective
against a fixed opponent decision.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .rules import (
    LineupDecision, Player, POS_MIN, POS_MAX,
    count_positions, is_valid_xi, validate_decision,
)
from .sampler import HistoricalEmpiricalSampler
from .simulate import MatchupStats, risk_objective, simulate_matchup_distribution


# ---------------------------------------------------------------------------
# Greedy initialisation
# ---------------------------------------------------------------------------

def greedy_lineup_initial(
    squad_map: Dict[int, Player],
    predicted_mean: Dict[int, float],
) -> LineupDecision:
    """
    Build an initial XI + bench ordered by predicted mean points.

    Guarantees a valid formation.  Captain = highest-mean XI player;
    vice = second-highest.
    """
    ranked = sorted(
        squad_map.keys(),
        key=lambda pid: predicted_mean.get(pid, 0.0),
        reverse=True,
    )

    # Always put exactly one GK in the XI
    gks = [pid for pid in ranked if squad_map[pid].position == "GK"]
    outfield = [pid for pid in ranked if squad_map[pid].position != "GK"]

    xi: List[int] = []
    bench_pool: List[int] = []

    if gks:
        xi.append(gks[0])
        bench_pool.extend(gks[1:])

    for pid in outfield:
        if len(xi) < 11:
            xi.append(pid)
        else:
            bench_pool.append(pid)

    xi, bench_pool = _repair_formation(xi, bench_pool, squad_map, predicted_mean)

    # Order bench: 3 best outfield (sub priority) + backup GK last
    bench_gks = [p for p in bench_pool if squad_map[p].position == "GK"]
    bench_out = sorted(
        [p for p in bench_pool if squad_map[p].position != "GK"],
        key=lambda p: predicted_mean.get(p, 0.0),
        reverse=True,
    )
    bench_gk = bench_gks[0] if bench_gks else bench_out[-1]
    bench = bench_out[:3] + [bench_gk]

    xi_by_mean = sorted(xi, key=lambda p: predicted_mean.get(p, 0.0), reverse=True)
    cap = xi_by_mean[0]
    vice = xi_by_mean[1] if len(xi_by_mean) > 1 else xi_by_mean[0]

    return LineupDecision(xi=xi, bench=bench, captain=cap, vice_captain=vice)


def _repair_formation(
    xi: List[int],
    bench_pool: List[int],
    squad_map: Dict[int, Player],
    predicted_mean: Dict[int, float],
    max_iters: int = 60,
) -> Tuple[List[int], List[int]]:
    """
    Iteratively swap players between XI and bench until the XI is valid.

    Handles the case where no position exceeds its maximum but one is below
    its minimum (e.g. DEF=2, MID=5, FWD=3): borrows from any position whose
    count exceeds its minimum requirement.
    """
    for _ in range(max_iters):
        if is_valid_xi(xi, squad_map):
            break
        c = count_positions(xi, squad_map)

        # Which position needs more players?
        need_pos: Optional[str] = None
        if c["GK"] < POS_MIN["GK"]:
            need_pos = "GK"
        elif c["DEF"] < POS_MIN["DEF"]:
            need_pos = "DEF"
        elif c["MID"] < POS_MIN["MID"]:
            need_pos = "MID"
        elif c["FWD"] < POS_MIN["FWD"]:
            need_pos = "FWD"

        if need_pos is None:
            break  # all positions meet minimums; shouldn't happen if is_valid_xi failed

        # Which position has too many (above its max)?
        over_pos: Optional[str] = None
        if c["GK"] > POS_MAX["GK"]:
            over_pos = "GK"
        elif c["DEF"] > POS_MAX["DEF"]:
            over_pos = "DEF"
        elif c["MID"] > POS_MAX["MID"]:
            over_pos = "MID"
        elif c["FWD"] > POS_MAX["FWD"]:
            over_pos = "FWD"

        if over_pos is None:
            # No position exceeds its max, but need_pos is below its min.
            # Donate from any position that has more than its minimum.
            donors = [
                pos for pos in ("GK", "DEF", "MID", "FWD")
                if pos != need_pos and c.get(pos, 0) > POS_MIN[pos]
            ]
            if not donors:
                break  # truly stuck; squad composition cannot satisfy constraints
            # Prefer donating from the most-represented position
            over_pos = max(donors, key=lambda pos: c.get(pos, 0))

        xi_over = [p for p in xi if squad_map[p].position == over_pos]
        bench_need = [p for p in bench_pool if squad_map[p].position == need_pos]
        if not xi_over or not bench_need:
            break

        victim = min(xi_over, key=lambda p: predicted_mean.get(p, 0.0))
        pick = max(bench_need, key=lambda p: predicted_mean.get(p, 0.0))
        xi.remove(victim)
        bench_pool.remove(pick)
        xi.append(pick)
        bench_pool.append(victim)

    return xi, bench_pool


# ---------------------------------------------------------------------------
# Bench ordering helper
# ---------------------------------------------------------------------------

def order_bench(
    bench_pool: List[int],
    squad_map: Dict[int, Player],
    predicted_mean: Dict[int, float],
) -> List[int]:
    """
    Given 4 bench players, return them ordered as
    [best_outfield, 2nd_outfield, 3rd_outfield, backup_gk].
    """
    bench_gks = [p for p in bench_pool if squad_map[p].position == "GK"]
    bench_out = sorted(
        [p for p in bench_pool if squad_map[p].position != "GK"],
        key=lambda p: predicted_mean.get(p, 0.0),
        reverse=True,
    )
    bench_gk = bench_gks[0] if bench_gks else bench_out[-1]
    return bench_out[:3] + [bench_gk]


# ---------------------------------------------------------------------------
# Main optimiser
# ---------------------------------------------------------------------------

def optimize_lineup_vs_opponent(
    my_squad_map: Dict[int, Player],
    opp_decision: LineupDecision,
    opp_squad_map: Dict[int, Player],
    sampler: HistoricalEmpiricalSampler,
    gw: int,
    risk: str = "balanced",
    n_sims_search: int = 1500,
    n_sims_final: int = 10_000,
    max_iter: int = 5,
    seed: int = 42,
) -> Tuple[LineupDecision, MatchupStats]:
    """
    Find the best lineup for my squad vs a fixed opponent decision.

    Algorithm
    ---------
    1. Greedy initialisation (rank by ``predicted_mean``).
    2. Exhaustive single-swap local search:
       - Try every valid XI ↔ bench swap.
       - Try all captain / vice-captain pairs among the top-6 XI players
         by predicted mean.
    3. Repeat until no improvement or *max_iter* outer rounds.
    4. Final evaluation with *n_sims_final* simulations.

    Parameters
    ----------
    my_squad_map : Dict[int, Player]
        My 15-player squad.
    opp_decision : LineupDecision
        Opponent's fixed lineup (XI, bench, captain, vice-captain).
    opp_squad_map : Dict[int, Player]
        Opponent's 15-player squad.
    sampler : HistoricalEmpiricalSampler
        Pre-built from history before *gw*.
    gw : int
        Target gameweek.
    risk : {"conservative", "balanced", "optimistic"}
        Optimisation objective.
    n_sims_search : int
        Simulations per candidate evaluation during search.
    n_sims_final : int
        Simulations for final evaluation.
    max_iter : int
        Maximum outer search iterations.
    seed : int
        Base RNG seed.

    Returns
    -------
    best_decision : LineupDecision
    final_stats : MatchupStats
    """
    predicted_mean = {pid: sampler.predicted_mean(pid) for pid in my_squad_map}

    def evaluate(dec: LineupDecision, n: int, s: int) -> Tuple[tuple, MatchupStats]:
        stats = simulate_matchup_distribution(
            gw, dec, my_squad_map, opp_decision, opp_squad_map,
            sampler, n_sims=n, seed=s,
        )
        return risk_objective(stats, risk), stats

    best = greedy_lineup_initial(my_squad_map, predicted_mean)
    best_obj, _ = evaluate(best, n_sims_search, seed)

    for iteration in range(max_iter):
        improved = False
        iter_seed = seed + (iteration + 1) * 997  # distinct seed per round

        # ----- Try all valid single XI ↔ bench swaps -----
        for xi_pid in list(best.xi):
            # Snapshot xi/bench at the start of this xi_pid iteration so that
            # candidate construction is always based on a consistent state even
            # if best is updated by an earlier swap within this loop.
            if xi_pid not in best.xi:
                continue
            snap_xi = list(best.xi)
            snap_bench = list(best.bench)

            xi_is_gk = my_squad_map[xi_pid].position == "GK"
            for bench_idx, bench_pid in enumerate(snap_bench):
                bench_is_gk = my_squad_map[bench_pid].position == "GK"
                bench_slot_gk = (bench_idx == 3)

                # Position-group consistency: GK↔GK, outfield↔outfield
                if xi_is_gk != bench_slot_gk:
                    continue
                if bench_is_gk and not bench_slot_gk:
                    continue
                if not bench_is_gk and bench_slot_gk:
                    continue

                # Build candidate from the snapshot (guarantees no duplicates)
                new_xi = [bench_pid if p == xi_pid else p for p in snap_xi]
                if not is_valid_xi(new_xi, my_squad_map):
                    continue

                new_bench = list(snap_bench)
                new_bench[bench_idx] = xi_pid

                # Fix captain/vice if they left the XI
                cap = best.captain if best.captain in new_xi else None
                vice = best.vice_captain if best.vice_captain in new_xi else None
                xi_by_mean = sorted(new_xi, key=lambda p: predicted_mean[p], reverse=True)
                if cap is None:
                    cap = xi_by_mean[0]
                if vice is None or vice == cap:
                    vice_candidates = [p for p in xi_by_mean if p != cap]
                    vice = vice_candidates[0] if vice_candidates else cap

                cand = LineupDecision(xi=new_xi, bench=new_bench, captain=cap, vice_captain=vice)
                obj, _ = evaluate(cand, n_sims_search, iter_seed)
                if obj > best_obj:
                    best, best_obj = cand, obj
                    improved = True

        # ----- Try all captain/vice pairs among top-6 XI players -----
        xi_by_mean = sorted(best.xi, key=lambda p: predicted_mean[p], reverse=True)
        cap_pool = xi_by_mean[:6]
        for cap in cap_pool:
            for vice in cap_pool:
                if vice == cap:
                    continue
                if cap == best.captain and vice == best.vice_captain:
                    continue  # already current best
                cand = best.copy()
                cand.captain = cap
                cand.vice_captain = vice
                obj, _ = evaluate(cand, n_sims_search, iter_seed)
                if obj > best_obj:
                    best, best_obj = cand, obj
                    improved = True

        if not improved:
            break

    # Final high-fidelity evaluation
    _, final_stats = evaluate(best, n_sims_final, seed)
    # Safety guard: the returned decision must always be valid
    validate_decision(best, my_squad_map)
    return best, final_stats
