from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
import random
import math
import numpy as np


# Core data structures

Position = str  # "GK", "DEF", "MID", "FWD"

@dataclass(frozen=True)
class Player:
    player_id: int
    name: str
    position: Position
    team: str
    price: float  # for transfer/budget constraints (optional)


@dataclass
class TeamState:
    """
    Represents a manager's 15-player squad and resources at a given gameweek deadline.
    """
    squad: List[Player]                # length 15
    bank: float                        # remaining budget (optional)
    free_transfers: int                # e.g., 1 or 2
    hits_taken: int                    # total hits this season (optional)
    chips_available: Dict[str, bool]   # e.g., {"WC": True, "FH": True, "TC": True}


@dataclass
class LineupDecision:
    xi: List[int]               # 11 player_ids
    bench: List[int]            # 4 player_ids, in order
    captain: int
    vice_captain: int


@dataclass
class GWOutcome:
    """
    One Monte Carlo sample outcome for a given GW and lineup decision.
    """
    gw_points: float
    effective_xi: List[int]     # final XI after autosub
    captain_effective: Optional[int]
    debug: Dict[str, object]


# Sampler interface

class PlayerSampler:

    def sample_minutes_points(self, player_id: int, gw: int) -> Tuple[int, float]:
        """
        Return one sample of (minutes, points) for the player at gameweek gw.
        points should already be FPL points for that GW.
        """
        raise NotImplementedError

    def sample_minutes(self, player_id: int, gw: int) -> int:
        raise NotImplementedError

    def sample_points_given_minutes(self, player_id: int, gw: int, minutes: int) -> float:
        raise NotImplementedError


# Formation rules

def count_positions(player_ids: List[int], squad_map: Dict[int, Player]) -> Dict[Position, int]:
    counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
    for pid in player_ids:
        pos = squad_map[pid].position
        counts[pos] += 1
    return counts


def is_valid_xi(player_ids: List[int], squad_map: Dict[int, Player]) -> bool:
    """
    Standard FPL XI constraints:
      - total 11
      - exactly 1 GK
      - DEF: 3-5
      - MID: 2-5
      - FWD: 1-3
    """
    if len(player_ids) != 11:
        return False
    c = count_positions(player_ids, squad_map)
    if c["GK"] != 1:
        return False
    if not (3 <= c["DEF"] <= 5):
        return False
    if not (2 <= c["MID"] <= 5):
        return False
    if not (1 <= c["FWD"] <= 3):
        return False
    # ensure counts sum to 11
    return sum(c.values()) == 11


# Auto-substitution logic

def apply_autosub(
    decision: LineupDecision,
    minutes: Dict[int, int],
    squad_map: Dict[int, Player],
) -> List[int]:
    """
    Apply standard-like autosub:
      - Bench players normally score 0
      - If any starter has minutes==0, attempt to replace using bench order
      - Goalkeeper: bench GK replaces starter GK if starter GK minutes==0
      - Outfield subs in order; replacement must keep a valid formation
    Returns effective XI (11 player_ids).
    """
    xi = list(decision.xi)

    # Identify bench GK vs outfield bench.
    # Convention: bench[-1] is GK.
    bench = list(decision.bench)
    bench_outfield = bench[:3]
    bench_gk = bench[3]

    # GK substitution
    # Find starter GK
    starter_gk = next((pid for pid in xi if squad_map[pid].position == "GK"), None)
    if starter_gk is None:
        # invalid input, but don't crash
        return xi

    if minutes.get(starter_gk, 0) == 0 and minutes.get(bench_gk, 0) > 0:
        # Replace GK
        xi.remove(starter_gk)
        xi.append(bench_gk)

    # Outfield substitutions
    # repeatedly try to replace non-playing outfield starters
    for sub in bench_outfield:
        if minutes.get(sub, 0) == 0:
            continue

        # find current non-playing outfield players in XI
        non_playing = [pid for pid in xi if minutes.get(pid, 0) == 0 and squad_map[pid].position != "GK"]
        if not non_playing:
            break

        # Try to replace one of them such that formation stays valid
        replaced = False
        for victim in non_playing:
            trial = [p for p in xi if p != victim] + [sub]
            if is_valid_xi(trial, squad_map):
                xi = trial
                replaced = True
                break

        # If sub can't be used without breaking formation, skip to next sub.
        # This matches FPL behavior: a sub only comes on if a legal formation can be maintained.
        if replaced:
            continue

    # Ensure 11 players
    if len(xi) != 11:
        # fallback: trim or pad not handled; keep as-is
        pass
    return xi


# Captain / vice-captain rule

def resolve_captain(
    decision: LineupDecision,
    effective_xi: List[int],
    minutes: Dict[int, int],
) -> Optional[int]:
    """
    Returns which player gets doubled points:
      - captain if played (>0 minutes)
      - else vice if played and is in effective XI
      - else None
    """
    cap = decision.captain
    vice = decision.vice_captain

    if cap in effective_xi and minutes.get(cap, 0) > 0:
        return cap
    if vice in effective_xi and minutes.get(vice, 0) > 0:
        return vice
    return None


# Score a single Monte Carlo sample

def simulate_one_gw_sample(
    gw: int,
    decision: LineupDecision,
    squad_map: Dict[int, Player],
    sampler: PlayerSampler,
) -> GWOutcome:
    """
    One simulation path for a GW:
      - sample (minutes, points) for all 15 players (or at least XI+bench)
      - apply autosub to get effective XI
      - sum points for effective XI that played
      - apply captain doubling
    """
    # sample minutes & points for all players involved (15)
    all_ids = set(decision.xi + decision.bench)
    minutes: Dict[int, int] = {}
    points: Dict[int, float] = {}

    for pid in all_ids:
        m, p = sampler.sample_minutes_points(pid, gw)
        minutes[pid] = int(m)
        points[pid] = float(p)

    effective_xi = apply_autosub(decision, minutes, squad_map)
    base = sum(points[pid] for pid in effective_xi if minutes.get(pid, 0) > 0)

    cap_effective = resolve_captain(decision, effective_xi, minutes)
    if cap_effective is not None:
        base += points[cap_effective]  # add one more time = doubling

    dbg = {
        "minutes": minutes,
        "points": points,
        "cap_effective": cap_effective,
    }
    return GWOutcome(gw_points=base, effective_xi=effective_xi, captain_effective=cap_effective, debug=dbg)


def simulate_gw_distribution(
    gw: int,
    decision: LineupDecision,
    squad: List[Player],
    sampler: PlayerSampler,
    n_sims: int = 10000,
    seed: int = 0,
) -> Dict[str, object]:
    """
    Monte Carlo distribution for GW team score.
    Returns summary stats + raw samples.
    """
    rng = random.Random(seed)
    random.seed(seed)
    np.random.seed(seed)

    squad_map = {p.player_id: p for p in squad}

    samples = np.empty(n_sims, dtype=float)
    for k in range(n_sims):
        out = simulate_one_gw_sample(gw, decision, squad_map, sampler)
        samples[k] = out.gw_points

    samples.sort()
    res = {
        "mean": float(samples.mean()),
        "std": float(samples.std(ddof=0)),
        "q10": float(np.quantile(samples, 0.10)),
        "q50": float(np.quantile(samples, 0.50)),
        "q90": float(np.quantile(samples, 0.90)),
        "samples": samples,
    }
    return res


# Matchup win probability

def win_probability(
    gw: int,
    decision_a: LineupDecision,
    squad_a: List[Player],
    decision_b: LineupDecision,
    squad_b: List[Player],
    sampler: PlayerSampler,
    n_sims: int = 10000,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Estimate P(scoreA > scoreB) and expected advantage in GW.
    Assumes sampler can sample for both squads (player_id space is global).
    """
    dist_a = simulate_gw_distribution(gw, decision_a, squad_a, sampler, n_sims=n_sims, seed=seed)
    dist_b = simulate_gw_distribution(gw, decision_b, squad_b, sampler, n_sims=n_sims, seed=seed + 1)

    a = dist_a["samples"]
    b = dist_b["samples"]
    # Align lengths if needed
    m = min(len(a), len(b))
    a = a[:m]
    b = b[:m]

    p_win = float(np.mean(a > b))
    exp_adv = float(np.mean(a - b))
    return {"p_win": p_win, "exp_adv": exp_adv}


# Simple lineup optimization skeleton

def greedy_lineup_initial(
    squad: List[Player],
    predicted_mean: Dict[int, float],
) -> LineupDecision:
    """
    Build a naive starting XI + bench order from predicted mean points.
    """
    squad_map = {p.player_id: p for p in squad}

    # Sort by predicted mean, descending
    ranked = sorted([p.player_id for p in squad], key=lambda pid: predicted_mean.get(pid, 0.0), reverse=True)

    # Start with top 11 then adjust to valid formation
    xi = ranked[:11]
    # Repair formation: try swaps from bench until valid
    bench_pool = ranked[11:]

    # Ensure exactly 1 GK in XI
    # If XI has 0 or 2 GK, swap accordingly.
    def gk_ids(ids): return [pid for pid in ids if squad_map[pid].position == "GK"]
    while True:
        c = count_positions(xi, squad_map)
        if c["GK"] == 1 and is_valid_xi(xi, squad_map):
            break
        # repair GK count first
        if c["GK"] == 0:
            # find best GK from bench_pool, swap with worst outfield in XI
            gk_candidates = [pid for pid in bench_pool if squad_map[pid].position == "GK"]
            if not gk_candidates:
                break
            best_gk = max(gk_candidates, key=lambda pid: predicted_mean.get(pid, 0.0))
            # remove worst outfield
            outfield = [pid for pid in xi if squad_map[pid].position != "GK"]
            worst = min(outfield, key=lambda pid: predicted_mean.get(pid, 0.0))
            xi.remove(worst)
            xi.append(best_gk)
            bench_pool.remove(best_gk)
            bench_pool.append(worst)
            continue
        if c["GK"] > 1:
            # move extra GK to bench, bring best outfield
            gks = gk_ids(xi)
            # keep best GK
            keep = max(gks, key=lambda pid: predicted_mean.get(pid, 0.0))
            drop = [pid for pid in gks if pid != keep][0]
            xi.remove(drop)
            # bring best outfield from bench_pool
            out_candidates = [pid for pid in bench_pool if squad_map[pid].position != "GK"]
            if not out_candidates:
                break
            best_out = max(out_candidates, key=lambda pid: predicted_mean.get(pid, 0.0))
            xi.append(best_out)
            bench_pool.remove(best_out)
            bench_pool.append(drop)
            continue

        # If GK ok but formation invalid, try swapping lowest xi player with best bench player of needed position
        # Simple approach: try a few random swaps
        repaired = False
        for _ in range(50):
            victim = min(xi, key=lambda pid: predicted_mean.get(pid, 0.0))
            cand = max(bench_pool, key=lambda pid: predicted_mean.get(pid, 0.0))
            trial = [p for p in xi if p != victim] + [cand]
            if is_valid_xi(trial, squad_map):
                xi = trial
                bench_pool.remove(cand)
                bench_pool.append(victim)
                repaired = True
                break
            else:
                # try another candidate
                bench_pool.remove(cand)
                bench_pool.append(cand)
        if not repaired:
            break

    # Bench: 3 outfield by mean, GK last
    bench_sorted = sorted(bench_pool, key=lambda pid: predicted_mean.get(pid, 0.0), reverse=True)
    bench_gk = [pid for pid in bench_sorted if squad_map[pid].position == "GK"]
    bench_out = [pid for pid in bench_sorted if squad_map[pid].position != "GK"]

    bench = bench_out[:3] + [bench_gk[0] if bench_gk else bench_out[3]]

    # Captain: best predicted in XI; vice: second best in XI
    xi_sorted = sorted(xi, key=lambda pid: predicted_mean.get(pid, 0.0), reverse=True)
    cap = xi_sorted[0]
    vice = xi_sorted[1] if len(xi_sorted) > 1 else xi_sorted[0]

    return LineupDecision(xi=xi, bench=bench, captain=cap, vice_captain=vice)


def local_search_lineup(
    gw: int,
    squad: List[Player],
    sampler: PlayerSampler,
    start: LineupDecision,
    n_iter: int = 50,
    n_sims: int = 3000,
    mode: str = "mean",  # "mean" or "q10" or "pwin" (needs opponent)
    opponent: Optional[Tuple[List[Player], LineupDecision]] = None,
) -> LineupDecision:
    squad_map = {p.player_id: p for p in squad}

    def objective(dec: LineupDecision) -> float:
        dist = simulate_gw_distribution(gw, dec, squad, sampler, n_sims=n_sims, seed=123)
        if mode == "mean":
            return dist["mean"]
        if mode == "q10":
            return dist["q10"]
        if mode == "pwin":
            assert opponent is not None
            opp_squad, opp_dec = opponent
            out = win_probability(gw, dec, squad, opp_dec, opp_squad, sampler, n_sims=n_sims, seed=123)
            return out["p_win"]
        raise ValueError("Unknown mode")

    best = start
    best_val = objective(best)

    for _ in range(n_iter):
        cand = LineupDecision(
            xi=list(best.xi),
            bench=list(best.bench),
            captain=best.captain,
            vice_captain=best.vice_captain,
        )

        # propose a random move
        move_type = random.choice(["swap_xi_bench", "swap_captain"])
        if move_type == "swap_xi_bench":
            # pick random xi player (not GK) and random bench outfield
            xi_out = [pid for pid in cand.xi if squad_map[pid].position != "GK"]
            bench_out = cand.bench[:3]
            if xi_out and bench_out:
                x = random.choice(xi_out)
                b = random.choice(bench_out)
                trial_xi = [p for p in cand.xi if p != x] + [b]
                if is_valid_xi(trial_xi, squad_map):
                    cand.xi = trial_xi
                    # bench swap
                    cand.bench = [x if pid == b else pid for pid in cand.bench]
        else:
            # change captain among top 5 predicted by mean using quick samples
            xi_sorted = list(cand.xi)
            random.shuffle(xi_sorted)
            cand.captain = xi_sorted[0]
            cand.vice_captain = xi_sorted[1] if len(xi_sorted) > 1 else xi_sorted[0]

        val = objective(cand)
        if val > best_val:
            best, best_val = cand, val

    return best


# Transfer optimization skeleton (rolling horizon)

@dataclass
class TransferAction:
    out_ids: List[int]
    in_players: List[Player]
    hit_cost: int  # points penalty (e.g., 4 per extra transfer)
    description: str = ""


def generate_transfer_candidates(
    state: TeamState,
    player_universe: List[Player],
    predicted_mean_nextk: Dict[Tuple[int, int], float],
    gw: int,
    max_in: int = 20,
) -> List[TransferAction]:
    """
    Generate a small list of plausible transfers (MVP heuristic):
    - consider transferring out lowest projected players
    - consider transferring in top projected players within budget/constraints
    This is only a skeleton; add price/team-limit/position constraints as needed.
    """
    squad_ids = {p.player_id for p in state.squad}
    # rank current squad by predicted mean (next GW only for simplicity)
    out_rank = sorted(state.squad, key=lambda p: predicted_mean_nextk.get((p.player_id, gw), 0.0))
    candidates_in = sorted(
        [p for p in player_universe if p.player_id not in squad_ids],
        key=lambda p: predicted_mean_nextk.get((p.player_id, gw), 0.0),
        reverse=True,
    )[:max_in]

    actions: List[TransferAction] = []
    # simplest: 0 transfer
    actions.append(TransferAction(out_ids=[], in_players=[], hit_cost=0, description="No transfer"))

    # 1-transfer candidates (transfer out 1 of bottom 5, in 1 of top N same position)
    for out_p in out_rank[:5]:
        for in_p in candidates_in:
            if in_p.position != out_p.position:
                continue
            # TODO: budget/team limits
            actions.append(TransferAction(out_ids=[out_p.player_id], in_players=[in_p], hit_cost=0, description=f"{out_p.name}->{in_p.name}"))

    # 2-transfer candidates can be added similarly (and hit cost if > free_transfers)
    return actions


def apply_transfer_action(state: TeamState, action: TransferAction) -> TeamState:
    new_squad = [p for p in state.squad if p.player_id not in set(action.out_ids)]
    new_squad.extend(action.in_players)
    # TODO: update bank/free_transfers/hits
    return TeamState(
        squad=new_squad,
        bank=state.bank,  # update later
        free_transfers=state.free_transfers,
        hits_taken=state.hits_taken,
        chips_available=dict(state.chips_available),
    )


def evaluate_plan_lookahead(
    state: TeamState,
    gw: int,
    horizon: int,
    sampler: PlayerSampler,
    objective_mode: str = "mean",  # "mean", "q10", "pwin"
    opponent_plan: Optional[Callable[[int], Tuple[List[Player], LineupDecision]]] = None,
    n_sims: int = 4000,
) -> float:
    """
    Evaluate a given state over K weeks by simulating best lineup each week (simplified).
    MVP: choose lineup by greedy mean each week; compute objective over horizon.
    """
    total_samples = []

    # MVP: treat weeks independent
    total_mean = 0.0
    total_q10 = 0.0
    pwin_vals = []

    for t in range(gw, gw + horizon):
        raise NotImplementedError("Connect your predicted_mean and lineup optimizer here")

    # return objective
    if objective_mode == "mean":
        return total_mean
    if objective_mode == "q10":
        return total_q10
    if objective_mode == "pwin":
        return float(np.mean(pwin_vals)) if pwin_vals else 0.0
    return total_mean


def optimize_transfers_and_lineup(
    state: TeamState,
    gw: int,
    player_universe: List[Player],
    sampler: PlayerSampler,
    predicted_mean_nextk: Dict[Tuple[int, int], float],
    horizon: int = 3,
    mode: str = "mean",
    opponent: Optional[Tuple[List[Player], LineupDecision]] = None,
) -> Tuple[TransferAction, LineupDecision]:
    """
    High-level skeleton:
      - generate transfer candidates
      - for each, apply, then choose best lineup (with local search)
      - evaluate lookahead objective
      - return best transfer + GW lineup decision
    """
    transfer_candidates = generate_transfer_candidates(
        state, player_universe, predicted_mean_nextk, gw, max_in=20
    )

    best_action = transfer_candidates[0]
    best_decision = None
    best_val = -1e18

    for action in transfer_candidates:
        st2 = apply_transfer_action(state, action)

        # Build a starting lineup using predicted_mean for current GW
        predicted_mean_gw = {p.player_id: predicted_mean_nextk.get((p.player_id, gw), 0.0) for p in st2.squad}
        start_dec = greedy_lineup_initial(st2.squad, predicted_mean_gw)

        # Refine lineup with local search using distribution simulator
        opp_tuple = opponent if mode == "pwin" else None
        dec = local_search_lineup(
            gw, st2.squad, sampler, start_dec,
            n_iter=30, n_sims=2500, mode=("pwin" if mode == "pwin" else ("q10" if mode=="q10" else "mean")),
            opponent=opp_tuple
        )

        # Evaluate objective for this GW only (MVP).
        if mode == "pwin" and opponent is not None:
            val = win_probability(gw, dec, st2.squad, opponent[1], opponent[0], sampler, n_sims=2500)["p_win"]
        else:
            dist = simulate_gw_distribution(gw, dec, st2.squad, sampler, n_sims=2500)
            val = dist["mean"] if mode == "mean" else dist["q10"]

        # subtract hit cost (points penalty)
        val -= action.hit_cost

        if val > best_val:
            best_val = val
            best_action = action
            best_decision = dec

    assert best_decision is not None
    return best_action, best_decision
