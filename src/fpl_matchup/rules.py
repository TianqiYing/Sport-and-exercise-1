"""
FPL game rules: types, formation validation, autosub, captain resolution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Position strings used throughout
Position = str  # "GK" | "DEF" | "MID" | "FWD"

POSITION_NORM: Dict[object, str] = {
    # String variants
    "GK": "GK", "GKP": "GK",
    "DEF": "DEF", "D": "DEF",
    "MID": "MID", "M": "MID",
    "FWD": "FWD", "F": "FWD", "ATT": "FWD",
    # Integer FPL API codes
    1: "GK", 2: "DEF", 3: "MID", 4: "FWD",
    "1": "GK", "2": "DEF", "3": "MID", "4": "FWD",
}


def normalise_position(raw: object) -> str:
    """Normalise any FPL position encoding to one of GK/DEF/MID/FWD."""
    if isinstance(raw, float):
        raw = int(raw)
    return POSITION_NORM.get(raw, "MID")


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Player:
    player_id: int
    name: str
    position: Position  # GK | DEF | MID | FWD
    team: str


@dataclass
class LineupDecision:
    """
    A manager's lineup choice.

    bench order: [1st_sub, 2nd_sub, 3rd_sub, bench_gk]
    bench[0..2] must be outfield players; bench[3] must be the backup GK.
    """
    xi: List[int]        # 11 player_ids
    bench: List[int]     # 4 player_ids
    captain: int
    vice_captain: int

    def copy(self) -> "LineupDecision":
        return LineupDecision(
            xi=list(self.xi),
            bench=list(self.bench),
            captain=self.captain,
            vice_captain=self.vice_captain,
        )


class PlayerSampler:
    """Abstract interface – subclasses must implement sample_minutes_points."""

    def sample_minutes_points(self, player_id: int, gw: int) -> Tuple[int, float]:
        """Return one sample of (minutes, points) for player at gameweek gw."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Formation rules
# ---------------------------------------------------------------------------

def count_positions(
    player_ids: List[int], squad_map: Dict[int, Player]
) -> Dict[str, int]:
    counts: Dict[str, int] = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
    for pid in player_ids:
        pos = squad_map[pid].position
        counts[pos] = counts.get(pos, 0) + 1
    return counts


POS_MIN: Dict[str, int] = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
POS_MAX: Dict[str, int] = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}


def is_valid_xi(player_ids: List[int], squad_map: Dict[int, Player]) -> bool:
    """
    Return True iff player_ids form a legal FPL starting XI:
      - exactly 11 players
      - exactly 1 GK
      - DEF 3–5, MID 2–5, FWD 1–3
    """
    if len(player_ids) != 11:
        return False
    c = count_positions(player_ids, squad_map)
    return (
        c["GK"] == 1
        and 3 <= c["DEF"] <= 5
        and 2 <= c["MID"] <= 5
        and 1 <= c["FWD"] <= 3
        and sum(c.values()) == 11
    )


def validate_decision(
    decision: "LineupDecision",
    squad_map: Dict[int, Player],
) -> None:
    """
    Raise ``ValueError`` if *decision* violates any FPL lineup constraint.

    Checks
    ------
    1. XI has exactly 11 players.
    2. XI satisfies formation rules (1 GK, 3–5 DEF, 2–5 MID, 1–3 FWD).
    3. Bench has exactly 4 players.
    4. Bench slot 3 (index 3) is the backup GK.
    5. Bench slots 0–2 are all non-GK (outfield) players.
    6. Captain is in the XI.
    7. Vice-captain is in the XI.
    """
    if len(decision.xi) != 11:
        raise ValueError(f"XI must have 11 players; got {len(decision.xi)}")
    if not is_valid_xi(decision.xi, squad_map):
        c = count_positions(decision.xi, squad_map)
        raise ValueError(
            f"Invalid XI formation: GK={c['GK']} DEF={c['DEF']} "
            f"MID={c['MID']} FWD={c['FWD']}"
        )
    if len(decision.bench) != 4:
        raise ValueError(f"Bench must have 4 players; got {len(decision.bench)}")
    bench_gk = decision.bench[3]
    if bench_gk not in squad_map:
        raise ValueError(f"Bench GK slot (index 3) has unknown player {bench_gk}")
    if squad_map[bench_gk].position != "GK":
        raise ValueError(
            f"Bench slot 3 must be a GK; got {squad_map[bench_gk].position} "
            f"(player {bench_gk})"
        )
    for i, pid in enumerate(decision.bench[:3]):
        if pid not in squad_map:
            raise ValueError(f"Bench slot {i} has unknown player {pid}")
        if squad_map[pid].position == "GK":
            raise ValueError(
                f"Bench slots 0–2 must be outfield; slot {i} has GK (player {pid})"
            )
    if decision.captain not in decision.xi:
        raise ValueError(
            f"Captain (player {decision.captain}) is not in the XI"
        )
    if decision.vice_captain not in decision.xi:
        raise ValueError(
            f"Vice-captain (player {decision.vice_captain}) is not in the XI"
        )


# ---------------------------------------------------------------------------
# Auto-substitution
# ---------------------------------------------------------------------------

def apply_autosub(
    decision: LineupDecision,
    minutes: Dict[int, int],
    squad_map: Dict[int, Player],
) -> List[int]:
    """
    Apply FPL auto-substitution rules; return effective XI.

    Rules
    -----
    - bench[0..2] are outfield subs in priority order.
    - bench[3] is the backup GK.
    - Bench GK replaces starter GK only if starter GK played 0 min and
      bench GK played > 0 min.
    - Outfield subs enter in order; a sub only enters if the resulting
      formation remains valid (≥3 DEF, ≥2 MID, ≥1 FWD, exactly 1 GK).
    """
    xi = list(decision.xi)
    bench_outfield = list(decision.bench[:3])
    bench_gk_id = decision.bench[3]

    # GK substitution
    starter_gk = next(
        (pid for pid in xi if squad_map[pid].position == "GK"), None
    )
    if (
        starter_gk is not None
        and minutes.get(starter_gk, 0) == 0
        and minutes.get(bench_gk_id, 0) > 0
    ):
        xi.remove(starter_gk)
        xi.append(bench_gk_id)

    # Outfield substitutions (in bench priority order)
    for sub in bench_outfield:
        if minutes.get(sub, 0) == 0:
            continue
        non_playing = [
            pid for pid in xi
            if minutes.get(pid, 0) == 0 and squad_map[pid].position != "GK"
        ]
        if not non_playing:
            break
        for victim in non_playing:
            trial = [p for p in xi if p != victim] + [sub]
            if is_valid_xi(trial, squad_map):
                xi = trial
                break  # sub used; move to next sub slot

    return xi


# ---------------------------------------------------------------------------
# Captain resolution
# ---------------------------------------------------------------------------

def resolve_captain(
    decision: LineupDecision,
    effective_xi: List[int],
    minutes: Dict[int, int],
) -> Optional[int]:
    """
    Return the player whose points are doubled:
      - captain if they played (> 0 min) and are in the effective XI;
      - else vice-captain if they played and are in the effective XI;
      - else None (no doubling).
    """
    cap, vice = decision.captain, decision.vice_captain
    if cap in effective_xi and minutes.get(cap, 0) > 0:
        return cap
    if vice in effective_xi and minutes.get(vice, 0) > 0:
        return vice
    return None
