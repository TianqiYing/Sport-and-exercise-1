"""
Load per-GW CSV data for a season and build player histories.
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List, Set

import pandas as pd

from .rules import Player, normalise_position

# Resolved relative to this file
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_data_root() -> str:
    """Best-effort discovery of the FPL dataset root.

    This repo is sometimes run from the project root (where data lives under
    ``Sport-and-exercise-1/data``) and sometimes from within the
    ``Sport-and-exercise-1`` folder itself (where data lives under ``data``).

    We search upwards from this file to support both layouts.
    """
    here = os.path.abspath(_THIS_DIR)
    # Search a few levels up to find either:
    #   - <...>/Sport-and-exercise-1/data
    #   - <...>/data (when code is inside Sport-and-exercise-1)
    for _ in range(8):
        cand1 = os.path.join(here, "data")
        if os.path.isdir(cand1):
            return os.path.normpath(cand1)

        cand2 = os.path.join(here, "Sport-and-exercise-1", "data")
        if os.path.isdir(cand2):
            return os.path.normpath(cand2)

        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent

    # Fallback to the historical default (repo root layout)
    return os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "Sport-and-exercise-1", "data"))


_DATA_ROOT = _resolve_data_root()


def _load_valid_player_ids_from_players_raw(
    season: str,
    data_root: str = _DATA_ROOT,
) -> Set[int] | None:
    """Return the set of valid *player* element IDs for a season.

    Some seasons/datasets include non-player "elements" (e.g., managers)
    which can appear in GW exports and in ``players_raw.csv`` with very low
    costs. For this project we treat only FPL footballers as valid players.

    We detect validity using ``players_raw.csv``:
    - element_type 1..4 = GK/DEF/MID/FWD (valid)
    - anything else (commonly 5) is excluded.

    Returns None if the file/columns are not available.
    """
    fpath = os.path.join(data_root, season, "players_raw.csv")
    if not os.path.exists(fpath):
        return None

    try:
        raw = pd.read_csv(fpath, usecols=["id", "element_type"])
    except Exception:
        return None

    if "id" not in raw.columns or "element_type" not in raw.columns:
        return None

    raw = raw.dropna(subset=["id", "element_type"]).copy()
    raw["id"] = pd.to_numeric(raw["id"], errors="coerce")
    raw["element_type"] = pd.to_numeric(raw["element_type"], errors="coerce")
    raw = raw.dropna(subset=["id", "element_type"])

    valid = raw[raw["element_type"].isin([1, 2, 3, 4])]["id"].astype(int)
    return set(valid.tolist())


def load_season_gws(
    season: str = "2024-25",
    data_dir: str | None = None,
) -> pd.DataFrame:
    """
    Load all gw*.csv files for *season* into a single DataFrame.

    Parameters
    ----------
    season : str
        Season folder name, e.g. ``"2024-25"``.
    data_dir : str, optional
        Override path to the gws/ directory.  Defaults to the canonical
        location under Sport-and-exercise-1/data.

    Returns
    -------
    pd.DataFrame
        Columns guaranteed present: gw, element, name, position, team,
        minutes, total_points.  xP, opponent_team, was_home included
        when available.
    """
    if data_dir is None:
        data_dir = os.path.join(_DATA_ROOT, season, "gws")

    pattern = os.path.join(data_dir, "gw*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No GW CSV files found in: {data_dir}\n"
            "Check --season / --data-dir arguments."
        )

    frames: list[pd.DataFrame] = []
    for fpath in files:
        df = pd.read_csv(fpath)
        # 'round' is the GW number in most exports
        if "round" in df.columns and "gw" not in df.columns:
            df = df.rename(columns={"round": "gw"})
        if "gw" not in df.columns:
            fname = os.path.basename(fpath)
            nums = "".join(c for c in fname if c.isdigit())
            df["gw"] = int(nums) if nums else None
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Coerce numeric columns
    for col in ("element", "gw", "minutes", "total_points"):
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    if "xP" in combined.columns:
        combined["xP"] = pd.to_numeric(combined["xP"], errors="coerce")

    # Normalise position encoding
    if "position" in combined.columns:
        combined["position"] = combined["position"].apply(normalise_position)

    combined = combined.dropna(subset=["element", "gw"])
    combined["element"] = combined["element"].astype(int)
    combined["gw"] = combined["gw"].astype(int)

    # Exclude non-player elements (e.g., managers) when the season includes them.
    valid_ids = _load_valid_player_ids_from_players_raw(season)
    if valid_ids is not None:
        combined = combined[combined["element"].isin(valid_ids)]

    return combined.reset_index(drop=True)


def build_player_histories(
    df: pd.DataFrame,
    target_gw: int,
) -> Dict[int, pd.DataFrame]:
    """
    Return per-player history using only GWs **strictly before** *target_gw*.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`load_season_gws`.
    target_gw : int
        Target gameweek; rows with ``gw >= target_gw`` are excluded.

    Returns
    -------
    Dict[int, pd.DataFrame]
        Maps element → DataFrame of historical rows.
    """
    history = df[df["gw"] < target_gw].copy()
    return {
        int(pid): grp.reset_index(drop=True)
        for pid, grp in history.groupby("element")
    }


def get_player_info(df: pd.DataFrame) -> Dict[int, dict]:
    """
    Build ``element → {name, position, team}`` from the most recent GW row.
    """
    result: Dict[int, dict] = {}
    for pid, grp in df.groupby("element"):
        row = grp.sort_values("gw").iloc[-1]
        result[int(pid)] = {
            "name": str(row.get("name", f"player_{pid}")),
            "position": str(row.get("position", "MID")),
            "team": str(row.get("team", "")),
        }
    return result


def build_squad_map(
    player_ids: List[int],
    player_info: Dict[int, dict],
) -> Dict[int, Player]:
    """
    Convert a list of element IDs + info dict into a squad_map ready for
    simulation and formation checks.
    """
    squad_map: Dict[int, Player] = {}
    for pid in player_ids:
        info = player_info.get(pid, {})
        squad_map[pid] = Player(
            player_id=pid,
            name=info.get("name", f"player_{pid}"),
            position=info.get("position", "MID"),
            team=info.get("team", ""),
        )
    return squad_map


def name_to_element(df: pd.DataFrame) -> Dict[str, int]:
    """Return a ``name → element`` mapping for convenient look-ups."""
    return {v["name"]: k for k, v in get_player_info(df).items()}


def load_player_costs(
    season: str = "2024-25",
    data_dir: str | None = None,
) -> Dict[int, int]:
    """
    Load player costs (``now_cost``) from ``players_raw.csv``.

    ``now_cost`` is in tenths of millions (e.g. 65 = £6.5 m).
    The FPL total squad budget is 1000 (£100 m).

    Parameters
    ----------
    season : str
        Season folder name.
    data_dir : str, optional
        Override path to the season directory containing ``players_raw.csv``.

    Returns
    -------
    Dict[int, int]
        Maps element id → ``now_cost`` in tenths of millions.

    Raises
    ------
    FileNotFoundError
        If ``players_raw.csv`` is not found.
    """
    if data_dir is None:
        fpath = os.path.join(_DATA_ROOT, season, "players_raw.csv")
    else:
        fpath = os.path.join(data_dir, "players_raw.csv")

    if not os.path.exists(fpath):
        raise FileNotFoundError(f"players_raw.csv not found at: {fpath}")

    # Filter out non-player entries (e.g., managers) via element_type when present.
    usecols = ["id", "now_cost", "element_type"]
    try:
        raw = pd.read_csv(fpath, usecols=usecols)
    except ValueError:
        # Older seasons may not have element_type in players_raw.csv
        raw = pd.read_csv(fpath, usecols=["id", "now_cost"])

    raw = raw.dropna(subset=["id", "now_cost"]).copy()
    if "element_type" in raw.columns:
        raw = raw.dropna(subset=["element_type"])
        raw["element_type"] = pd.to_numeric(raw["element_type"], errors="coerce")
        raw = raw[raw["element_type"].isin([1, 2, 3, 4])]

    return {int(row.id): int(row.now_cost) for row in raw.itertuples()}
