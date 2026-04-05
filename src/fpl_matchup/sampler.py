"""
HistoricalEmpiricalSampler: sample (minutes, points) from per-player history.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .rules import PlayerSampler


class HistoricalEmpiricalSampler(PlayerSampler):
    """
    Sample (minutes, points) for a player using only historical GW data.

    Strategy
    --------
    1. Estimate ``p_play`` as the fraction of the last *recent_n* GWs
       where the player had ``minutes > 0``.
    2. If the player plays: sample uniformly from that player's historical
       ``total_points`` values when ``minutes > 0``.
    3. Backoff chain when player history is thin:
       player history → position-level empirical distribution → global.

    Parameters
    ----------
    histories : Dict[int, pd.DataFrame]
        element → DataFrame with at least ``minutes`` and
        ``total_points`` columns.  Build with
        :func:`fpl_matchup.data.build_player_histories`.
    player_positions : Dict[int, str]
        element → one of ``"GK"``, ``"DEF"``, ``"MID"``, ``"FWD"``.
    seed : int
        RNG seed (reproducible across calls when the seed is the same).
    recent_n : int
        Window size for ``p_play`` estimation (most recent N GWs).
    min_hist : int
        Minimum number of played GWs required to use player-level
        distribution; falls back to position-level otherwise.
    """

    def __init__(
        self,
        histories: Dict[int, pd.DataFrame],
        player_positions: Dict[int, str],
        seed: int = 42,
        recent_n: int = 5,
        min_hist: int = 2,
    ) -> None:
        self.histories = histories
        self.player_positions = player_positions
        self.rng = np.random.default_rng(seed)
        self.recent_n = recent_n
        self.min_hist = min_hist

        # Pre-compute per-player arrays for fast batch sampling
        self._played_pts: Dict[int, np.ndarray] = {}
        self._played_mins: Dict[int, np.ndarray] = {}
        self._p_play_cache: Dict[int, float] = {}

        for pid, h in histories.items():
            played = h[h["minutes"] > 0]
            self._played_pts[pid] = played["total_points"].dropna().values.astype(float)
            self._played_mins[pid] = played["minutes"].values.astype(int)
            recent = h.tail(recent_n)
            self._p_play_cache[pid] = float((recent["minutes"] > 0).mean())

        # Build position-level and global fallback distributions
        pos_buckets: Dict[str, List[float]] = {
            "GK": [], "DEF": [], "MID": [], "FWD": []
        }
        all_pts: List[float] = []
        for pid, h in histories.items():
            pos = player_positions.get(pid, "MID")
            pts = h[h["minutes"] > 0]["total_points"].dropna().tolist()
            if pos in pos_buckets:
                pos_buckets[pos].extend(pts)
            all_pts.extend(pts)

        self._pos_pts: Dict[str, np.ndarray] = {
            pos: np.array(pts, dtype=float)
            for pos, pts in pos_buckets.items()
            if pts
        }
        self._global_pts = np.array(all_pts, dtype=float) if all_pts else np.array([2.0])

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def sample_minutes_points(self, player_id: int, gw: int) -> Tuple[int, float]:
        """Sample ``(minutes, points)`` for one player at gameweek *gw*."""
        p = self._p_play_cache.get(player_id, 0.7)
        if self.rng.random() >= p:
            return (0, 0.0)
        return (self._sample_mins(player_id), self._sample_pts(player_id))

    def predicted_mean(self, player_id: int) -> float:
        """``E[points] = p_play × E[points | played]``."""
        p = self._p_play_cache.get(player_id, 0.7)
        pts_arr = self._effective_pts_arr(player_id)
        return p * float(pts_arr.mean())

    # ------------------------------------------------------------------
    # Batch sampling (vectorised; used by simulate.py)
    # ------------------------------------------------------------------

    def batch_sample(
        self, player_ids: List[int], n_sims: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample ``(minutes, points)`` for *all* players across *n_sims* sims
        in one vectorised call.

        Parameters
        ----------
        player_ids : list of int
        n_sims : int

        Returns
        -------
        mins_matrix : np.ndarray, shape ``(len(player_ids), n_sims)``
        pts_matrix  : np.ndarray, shape ``(len(player_ids), n_sims)``
        """
        n_p = len(player_ids)
        mins_matrix = np.zeros((n_p, n_sims), dtype=int)
        pts_matrix = np.zeros((n_p, n_sims), dtype=float)

        # Vectorised play/no-play draw
        p_plays = np.array([self._p_play_cache.get(pid, 0.7) for pid in player_ids])
        plays = self.rng.random((n_p, n_sims)) < p_plays[:, np.newaxis]

        for i, pid in enumerate(player_ids):
            if not plays[i].any():
                continue

            pts_arr = self._effective_pts_arr(pid)
            mins_arr = self._played_mins.get(pid)

            idx = self.rng.integers(0, len(pts_arr), size=n_sims)
            raw_pts = pts_arr[idx]

            if mins_arr is not None and len(mins_arr) > 0:
                m_idx = self.rng.integers(0, len(mins_arr), size=n_sims)
                raw_mins = mins_arr[m_idx]
            else:
                raw_mins = np.full(n_sims, 60, dtype=int)

            pts_matrix[i] = np.where(plays[i], raw_pts, 0.0)
            mins_matrix[i] = np.where(plays[i], raw_mins, 0)

        return mins_matrix, pts_matrix

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _effective_pts_arr(self, player_id: int) -> np.ndarray:
        """Return the points array to sample from (player → pos → global)."""
        pts_arr = self._played_pts.get(player_id)
        if pts_arr is not None and len(pts_arr) >= self.min_hist:
            return pts_arr
        pos = self.player_positions.get(player_id, "MID")
        return self._pos_pts.get(pos, self._global_pts)

    def _sample_pts(self, player_id: int) -> float:
        arr = self._effective_pts_arr(player_id)
        return float(self.rng.choice(arr))

    def _sample_mins(self, player_id: int) -> int:
        mins_arr = self._played_mins.get(player_id)
        if mins_arr is not None and len(mins_arr) > 0:
            return int(self.rng.choice(mins_arr))
        return 60
