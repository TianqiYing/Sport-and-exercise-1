"""
Simple backtest: for each GW t, optimise lineup using history < t,
then score against actual total_points from GW t.

Run:
    python -m fpl_matchup.backtest --season 2024-25 \\
        --my-squad '[1,2,...,15]' --opp-squad '[...]' --opp-xi '[...]' \\
        --opp-captain 123 --risk balanced
"""
from __future__ import annotations

import argparse
import json
from typing import List

import numpy as np
import pandas as pd

from .data import build_player_histories, build_squad_map, get_player_info, load_season_gws
from .optimize import greedy_lineup_initial, optimize_lineup_vs_opponent, order_bench
from .rules import LineupDecision
from .sampler import HistoricalEmpiricalSampler
from .simulate import simulate_matchup_distribution, _score_lineup


def _parse_ids(raw: str) -> List[int]:
    return [int(x) for x in json.loads(raw)]


def score_realised(
    decision: LineupDecision,
    squad_map: dict,
    gw_row: pd.DataFrame,
) -> float:
    """Score a lineup using the actual GW outcome (ground truth)."""
    minutes = dict(zip(gw_row["element"].astype(int), gw_row["minutes"].astype(int)))
    points = dict(zip(gw_row["element"].astype(int), gw_row["total_points"].astype(float)))
    return _score_lineup(decision, squad_map, minutes, points)


def run_backtest(
    season: str,
    my_ids: List[int],
    opp_ids: List[int],
    opp_xi_ids: List[int],
    opp_captain: int,
    opp_vice: int,
    risk: str = "balanced",
    n_sims: int = 3000,
    seed: int = 42,
    start_gw: int = 5,   # need enough history
    end_gw: int | None = None,
    data_dir: str | None = None,
) -> pd.DataFrame:
    """
    Run a walkforward backtest.

    For each GW t in [start_gw, end_gw]:
      - Build sampler from GWs < t
      - Optimise my lineup vs opponent
      - Also compute a baseline (greedy by expected points)
      - Score both lineups against the realised GW t outcome
      - Record margin for optimised vs baseline

    Returns a DataFrame with one row per GW.
    """
    df = load_season_gws(season=season, data_dir=data_dir)
    all_ids = list(set(my_ids + opp_ids))
    player_info = get_player_info(df)
    my_squad_map = build_squad_map(my_ids, player_info)
    opp_squad_map = build_squad_map(opp_ids, player_info)
    player_positions = {pid: player_info.get(pid, {}).get("position", "MID") for pid in all_ids}

    gws = sorted(df["gw"].unique())
    if end_gw is None:
        end_gw = max(gws)
    eval_gws = [g for g in gws if start_gw <= g <= end_gw]

    records = []
    for gw in eval_gws:
        gw_actual = df[df["gw"] == gw]
        if gw_actual.empty:
            continue

        histories = build_player_histories(df, target_gw=gw)
        hist_subset = {pid: histories[pid] for pid in all_ids if pid in histories}
        if not hist_subset:
            continue

        sampler = HistoricalEmpiricalSampler(
            histories=hist_subset,
            player_positions=player_positions,
            seed=seed,
        )

        predicted_mean_opp = {pid: sampler.predicted_mean(pid) for pid in opp_ids}
        opp_bench_pool = [pid for pid in opp_ids if pid not in opp_xi_ids]
        opp_bench = order_bench(opp_bench_pool, opp_squad_map, predicted_mean_opp)
        opp_decision = LineupDecision(
            xi=opp_xi_ids, bench=opp_bench,
            captain=opp_captain, vice_captain=opp_vice,
        )

        # Optimised lineup
        try:
            opt_dec, _ = optimize_lineup_vs_opponent(
                my_squad_map=my_squad_map,
                opp_decision=opp_decision,
                opp_squad_map=opp_squad_map,
                sampler=sampler,
                gw=gw,
                risk=risk,
                n_sims_search=max(500, n_sims // 3),
                n_sims_final=n_sims,
                max_iter=3,
                seed=seed,
            )
        except Exception as exc:
            print(f"  GW {gw}: optimiser failed ({exc}); skipping")
            continue

        # Baseline: greedy by predicted mean only
        predicted_mean_my = {pid: sampler.predicted_mean(pid) for pid in my_ids}
        base_dec = greedy_lineup_initial(my_squad_map, predicted_mean_my)

        # Realised scores
        my_opt_pts = score_realised(opt_dec, my_squad_map, gw_actual)
        my_base_pts = score_realised(base_dec, my_squad_map, gw_actual)
        opp_pts = score_realised(opp_decision, opp_squad_map, gw_actual)

        records.append({
            "gw": gw,
            "opt_my_pts": my_opt_pts,
            "base_my_pts": my_base_pts,
            "opp_pts": opp_pts,
            "opt_margin": my_opt_pts - opp_pts,
            "base_margin": my_base_pts - opp_pts,
            "opt_win": int(my_opt_pts > opp_pts),
            "base_win": int(my_base_pts > opp_pts),
        })
        print(
            f"  GW {gw:2d}:  opt={my_opt_pts:.0f}  base={my_base_pts:.0f}"
            f"  opp={opp_pts:.0f}"
            f"  opt_margin={my_opt_pts - opp_pts:+.0f}"
        )

    return pd.DataFrame(records)


def print_summary(results: pd.DataFrame) -> None:
    if results.empty:
        print("No results to summarise.")
        return
    n = len(results)
    print("\n" + "=" * 55)
    print("Backtest summary")
    print("=" * 55)
    print(f"  GWs evaluated:         {n}")
    print(f"  Optimised win rate:    {results['opt_win'].mean():.3f}")
    print(f"  Baseline win rate:     {results['base_win'].mean():.3f}")
    print(f"  Avg opt margin:        {results['opt_margin'].mean():+.1f}")
    print(f"  Avg base margin:       {results['base_margin'].mean():+.1f}")
    print(f"  Avg improvement over baseline: "
          f"{(results['opt_margin'] - results['base_margin']).mean():+.1f} pts")


def main() -> None:
    parser = argparse.ArgumentParser(description="FPL matchup backtest")
    parser.add_argument("--season", default="2024-25")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--my-squad", required=True)
    parser.add_argument("--opp-squad", required=True)
    parser.add_argument("--opp-xi", required=True)
    parser.add_argument("--opp-captain", type=int, required=True)
    parser.add_argument("--opp-vice", type=int, default=None)
    parser.add_argument("--risk", choices=["conservative", "balanced", "optimistic"],
                        default="balanced")
    parser.add_argument("--n-sims", type=int, default=2000)
    parser.add_argument("--start-gw", type=int, default=5)
    parser.add_argument("--end-gw", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    my_ids = _parse_ids(args.my_squad)
    opp_ids = _parse_ids(args.opp_squad)
    opp_xi_ids = _parse_ids(args.opp_xi)
    opp_vice = args.opp_vice if args.opp_vice is not None else args.opp_captain

    print(f"Running backtest: season={args.season}, risk={args.risk}")
    results = run_backtest(
        season=args.season,
        my_ids=my_ids,
        opp_ids=opp_ids,
        opp_xi_ids=opp_xi_ids,
        opp_captain=args.opp_captain,
        opp_vice=opp_vice,
        risk=args.risk,
        n_sims=args.n_sims,
        seed=args.seed,
        start_gw=args.start_gw,
        end_gw=args.end_gw,
        data_dir=args.data_dir,
    )
    print_summary(results)


if __name__ == "__main__":
    main()
