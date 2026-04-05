"""
CLI entry point for the FPL Matchup Optimiser.

Usage examples
--------------
# Basic – provide my squad and opponent's XI as JSON arrays of element IDs.
python -m fpl_matchup.cli \\
    --season 2024-25 \\
    --gw 20 \\
    --my-squad '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]' \\
    --opp-squad '[10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]' \\
    --opp-xi '[10,11,12,13,14,15,16,17,18,19,20]' \\
    --opp-captain 10 \\
    --risk balanced \\
    --n-sims 5000

# With transfer planning (1 free transfer)
python -m fpl_matchup.cli \\
    --season 2024-25 --gw 20 \\
    --my-squad '[...]' --opp-squad '[...]' --opp-xi '[...]' --opp-captain 10 \\
    --free-transfers 1

# List all known player names for a season (to look up element IDs)
python -m fpl_matchup.cli --season 2024-25 --list-players
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .data import (
    build_player_histories,
    build_squad_map,
    get_player_info,
    load_player_costs,
    load_season_gws,
    name_to_element,
)
from .optimize import greedy_lineup_initial, optimize_lineup_vs_opponent, order_bench
from .rules import LineupDecision, Player
from .sampler import HistoricalEmpiricalSampler
from .simulate import simulate_matchup_distribution
from .transfers import optimize_transfers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_ids(raw: str) -> List[int]:
    """Parse a JSON array of ints from a CLI string argument."""
    try:
        parsed = json.loads(raw)
        return [int(x) for x in parsed]
    except (json.JSONDecodeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"Expected a JSON array of integers (e.g. '[1,2,3]'), got: {raw!r}"
        ) from exc


def _print_decision(
    decision: LineupDecision,
    squad_map: dict,
    label: str = "Recommended lineup",
) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print("\nStarting XI:")
    for pid in decision.xi:
        p = squad_map[pid]
        cap_flag = ""
        if pid == decision.captain:
            cap_flag = " [C]"
        elif pid == decision.vice_captain:
            cap_flag = " [V]"
        print(f"  {p.position:3s}  {p.name}{cap_flag}")

    print("\nBench (sub priority order):")
    for i, pid in enumerate(decision.bench[:3], 1):
        p = squad_map[pid]
        print(f"  {i}. {p.position:3s}  {p.name}")
    gk_pid = decision.bench[3]
    p = squad_map[gk_pid]
    print(f"  GK  {p.name}")

    print(f"\nCaptain:      {squad_map[decision.captain].name}")
    print(f"Vice-captain: {squad_map[decision.vice_captain].name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="fpl_matchup",
        description="H2H FPL matchup optimiser – recommend best XI vs opponent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data args
    parser.add_argument("--season", default="2024-25", help="Season folder (default: 2024-25)")
    parser.add_argument("--data-dir", default=None, help="Override path to the gws/ directory")
    parser.add_argument("--gw", type=int, required=False, default=None,
                        help="Target gameweek number (required unless --list-players)")

    # Squad args
    parser.add_argument("--my-squad", default=None,
                        help="JSON array of 15 element IDs for my squad")
    parser.add_argument("--opp-squad", default=None,
                        help="JSON array of 15 element IDs for opponent's squad")
    parser.add_argument("--opp-xi", default=None,
                        help="JSON array of 11 element IDs in opponent's starting XI")
    parser.add_argument("--opp-captain", type=int, default=None,
                        help="Opponent's captain element ID")
    parser.add_argument("--opp-vice", type=int, default=None,
                        help="Opponent's vice-captain element ID (defaults to captain)")

    # Optimisation args
    parser.add_argument(
        "--risk", choices=["conservative", "balanced", "optimistic"],
        default="balanced",
        help="Risk profile (default: balanced)",
    )
    parser.add_argument("--n-sims", type=int, default=5000,
                        help="Final simulation count (default: 5000)")
    parser.add_argument("--n-sims-search", type=int, default=1000,
                        help="Simulation count during search (default: 1000)")
    parser.add_argument("--max-iter", type=int, default=5,
                        help="Max local-search iterations (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")

    # Transfer planning
    parser.add_argument(
        "--free-transfers", type=int, default=0, metavar="INT",
        help="Number of free transfers to consider (default: 0)",
    )

    # Utility
    parser.add_argument("--list-players", action="store_true",
                        help="Print all player names and element IDs then exit")
    parser.add_argument("--no-optimize", action="store_true",
                        help="Skip optimisation; just show greedy lineup and simulate it")

    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"Loading season {args.season} data…", end=" ", flush=True)
    df = load_season_gws(season=args.season, data_dir=args.data_dir)
    print(f"done ({len(df):,} rows, GWs {df['gw'].min()}–{df['gw'].max()})")

    if args.list_players:
        player_info = get_player_info(df)
        lines: List[str] = [f"{'Element':>8}  {'Pos':3}  Name", "-" * 50]
        for pid in sorted(player_info):
            p = player_info[pid]
            lines.append(f"{pid:>8}  {p['position']:3}  {p['name']}")
        out_path = Path(f"players_{args.season}_list.txt")
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            print(f"\nWrote {len(player_info)} players to {out_path.resolve()}")
            print("\n" + "\n".join(lines))
        except BrokenPipeError:
            try:
                sys.stdout.close()
            finally:
                return
        return

    # ------------------------------------------------------------------
    # Validate required args
    # ------------------------------------------------------------------
    if args.gw is None:
        parser.error("--gw is required (unless --list-players)")
    if args.my_squad is None:
        parser.error("--my-squad is required")
    if args.opp_squad is None:
        parser.error("--opp-squad is required")
    if args.opp_xi is None:
        parser.error("--opp-xi is required")
    if args.opp_captain is None:
        parser.error("--opp-captain is required")

    my_ids = _parse_ids(args.my_squad)
    opp_ids = _parse_ids(args.opp_squad)
    opp_xi_ids = _parse_ids(args.opp_xi)
    opp_cap = args.opp_captain
    opp_vice = args.opp_vice if args.opp_vice is not None else opp_cap

    if len(my_ids) != 15:
        sys.exit(f"--my-squad must contain exactly 15 element IDs, got {len(my_ids)}")
    if len(opp_ids) != 15:
        sys.exit(f"--opp-squad must contain exactly 15 element IDs, got {len(opp_ids)}")
    if len(opp_xi_ids) != 11:
        sys.exit(f"--opp-xi must contain exactly 11 element IDs, got {len(opp_xi_ids)}")

    # ------------------------------------------------------------------
    # Build histories and sampler
    # ------------------------------------------------------------------
    player_info = get_player_info(df)
    all_ids = list(set(my_ids + opp_ids))

    print(f"Building player histories (GWs < {args.gw})…", end=" ", flush=True)
    histories = build_player_histories(df, target_gw=args.gw)
    player_positions = {
        pid: player_info.get(pid, {}).get("position", "MID")
        for pid in player_info  # include full universe for transfer search
    }
    sampler = HistoricalEmpiricalSampler(
        histories=histories,
        player_positions=player_positions,
        seed=args.seed,
    )
    print("done")

    # ------------------------------------------------------------------
    # Build squad maps
    # ------------------------------------------------------------------
    my_squad_map = build_squad_map(my_ids, player_info)
    opp_squad_map = build_squad_map(opp_ids, player_info)

    opp_bench_pool = [pid for pid in opp_ids if pid not in opp_xi_ids]
    predicted_mean_opp = {pid: sampler.predicted_mean(pid) for pid in opp_ids}
    opp_bench = order_bench(opp_bench_pool, opp_squad_map, predicted_mean_opp)

    opp_decision = LineupDecision(
        xi=opp_xi_ids,
        bench=opp_bench,
        captain=opp_cap,
        vice_captain=opp_vice,
    )

    # ------------------------------------------------------------------
    # Transfer planning (optional)
    # ------------------------------------------------------------------
    if args.free_transfers > 0:
        print(f"\nSearching for up to {args.free_transfers} transfer(s)…")
        costs: Optional[dict] = None
        try:
            costs = load_player_costs(season=args.season)
        except FileNotFoundError:
            print("  (players_raw.csv not found; skipping budget constraint)")

        universe = [
            Player(
                player_id=pid,
                name=info["name"],
                position=info["position"],
                team=info["team"],
            )
            for pid, info in player_info.items()
        ]

        transfer_results = optimize_transfers(
            my_squad_map=my_squad_map,
            opp_decision=opp_decision,
            opp_squad_map=opp_squad_map,
            sampler=sampler,
            gw=args.gw,
            free_transfers=args.free_transfers,
            universe=universe,
            costs=costs,
            risk=args.risk,
            n_sims_inner=max(300, args.n_sims_search // 3),
            n_sims_final=args.n_sims,
            seed=args.seed,
        )

        if not transfer_results:
            print("  No beneficial transfer found; keeping current squad.")
        else:
            print(f"\n  Suggested transfer(s):")
            for i, tr in enumerate(transfer_results, 1):
                out_name = player_info.get(tr.transfer_out, {}).get("name", f"#{tr.transfer_out}")
                in_name = player_info.get(tr.transfer_in, {}).get("name", f"#{tr.transfer_in}")
                print(f"  {i}. OUT: {out_name}  →  IN: {in_name}")
                print(f"     P(win)={tr.stats.p_win:.3f}  E[margin]={tr.stats.exp_margin:+.1f}")

            # Use the final transfer's squad and lineup
            last = transfer_results[-1]
            my_squad_map = build_squad_map(last.new_squad_ids, player_info)
            best_decision = last.lineup
            final_stats = last.stats

            _print_decision(
                best_decision, my_squad_map,
                label=f"Recommended lineup after transfers (GW {args.gw})",
            )
            print(f"\nMatchup statistics vs opponent:")
            print(f"  {final_stats.summary()}")
            print(f"\n  My expected score:  {final_stats.my_scores.mean():.1f} pts")
            print(f"  Opp expected score: {final_stats.opp_scores.mean():.1f} pts")
            print()
            return

    # ------------------------------------------------------------------
    # Lineup optimisation (no transfers)
    # ------------------------------------------------------------------
    if args.no_optimize:
        print("Skipping optimisation (--no-optimize); using greedy lineup.")
        predicted_mean_my = {pid: sampler.predicted_mean(pid) for pid in my_ids}
        best_decision = greedy_lineup_initial(my_squad_map, predicted_mean_my)
        print("Simulating…", end=" ", flush=True)
        final_stats = simulate_matchup_distribution(
            args.gw, best_decision, my_squad_map,
            opp_decision, opp_squad_map, sampler,
            n_sims=args.n_sims, seed=args.seed,
        )
        print("done")
    else:
        print(
            f"Optimising lineup ({args.risk} risk, "
            f"{args.n_sims_search} search sims, {args.n_sims} final sims)…"
        )
        best_decision, final_stats = optimize_lineup_vs_opponent(
            my_squad_map=my_squad_map,
            opp_decision=opp_decision,
            opp_squad_map=opp_squad_map,
            sampler=sampler,
            gw=args.gw,
            risk=args.risk,
            n_sims_search=args.n_sims_search,
            n_sims_final=args.n_sims,
            max_iter=args.max_iter,
            seed=args.seed,
        )
        print("done")

    _print_decision(best_decision, my_squad_map, label=f"Recommended lineup (GW {args.gw})")
    print("\nMatchup statistics vs opponent:")
    print(f"  {final_stats.summary()}")
    print(f"\n  My expected score:  {final_stats.my_scores.mean():.1f} pts")
    print(f"  Opp expected score: {final_stats.opp_scores.mean():.1f} pts")
    print()


if __name__ == "__main__":
    main()
