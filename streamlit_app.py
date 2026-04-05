"""
FPL H2H Matchup Optimiser — local Streamlit web UI.

Run with:
    streamlit run streamlit_app.py

Tabs
----
- Manual  : select squads by position, then optimise.
- Auto A  : loads your squad from my_squad.json (saved via "Save my squad").
- Auto B  : builds the best 15-player squad from scratch vs the opponent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Make local package importable when running from the repo root
sys.path.insert(0, str(Path(__file__).parent))

from src.fpl_matchup.data import (
    build_player_histories,
    build_squad_map,
    get_player_info,
    load_player_costs,
    load_season_gws,
)
from src.fpl_matchup.optimize import (
    greedy_lineup_initial,
    optimize_lineup_vs_opponent,
    order_bench,
)
from src.fpl_matchup.rules import LineupDecision, Player, is_valid_xi, validate_decision
from src.fpl_matchup.sampler import HistoricalEmpiricalSampler
from src.fpl_matchup.simulate import simulate_matchup_distribution
from src.fpl_matchup.transfers import build_squad_vs_opponent, optimize_transfers

MY_SQUAD_FILE = Path(__file__).parent / "my_squad.json"

POSITION_QUOTAS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FPL H2H Optimiser",
    page_icon="⚽",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached data loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_season(season: str):
    df = load_season_gws(season=season)
    player_info = get_player_info(df)
    return df, player_info


@st.cache_data(show_spinner=False)
def _load_costs(season: str) -> Optional[Dict[int, int]]:
    try:
        return load_player_costs(season=season)
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label(pid: int, player_info: Dict[int, dict]) -> str:
    info = player_info.get(pid, {})
    return (
        f"{info.get('name', f'#{pid}')} "
        f"({info.get('position','?')}, {info.get('team','')})"
    )


def _build_sampler(
    df: pd.DataFrame,
    player_info: Dict[int, dict],
    gw: int,
    seed: int,
) -> HistoricalEmpiricalSampler:
    histories = build_player_histories(df, target_gw=gw)
    positions = {pid: info["position"] for pid, info in player_info.items()}
    return HistoricalEmpiricalSampler(
        histories=histories, player_positions=positions, seed=seed
    )


def _build_opp_decision(
    opp_ids: List[int],
    opp_xi_ids: List[int],
    opp_squad_map: Dict[int, Player],
    opp_cap: int,
    opp_vice: int,
    sampler: HistoricalEmpiricalSampler,
) -> LineupDecision:
    opp_bench_pool = [pid for pid in opp_ids if pid not in opp_xi_ids]
    pred_opp = {pid: sampler.predicted_mean(pid) for pid in opp_ids}
    opp_bench = order_bench(opp_bench_pool, opp_squad_map, pred_opp)
    return LineupDecision(
        xi=opp_xi_ids, bench=opp_bench,
        captain=opp_cap, vice_captain=opp_vice,
    )


def _display_lineup(decision: LineupDecision, squad_map: Dict[int, Player]) -> None:
    rows = []
    for pid in decision.xi:
        p = squad_map[pid]
        role = "[C]" if pid == decision.captain else ("[V]" if pid == decision.vice_captain else "")
        rows.append({"Pos": p.position, "Name": p.name, "Role": role})
    st.write("**Starting XI**")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    bench_rows = []
    for i, pid in enumerate(decision.bench[:3], 1):
        p = squad_map[pid]
        bench_rows.append({"Priority": i, "Pos": p.position, "Name": p.name})
    gk = squad_map[decision.bench[3]]
    bench_rows.append({"Priority": "GK", "Pos": gk.position, "Name": gk.name})
    st.write("**Bench**")
    st.dataframe(pd.DataFrame(bench_rows), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    c1.metric("Captain", squad_map[decision.captain].name)
    c2.metric("Vice-captain", squad_map[decision.vice_captain].name)


def _display_stats(stats) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("P(win)", f"{stats.p_win:.1%}")
    c2.metric("E[margin]", f"{stats.exp_margin:+.1f}")
    c3.metric("q10", f"{stats.q10:+.1f}")
    c4.metric("q50", f"{stats.q50:+.1f}")
    c5.metric("q90", f"{stats.q90:+.1f}")


# ---------------------------------------------------------------------------
# Position-split squad selector
# ---------------------------------------------------------------------------

def _position_squad_selector(
    key_prefix: str,
    player_info: Dict[int, dict],
    label: str = "squad",
    default_ids: Optional[List[int]] = None,
) -> List[int]:
    """
    Render 4 multiselects (one per position) and return selected element IDs.

    Parameters
    ----------
    key_prefix : str
        Unique Streamlit key prefix (e.g. "my_squad" or "opp_squad").
    player_info : dict
        Mapping from element id → {name, position, team}.
    label : str
        Display label (shown in count validation messages).
    default_ids : list of int, optional
        Pre-select these IDs (used for Auto A loaded squad).
    """
    st.caption(f"Select exactly: 2 GK · 5 DEF · 5 MID · 3 FWD = 15 players")

    selected: List[int] = []
    for pos, quota in POSITION_QUOTAS.items():
        # Candidates for this position
        candidates = sorted(
            [pid for pid, info in player_info.items() if info["position"] == pos],
            key=lambda pid: player_info[pid]["name"],
        )
        defaults = []
        if default_ids:
            defaults = [pid for pid in default_ids if pid in candidates]

        chosen = st.multiselect(
            f"{pos} (select {quota})",
            options=candidates,
            default=defaults if defaults else None,
            format_func=lambda pid: _label(pid, player_info),
            max_selections=quota,
            key=f"{key_prefix}_{pos}",
        )
        count = len(chosen)
        if count == quota:
            st.caption(f"✅ {pos}: {count}/{quota}")
        elif count > 0:
            st.warning(f"You selected {count} {pos}, need exactly {quota}")
        selected.extend(chosen)

    return selected


def _validate_squad(ids: List[int], player_info: Dict[int, dict]) -> Optional[str]:
    """Return error string if squad is invalid, else None."""
    if len(ids) != 15:
        return f"Squad has {len(ids)} players, need exactly 15"
    counts: Dict[str, int] = {}
    for pid in ids:
        pos = player_info.get(pid, {}).get("position", "MID")
        counts[pos] = counts.get(pos, 0) + 1
    for pos, quota in POSITION_QUOTAS.items():
        if counts.get(pos, 0) != quota:
            return f"Need exactly {quota} {pos}, have {counts.get(pos, 0)}"
    return None


def _validate_opp_xi(
    opp_xi_ids: List[int],
    opp_ids: List[int],
    opp_cap: Optional[int],
    player_info: Dict[int, dict],
) -> Optional[str]:
    """Return error string or None."""
    if len(opp_xi_ids) != 11:
        return f"Opponent XI must have exactly 11 players (have {len(opp_xi_ids)})"
    # Must be subset of opponent squad
    opp_set = set(opp_ids)
    bad = [pid for pid in opp_xi_ids if pid not in opp_set]
    if bad:
        return f"Opp XI contains players not in opp squad: {bad}"
    # Formation check
    tmp_map = build_squad_map(opp_xi_ids, player_info)
    if not is_valid_xi(opp_xi_ids, tmp_map):
        from src.fpl_matchup.rules import count_positions
        c = count_positions(opp_xi_ids, tmp_map)
        return (
            f"Opponent XI formation invalid: "
            f"GK={c['GK']} DEF={c['DEF']} MID={c['MID']} FWD={c['FWD']} "
            "(need 1 GK, 3–5 DEF, 2–5 MID, 1–3 FWD)"
        )
    if opp_cap is None:
        return "Please select the opponent's captain"
    if opp_cap not in opp_xi_ids:
        return "Opponent captain must be in their XI"
    return None


# ---------------------------------------------------------------------------
# Opponent inputs section (shared across tabs)
# ---------------------------------------------------------------------------

def _opp_section(key_suffix: str, player_info: Dict[int, dict]):
    """Render opponent squad / XI / captain inputs; return (opp_ids, opp_xi_ids, cap, vice)."""
    st.subheader("Opponent squad")
    opp_ids = _position_squad_selector(
        f"opp_squad_{key_suffix}", player_info, label="opponent squad"
    )

    st.subheader("Opponent starting XI")
    opp_xi_pool = opp_ids if len(opp_ids) == 15 else sorted(player_info.keys())
    opp_xi_by_pos = {
        pos: [pid for pid in opp_xi_pool if player_info.get(pid, {}).get("position") == pos]
        for pos in ("GK", "DEF", "MID", "FWD")
    }
    xi_quotas = {"GK": 1, "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
    st.caption("Select exactly 11 players with valid formation (1 GK, 3–5 DEF, 2–5 MID, 1–3 FWD)")
    opp_xi_ids: List[int] = []
    for pos in ("GK", "DEF", "MID", "FWD"):
        cands = opp_xi_by_pos[pos]
        quota_info = xi_quotas[pos]
        max_sel = quota_info if isinstance(quota_info, int) else quota_info[1]
        chosen = st.multiselect(
            f"XI {pos}",
            options=cands,
            format_func=lambda pid: _label(pid, player_info),
            max_selections=max_sel,
            key=f"opp_xi_{key_suffix}_{pos}",
        )
        opp_xi_ids.extend(chosen)

    cap_options = opp_xi_ids if opp_xi_ids else opp_xi_pool
    opp_cap = st.selectbox(
        "Opponent captain",
        options=[None] + cap_options,
        format_func=lambda p: "— select —" if p is None else _label(p, player_info),
        key=f"opp_cap_{key_suffix}",
    )
    opp_vice = st.selectbox(
        "Opponent vice-captain (defaults to captain)",
        options=[None] + cap_options,
        format_func=lambda p: "— same as captain —" if p is None else _label(p, player_info),
        key=f"opp_vice_{key_suffix}",
    )
    return opp_ids, opp_xi_ids, opp_cap, opp_vice or opp_cap


# ---------------------------------------------------------------------------
# Core run-and-display logic
# ---------------------------------------------------------------------------

def _run_optimise(
    my_ids: List[int],
    opp_ids: List[int],
    opp_xi_ids: List[int],
    opp_cap: int,
    opp_vice: int,
    df: pd.DataFrame,
    player_info: Dict[int, dict],
    costs: Optional[Dict[int, int]],
    gw: int,
    risk: str,
    free_transfers: int,
    n_sims: int,
    n_sims_search: int,
    seed: int,
) -> None:
    # Validate
    err = _validate_squad(my_ids, player_info)
    if err:
        st.error(f"My squad: {err}")
        return
    err = _validate_squad(opp_ids, player_info)
    if err:
        st.error(f"Opp squad: {err}")
        return
    err = _validate_opp_xi(opp_xi_ids, opp_ids, opp_cap, player_info)
    if err:
        st.error(err)
        return

    with st.spinner("Building sampler…"):
        sampler = _build_sampler(df, player_info, gw=gw, seed=seed)

    my_squad_map = build_squad_map(my_ids, player_info)
    opp_squad_map = build_squad_map(opp_ids, player_info)
    opp_decision = _build_opp_decision(
        opp_ids, opp_xi_ids, opp_squad_map, opp_cap, opp_vice, sampler
    )

    transferred = False
    if free_transfers > 0:
        with st.spinner(f"Searching {free_transfers} transfer(s)…"):
            universe = [
                Player(pid, info["name"], info["position"], info["team"])
                for pid, info in player_info.items()
            ]
            transfer_results = optimize_transfers(
                my_squad_map=my_squad_map,
                opp_decision=opp_decision,
                opp_squad_map=opp_squad_map,
                sampler=sampler,
                gw=gw,
                free_transfers=free_transfers,
                universe=universe,
                costs=costs,
                risk=risk,
                n_sims_inner=max(200, n_sims_search // 3),
                n_sims_final=n_sims,
                seed=seed,
            )

        if transfer_results:
            st.subheader("Suggested transfers")
            for i, tr in enumerate(transfer_results, 1):
                out_name = player_info.get(tr.transfer_out, {}).get("name", f"#{tr.transfer_out}")
                in_name = player_info.get(tr.transfer_in, {}).get("name", f"#{tr.transfer_in}")
                st.write(f"**Transfer {i}:** OUT {out_name} → IN {in_name}")
                st.caption(f"P(win)={tr.stats.p_win:.1%}  E[margin]={tr.stats.exp_margin:+.1f}")

            last = transfer_results[-1]
            my_squad_map = build_squad_map(last.new_squad_ids, player_info)
            best_decision = last.lineup
            final_stats = last.stats
            transferred = True
        else:
            st.info("No beneficial transfer found; keeping current squad.")

    if not transferred:
        with st.spinner("Optimising lineup…"):
            best_decision, final_stats = optimize_lineup_vs_opponent(
                my_squad_map=my_squad_map,
                opp_decision=opp_decision,
                opp_squad_map=opp_squad_map,
                sampler=sampler,
                gw=gw,
                risk=risk,
                n_sims_search=n_sims_search,
                n_sims_final=n_sims,
                seed=seed,
            )

    st.subheader(f"Recommended lineup — GW {gw}")
    _display_lineup(best_decision, my_squad_map)

    st.subheader("Matchup statistics vs opponent")
    _display_stats(final_stats)
    st.caption(
        f"My E[score]: {final_stats.my_scores.mean():.1f} pts · "
        f"Opp E[score]: {final_stats.opp_scores.mean():.1f} pts · "
        f"std(margin)={final_stats.std_margin:.1f}"
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("⚽ FPL H2H Optimiser")
season = st.sidebar.text_input("Season", "2024-25")
gw = int(st.sidebar.number_input("Gameweek", min_value=1, max_value=38, value=20))
risk = st.sidebar.selectbox("Risk profile", ["balanced", "conservative", "optimistic"])
free_transfers = int(st.sidebar.number_input("Free transfers", min_value=0, max_value=5, value=0))

with st.sidebar.expander("Advanced"):
    n_sims = int(st.number_input("Final sims", min_value=500, max_value=30_000, value=3000, step=500))
    n_sims_search = int(st.number_input("Search sims", min_value=200, max_value=5000, value=800, step=200))
    seed = int(st.number_input("RNG seed", min_value=0, value=42))

# Load data
with st.spinner(f"Loading season {season} data…"):
    try:
        df, player_info = _load_season(season)
        costs = _load_costs(season)
    except FileNotFoundError as exc:
        st.error(f"Data not found: {exc}")
        st.stop()

st.sidebar.caption(
    f"{len(player_info)} players · GWs {df['gw'].min()}–{df['gw'].max()}"
    + (" · budget available" if costs else " · no budget data")
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_manual, tab_auto_a, tab_auto_b = st.tabs(
    ["Manual selection", "Auto A – use saved squad", "Auto B – build from scratch"]
)


# ===========================================================================
# TAB 1: MANUAL
# ===========================================================================
with tab_manual:
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("My squad")
        my_ids = _position_squad_selector("my_manual", player_info, label="my squad")
        if st.button("💾 Save my squad", key="save_manual"):
            MY_SQUAD_FILE.write_text(json.dumps(my_ids), encoding="utf-8")
            st.success(f"Saved {len(my_ids)} players to {MY_SQUAD_FILE.name}")

    with col_right:
        opp_ids_m, opp_xi_m, opp_cap_m, opp_vice_m = _opp_section("manual", player_info)

    if st.button("⚽ Optimise", key="opt_manual", type="primary"):
        _run_optimise(
            my_ids=my_ids,
            opp_ids=opp_ids_m, opp_xi_ids=opp_xi_m,
            opp_cap=opp_cap_m, opp_vice=opp_vice_m or opp_cap_m,
            df=df, player_info=player_info, costs=costs,
            gw=gw, risk=risk, free_transfers=free_transfers,
            n_sims=n_sims, n_sims_search=n_sims_search, seed=seed,
        )


# ===========================================================================
# TAB 2: AUTO A — load saved squad
# ===========================================================================
with tab_auto_a:
    st.info(
        "**Auto A**: loads your saved squad from `my_squad.json` and optimises "
        "lineup (+ suggested transfers if free_transfers > 0) vs the opponent. "
        "Save your squad from the **Manual** tab first."
    )

    saved_ids: Optional[List[int]] = None
    if MY_SQUAD_FILE.exists():
        try:
            raw = json.loads(MY_SQUAD_FILE.read_text())
            saved_ids = [int(x) for x in raw]
            st.success(f"Loaded {len(saved_ids)} players from `{MY_SQUAD_FILE.name}`")
            err = _validate_squad(saved_ids, player_info)
            if err:
                st.warning(f"Saved squad issue: {err}")
            else:
                names = [player_info.get(pid, {}).get("name", str(pid)) for pid in saved_ids]
                with st.expander("Saved squad"):
                    st.write(", ".join(names))
        except Exception as exc:
            st.error(f"Could not parse {MY_SQUAD_FILE}: {exc}")
    else:
        st.warning(f"`{MY_SQUAD_FILE.name}` not found. Use the Manual tab to save your squad.")

    st.divider()
    opp_ids_a, opp_xi_a, opp_cap_a, opp_vice_a = _opp_section("auto_a", player_info)

    if st.button("⚽ Optimise (Auto A)", key="opt_auto_a", type="primary"):
        if saved_ids is None:
            st.error("No saved squad found.")
        else:
            _run_optimise(
                my_ids=saved_ids,
                opp_ids=opp_ids_a, opp_xi_ids=opp_xi_a,
                opp_cap=opp_cap_a, opp_vice=opp_vice_a or opp_cap_a,
                df=df, player_info=player_info, costs=costs,
                gw=gw, risk=risk, free_transfers=free_transfers,
                n_sims=n_sims, n_sims_search=n_sims_search, seed=seed,
            )


# ===========================================================================
# TAB 3: AUTO B — build squad from scratch
# ===========================================================================
with tab_auto_b:
    st.info(
        "**Auto B**: builds the best 15-player squad targeting your opponent "
        "from scratch (2 GK / 5 DEF / 5 MID / 3 FWD, max 3 per team"
        + (", budget constraint applied" if costs else "")
        + ")."
    )

    if costs is not None:
        budget_b = int(st.slider(
            "Budget (× £0.1m, default 1000 = £100m)",
            min_value=800, max_value=1000, value=1000, step=10,
        ))
    else:
        budget_b = 1000
        st.caption("Budget constraint not available (players_raw.csv not found).")

    st.divider()
    opp_ids_b, opp_xi_b, opp_cap_b, opp_vice_b = _opp_section("auto_b", player_info)

    if st.button("⚽ Build & Optimise (Auto B)", key="opt_auto_b", type="primary"):
        err = _validate_squad(opp_ids_b, player_info)
        if err:
            st.error(f"Opp squad: {err}")
        else:
            err2 = _validate_opp_xi(opp_xi_b, opp_ids_b, opp_cap_b, player_info)
            if err2:
                st.error(err2)
            else:
                with st.spinner("Building squad and optimising…"):
                    sampler_b = _build_sampler(df, player_info, gw=gw, seed=seed)
                    opp_squad_map_b = build_squad_map(opp_ids_b, player_info)
                    opp_decision_b = _build_opp_decision(
                        opp_ids_b, opp_xi_b, opp_squad_map_b,
                        opp_cap_b, opp_vice_b or opp_cap_b, sampler_b,
                    )
                    universe_b = [
                        Player(pid, info["name"], info["position"], info["team"])
                        for pid, info in player_info.items()
                    ]
                    try:
                        squad_ids_b, best_dec_b, stats_b = build_squad_vs_opponent(
                            universe=universe_b,
                            sampler=sampler_b,
                            opp_decision=opp_decision_b,
                            opp_squad_map=opp_squad_map_b,
                            gw=gw,
                            risk=risk,
                            costs=costs,
                            budget=budget_b,
                            n_sims_final=n_sims,
                            seed=seed,
                        )
                    except ValueError as exc:
                        st.error(f"Could not build squad: {exc}")
                        st.stop()

                my_squad_map_b = build_squad_map(squad_ids_b, player_info)

                st.subheader("Built squad")
                squad_rows = [
                    {
                        "Pos": player_info[pid]["position"],
                        "Name": player_info[pid]["name"],
                        "Team": player_info[pid]["team"],
                        **({"Cost (£m)": f"{costs[pid]/10:.1f}"} if costs else {}),
                    }
                    for pid in squad_ids_b
                ]
                st.dataframe(pd.DataFrame(squad_rows), use_container_width=True, hide_index=True)
                if costs:
                    total_cost = sum(costs.get(pid, 0) for pid in squad_ids_b)
                    st.caption(f"Total squad cost: £{total_cost/10:.1f}m / £{budget_b/10:.1f}m")

                st.subheader(f"Recommended lineup — GW {gw}")
                _display_lineup(best_dec_b, my_squad_map_b)

                st.subheader("Matchup statistics vs opponent")
                _display_stats(stats_b)
