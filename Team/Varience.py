import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests


# -----------------------------
# StatsBomb Open Data endpoints
# -----------------------------
BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
COMPETITIONS_URL = f"{BASE}/competitions.json"


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def read_json_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_file(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def _safe_filename(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _get_json(
    session: requests.Session,
    url: str,
    cache_dir: str,
    timeout: int = 180,
    retries: int = 5,
    backoff: float = 1.8,
    sleep_after: float = 0.05,
):
    """
    Robust JSON fetcher:
      - local cache
      - longer timeout
      - retries with exponential backoff
    """
    ensure_dir(cache_dir)
    cache_path = os.path.join(cache_dir, _safe_filename(url) + ".json")
    if os.path.exists(cache_path):
        return read_json_file(cache_path)

    last_err = None
    for i in range(retries):
        try:
            # stream=True avoids some large-response edge cases; we still read fully
            r = session.get(url, timeout=timeout, stream=True)
            r.raise_for_status()
            text = r.content.decode("utf-8")
            obj = json.loads(text)
            write_json_file(cache_path, obj)
            time.sleep(sleep_after)
            return obj
        except Exception as e:
            last_err = e
            wait = (backoff ** i)
            print(f"[WARN] fetch failed ({i+1}/{retries}) url={url} err={type(e).__name__}: {e} -> retry in {wait:.1f}s")
            time.sleep(wait)

    raise RuntimeError(f"Failed to fetch after retries: {url}") from last_err


def find_competition_season(
    session: requests.Session, cache_dir: str, competition_name: str, season_name: str
) -> Tuple[int, int]:
    comps = _get_json(session, COMPETITIONS_URL, cache_dir=cache_dir)
    matches = []
    for c in comps:
        if str(c.get("competition_name", "")).lower() == competition_name.lower() and \
           str(c.get("season_name", "")).lower() == season_name.lower():
            matches.append((int(c["competition_id"]), int(c["season_id"])))
    if not matches:
        raise ValueError(
            f"Could not find competition='{competition_name}', season='{season_name}'. "
            f"Try a different season_name, or pass --competition-id/--season-id."
        )
    return matches[0]


def load_matches(session: requests.Session, cache_dir: str, competition_id: int, season_id: int) -> List[dict]:
    url = f"{BASE}/matches/{competition_id}/{season_id}.json"
    return _get_json(session, url, cache_dir=cache_dir)


def load_events(session: requests.Session, cache_dir: str, match_id: int) -> List[dict]:
    url = f"{BASE}/events/{match_id}.json"
    return _get_json(session, url, cache_dir=cache_dir)


# -----------------------------
# FPL proxy scoring (no BPS)
# -----------------------------
def map_position_to_fpl(pos_name: str) -> str:
    s = (pos_name or "").lower()
    if "goalkeeper" in s:
        return "GK"
    if "back" in s or "wing back" in s or "centre back" in s or "full back" in s:
        return "DEF"
    if "midfield" in s or "winger" in s or "wide" in s:
        return "MID"
    if "forward" in s or "striker" in s:
        return "FWD"
    return "MID"


def fpl_points_proxy(
    position: str,
    minutes: int,
    goals: int,
    assists: int,
    goals_conceded: int,
    yellow: int,
    red: int,
    own_goals: int,
    pen_miss: int,
    pen_save: int,
    saves: int,
    clean_sheet: bool,
) -> int:
    pts = 0
    if minutes > 0:
        pts += 1
    if minutes >= 60:
        pts += 1

    if position in ("GK", "DEF"):
        pts += 6 * goals
    elif position == "MID":
        pts += 5 * goals
    else:
        pts += 4 * goals

    pts += 3 * assists

    if clean_sheet and minutes >= 60:
        if position in ("GK", "DEF"):
            pts += 4
        elif position == "MID":
            pts += 1

    if position in ("GK", "DEF") and minutes >= 60 and goals_conceded > 0:
        pts -= (goals_conceded // 2)

    if position == "GK":
        pts += 5 * pen_save
        pts += (saves // 3)

    pts -= 1 * yellow
    pts -= 3 * red
    pts -= 2 * own_goals
    pts -= 2 * pen_miss
    return int(pts)


# -----------------------------
# Event extraction helpers
# -----------------------------
def build_minutes_from_subs(events: List[dict]) -> Dict[int, int]:
    max_minute = 90
    for e in events:
        m = e.get("minute")
        if isinstance(m, int):
            max_minute = max(max_minute, m)

    starters = set()
    for e in events:
        if e.get("type", {}).get("name") == "Starting XI":
            lineup = e.get("tactics", {}).get("lineup", [])
            for p in lineup:
                pid = p.get("player", {}).get("id")
                if pid is not None:
                    starters.add(int(pid))

    start_min = {pid: 0 for pid in starters}
    end_min = {pid: max_minute for pid in starters}

    for e in events:
        if e.get("type", {}).get("name") == "Substitution":
            minute = int(e.get("minute", 0))
            off_id = e.get("player", {}).get("id")
            on_id = e.get("substitution", {}).get("replacement", {}).get("id")
            if off_id is not None:
                off_id = int(off_id)
                end_min[off_id] = min(end_min.get(off_id, max_minute), minute)
                start_min.setdefault(off_id, 0)
            if on_id is not None:
                on_id = int(on_id)
                start_min[on_id] = minute
                end_min[on_id] = max_minute

    minutes = {}
    for pid, s in start_min.items():
        e = end_min.get(pid, max_minute)
        minutes[pid] = max(0, int(e) - int(s))
    return minutes


def parse_player_events(events: List[dict]) -> Dict[int, dict]:
    pass_by_id = {}
    for e in events:
        if e.get("type", {}).get("name") == "Pass":
            pass_by_id[e.get("id")] = e

    stats: Dict[int, dict] = {}

    def _ensure(pid: int):
        if pid not in stats:
            stats[pid] = dict(goals=0, assists=0, yellow=0, red=0, own_goals=0, pen_miss=0, pen_save=0, saves=0)

    for e in events:
        etype = e.get("type", {}).get("name")

        if etype == "Shot":
            shooter = e.get("player", {}).get("id")
            if shooter is not None:
                shooter = int(shooter)
                _ensure(shooter)

            shot = e.get("shot", {}) or {}
            outcome = (shot.get("outcome", {}) or {}).get("name", "").lower()
            stype = (shot.get("type", {}) or {}).get("name", "").lower()

            if outcome == "goal":
                if shooter is not None:
                    stats[shooter]["goals"] += 1
                key_pass_id = shot.get("key_pass_id")
                if key_pass_id in pass_by_id:
                    passer = pass_by_id[key_pass_id].get("player", {}).get("id")
                    if passer is not None:
                        passer = int(passer)
                        _ensure(passer)
                        stats[passer]["assists"] += 1

            if "penalty" in stype and outcome != "goal" and shooter is not None:
                stats[shooter]["pen_miss"] += 1

        if etype in ("Foul Committed", "Bad Behaviour"):
            pid = e.get("player", {}).get("id")
            if pid is None:
                continue
            pid = int(pid)
            _ensure(pid)
            if etype == "Foul Committed":
                card = (e.get("foul_committed", {}) or {}).get("card", {})
            else:
                card = (e.get("bad_behaviour", {}) or {}).get("card", {})
            cname = (card.get("name") or "").lower() if card else ""
            if "yellow" in cname:
                stats[pid]["yellow"] += 1
            if "red" in cname:
                stats[pid]["red"] += 1

        if etype == "Goal Keeper":
            pid = e.get("player", {}).get("id")
            if pid is None:
                continue
            pid = int(pid)
            _ensure(pid)
            gk = e.get("goalkeeper", {}) or {}
            gk_type = (gk.get("type", {}) or {}).get("name", "").lower()
            if "save" in gk_type:
                stats[pid]["saves"] += 1
            if "penalty" in gk_type and "save" in gk_type:
                stats[pid]["pen_save"] += 1

    return stats


# -----------------------------
# Rolling distribution
# -----------------------------
def rolling_distribution_features(s: pd.Series, window: int, prefix: str) -> pd.DataFrame:
    s_shift = s.shift(1)
    rp = s_shift.rolling(window, min_periods=1)
    rf = s_shift.rolling(window, min_periods=window)
    return pd.DataFrame({
        f"{prefix}mean_roll{window}_partial": rp.mean(),
        f"{prefix}std_roll{window}_partial": rp.std(ddof=0),
        f"{prefix}q10_roll{window}_partial": rp.quantile(0.10),
        f"{prefix}q50_roll{window}_partial": rp.quantile(0.50),
        f"{prefix}q90_roll{window}_partial": rp.quantile(0.90),

        f"{prefix}mean_roll{window}_full": rf.mean(),
        f"{prefix}std_roll{window}_full": rf.std(ddof=0),
        f"{prefix}q10_roll{window}_full": rf.quantile(0.10),
        f"{prefix}q50_roll{window}_full": rf.quantile(0.50),
        f"{prefix}q90_roll{window}_full": rf.quantile(0.90),
    })


def add_player_rolling(df: pd.DataFrame, windows: Tuple[int, ...]) -> pd.DataFrame:
    df = df.copy()
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values(["player_id", "match_date"])
    parts = [df]
    for w in windows:
        parts.append(df.groupby("player_id", group_keys=False)["points"].apply(lambda s: rolling_distribution_features(s, w, "pts_")))
        parts.append(df.groupby("player_id", group_keys=False)["minutes"].apply(lambda s: rolling_distribution_features(s, w, "min_")))
    return pd.concat(parts, axis=1)


def build_leaderboards(df_roll: pd.DataFrame, window: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    latest = df_roll.sort_values(["player_id", "match_date"]).groupby("player_id").tail(1).copy()
    latest["risk_spread"] = latest[f"pts_q90_roll{window}_full"] - latest[f"pts_q10_roll{window}_full"]
    latest["pts_mean"] = latest[f"pts_mean_roll{window}_full"]
    latest["pts_q10"] = latest[f"pts_q10_roll{window}_full"]
    latest["pts_q90"] = latest[f"pts_q90_roll{window}_full"]

    consistent = latest.dropna(subset=["pts_q10", "risk_spread"]).sort_values(
        ["pts_q10", "risk_spread"], ascending=[False, True]
    ).head(50)

    high_risk = latest.dropna(subset=["pts_q90", "risk_spread"]).sort_values(
        ["pts_q90", "risk_spread"], ascending=[False, False]
    ).head(50)

    cols = ["player_name", "position", "team", "pts_mean", "pts_q10", "pts_q90", "risk_spread", "minutes"]
    return consistent[cols], high_risk[cols]


# -----------------------------
# Main
# -----------------------------
@dataclass
class Config:
    outdir: str
    cache_dir: str
    windows: Tuple[int, ...]
    competition_name: str
    season_name: str
    competition_id: Optional[int]
    season_id: Optional[int]
    max_matches: Optional[int]  # for quick test


def main(cfg: Config) -> None:
    ensure_dir(cfg.outdir)
    ensure_dir(cfg.cache_dir)

    session = requests.Session()
    session.headers.update({"User-Agent": "MDM3C-Variance/1.0"})

    if cfg.competition_id is None or cfg.season_id is None:
        comp_id, season_id = find_competition_season(session, cfg.cache_dir, cfg.competition_name, cfg.season_name)
    else:
        comp_id, season_id = cfg.competition_id, cfg.season_id

    matches = load_matches(session, cfg.cache_dir, comp_id, season_id)
    if cfg.max_matches is not None:
        matches = matches[: cfg.max_matches]

    rows = []
    for m in matches:
        match_id = int(m["match_id"])
        match_date = (m.get("match_date") or m.get("kick_off") or "")[:10]
        match_week = m.get("match_week")
        home_team = m["home_team"]["home_team_name"]
        away_team = m["away_team"]["away_team_name"]
        home_score = int(m.get("home_score", 0))
        away_score = int(m.get("away_score", 0))

        events = load_events(session, cfg.cache_dir, match_id)
        minutes_map = build_minutes_from_subs(events)
        pstats = parse_player_events(events)

        # positions & names from starting XI
        pos_map: Dict[int, str] = {}
        name_map: Dict[int, str] = {}
        team_map: Dict[int, str] = {}
        homeflag: Dict[int, int] = {}

        for e in events:
            if e.get("type", {}).get("name") == "Starting XI":
                team_name = e.get("team", {}).get("name", "")
                lineup = e.get("tactics", {}).get("lineup", [])
                for p in lineup:
                    pid = p.get("player", {}).get("id")
                    if pid is None:
                        continue
                    pid = int(pid)
                    name_map[pid] = p.get("player", {}).get("name", f"player_{pid}")
                    team_map[pid] = team_name
                    homeflag[pid] = 1 if team_name == home_team else 0
                    pos_map[pid] = map_position_to_fpl((p.get("position") or {}).get("name", ""))

        # include subs (fallback MID)
        for e in events:
            if e.get("type", {}).get("name") == "Substitution":
                team_name = e.get("team", {}).get("name", "")
                repl = (e.get("substitution", {}) or {}).get("replacement", {}) or {}
                pid = repl.get("id")
                if pid is None:
                    continue
                pid = int(pid)
                name_map.setdefault(pid, repl.get("name", f"player_{pid}"))
                team_map.setdefault(pid, team_name)
                homeflag.setdefault(pid, 1 if team_name == home_team else 0)
                pos_map.setdefault(pid, "MID")

        # rows
        for pid, mins in minutes_map.items():
            team = team_map.get(pid)
            if not team:
                continue
            is_home = homeflag.get(pid, 1 if team == home_team else 0)
            opp = away_team if is_home == 1 else home_team
            ga = away_score if team == home_team else home_score
            clean = 1 if (ga == 0 and mins >= 60) else 0

            st = pstats.get(pid, {})
            goals = int(st.get("goals", 0))
            assists = int(st.get("assists", 0))
            yellow = int(st.get("yellow", 0))
            red = int(st.get("red", 0))
            own_goals = int(st.get("own_goals", 0))
            pen_miss = int(st.get("pen_miss", 0))
            pen_save = int(st.get("pen_save", 0))
            saves = int(st.get("saves", 0))

            pos = pos_map.get(pid, "MID")
            pts = fpl_points_proxy(
                position=pos,
                minutes=int(mins),
                goals=goals,
                assists=assists,
                goals_conceded=int(ga),
                yellow=yellow,
                red=red,
                own_goals=own_goals,
                pen_miss=pen_miss,
                pen_save=pen_save,
                saves=saves,
                clean_sheet=bool(clean),
            )

            rows.append({
                "competition": cfg.competition_name,
                "season": cfg.season_name,
                "match_id": match_id,
                "match_date": match_date,
                "match_week": int(match_week) if match_week is not None else None,
                "team": team,
                "opponent": opp,
                "is_home": is_home,
                "player_id": int(pid),
                "player_name": name_map.get(pid, f"player_{pid}"),
                "position": pos,
                "minutes": int(mins),
                "goals": goals,
                "assists": assists,
                "goals_conceded": int(ga),
                "clean_sheet": int(clean),
                "yellow": yellow,
                "red": red,
                "own_goals": own_goals,
                "pen_miss": pen_miss,
                "pen_save": pen_save,
                "saves": saves,
                "points": int(pts),
            })

        print(f"[OK] match_id={match_id} date={match_date} players={len(minutes_map)}")

    df = pd.DataFrame(rows)
    base_path = os.path.join(cfg.outdir, "player_match_points.csv")
    df.to_csv(base_path, index=False)

    df_roll = add_player_rolling(df, cfg.windows)
    roll_path = os.path.join(cfg.outdir, "player_match_with_rolling.csv")
    df_roll.to_csv(roll_path, index=False)

    w0 = cfg.windows[0]
    cons, risk = build_leaderboards(df_roll, w0)
    cons.to_csv(os.path.join(cfg.outdir, f"leaderboard_consistent_roll{w0}.csv"), index=False)
    risk.to_csv(os.path.join(cfg.outdir, f"leaderboard_highrisk_roll{w0}.csv"), index=False)

    print("Done.")
    print("Saved:", base_path)
    print("Saved:", roll_path)
    print("Tip: First run with --max-matches 5 to warm cache; then remove it.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs_player_variance", help="Output directory")
    ap.add_argument("--cache-dir", default="cache_statsbomb", help="Cache directory for downloaded JSON")
    ap.add_argument("--windows", default="5,8", help="Rolling windows, e.g. 5,8,10")
    ap.add_argument("--competition-name", default="Premier League", help="Competition name in competitions.json")
    ap.add_argument("--season-name", default="2015/2016", help="Season name in competitions.json")
    ap.add_argument("--competition-id", type=int, default=None)
    ap.add_argument("--season-id", type=int, default=None)
    ap.add_argument("--max-matches", type=int, default=5, help="Limit matches for quick test (warm cache)")
    args = ap.parse_args()

    windows = tuple(int(x.strip()) for x in args.windows.split(",") if x.strip())
    cfg = Config(
        outdir=args.outdir,
        cache_dir=args.cache_dir,
        windows=windows,
        competition_name=args.competition_name,
        season_name=args.season_name,
        competition_id=args.competition_id,
        season_id=args.season_id,
        max_matches=args.max_matches,
    )
    main(cfg)