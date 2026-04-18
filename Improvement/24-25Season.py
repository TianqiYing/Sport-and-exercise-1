import math
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Parameters
# Use the enhanced dataset (FPL + Understat)
DATA_PATH = "active_perfect_understat_enhanced.csv"

# Research season
SEASON = "2024-25"

# Budget for the best XI only (83.0m). In this dataset, value is in tenths of £m.
BUDGET = 830
FORM_WINDOW = 5
EWMA_ALPHA = 0.6

# Season Monte Carlo: running 1000 sims for every GW is expensive.
# This smaller value still gives a stable frequency ranking.
N_SIM_SEASON = 200
RANDOM_SEED = 42

# Starting XI composition
CONSTRAINTS = {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2}
MAX_PER_TEAM = 3

# Home advantage multipliers (simple baseline)
HOME_MULT = 1.10
AWAY_MULT = 0.90

# Fixture factor stabilisation
FIXTURE_CLIP = (0.70, 1.40)      # hard cap to avoid extreme scaling
FIXTURE_SHRINK = 0.60            # shrinkage toward 1.0 (0=no effect, 1=keep as-is)

# "Ceiling/boom" bonus
BOOM_THRESHOLD = 8               # points threshold for a "haul"
BOOM_WINDOW = 20                 # lookback games for haul probability
BOOM_WEIGHT = 0.6                # points bonus per unit haul-probability

# Plot annotation controls (season risk-return)
ANNOTATE_TOPK = 10
ANNOTATE_MIN_FREQ = 0.15

# Helpers

def make_player_key(df: pd.DataFrame) -> pd.Series:
    """Stable key to identify a unique FPL player in a GW candidate set."""
    # Prefer stable FPL element id if present (prevents hidden duplicates and name collisions)
    if "element" in df.columns:
        return df["element"].astype(str).str.strip()

    return (
        df["name"].astype(str).str.strip()
        + "__"
        + df["team"].astype(str).str.strip()
        + "__"
        + df["position"].astype(str).str.strip()
    )


def load_and_standardize_enhanced_dataset(path: str) -> pd.DataFrame:
    """Load the enhanced dataset and standardise to this script's schema.

    Output columns used by the pipeline:
      season_x, GW, element, name, team_x, opp_team_name, was_home,
      position (GK/DEF/MID/FWD), total_points, value,
      goals_scored, goals_conceded
    """
    raw = pd.read_csv(path, low_memory=False)

    df = raw.rename(
        columns={
            "round": "GW",
            "player": "name",
            "team": "team_x",
            "opponent": "opp_team_name",
        }
    ).copy()


    # minutes: keep a unified column name if present
    if "minutes" not in df.columns:
        if "minutes_x" in df.columns:
            df["minutes"] = df["minutes_x"]
        elif "minutes_y" in df.columns:
            df["minutes"] = df["minutes_y"]
        else:
            df["minutes"] = np.nan

    if "season_x" not in df.columns and "season" in df.columns:
        df["season_x"] = df["season"]

    # Fill FPL position if missing: element-wise mode across all seasons
    if "FPL_pos" in df.columns:
        pos_map = (
            df.dropna(subset=["FPL_pos"])
            .groupby("element")["FPL_pos"]
            .agg(lambda s: s.value_counts().idxmax())
        )
        df["position"] = df["FPL_pos"]
        miss = df["position"].isna()
        df.loc[miss, "position"] = df.loc[miss, "element"].map(pos_map)
    else:
        df["position"] = np.nan

    df["position"] = df["position"].astype(str)
    df.loc[~df["position"].isin(["GK", "DEF", "MID", "FWD"]), "position"] = np.nan

    for c in ["GW", "total_points", "value", "goals_scored", "goals_conceded", "element", "minutes"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    required = [
        "season_x",
        "GW",
        "name",
        "team_x",
        "opp_team_name",
        "was_home",
        "position",
        "total_points",
        "value",
        "goals_scored",
        "goals_conceded",
    ]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise RuntimeError(f"Dataset missing required columns after standardisation: {missing_cols}")

    if "element" not in df.columns:
        df["element"] = np.nan

    df = df.dropna(subset=["season_x", "GW", "name", "team_x", "opp_team_name", "position"]).copy()
    df["GW"] = df["GW"].astype(int)
    for c in ["name", "team_x", "opp_team_name", "position"]:
        df[c] = df[c].astype(str).str.strip()

    return df


def aggregate_player_rows_for_gw(agg_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure each player appears exactly once in the GW candidate table.

    Why this is necessary:
      - The enhanced dataset can contain duplicate rows per player for the target GW
        (e.g., double fixtures, merge duplicates).

    Strategy:
      - Create player_key = name__team__position.
      - If a player has multiple rows in the target GW, SUM expected_points across rows
        (double fixture => additive expectation).
      - Combine std_points conservatively as std * sqrt(n_rows).
      - Concatenate opponent/home info for reporting.
    """
    df = agg_df.copy()
    if "player_key" not in df.columns:
        df["player_key"] = make_player_key(df)

    df["expected_points"] = pd.to_numeric(df["expected_points"], errors="coerce").fillna(0.0)
    df["std_points"] = pd.to_numeric(df["std_points"], errors="coerce").fillna(0.0)

    def _join_unique(x):
        vals = [str(v) for v in x.dropna().tolist()]
        # preserve order but unique
        seen = set()
        out = []
        for v in vals:
            if v not in seen:
                seen.add(v)
                out.append(v)
        return ";".join(out) if out else ""

    g = df.groupby("player_key", as_index=False)
    out = g.agg(
        name=("name", "first"),
        team=("team", "first"),
        position=("position", "first"),
        latest_value=("latest_value", "first"),
        expected_points=("expected_points", "sum"),
        std_points=("std_points", "first"),
        fixture_factor=("fixture_factor", "mean"),
        def_factor=("def_factor", "mean"),
        atk_factor=("atk_factor", "mean"),
        opp_team_name=("opp_team_name", _join_unique),
        was_home=("was_home", _join_unique),
    )

    # inflate risk for multi-row players (e.g., double fixtures)
    counts = df.groupby("player_key").size().rename("n_rows").reset_index()
    out = out.merge(counts, on="player_key", how="left")
    out["n_rows"] = out["n_rows"].fillna(1).astype(int)
    out["std_points"] = out["std_points"] * np.sqrt(out["n_rows"].clip(lower=1))

    # if was_home became 'True;False' etc, keep as string for reporting
    return out

def hybrid_form_expected_points(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Player-level expected base points from historical total_points, using a gap-aware, minutes-filtered EWMA.

    Fixes:
      - Long-term injured / long absence players: do NOT treat DNP as 0 in EWMA.
      - Transfers: output is keyed by player_key (prefer FPL element id).

    Returns: DataFrame with columns [player_key, expected_points_base, last_play_gw]
    """
    df = train_df.sort_values(["GW"]).copy()
    df["player_key"] = make_player_key(df)

    # minutes column. If missing, treat all rows as "played".
    if "minutes" in df.columns:
        played = df[df["minutes"].fillna(0) > 0].copy()
    else:
        played = df.copy()

    if played.empty:
        out = df[["player_key"]].drop_duplicates().copy()
        out["expected_points_base"] = 0.0
        out["last_play_gw"] = np.nan
        return out

    played = played.sort_values(["player_key", "GW"]).copy()

    played["ewma_points"] = played.groupby("player_key")["total_points"].transform(
        lambda x: x.ewm(alpha=EWMA_ALPHA, adjust=False).mean()
    )

    ewma_latest = played.loc[played.groupby("player_key")["GW"].idxmax()][
        ["player_key", "ewma_points", "GW"]
    ].rename(columns={"GW": "last_play_gw"})

    played["gw_rank"] = played.groupby("player_key")["GW"].rank(ascending=False, method="first")
    recent_avg = (
        played[played["gw_rank"] <= FORM_WINDOW]
        .groupby("player_key")["total_points"]
        .mean()
        .reset_index()
        .rename(columns={"total_points": "recent_avg_points"})
    )

    form_df = ewma_latest.merge(recent_avg, on="player_key", how="left")
    form_df["recent_avg_points"] = form_df["recent_avg_points"].fillna(form_df["ewma_points"])

    base = 0.6 * form_df["ewma_points"] + 0.4 * form_df["recent_avg_points"]

    # Mild decay for long gaps (keeps returners from being crushed to ~0)
    GAP_DECAY = 0.97   # per missed GW
    MIN_DECAY = 0.70   # never decay below this multiplier
    current_gw = int(df["GW"].max())
    gap = (current_gw - form_df["last_play_gw"].fillna(current_gw)).clip(lower=0)
    decay = np.power(GAP_DECAY, gap).clip(lower=MIN_DECAY)

    form_df["expected_points_base"] = (base * decay).astype(float)

    return form_df[["player_key", "expected_points_base", "last_play_gw"]]



def latest_price(train_df: pd.DataFrame) -> pd.DataFrame:
    """Use the latest available 'value' in the training history as player price (by player_key)."""
    df = train_df.sort_values(["GW"]).copy()
    df["player_key"] = make_player_key(df)
    lp = df.loc[df.groupby("player_key")["GW"].idxmax()][["player_key", "value"]]
    return lp.rename(columns={"value": "latest_value"})



def build_position_aware_factors(train_df: pd.DataFrame, agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build opponent difficulty factors:
    - def_factor for MID/FWD: opponent goals_conceded / league average
      (opponent concedes more => easier => factor > 1)
    - atk_factor for GK/DEF: league avg goals_scored / opponent goals_scored
      (opponent scores more => harder => factor < 1)

    Requires:
      train_df has team_x, goals_conceded, goals_scored
      agg_df has opp_team_name, position
    """
    # Opponent defence (conceded) - higher conceded => easier for attackers
    team_def = train_df.groupby("team_x")["goals_conceded"].mean()
    avg_def = team_def.mean()
    agg_df["def_factor"] = agg_df["opp_team_name"].map(team_def) / avg_def
    agg_df["def_factor"] = agg_df["def_factor"].fillna(1.0)

    # Opponent attack (scored) - higher scored => harder for defenders/GK
    team_atk = train_df.groupby("team_x")["goals_scored"].mean()
    avg_atk = team_atk.mean()
    opp_atk = agg_df["opp_team_name"].map(team_atk)
    agg_df["atk_factor"] = avg_atk / opp_atk
    agg_df["atk_factor"] = agg_df["atk_factor"].replace([np.inf, -np.inf], np.nan).fillna(1.0)

    def get_factor(row):
        if row["position"] in ("GK", "DEF"):
            return row["atk_factor"]
        return row["def_factor"]

    agg_df["fixture_factor"] = agg_df.apply(get_factor, axis=1)

    # Stabilise: cap extremes then shrink towards 1.0
    lo, hi = FIXTURE_CLIP
    agg_df["fixture_factor"] = agg_df["fixture_factor"].clip(lo, hi)
    agg_df["fixture_factor"] = 1.0 + (agg_df["fixture_factor"] - 1.0) * FIXTURE_SHRINK
    return agg_df


def ceiling_features(train_df: pd.DataFrame) -> pd.DataFrame:
    """Estimate an upside proxy per player on *played* matches only."""
    tmp = train_df.sort_values(["GW"]).copy()
    tmp["player_key"] = make_player_key(tmp)

    if "minutes" in tmp.columns:
        tmp = tmp[tmp["minutes"].fillna(0) > 0].copy()

    tmp = tmp.sort_values(["player_key", "GW"]).copy()
    tmp["gw_rank"] = tmp.groupby("player_key")["GW"].rank(ascending=False, method="first")
    recent = tmp[tmp["gw_rank"] <= BOOM_WINDOW].copy()

    if recent.empty:
        out = tmp[["player_key"]].drop_duplicates().copy()
        out["p_haul"] = 0.0
        out["p90_points"] = 0.0
        return out

    p_haul = (
        recent.assign(is_haul=(recent["total_points"] >= BOOM_THRESHOLD).astype(int))
        .groupby("player_key")["is_haul"]
        .mean()
    )
    p90 = recent.groupby("player_key")["total_points"].quantile(0.90)

    out = pd.DataFrame({"player_key": p_haul.index, "p_haul": p_haul.values, "p90_points": p90.values})
    out["p_haul"] = out["p_haul"].fillna(0.0)
    out["p90_points"] = out["p90_points"].fillna(0.0)
    return out



def availability_features(train_df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Build per-player availability proxies from recent minutes.

    Output columns:
      name, p_play, avg_mins_nonzero, p60, availability, dnp_rate

    - p_play: share of last window games where minutes > 0
    - avg_mins_nonzero: avg minutes when played (minutes>0)
    - p60: share of last window games where minutes >= 60
    - availability: p_play * (0.2 + 0.8 * avg_mins_nonzero/90)
    - dnp_rate: 1 - p_play (for Monte Carlo: chance of 0 points)
    """

    df = train_df.copy()

    # robust minutes column detection
    if "minutes" in df.columns:
        min_col = "minutes"
    elif "minutes_x" in df.columns:
        min_col = "minutes_x"
    else:
        raise KeyError("No minutes column found. Expected 'minutes' or 'minutes_x' in train_df.")

    df[min_col] = pd.to_numeric(df[min_col], errors="coerce").fillna(0.0)

    # robust sorting keys
    sort_cols = []
    if "season_x" in df.columns:
        sort_cols.append("season_x")
    if "season" in df.columns:
        sort_cols.append("season")
    if "GW" in df.columns:
        sort_cols.append("GW")

    if sort_cols:
        df = df.sort_values(["name"] + sort_cols)

    def _calc(g: pd.DataFrame) -> pd.Series:
        recent = g.tail(window)
        mins = recent[min_col].to_numpy(dtype=float)

        p_play = float(np.mean(mins > 0))
        nonzero = mins[mins > 0]
        avg_mins_nonzero = float(nonzero.mean()) if len(nonzero) else 0.0
        p60 = float(np.mean(mins >= 60))

        availability = p_play * (0.2 + 0.8 * (avg_mins_nonzero / 90.0))
        dnp_rate = 1.0 - p_play

        return pd.Series({
            "p_play": p_play,
            "avg_mins_nonzero": avg_mins_nonzero,
            "p60": p60,
            "availability": availability,
            "dnp_rate": dnp_rate
        })

    out = df.groupby("name", as_index=False).apply(_calc).reset_index(drop=True)
    return out


def apply_home_factor(agg_df: pd.DataFrame) -> pd.DataFrame:
    agg_df["home_factor"] = agg_df["was_home"].apply(lambda x: HOME_MULT if bool(x) else AWAY_MULT)
    return agg_df


# ILP solver

def solve_xi_ilp(agg_df: pd.DataFrame, points_col: str) -> pd.DataFrame:
    """
    Solve for the best XI (given CONSTRAINTS, MAX_PER_TEAM, BUDGET) maximizing points_col.
    Tries OR-Tools CP-SAT first, then PuLP. If neither is available, raises.
    """
    agg_df = agg_df.copy()
    # Enforce a unique player identifier, even if upstream still contains duplicates.
    if "player_key" not in agg_df.columns:
        agg_df["player_key"] = make_player_key(agg_df)

    agg_df = agg_df.reset_index(drop=True)
    n = len(agg_df)

    # Try OR-Tools CP-SAT
    try:
        from ortools.sat.python import cp_model
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        # objective
        scale = 1000
        pts = agg_df[points_col].astype(float).fillna(0.0).to_numpy()
        model.Maximize(sum(int(round(pts[i]*scale)) * x[i] for i in range(n)))

        # budget
        costs = agg_df["latest_value"].astype(float).fillna(0.0).to_numpy()
        model.Add(sum(int(round(costs[i])) * x[i] for i in range(n)) <= int(round(BUDGET)))

        # position constraints
        for pos, cnt in CONSTRAINTS.items():
            idx = agg_df.index[agg_df["position"] == pos].tolist()
            model.Add(sum(x[i] for i in idx) == cnt)

        # total players
        model.Add(sum(x) == sum(CONSTRAINTS.values()))

        # team limit
        for team, g in agg_df.groupby("team"):
            idx = g.index.tolist()
            model.Add(sum(x[i] for i in idx) <= MAX_PER_TEAM)

        # player uniqueness (prevents selecting the same player twice if duplicates exist)
        for pk, g in agg_df.groupby("player_key"):
            idx = g.index.tolist()
            model.Add(sum(x[i] for i in idx) <= 1)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0
        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError("No feasible solution found (OR-Tools).")

        selected = [i for i in range(n) if solver.Value(x[i]) == 1]
        return agg_df.loc[selected].copy().sort_values(points_col, ascending=False)

    except Exception:
        pass

    # Try PuLP
    try:
        from pulp import LpMaximize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD

        prob = LpProblem("FPL_XI", LpMaximize)
        x = [LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

        prob += lpSum(float(agg_df.loc[i, points_col]) * x[i] for i in range(n))
        prob += lpSum(float(agg_df.loc[i, "latest_value"]) * x[i] for i in range(n)) <= float(BUDGET)
        prob += lpSum(x) == sum(CONSTRAINTS.values())

        for pos, cnt in CONSTRAINTS.items():
            idx = agg_df.index[agg_df["position"] == pos].tolist()
            prob += lpSum(x[i] for i in idx) == cnt

        for team, g in agg_df.groupby("team"):
            idx = g.index.tolist()
            prob += lpSum(x[i] for i in idx) <= MAX_PER_TEAM

        # player uniqueness
        for pk, g in agg_df.groupby("player_key"):
            idx = g.index.tolist()
            prob += lpSum(x[i] for i in idx) <= 1

        prob.solve(PULP_CBC_CMD(msg=False))

        selected = [i for i in range(n) if x[i].value() == 1]
        return agg_df.loc[selected].copy().sort_values(points_col, ascending=False)

    except ImportError as e:
        raise RuntimeError(
            "No ILP solver available. Install one of:\n"
            "  pip install ortools\n"
            "  pip install pulp\n"
        ) from e


def _safe_series(df: pd.DataFrame, col: str, default=0.0) -> pd.Series:
    """Return numeric series, fill missing with default."""
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    s = pd.to_numeric(df[col], errors="coerce")
    return s.fillna(default)


def plot_improved_model_figures(
    agg_df: pd.DataFrame,
    out_dir: str = "plots",
    deterministic_xi: pd.DataFrame | None = None,
    top_n_mc: int = 30,
) -> dict:

    os.makedirs(out_dir, exist_ok=True)

    df = agg_df.copy()

    # Ensure needed numeric columns exist
    df["expected_points"] = _safe_series(df, "expected_points", 0.0)
    df["std_points"] = _safe_series(df, "std_points", 0.0)
    df["latest_value"] = _safe_series(df, "latest_value", 0.0)
    df["selection_freq"] = _safe_series(df, "selection_freq", 0.0)
    df["fixture_factor"] = _safe_series(df, "fixture_factor", 1.0)

    # Value per cost
    price = df["latest_value"].replace(0, np.nan)
    df["value_per_cost"] = (df["expected_points"] / price).fillna(0.0)

    # For consistent position ordering
    pos_order = ["GK", "DEF", "MID", "FWD"]
    if "position" in df.columns:
        df["position"] = df["position"].astype(str)
    else:
        df["position"] = "UNK"

    saved = {}

    # Risk–return scatter
    plt.figure()
    sc = plt.scatter(
        df["std_points"],
        df["expected_points"],
        c=df["selection_freq"],
        s=20,
        alpha=0.8,
    )
    plt.xlabel("std_points (risk)")
    plt.ylabel("expected_points (return)")
    plt.title("Risk–Return (color = Monte Carlo selection frequency)")
    plt.colorbar(sc, label="selection_freq")

    # highlight deterministic XI if provided
    if deterministic_xi is not None and len(deterministic_xi) > 0 and "name" in deterministic_xi.columns:
        xi_names = set(deterministic_xi["name"].astype(str).tolist())
        mask = df["name"].astype(str).isin(xi_names) if "name" in df.columns else np.zeros(len(df), dtype=bool)
        if mask.any():
            plt.scatter(
                df.loc[mask, "std_points"],
                df.loc[mask, "expected_points"],
                s=60,
                facecolors="none",
                edgecolors="black",
                linewidths=1.5,
                alpha=1.0,
            )
    ANNOTATE_TOPK = 8
    ANNOTATE_MIN_FREQ = 0.15
    cand = df[df["selection_freq"] >= ANNOTATE_MIN_FREQ].sort_values("std_points", ascending=False).head(ANNOTATE_TOPK)
    for _, r in cand.iterrows():
        plt.annotate(
            str(r["name"]),
            (r["std_points"], r["expected_points"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )
    # explicitly highlight Haaland if exists
    haal = df[df["name"].astype(str).str.contains("Haaland", case=False, na=False)]
    if len(haal) > 0:
        r = haal.iloc[0]
        plt.scatter(
            [r["std_points"]],
            [r["expected_points"]],
            s=120,
            facecolors="none",
            edgecolors="red",
            linewidths=2,
            alpha=1.0,
        )
        plt.annotate(
            "Haaland",
            (r["std_points"], r["expected_points"]),
            textcoords="offset points",
            xytext=(8, -10),
            fontsize=10,
        )
    else:
        print("[plot] Haaland not found in df['name'] (check spelling / dataset coverage).")
    p1 = os.path.join(out_dir, "risk_return.png")
    plt.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close()
    saved["risk_return"] = p1

    # 2) Price vs expected points (value)
    plt.figure()
    sc = plt.scatter(
        df["latest_value"],
        df["expected_points"],
        c=df["selection_freq"],
        s=(df["std_points"] * 12).clip(12, 250),
        alpha=0.75,
    )
    plt.xlabel("latest_value (price)")
    plt.ylabel("expected_points")
    plt.title("Price vs Expected Points (size = risk, color = selection_freq)")
    plt.colorbar(sc, label="selection_freq")
    p2 = os.path.join(out_dir, "price_vs_points.png")
    plt.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close()
    saved["price_vs_points"] = p2

    # 3) Value per cost distribution by position
    plt.figure()
    data = []
    labels = []
    for pos in pos_order:
        vals = df.loc[df["position"] == pos, "value_per_cost"].to_numpy()
        if len(vals) > 0:
            data.append(vals)
            labels.append(pos)
    if len(data) == 0:
        data = [df["value_per_cost"].to_numpy()]
        labels = ["ALL"]
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.xlabel("position")
    plt.ylabel("expected_points / price")
    plt.title("Value per cost distribution by position")
    p3 = os.path.join(out_dir, "value_boxplot_by_position.png")
    plt.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close()
    saved["value_boxplot_by_position"] = p3

    # 4) Team × position heatmap (mean expected points)
    if "team" in df.columns:
        pivot = (
            df.pivot_table(index="team", columns="position", values="expected_points", aggfunc="mean")
            .reindex(columns=[c for c in pos_order if c in df["position"].unique()])
        )
        if pivot.shape[0] > 0 and pivot.shape[1] > 0:
            plt.figure(figsize=(7, max(4, 0.22 * len(pivot))))
            plt.imshow(pivot.fillna(0).values, aspect="auto")
            plt.yticks(range(len(pivot.index)), pivot.index)
            plt.xticks(range(len(pivot.columns)), pivot.columns)
            plt.xlabel("position")
            plt.title("Team × position heatmap (mean expected_points)")
            plt.colorbar(label="mean expected_points")
            p4 = os.path.join(out_dir, "team_position_heatmap.png")
            plt.savefig(p4, dpi=200, bbox_inches="tight")
            plt.close()
            saved["team_position_heatmap"] = p4

    # 4b) Opponent defence/attack factor ranking + heatmap
    # Requires: opp_team_name, def_factor, atk_factor
    if all(c in df.columns for c in ["opp_team_name", "def_factor", "atk_factor"]):
        opp = (
            df.groupby("opp_team_name", as_index=False)[["def_factor", "atk_factor"]]
              .mean()
              .rename(columns={"opp_team_name": "opponent"})
        )

        # Rankings:
        # - For attackers (MID/FWD): higher def_factor => opponent concedes more => easier
        opp["attacker_easiness_rank"] = opp["def_factor"].rank(ascending=False, method="min").astype(int)

        # - For defenders (GK/DEF): higher atk_factor => opponent scores less => easier
        # because atk_factor = league_avg_scored / opp_scored
        opp["defender_easiness_rank"] = opp["atk_factor"].rank(ascending=False, method="min").astype(int)

        # Save ranking CSV
        p_rank = os.path.join(out_dir, "opponent_factor_rankings.csv")
        opp.sort_values(["attacker_easiness_rank", "defender_easiness_rank"]).to_csv(p_rank, index=False)
        saved["opponent_factor_rankings_csv"] = p_rank

        # Heatmap: rows=opponent, cols=[def_factor, atk_factor]
        opp_hm = opp.sort_values("attacker_easiness_rank").set_index("opponent")[["def_factor", "atk_factor"]]

        plt.figure(figsize=(6, max(4, 0.25 * len(opp_hm))))
        plt.imshow(opp_hm.values, aspect="auto")
        plt.yticks(range(len(opp_hm.index)), opp_hm.index)
        plt.xticks([0, 1], ["def_factor (attackers)", "atk_factor (defenders)"])
        plt.title("Opponent difficulty factors (higher = easier)")
        plt.colorbar(label="factor")
        p_hm = os.path.join(out_dir, "opponent_factors_heatmap.png")
        plt.savefig(p_hm, dpi=200, bbox_inches="tight")
        plt.close()
        saved["opponent_factors_heatmap"] = p_hm

        # print top/bottom 5
        print("\n=== Easiest opponents for attackers (highest def_factor) ===")
        print(opp.sort_values("def_factor", ascending=False).head(5)[["opponent", "def_factor"]])
        print("\n=== Hardest opponents for attackers (lowest def_factor) ===")
        print(opp.sort_values("def_factor", ascending=True).head(5)[["opponent", "def_factor"]])

        print("\n=== Easiest opponents for defenders/GK (highest atk_factor) ===")
        print(opp.sort_values("atk_factor", ascending=False).head(5)[["opponent", "atk_factor"]])
        print("\n=== Hardest opponents for defenders/GK (lowest atk_factor) ===")
        print(opp.sort_values("atk_factor", ascending=True).head(5)[["opponent", "atk_factor"]])

    # 5) Monte Carlo selection frequency ranking (top N)
    plt.figure(figsize=(8, 8))
    top = df.sort_values("selection_freq", ascending=False).head(top_n_mc)
    names = top["name"].astype(str).tolist() if "name" in top.columns else [f"p{i}" for i in range(len(top))]
    plt.barh(names[::-1], top["selection_freq"].to_numpy()[::-1])
    plt.xlabel("selection_freq")
    plt.title(f"Monte Carlo selection frequency (Top {top_n_mc})")
    p5 = os.path.join(out_dir, "mc_top_barh.png")
    plt.savefig(p5, dpi=200, bbox_inches="tight")
    plt.close()
    saved["mc_top_barh"] = p5

    return saved


def plot_season_figures(
    player_season: pd.DataFrame,
    season_xi: pd.DataFrame,
    mae_pos: pd.DataFrame,
    opponent_rank: pd.DataFrame,
    out_dir: str = "plots_season",
    highlight_name_substr: str = "Haaland",
) -> dict:
    """Season-level plots matching the same *types* as the single-GW report.

    player_season is one row per player (aggregated over the season) and should include:
      name, position, team, latest_value, mean_expected_points, mean_std_points, selection_freq

    season_xi is one row per GW with:
      GW, pred_xi_points, true_xi_points
    """
    os.makedirs(out_dir, exist_ok=True)
    df = player_season.copy()

    # Ensure numeric
    for c in ["mean_expected_points", "mean_std_points", "latest_value", "selection_freq"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Value per cost
    price = df["latest_value"].replace(0, np.nan)
    df["value_per_cost"] = (df["mean_expected_points"] / price).fillna(0.0)

    saved: dict[str, str] = {}

    # 1) Risk–return (season)
    plt.figure()
    sc = plt.scatter(
        df["mean_std_points"],
        df["mean_expected_points"],
        c=df["selection_freq"],
        s=25,
        alpha=0.85,
    )
    plt.xlabel("mean std_points (risk, season avg)")
    plt.ylabel("mean expected_points (return, season avg)")
    plt.title("Season Risk–Return (color = season selection frequency)")
    plt.colorbar(sc, label="selection_freq")

    # Annotate high-risk frequent players
    if "name" in df.columns:
        cand = (
            df[df["selection_freq"] >= ANNOTATE_MIN_FREQ]
            .sort_values("mean_std_points", ascending=False)
            .head(ANNOTATE_TOPK)
        )
        for _, r in cand.iterrows():
            plt.annotate(
                str(r["name"]),
                (float(r["mean_std_points"]), float(r["mean_expected_points"])),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
            )

    p1 = os.path.join(out_dir, "season_risk_return.png")
    plt.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close()
    saved["season_risk_return"] = p1

    # 2) Price vs points (season)
    plt.figure()
    sc = plt.scatter(
        df["latest_value"],
        df["mean_expected_points"],
        c=df["selection_freq"],
        s=(df["mean_std_points"] * 12).clip(12, 260),
        alpha=0.75,
    )
    plt.xlabel("latest_value (price)")
    plt.ylabel("mean expected_points")
    plt.title("Season Price vs Expected Points (size=risk, color=selection_freq)")
    plt.colorbar(sc, label="selection_freq")
    p2 = os.path.join(out_dir, "season_price_vs_points.png")
    plt.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close()
    saved["season_price_vs_points"] = p2

    # 2b) Value distribution by position
    pos_order = ["GK", "DEF", "MID", "FWD"]
    plt.figure()
    data, labels = [], []
    if "position" in df.columns:
        for pos in pos_order:
            vals = df.loc[df["position"] == pos, "value_per_cost"].to_numpy()
            if len(vals) > 0:
                data.append(vals)
                labels.append(pos)
    if not data:
        data = [df["value_per_cost"].to_numpy()]
        labels = ["ALL"]
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.xlabel("position")
    plt.ylabel("mean expected_points / price")
    plt.title("Season value-per-cost distribution by position")
    p3 = os.path.join(out_dir, "season_value_boxplot_by_position.png")
    plt.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close()
    saved["season_value_boxplot_by_position"] = p3

    # 2c) Team × position heatmap (season)
    if "team" in df.columns and "position" in df.columns:
        pivot = df.pivot_table(index="team", columns="position", values="mean_expected_points", aggfunc="mean")
        cols = [c for c in pos_order if c in pivot.columns]
        pivot = pivot.reindex(columns=cols)
        if pivot.shape[0] > 0 and pivot.shape[1] > 0:
            plt.figure(figsize=(7, max(4, 0.22 * len(pivot))))
            plt.imshow(pivot.fillna(0).values, aspect="auto")
            plt.yticks(range(len(pivot.index)), pivot.index)
            plt.xticks(range(len(pivot.columns)), pivot.columns)
            plt.xlabel("position")
            plt.title("Season Team × position heatmap (mean expected_points)")
            plt.colorbar(label="mean expected_points")
            p4 = os.path.join(out_dir, "season_team_position_heatmap.png")
            plt.savefig(p4, dpi=200, bbox_inches="tight")
            plt.close()
            saved["season_team_position_heatmap"] = p4

    # 3) Season selection frequency ranking (top 30)
    if "name" in df.columns:
        top = df.sort_values("selection_freq", ascending=False).head(30)
        plt.figure(figsize=(8, 8))
        plt.barh(top["name"].astype(str).tolist()[::-1], top["selection_freq"].to_numpy()[::-1])
        plt.xlabel("selection_freq")
        plt.title("Season selection frequency (Top 30)")
        p5 = os.path.join(out_dir, "season_selection_freq_top30.png")
        plt.savefig(p5, dpi=200, bbox_inches="tight")
        plt.close()
        saved["season_selection_freq_top30"] = p5

    # 4) Opponent factors heatmap + rankings CSV
    if not opponent_rank.empty:
        # Save ranking CSV
        p_rank = os.path.join(out_dir, "season_opponent_factor_rankings.csv")
        opponent_rank.to_csv(p_rank, index=False)
        saved["season_opponent_factor_rankings_csv"] = p_rank

        hm = opponent_rank.sort_values("attacker_easiness_rank").set_index("opponent")[
            ["def_factor", "atk_factor"]
        ]
        plt.figure(figsize=(6, max(4, 0.25 * len(hm))))
        plt.imshow(hm.fillna(0).values, aspect="auto")
        plt.yticks(range(len(hm.index)), hm.index)
        plt.xticks([0, 1], ["def_factor (attackers)", "atk_factor (defenders)"])
        plt.title("Season opponent factors (higher = easier)")
        plt.colorbar(label="factor")
        p6 = os.path.join(out_dir, "season_opponent_factors_heatmap.png")
        plt.savefig(p6, dpi=200, bbox_inches="tight")
        plt.close()
        saved["season_opponent_factors_heatmap"] = p6

    # Season team-score curve
    if not season_xi.empty and "GW" in season_xi.columns:
        plt.figure()
        plt.plot(season_xi["GW"], season_xi["true_xi_points"], marker="o")
        plt.xlabel("GW")
        plt.ylabel("true XI points")
        plt.title("Season team score by GW (true XI points)")
        p7 = os.path.join(out_dir, "season_team_score_curve.png")
        plt.savefig(p7, dpi=200, bbox_inches="tight")
        plt.close()
        saved["season_team_score_curve"] = p7

    # Positional MAE bar
    if not mae_pos.empty and "position" in mae_pos.columns:
        plt.figure()
        plt.bar(mae_pos["position"].astype(str), mae_pos["MAE"].astype(float))
        plt.xlabel("position")
        plt.ylabel("MAE")
        plt.title("Positional MAE (walk-forward over season)")
        p8 = os.path.join(out_dir, "season_positional_mae.png")
        plt.savefig(p8, dpi=200, bbox_inches="tight")
        plt.close()
        saved["season_positional_mae"] = p8

    return saved


def build_predictions_for_gw(df_season: pd.DataFrame, target_gw: int) -> tuple[pd.DataFrame, pd.Series]:
    """Walk-forward prediction builder for a single GW.

    Returns:
      agg_df: one row per player_key with expected_points/std/etc for that GW
      true_points_by_key: aggregated true points for that GW (sum over fixtures)
    """
    train_df = df_season[df_season["GW"] < target_gw].copy()
    gw_df = df_season[df_season["GW"] == target_gw].copy()
    if train_df.empty or gw_df.empty:
        raise RuntimeError(f"Empty train/gw split at GW={target_gw}.")

    # base expected points from form (player_key-based, minutes-filtered EWMA)
    form_df = hybrid_form_expected_points(train_df)
    price_df = latest_price(train_df)
    ceil_df = ceiling_features(train_df)

    # Use the GW fixtures as the *base table* (prevents dropping winter transfers).
    fixtures = gw_df[["name", "team_x", "position", "opp_team_name", "was_home", "element"]].rename(columns={"team_x": "team"}).copy()
    fixtures["player_key"] = make_player_key(fixtures)

    # Merge everything onto fixtures by player_key (LEFT joins keep transferred players).
    agg_df = (
        fixtures.merge(form_df, on="player_key", how="left")
        .merge(price_df, on="player_key", how="left")
        .merge(ceil_df, on="player_key", how="left")
    )
    agg_df["p_haul"] = agg_df["p_haul"].fillna(0.0)
    agg_df["p90_points"] = agg_df["p90_points"].fillna(0.0)

    # If a player has no history yet (early season), fall back to 0 expected points.
    agg_df["expected_points_base"] = agg_df["expected_points_base"].fillna(0.0)
    agg_df["latest_value"] = agg_df["latest_value"].fillna(9999)  # effectively unselectable if missing price

    agg_df = build_position_aware_factors(train_df, agg_df)
    agg_df = apply_home_factor(agg_df)

    base_scaled = agg_df["expected_points_base"] * agg_df["fixture_factor"] * agg_df["home_factor"]
    # Upside bonus (avoid double-counting): scale only *excess* above the base
    boom_bonus = BOOM_WEIGHT * agg_df["p_haul"] * (agg_df["p90_points"] - base_scaled).clip(lower=0.0)
    agg_df["expected_points"] = base_scaled + boom_bonus

    # std proxy from recent window
    tmp = train_df.sort_values(["GW"]).copy()
    tmp["player_key"] = make_player_key(tmp)
    if "minutes" in tmp.columns:
        tmp = tmp[tmp["minutes"].fillna(0) > 0].copy()

    tmp["gw_rank"] = tmp.groupby("player_key")["GW"].rank(ascending=False, method="first")
    recent = tmp[tmp["gw_rank"] <= FORM_WINDOW].copy()
    std_map = recent.groupby("player_key")["total_points"].std(ddof=0)
    agg_df["std_points"] = agg_df["player_key"].map(std_map).fillna(0.0)

    # deduplicate (double fixtures)
    agg_df = aggregate_player_rows_for_gw(agg_df)

    # true points per player_key for this GW (sum if double fixtures)
    # Use element if available for stable key; fall back to name__team.
    if "element" in gw_df.columns:
        gw_df = gw_df.copy()
        gw_df["player_key"] = gw_df["element"].astype("Int64").astype(str)
    else:
        gw_df = gw_df.copy()
        gw_df["player_key"] = gw_df["name"].astype(str) + "__" + gw_df["team_x"].astype(str)
    true_points_by_key = gw_df.groupby("player_key")["total_points"].sum()

    return agg_df, true_points_by_key


# Main pipeline

def main():
    rng = np.random.default_rng(RANDOM_SEED)

    # Load enhanced dataset and standardise schema
    df = load_and_standardize_enhanced_dataset(DATA_PATH)
    df = df[df["season_x"] == SEASON].copy()

    # Season (24/25) walk-forward evaluation
    gws = sorted(df["GW"].unique().tolist())
    if len(gws) >= 2:
        min_gw, max_gw = min(gws), max(gws)
        start_gw = min_gw + 1  # need at least one GW of history

        season_rows = []
        abs_err_rows = []

        # Season-level player aggregation for "season plots/tables"
        player_sum_exp: dict[str, float] = {}
        player_sum_std: dict[str, float] = {}
        player_obs: dict[str, int] = {}
        player_meta: dict[str, dict] = {}

        # deterministic XI selection count per GW
        det_select_count: dict[str, int] = {}

        # Monte Carlo selection count (season-level; per GW use N_SIM_SEASON)
        mc_select_count: dict[str, int] = {}
        mc_total_draws = 0

        for gw in range(start_gw, max_gw + 1):
            try:
                gw_pred, true_pts = build_predictions_for_gw(df, gw)
            except Exception:
                # skip malformed weeks
                continue

            # attach true points for MAE
            gw_pred = gw_pred.copy()
            gw_pred["true_points"] = gw_pred["player_key"].map(true_pts).fillna(0.0)
            gw_pred["abs_err"] = (gw_pred["expected_points"] - gw_pred["true_points"]).abs()
            abs_err_rows.append(gw_pred[["position", "abs_err"]].assign(GW=gw))

            # accumulate per-player season aggregates
            for _, r in gw_pred.iterrows():
                k = str(r["player_key"])
                player_sum_exp[k] = player_sum_exp.get(k, 0.0) + float(r.get("expected_points", 0.0))
                player_sum_std[k] = player_sum_std.get(k, 0.0) + float(r.get("std_points", 0.0))
                player_obs[k] = player_obs.get(k, 0) + 1
                if k not in player_meta:
                    player_meta[k] = {
                        "player_key": k,
                        "name": r.get("name"),
                        "position": r.get("position"),
                        "team": r.get("team"),
                        "latest_value": float(r.get("latest_value", 0.0)),
                    }
                else:
                    # keep the most recent price
                    player_meta[k]["latest_value"] = float(r.get("latest_value", player_meta[k]["latest_value"]))

            # pick best XI under 83m budget using predicted points
            team_gw = solve_xi_ilp(gw_pred, "expected_points")
            pred_team_points = float(team_gw["expected_points"].sum())
            true_team_points = float(team_gw["player_key"].map(true_pts).fillna(0.0).sum())

            # deterministic selection count
            for k in team_gw["player_key"].astype(str).tolist():
                det_select_count[k] = det_select_count.get(k, 0) + 1

            # season Monte Carlo selection frequency (reduced sims per GW)
            for _ in range(N_SIM_SEASON):
                sim_pts = gw_pred["expected_points"].to_numpy(dtype=float) + rng.normal(
                    0.0, gw_pred["std_points"].to_numpy(dtype=float)
                )
                gw_pred["sim_points"] = sim_pts
                team_sim = solve_xi_ilp(gw_pred, "sim_points")
                for k in team_sim["player_key"].astype(str).tolist():
                    mc_select_count[k] = mc_select_count.get(k, 0) + 1
            mc_total_draws += N_SIM_SEASON

            season_rows.append(
                {
                    "GW": gw,
                    "budget": BUDGET,
                    "pred_xi_points": pred_team_points,
                    "true_xi_points": true_team_points,
                }
            )

        season_df = pd.DataFrame(season_rows).sort_values("GW") if season_rows else pd.DataFrame()

        if not season_df.empty:
            season_df.to_csv("season_xi_scores_2425.csv", index=False)
            print("\nSaved: season_xi_scores_2425.csv")
            print(
                f"\nSeason XI score (sum of true XI points over evaluated GWs): {season_df['true_xi_points'].sum():.1f}"
            )

        if abs_err_rows:
            ae = pd.concat(abs_err_rows, ignore_index=True)
            mae_pos = ae.groupby("position")["abs_err"].mean().reset_index().rename(columns={"abs_err": "MAE"})
            mae_pos.to_csv("positional_mae_2425.csv", index=False)
            print("Saved: positional_mae_2425.csv")
            print("\nPositional MAE (walk-forward over season):")
            print(mae_pos.to_string(index=False))

            # Build season-level player summary table (one row per player)
            player_rows = []
            n_gws_eval = len(season_df) if not season_df.empty else 0
            for k, meta in player_meta.items():
                obs = player_obs.get(k, 0)
                if obs <= 0:
                    continue
                row = dict(meta)
                row["mean_expected_points"] = player_sum_exp.get(k, 0.0) / obs
                row["mean_std_points"] = player_sum_std.get(k, 0.0) / obs
                row["det_selection_rate"] = (det_select_count.get(k, 0) / n_gws_eval) if n_gws_eval else 0.0
                row["selection_freq"] = (mc_select_count.get(k, 0) / (mc_total_draws * sum(CONSTRAINTS.values()))) if mc_total_draws else 0.0
                player_rows.append(row)

            player_season = pd.DataFrame(player_rows)
            if not player_season.empty:
                player_season = player_season.sort_values("selection_freq", ascending=False)
                player_season.to_csv("season_player_report_2425.csv", index=False)
                print("Saved: season_player_report_2425.csv")

            # Season-level opponent rankings (computed from the whole season data, not a single GW)
            team_def = df.groupby("team_x")["goals_conceded"].mean()
            team_atk = df.groupby("team_x")["goals_scored"].mean()
            avg_def = float(team_def.mean()) if len(team_def) else 1.0
            avg_atk = float(team_atk.mean()) if len(team_atk) else 1.0
            opponent_rank = pd.DataFrame(
                {
                    "opponent": team_def.index.astype(str),
                    "def_factor": (team_def / avg_def).values,
                    "atk_factor": (avg_atk / team_atk.reindex(team_def.index)).values,
                }
            )
            opponent_rank["attacker_easiness_rank"] = opponent_rank["def_factor"].rank(ascending=False, method="min").astype(int)
            opponent_rank["defender_easiness_rank"] = opponent_rank["atk_factor"].rank(ascending=False, method="min").astype(int)

            # Season plots (all plots/tables become season-level)
            saved = plot_season_figures(
                player_season=player_season,
                season_xi=season_df,
                mae_pos=mae_pos,
                opponent_rank=opponent_rank,
                out_dir="plots_season",
                highlight_name_substr="Haaland",
            )
            if saved:
                print("\nSaved season plots:")
                for k, v in saved.items():
                    print(f" - {k}: {v}")
    else:
        print("\n[season-eval] Not enough GWs in this season to run walk-forward evaluation.")


if __name__ == "__main__":
    main()
