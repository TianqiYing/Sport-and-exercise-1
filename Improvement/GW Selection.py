import math
import numpy as np
import pandas as pd

# Parameters
SEASON = "2023-24"
TARGET_GW = 30
BUDGET = 1000
FORM_WINDOW = 5
EWMA_ALPHA = 0.6
N_SIM = 1000
RANDOM_SEED = 42

# Starting XI composition (example)
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
BOOM_WEIGHT = 2.0                # points bonus per unit haul-probability

# Helpers

def hybrid_form_expected_points(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build player-level expected base points from historical total_points:
    expected_points = 0.6 * EWMA + 0.4 * last-N mean
    Returns dataframe with columns: name, expected_points_base
    """
    df = train_df.sort_values(["name", "GW"]).copy()

    df["ewma_points"] = df.groupby("name")["total_points"].transform(
        lambda x: x.ewm(alpha=EWMA_ALPHA).mean()
    )

    ewma_latest = df.loc[df.groupby("name")["GW"].idxmax()][["name", "ewma_points"]]

    df["gw_rank"] = df.groupby("name")["GW"].rank(ascending=False, method="first")
    recent_avg = (
        df[df["gw_rank"] <= FORM_WINDOW]
        .groupby("name")["total_points"]
        .mean()
        .reset_index()
        .rename(columns={"total_points": "recent_avg_points"})
    )

    form_df = ewma_latest.merge(recent_avg, on="name", how="left")
    form_df["recent_avg_points"] = form_df["recent_avg_points"].fillna(form_df["ewma_points"])

    form_df["expected_points_base"] = 0.6 * form_df["ewma_points"] + 0.4 * form_df["recent_avg_points"]
    return form_df[["name", "expected_points_base"]]


def latest_price(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Use latest 'value' before TARGET_GW as player price.
    """
    lp = train_df.loc[train_df.groupby("name")["GW"].idxmax()][["name", "value"]]
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
    """Estimate a simple "boom" probability per player (chance to exceed BOOM_THRESHOLD).

    This helps high-upside players whose mean can look similar to cheaper options.
    """
    tmp = train_df.sort_values(["name", "GW"]).copy()
    tmp["gw_rank"] = tmp.groupby("name")["GW"].rank(ascending=False, method="first")
    recent = tmp[tmp["gw_rank"] <= BOOM_WINDOW].copy()

    # probability of a haul (>= threshold) in recent window
    p_haul = (
        recent.assign(is_haul=(recent["total_points"] >= BOOM_THRESHOLD).astype(int))
        .groupby("name")["is_haul"]
        .mean()
    )

    # 90th percentile as another proxy for ceiling
    p90 = recent.groupby("name")["total_points"].quantile(0.90)

    out = pd.DataFrame({"name": p_haul.index, "p_haul": p_haul.values, "p90_points": p90.values})
    out["p_haul"] = out["p_haul"].fillna(0.0)
    out["p90_points"] = out["p90_points"].fillna(0.0)
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

        prob.solve(PULP_CBC_CMD(msg=False))

        selected = [i for i in range(n) if x[i].value() == 1]
        return agg_df.loc[selected].copy().sort_values(points_col, ascending=False)

    except ImportError as e:
        raise RuntimeError(
            "No ILP solver available. Install one of:\n"
            "  pip install ortools\n"
            "  pip install pulp\n"
        ) from e


# Main pipeline

def main():
    rng = np.random.default_rng(RANDOM_SEED)

    df = pd.read_csv("cleaned_merged_seasons_team_aggregated.csv", encoding="latin1", low_memory=False)
    df = df[df["season_x"] == SEASON].copy()

    train_df = df[df["GW"] < TARGET_GW].copy()
    gw_df = df[df["GW"] == TARGET_GW].copy()
    if gw_df.empty:
        raise RuntimeError(f"No rows for SEASON={SEASON}, TARGET_GW={TARGET_GW}. Check your data.")

    # base expected points from form
    form_df = hybrid_form_expected_points(train_df)
    price_df = latest_price(train_df)
    ceil_df = ceiling_features(train_df)

    # fixture info for TARGET_GW
    fixtures = gw_df[["name", "team_x", "position", "opp_team_name", "was_home"]].rename(columns={"team_x": "team"})

    # merge into agg_df
    agg_df = (
        form_df.merge(price_df, on="name", how="inner")
        .merge(ceil_df, on="name", how="left")
        .merge(fixtures, on="name", how="inner")
    )
    agg_df["p_haul"] = agg_df["p_haul"].fillna(0.0)

    # position-aware opponent difficulty + home factor
    agg_df = build_position_aware_factors(train_df, agg_df)
    agg_df = apply_home_factor(agg_df)

    # expected points (mean) + ceiling bonus
    # base mean scaled by fixture/home + bonus for high-upside players
    base_scaled = agg_df["expected_points_base"] * agg_df["fixture_factor"] * agg_df["home_factor"]
    boom_bonus = BOOM_WEIGHT * agg_df["p_haul"]
    agg_df["expected_points"] = base_scaled + boom_bonus

    # estimate per-player std from recent window (simple proxy)
    tmp = train_df.sort_values(["name", "GW"]).copy()
    tmp["gw_rank"] = tmp.groupby("name")["GW"].rank(ascending=False, method="first")
    recent = tmp[tmp["gw_rank"] <= FORM_WINDOW].copy()
    std_map = recent.groupby("name")["total_points"].std(ddof=0)
    agg_df["std_points"] = agg_df["name"].map(std_map).fillna(0.0)

    # Deterministic ILP
    team_det = solve_xi_ilp(agg_df, "expected_points")
    print("\n=== Deterministic XI (max expected points) ===")
    print(team_det[["name","position","team","opp_team_name","was_home","latest_value","expected_points","std_points","fixture_factor"]])

    # Monte Carlo robustness (no lambda)
    selection_count = np.zeros(len(agg_df), dtype=int)

    for _ in range(N_SIM):
        sim_pts = agg_df["expected_points"].to_numpy(dtype=float) + rng.normal(0.0, agg_df["std_points"].to_numpy(dtype=float))
        agg_df["sim_points"] = sim_pts
        team_sim = solve_xi_ilp(agg_df, "sim_points")
        selected_names = set(team_sim["name"].tolist())
        selection_count += agg_df["name"].isin(selected_names).to_numpy(dtype=int)

    agg_df["selection_freq"] = selection_count / N_SIM
    out = agg_df.sort_values("selection_freq", ascending=False)[
        ["name","position","team","latest_value","expected_points","std_points","selection_freq"]
    ]

    print("\n=== Monte Carlo selection frequency (top 30) ===")
    print(out.head(30).to_string(index=False))

    # buckets
    print("\nBuckets:")
    print(">=0.80  core starters:", int((agg_df["selection_freq"] >= 0.80).sum()))
    print("0.40-0.80 valuable/risky:", int(((agg_df["selection_freq"] >= 0.40) & (agg_df["selection_freq"] < 0.80)).sum()))

    out.to_csv("improved_gw_selection_report.csv", index=False)
    print("\nSaved: improved_gw_selection_report.csv")


if __name__ == "__main__":
    main()
