import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# =============================================================================
# 1. LOAD DATA
# =============================================================================
df = pd.read_csv("Data/active_perfect_understat_enhanced.csv")
df = df.loc[:, ~df.columns.duplicated()]
df = df.drop(columns=[
    "league_id", "player_id", "position_id", "minutes_y",
    "own_goals_y", "game_id", "season_id",
    "game", "venue",
    "ea_index", "loaned_in", "loaned_out",
    "kickoff_time_formatted",
    "id",
], errors="ignore")
df = df.sort_values(["player", "date"]).reset_index(drop=True)

print(f"Duplicate columns : {df.columns[df.columns.duplicated()].tolist()}")
print(f"Duplicate indices : {df.index.duplicated().sum()}")

# =============================================================================
# 2. OPPONENT STRENGTH FEATURES
# Aggregate to team level first to avoid player-level duplicates
# =============================================================================

# One row per team per game
team_level = (
    df.groupby(["team", "date", "opponent", "season_x"], as_index=False)
    .agg(
        goals_scored_team = ("goals_scored", "sum"),
        xg_team           = ("xg",           "sum"),
    )
)

# Build opponent-conceded view by flipping team/opponent
opp_conceded = (
    team_level
    .rename(columns={
        "team"            : "opponent",
        "opponent"        : "team",
        "goals_scored_team": "opp_goals_conceded",
        "xg_team"         : "opp_xg_conceded",
    })
    [["team", "date", "opp_goals_conceded", "opp_xg_conceded"]]
)

# Merge back — now guaranteed one row per (team, date)
team_level = team_level.merge(opp_conceded, on=["team", "date"], how="left")

# Sort and compute rolling opponent form
team_level = team_level.sort_values(["team", "date"]).reset_index(drop=True)
for w in [3, 5]:
    team_level[f"opp_goals_conceded_form_{w}"] = (
        team_level.groupby("team")["opp_goals_conceded"]
        .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
    )
    team_level[f"opp_xg_conceded_form_{w}"] = (
        team_level.groupby("team")["opp_xg_conceded"]
        .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
    )

# Verify no duplicates before merging to player level
assert team_level.duplicated(subset=["team", "date"]).sum() == 0, \
    "Duplicate (team, date) rows in team_level — investigate before merging"

# Merge opponent form onto player-level df via the player's opponent
opp_form_cols = [
    "opp_goals_conceded_form_3", "opp_xg_conceded_form_3",
    "opp_goals_conceded_form_5", "opp_xg_conceded_form_5",
]

df = df.merge(
    team_level[["team", "date"] + opp_form_cols],
    left_on  = ["opponent", "date"],   # player faces opponent → look up opponent's defensive form
    right_on = ["team", "date"],
    how      = "left",
    suffixes = ("", "_drop"),
)
df = df.drop(columns=[c for c in df.columns if c.endswith("_drop")])
df = df.loc[:, ~df.columns.duplicated()].reset_index(drop=True)

print(f"After opp merge — duplicate indices: {df.index.duplicated().sum()}")

# =============================================================================
# 3. SEASON-LEVEL FEATURES
# =============================================================================
season_order = sorted(df["season_x"].unique())
season_rank  = {s: i for i, s in enumerate(season_order)}
df["season_rank"] = df["season_x"].map(season_rank)

season_summary = (
    df.groupby(["player", "season_x", "season_rank"], as_index=False)
    .agg(
        prev_season_pts     = ("total_points", "sum"),
        prev_season_goals   = ("goals_scored",  "sum"),
        prev_season_assists = ("assists_x",      "sum"),
        prev_season_minutes = ("minutes_x",      "sum"),
        prev_season_xg      = ("xg",             "mean"),
        prev_season_xa      = ("xa",             "mean"),
        prev_season_games   = ("played",          "sum"),
    )
)

season_summary["season_rank"] += 1
season_summary = season_summary.drop_duplicates(subset=["player", "season_rank"], keep="first")

df = df.merge(
    season_summary[[
        "player", "season_rank",
        "prev_season_pts", "prev_season_goals", "prev_season_assists",
        "prev_season_minutes", "prev_season_xg", "prev_season_xa",
        "prev_season_games",
    ]],
    on=["player", "season_rank"],
    how="left"
)

df["prev_season_complete"] = (df["prev_season_games"] >= 25).astype(float)
df = df.loc[:, ~df.columns.duplicated()].reset_index(drop=True)

print(f"After season merge — duplicate indices: {df.index.duplicated().sum()}")

# =============================================================================
# 4. FEATURE SET
# =============================================================================
features = [
    # Context
    "is_home", "price", "gameweek",

    # Minutes
    "form_minutes_3", "form_minutes_5", "season_minutes",

    # Attacking
    "form_goals_3", "form_goals_5",
    "form_xg_3", "form_xg_5",
    "form_shots_3", "form_shots_5",

    # Creativity
    "form_assists_3", "form_assists_5",
    "form_xa_3", "form_xa_5",
    "form_key_passes_y_3", "form_key_passes_y_5",

    # Bonus proxies
    "form_bps_3", "form_bps_5",
    "form_influence_3", "form_influence_5",
    "form_creativity_3", "form_creativity_5",

    # Opponent strength
    "opp_xg_conceded_form_3", "opp_xg_conceded_form_5",
    "opp_goals_conceded_form_3", "opp_goals_conceded_form_5",

    # Season totals
    "season_goals", "season_assists", "clean_sheets_season",

    # Previous season summary
    "prev_season_pts", "prev_season_goals", "prev_season_assists",
    "prev_season_minutes", "prev_season_xg", "prev_season_xa",
    "prev_season_games", "prev_season_complete",
]

missing = [f for f in features if f not in df.columns]
if missing:
    raise ValueError(f"Missing feature columns: {missing}")

print(f"Features : {len(features)}")
print(f"Rows     : {len(df):,}")
print(f"Seasons  : {sorted(df['season_x'].unique())}")

# =============================================================================
# 5. ROLLING SEASON-BY-SEASON TRAINING
# =============================================================================
MIN_TRAIN_SEASONS = 2
all_predictions   = []

for i, pred_season in enumerate(season_order):
    if i < MIN_TRAIN_SEASONS:
        continue

    train_seasons = season_order[:i]
    train_df      = df[df["season_x"].isin(train_seasons)]
    test_df       = df[df["season_x"] == pred_season]

    print(f"\n{'='*55}")
    print(f"Predicting : {pred_season}")
    print(f"Training on: {train_seasons}")
    print(f"{'='*55}")

    for pos in ["FWD", "MID", "DEF", "GK"]:
        tr = (
            train_df[train_df["FPL_pos"] == pos]
            [features + ["total_points"]]
            .dropna()
        )
        tr = tr.loc[:, ~tr.columns.duplicated()].reset_index(drop=True)

        te = (
            test_df[test_df["FPL_pos"] == pos]
            [features + ["total_points", "player", "date", "season_x", "FPL_pos", "price"]]
            .dropna(subset=features)
            .copy()
        )
        te = te.loc[:, ~te.columns.duplicated()].reset_index(drop=True)

        if len(tr) < 200 or len(te) == 0:
            print(f"  {pos}: insufficient data, skipping")
            continue

        model = HistGradientBoostingRegressor(
            max_depth     = 6,
            learning_rate = 0.05,
            max_iter      = 300,
            random_state  = 42,
        )
        model.fit(tr[features].values, tr["total_points"].values)

        te["xPts"] = model.predict(te[features].values).round(3)

        mae  = mean_absolute_error(te["total_points"], te["xPts"])
        corr = np.corrcoef(te["xPts"], te["total_points"])[0, 1]
        print(f"  {pos:3s}  |  MAE: {mae:.3f}  |  Corr: {corr:.3f}  |  n={len(te):,}")

        all_predictions.append(
            te[["player", "FPL_pos", "date", "gameweek", "season_x", "xPts", "total_points", "price"]]
        )

# =============================================================================
# 6. COMBINE & EXPORT PREDICTIONS
# =============================================================================
output = pd.concat(all_predictions, ignore_index=True)
output.to_csv("Data/xpts_season_by_season.csv", index=False)
print(f"\nSaved {len(output):,} rows → Data/xpts_season_by_season.csv")

print("\nTop 20 xPts predictions across all seasons:")
print(output.sort_values("xPts", ascending=False).head(20).to_string(index=False))

print("\nPer-season summary:")
print(
    output.groupby("season_x")
    .apply(lambda x: pd.Series({
        "MAE" : mean_absolute_error(x["total_points"], x["xPts"]),
        "Corr": np.corrcoef(x["xPts"], x["total_points"])[0, 1],
        "n"   : len(x),
    }), include_groups=False)
    .round(3)
    .to_string()
)

# =============================================================================
# 7. BEST XI SELECTOR
# =============================================================================
def get_best_xi(df, gameweek, season=None):
    FORMATIONS = [(3,4,3),(3,5,2),(4,4,2),(4,3,3),(5,3,2),(5,4,1)]

    df_gw = df.copy()
    if season:
        df_gw = df_gw[df_gw["season_x"] == season]
    df_gw = df_gw[df_gw["gameweek"] == gameweek].dropna(subset=["xPts"])

    gk = df_gw[df_gw["FPL_pos"] == "GK"].nlargest(1, "xPts")
    if len(gk) == 0:
        print("No GK available")
        return None

    best_team, best_score, best_formation = None, -np.inf, None

    for d, m, f in FORMATIONS:
        team = pd.concat([
            gk,
            df_gw[df_gw["FPL_pos"] == "DEF"].nlargest(d, "xPts"),
            df_gw[df_gw["FPL_pos"] == "MID"].nlargest(m, "xPts"),
            df_gw[df_gw["FPL_pos"] == "FWD"].nlargest(f, "xPts"),
        ])
        if len(team) != 11:
            continue
        score = team["xPts"].sum()
        if score > best_score:
            best_score, best_team, best_formation = score, team, (d, m, f)

    if best_team is None:
        print("No valid team found")
        return None

    captain = best_team.loc[best_team["xPts"].idxmax()]
    total   = best_score + captain["xPts"]

    label = f"GW{gameweek}" + (f" {season}" if season else "")
    print(f"\nBest XI — {label}")
    print(f"Formation : {best_formation[0]}-{best_formation[1]}-{best_formation[2]}")
    print(
        best_team
        .sort_values("xPts", ascending=False)
        [["player", "FPL_pos", "xPts"]]
        .to_string(index=False)
    )
    print(f"\nCaptain   : {captain['player']} ({captain['xPts']:.2f} xPts)")
    print(f"Total xPts: {total:.2f}  (with captain double)")

    return best_team, best_formation, total

# Example usage
get_best_xi(output, gameweek=10, season="2023-24")
get_best_xi(output, gameweek=38, season="2024-25")

# =============================================================================
# 8. BEST XI SELECTOR WITH BUDGET CONSTRAINT
# =============================================================================
def get_best_xi_budget_greedy(df, gameweek, season=None, budget=83.0):
    """
    Select the best XI within a budget cap using a greedy approach
    that iteratively fills each position with the best available player
    that keeps the remaining budget feasible.

    Parameters
    ----------
    df       : DataFrame output from the model (must contain 'price' column)
    gameweek : int
    season   : str or None
    budget   : float — total squad budget in £m (default 83.0, i.e. 100m minus 4 subs)
    """
    FORMATIONS = [(3,4,3),(3,5,2),(4,4,2),(4,3,3),(5,3,2),(5,4,1)]

    # Filter to the right gameweek/season
    df_gw = df.copy()
    if season:
        df_gw = df_gw[df_gw["season_x"] == season]
    df_gw = df_gw[df_gw["gameweek"] == gameweek].dropna(subset=["xPts", "price"])

    # Sort each position group by xPts descending for greedy selection
    by_pos = {pos: df_gw[df_gw["FPL_pos"] == pos].sort_values("xPts", ascending=False)
              for pos in ["GK", "DEF", "MID", "FWD"]}

    def greedy_pick(pos_group, n, budget_remaining, slots_remaining):
        """
        Pick n players from pos_group such that budget_remaining is not
        exceeded AND enough budget is left for the remaining slots.
        Returns selected rows or None if impossible.
        """
        selected = []
        remaining_pool = pos_group.copy()

        for _ in range(n):
            slots_after_this = slots_remaining - 1
            # After picking this player we need at least £4.0m per remaining slot
            min_budget_after = slots_after_this * 4.0

            feasible = remaining_pool[
                remaining_pool["price"] <= (budget_remaining - min_budget_after)
            ]
            if feasible.empty:
                return None

            pick = feasible.iloc[0]  # best xPts within budget
            selected.append(pick)
            budget_remaining -= pick["price"]
            slots_remaining  -= 1
            remaining_pool    = remaining_pool[remaining_pool["player"] != pick["player"]]

        return pd.DataFrame(selected), budget_remaining

    best_team, best_score, best_formation = None, -np.inf, None

    for d, m, f in FORMATIONS:
        slots = [("GK", 1), ("DEF", d), ("MID", m), ("FWD", f)]
        total_slots = 11
        budget_left = budget
        team_parts  = []
        failed      = False

        for pos, n in slots:
            slots_left_after = total_slots - sum(s for _, s in slots[:slots.index((pos,n))+1])
            result = greedy_pick(by_pos[pos], n, budget_left, n + slots_left_after)
            if result is None:
                failed = True
                break
            picked, budget_left = result
            team_parts.append(picked)

        if failed:
            continue

        team  = pd.concat(team_parts)
        if len(team) != 11:
            continue

        score = team["xPts"].sum()
        if score > best_score:
            best_score, best_team, best_formation = score, team, (d, m, f)

    if best_team is None:
        print(f"No valid team found within £{budget}m budget")
        return None

    captain   = best_team.loc[best_team["xPts"].idxmax()]
    total_pts = best_score + captain["xPts"]
    spent     = best_team["price"].sum()

    label = f"GW{gameweek}" + (f" {season}" if season else "")
    print(f"\nBest XI (Budget: £{budget}m) — {label}")
    print(f"Formation  : {best_formation[0]}-{best_formation[1]}-{best_formation[2]}")
    print(
        best_team
        .sort_values("xPts", ascending=False)
        [["player", "FPL_pos", "xPts", "price"]]
        .to_string(index=False)
    )
    print(f"\nCaptain    : {captain['player']} ({captain['xPts']:.2f} xPts)")
    print(f"Total xPts : {total_pts:.2f}  (with captain double)")
    print(f"Spent      : £{spent:.1f}m / £{budget:.1f}m  (£{budget - spent:.1f}m remaining)")

    return best_team, best_formation, total_pts


# Example usage
get_best_xi_budget_greedy(output, gameweek=10, season="2023-24", budget=83.0)
get_best_xi_budget_greedy(output, gameweek=38, season="2024-25", budget=83.0)

# =============================================================================
# 10. BEST XI SELECTOR WITH BUDGET CONSTRAINT (ILP - OPTIMAL)
# =============================================================================
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, value, PULP_CBC_CMD

def get_best_xi_budget(df, gameweek, season=None, budget=83.0):
    """
    Select the globally optimal XI within a budget cap using Integer Linear
    Programming. Guaranteed to find the best possible team unlike greedy.

    Parameters
    ----------
    df       : DataFrame output from the model (must contain 'price' column)
    gameweek : int
    season   : str or None
    budget   : float — total XI budget in £m (default 83.0)
    """
    FORMATIONS = [(3,4,3),(3,5,2),(4,4,2),(4,3,3),(5,3,2),(5,4,1)]

    # Filter to gameweek/season
    df_gw = df.copy()
    if season:
        df_gw = df_gw[df_gw["season_x"] == season]
    df_gw = (
        df_gw[df_gw["gameweek"] == gameweek]
        .dropna(subset=["xPts", "price"])
        .reset_index(drop=True)
    )

    if df_gw.empty:
        print("No players found for this gameweek/season.")
        return None

    players  = df_gw.index.tolist()
    xpts     = df_gw["xPts"].to_dict()
    prices   = df_gw["price"].to_dict()
    pos_map  = df_gw["FPL_pos"].to_dict()

    best_team, best_score, best_formation = None, -np.inf, None

    for (d, m, f) in FORMATIONS:
        prob = LpProblem(f"BestXI_{d}{m}{f}", LpMaximize)

        # Binary variable: 1 if player is selected
        x = {i: LpVariable(f"x_{i}", cat="Binary") for i in players}

        # Objective: maximise total xPts
        prob += lpSum(xpts[i] * x[i] for i in players)

        # Budget constraint
        prob += lpSum(prices[i] * x[i] for i in players) <= budget

        # Exactly 11 players
        prob += lpSum(x[i] for i in players) == 11

        # Positional constraints
        gk_players  = [i for i in players if pos_map[i] == "GK"]
        def_players = [i for i in players if pos_map[i] == "DEF"]
        mid_players = [i for i in players if pos_map[i] == "MID"]
        fwd_players = [i for i in players if pos_map[i] == "FWD"]

        prob += lpSum(x[i] for i in gk_players)  == 1
        prob += lpSum(x[i] for i in def_players) == d
        prob += lpSum(x[i] for i in mid_players) == m
        prob += lpSum(x[i] for i in fwd_players) == f

        # Solve (suppress solver output)
        prob.solve(PULP_CBC_CMD(msg=0))

        # Check if a valid solution was found
        if prob.status != 1:
            continue

        selected  = [i for i in players if value(x[i]) == 1]
        team      = df_gw.loc[selected]
        score     = team["xPts"].sum()

        if score > best_score:
            best_score     = score
            best_team      = team
            best_formation = (d, m, f)

    if best_team is None:
        print(f"No valid team found within £{budget}m budget")
        return None

    captain   = best_team.loc[best_team["xPts"].idxmax()]
    total_pts = best_score + captain["xPts"]
    spent     = best_team["price"].sum()

    label = f"GW{gameweek}" + (f" {season}" if season else "")
    print(f"\nBest XI (Budget: £{budget}m) — {label}")
    print(f"Formation  : {best_formation[0]}-{best_formation[1]}-{best_formation[2]}")
    print(
        best_team
        .sort_values("xPts", ascending=False)
        [["player", "FPL_pos", "xPts", "price"]]
        .to_string(index=False)
    )
    print(f"\nCaptain    : {captain['player']} ({captain['xPts']:.2f} xPts)")
    print(f"Total xPts : {total_pts:.2f}  (with captain double)")
    print(f"Spent      : £{spent:.1f}m / £{budget:.1f}m  (£{budget - spent:.1f}m remaining)")

    return best_team, best_formation, total_pts


def get_actual_best_xi_budget(df, gameweek, season=None, budget=83.0):
    """
    Select the actual best XI for a specific gameweek based on real total_points
    scored, within a budget cap. Use this to compare against xPts predictions.

    Parameters
    ----------
    df       : DataFrame output from the model (must contain 'price' column)
    gameweek : int
    season   : str or None
    budget   : float — total XI budget in £m (default 83.0)
    """
    FORMATIONS = [(3,4,3),(3,5,2),(4,4,2),(4,3,3),(5,3,2),(5,4,1)]

    df_gw = df.copy()
    if season:
        df_gw = df_gw[df_gw["season_x"] == season]
    df_gw = (
        df_gw[df_gw["gameweek"] == gameweek]
        .dropna(subset=["total_points", "price"])
        .reset_index(drop=True)
    )

    if df_gw.empty:
        print("No players found for this gameweek/season.")
        return None

    players  = df_gw.index.tolist()
    points   = df_gw["total_points"].to_dict()
    prices   = df_gw["price"].to_dict()
    pos_map  = df_gw["FPL_pos"].to_dict()

    best_team, best_score, best_formation = None, -np.inf, None

    for (d, m, f) in FORMATIONS:
        prob = LpProblem(f"ActualBestXI_{d}{m}{f}", LpMaximize)

        x = {i: LpVariable(f"x_{i}", cat="Binary") for i in players}

        # Objective: maximise actual total_points
        prob += lpSum(points[i] * x[i] for i in players)

        prob += lpSum(prices[i] * x[i] for i in players) <= budget
        prob += lpSum(x[i] for i in players) == 11

        gk_players  = [i for i in players if pos_map[i] == "GK"]
        def_players = [i for i in players if pos_map[i] == "DEF"]
        mid_players = [i for i in players if pos_map[i] == "MID"]
        fwd_players = [i for i in players if pos_map[i] == "FWD"]

        prob += lpSum(x[i] for i in gk_players)  == 1
        prob += lpSum(x[i] for i in def_players) == d
        prob += lpSum(x[i] for i in mid_players) == m
        prob += lpSum(x[i] for i in fwd_players) == f

        prob.solve(PULP_CBC_CMD(msg=0))

        if prob.status != 1:
            continue

        selected = [i for i in players if value(x[i]) == 1]
        team     = df_gw.loc[selected]
        score    = team["total_points"].sum()

        if score > best_score:
            best_score     = score
            best_team      = team
            best_formation = (d, m, f)

    if best_team is None:
        print(f"No valid team found within £{budget}m budget")
        return None

    captain   = best_team.loc[best_team["total_points"].idxmax()]
    total_pts = best_score + captain["total_points"]
    spent     = best_team["price"].sum()

    label = f"GW{gameweek}" + (f" {season}" if season else "")
    print(f"\nActual Best XI (Budget: £{budget}m) — {label}")
    print(f"Formation  : {best_formation[0]}-{best_formation[1]}-{best_formation[2]}")
    print(
        best_team
        .sort_values("total_points", ascending=False)
        [["player", "FPL_pos", "total_points", "xPts", "price"]]
        .to_string(index=False)
    )
    print(f"\nCaptain    : {captain['player']} ({captain['total_points']} pts)")
    print(f"Total pts  : {total_pts:.0f}  (with captain double)")
    print(f"Spent      : £{spent:.1f}m / £{budget:.1f}m  (£{budget - spent:.1f}m remaining)")

    return best_team, best_formation, total_pts


def compare_actual_vs_predicted(df, gameweek, season=None, budget=83.0):
    """
    Side-by-side comparison of the actual best XI vs the xPts predicted best XI
    for a given gameweek, both within the same budget.
    """
    print(f"\n{'='*55}")
    print(f"PREDICTED best XI (based on xPts)")
    print(f"{'='*55}")
    pred_result   = get_best_xi_budget(df, gameweek, season, budget)

    print(f"\n{'='*55}")
    print(f"ACTUAL best XI (based on total_points)")
    print(f"{'='*55}")
    actual_result = get_actual_best_xi_budget(df, gameweek, season, budget)

    if pred_result is None or actual_result is None:
        return

    pred_team,   _, pred_total   = pred_result
    actual_team, _, actual_total = actual_result

    # How many points did the predicted team actually score?
    predicted_team_actual_pts = (
        df[(df["gameweek"] == gameweek) & (df["season_x"] == season)]
        .set_index("player")["total_points"]
        .reindex(pred_team["player"].values)
        .sum()
    )
    predicted_team_actual_pts += predicted_team_actual_pts  # captain double
    captain_actual = pred_team.loc[pred_team["xPts"].idxmax(), "player"]
    captain_actual_pts = (
        df[(df["gameweek"] == gameweek) & (df["season_x"] == season)]
        .set_index("player")["total_points"]
        .get(captain_actual, 0)
    )
    predicted_team_actual_pts = (
        pred_team.merge(
            df[(df["gameweek"] == gameweek) & (df["season_x"] == season)]
            [["player", "total_points"]],
            on="player", how="left", suffixes=("_pred", "_actual")
        )["total_points_actual"].sum() + captain_actual_pts
    )

    print(f"\n{'='*55}")
    print(f"COMPARISON SUMMARY — GW{gameweek}" + (f" {season}" if season else ""))
    print(f"{'='*55}")
    print(f"Predicted XI actual pts scored : {predicted_team_actual_pts:.0f}")
    print(f"Optimal XI actual pts scored   : {actual_total:.0f}")
    print(f"Points left on table           : {actual_total - predicted_team_actual_pts:.0f}")

    # Players in optimal but not in predicted
    missed = set(actual_team["player"]) - set(pred_team["player"])
    gained = set(pred_team["player"])  - set(actual_team["player"])
    print(f"\nIn optimal XI but NOT in predicted XI : {', '.join(missed) if missed else 'none'}")
    print(f"In predicted XI but NOT in optimal XI : {', '.join(gained) if gained else 'none'}")


# Example
compare_actual_vs_predicted(output, gameweek=10, season="2023-24", budget=83.0)


# Example — best team you could pick going into GW15 of 2023-24
get_best_xi_budget(output, gameweek=10, season="2023-24", budget=83.0)
get_best_xi_budget(output, gameweek=38, season="2024-25", budget=83.0)
get_best_xi_budget(output, gameweek=15, season="2023-24", budget=83.0)
get_best_xi(output, gameweek=15, season="2023-24")
compare_actual_vs_predicted(output, gameweek=15, season="2023-24", budget=83.0)

