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
        te = (
            test_df[test_df["FPL_pos"] == pos]
            [features + ["total_points", "player", "date", "season_x", "FPL_pos"]]
            .dropna(subset=features)
            .copy()
        )

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
            te[["player", "FPL_pos", "date", "gameweek", "season_x", "xPts", "total_points"]]
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