import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import optuna
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, value, PULP_CBC_CMD

optuna.logging.set_verbosity(optuna.logging.WARNING)

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
# =============================================================================
team_level = (
    df.groupby(["team", "date", "opponent", "season_x"], as_index=False)
    .agg(
        goals_scored_team = ("goals_scored", "sum"),
        xg_team           = ("xg",           "sum"),
    )
)

opp_conceded = (
    team_level
    .rename(columns={
        "team"             : "opponent",
        "opponent"         : "team",
        "goals_scored_team": "opp_goals_conceded",
        "xg_team"          : "opp_xg_conceded",
    })
    [["team", "date", "opp_goals_conceded", "opp_xg_conceded"]]
)

team_level = team_level.merge(opp_conceded, on=["team", "date"], how="left")
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

assert team_level.duplicated(subset=["team", "date"]).sum() == 0, \
    "Duplicate (team, date) rows in team_level"

opp_form_cols = [
    "opp_goals_conceded_form_3", "opp_xg_conceded_form_3",
    "opp_goals_conceded_form_5", "opp_xg_conceded_form_5",
]

df = df.merge(
    team_level[["team", "date"] + opp_form_cols],
    left_on  = ["opponent", "date"],
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
season_summary = season_summary.drop_duplicates(
    subset=["player", "season_rank"], keep="first"
)

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
# 4. TEAM-LEVEL INTER-PLAYER FEATURES
# Captures the collective quality of teammates around each player.
# All rolling windows are shift(1) to avoid leaking current GW data.
# =============================================================================
print("\nBuilding team-level inter-player features...")

# Aggregate per team per game — one row per (team, date)
team_context = (
    df.groupby(["team", "date", "season_x"], as_index=False)
    .agg(
        # Attacking context — how dangerous is this team collectively?
        team_xg_sum_3          = ("form_xg_3",          "sum"),
        team_xg_sum_5          = ("form_xg_5",          "sum"),
        team_shots_sum_3       = ("form_shots_3",        "sum"),
        team_shots_sum_5       = ("form_shots_5",        "sum"),
        team_goals_sum_3       = ("form_goals_3",        "sum"),
        team_goals_sum_5       = ("form_goals_5",        "sum"),

        # Creativity context — how well does this team create chances?
        team_xa_sum_3          = ("form_xa_3",           "sum"),
        team_xa_sum_5          = ("form_xa_5",           "sum"),
        team_creativity_sum_3  = ("form_creativity_3",   "sum"),
        team_creativity_sum_5  = ("form_creativity_5",   "sum"),
        team_key_passes_sum_3  = ("form_key_passes_y_3", "sum"),
        team_key_passes_sum_5  = ("form_key_passes_y_5", "sum"),

        # Influence context — general contribution of the squad
        team_influence_sum_3   = ("form_influence_3",    "sum"),
        team_influence_sum_5   = ("form_influence_5",    "sum"),
        team_bps_sum_3         = ("form_bps_3",          "sum"),
        team_bps_sum_5         = ("form_bps_5",          "sum"),

        # In-form teammates — how many teammates are scoring/assisting?
        teammates_scoring_3    = ("form_goals_3",   lambda x: (x > 0.3).sum()),
        teammates_scoring_5    = ("form_goals_5",   lambda x: (x > 0.3).sum()),
        teammates_assisting_3  = ("form_assists_3", lambda x: (x > 0.2).sum()),
        teammates_assisting_5  = ("form_assists_5", lambda x: (x > 0.2).sum()),

        # Minutes context — how many teammates are getting regular minutes?
        teammates_playing_3    = ("form_minutes_3", lambda x: (x > 45).sum()),
        teammates_playing_5    = ("form_minutes_5", lambda x: (x > 45).sum()),

        # Squad size tracked (useful normalisation denominator)
        squad_size             = ("player", "count"),
    )
)

# Sort and apply rolling shift to avoid leaking current GW into features
team_context = team_context.sort_values(["team", "date"]).reset_index(drop=True)

# Normalise by squad size to make features comparable across teams
# (a team with 15 players tracked vs 10 shouldn't look artificially better)
norm_cols = [
    "team_xg_sum_3", "team_xg_sum_5",
    "team_shots_sum_3", "team_shots_sum_5",
    "team_goals_sum_3", "team_goals_sum_5",
    "team_xa_sum_3", "team_xa_sum_5",
    "team_creativity_sum_3", "team_creativity_sum_5",
    "team_key_passes_sum_3", "team_key_passes_sum_5",
    "team_influence_sum_3", "team_influence_sum_5",
    "team_bps_sum_3", "team_bps_sum_5",
]
for col in norm_cols:
    team_context[f"{col}_per_player"] = (
        team_context[col] / team_context["squad_size"].clip(lower=1)
    )

team_context_cols = (
    [c + "_per_player" for c in norm_cols] +
    [
        "teammates_scoring_3", "teammates_scoring_5",
        "teammates_assisting_3", "teammates_assisting_5",
        "teammates_playing_3", "teammates_playing_5",
    ]
)

assert team_context.duplicated(subset=["team", "date"]).sum() == 0, \
    "Duplicate (team, date) rows in team_context"

# Merge team context onto player level
df = df.merge(
    team_context[["team", "date"] + team_context_cols],
    on     = ["team", "date"],
    how    = "left",
    suffixes = ("", "_drop"),
)
df = df.drop(columns=[c for c in df.columns if c.endswith("_drop")])
df = df.loc[:, ~df.columns.duplicated()].reset_index(drop=True)

print(f"After team context merge — duplicate indices: {df.index.duplicated().sum()}")
print(f"Team context features added: {len(team_context_cols)}")

# =============================================================================
# 5. FEATURE SET
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

    # ── NEW: Team-level inter-player features ─────────────────────────
    # Attacking context (per player normalised)
    "team_xg_sum_3_per_player", "team_xg_sum_5_per_player",
    "team_shots_sum_3_per_player", "team_shots_sum_5_per_player",
    "team_goals_sum_3_per_player", "team_goals_sum_5_per_player",

    # Creativity context
    "team_xa_sum_3_per_player", "team_xa_sum_5_per_player",
    "team_creativity_sum_3_per_player", "team_creativity_sum_5_per_player",
    "team_key_passes_sum_3_per_player", "team_key_passes_sum_5_per_player",

    # Influence context
    "team_influence_sum_3_per_player", "team_influence_sum_5_per_player",
    "team_bps_sum_3_per_player", "team_bps_sum_5_per_player",

    # In-form teammate counts
    "teammates_scoring_3", "teammates_scoring_5",
    "teammates_assisting_3", "teammates_assisting_5",
    "teammates_playing_3", "teammates_playing_5",
]

missing = [f for f in features if f not in df.columns]
if missing:
    raise ValueError(f"Missing feature columns: {missing}")

print(f"Features : {len(features)}")
print(f"Rows     : {len(df):,}")
print(f"Seasons  : {sorted(df['season_x'].unique())}")

# =============================================================================
# 6. CUSTOM METRICS
# =============================================================================
def rank_weighted_mae(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ranks  = pd.Series(y_true).rank(pct=True).values
    return np.mean(np.abs(y_true - y_pred) * ranks)


def topk_recall(y_true, y_pred, k=11):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    k      = min(k, len(y_true))
    return len(set(np.argsort(y_true)[-k:]) & set(np.argsort(y_pred)[-k:])) / k


def topk_recall_by_gameweek(df_with_preds, k=11):
    recalls = []
    for _, gw_df in df_with_preds.groupby("gameweek"):
        if len(gw_df) < k:
            continue
        recalls.append(topk_recall(
            gw_df["total_points"].values,
            gw_df["xPts"].values, k=k
        ))
    return np.mean(recalls) if recalls else np.nan


def topk_recall_by_position(df_with_preds):
    pos_k   = {"GK": 1, "DEF": 3, "MID": 4, "FWD": 2}
    results = {}
    for pos, k in pos_k.items():
        pos_df  = df_with_preds[df_with_preds["FPL_pos"] == pos]
        recalls = []
        for _, gw_df in pos_df.groupby("gameweek"):
            if len(gw_df) < k:
                continue
            recalls.append(topk_recall(
                gw_df["total_points"].values,
                gw_df["xPts"].values, k=k
            ))
        results[pos] = round(np.mean(recalls), 3) if recalls else np.nan
    return results

# =============================================================================
# 7. HYPERPARAMETER TUNING WITH OPTUNA (rank-weighted MAE objective)
# =============================================================================
def tune_hyperparams(train_df, pos, features, n_trials=50, val_seasons=2):
    pos_df = (
        train_df[train_df["FPL_pos"] == pos]
        [features + ["total_points", "season_x"]]
        .dropna()
    )

    if len(pos_df) < 200:
        print(f"  {pos}: insufficient data for tuning, using defaults")
        return {}

    all_seasons = sorted(pos_df["season_x"].unique())
    if len(all_seasons) <= val_seasons:
        print(f"  {pos}: not enough seasons to hold out validation fold")
        return {}

    val_szns     = all_seasons[-val_seasons:]
    train_fold   = pos_df[~pos_df["season_x"].isin(val_szns)]
    val_fold     = pos_df[ pos_df["season_x"].isin(val_szns)]

    X_tr,  y_tr  = train_fold[features].values, train_fold["total_points"].values
    X_val, y_val = val_fold[features].values,   val_fold["total_points"].values

    def objective(trial):
        # Replace the params dict inside the Optuna objective with:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "random_state": 42,
            "verbosity": 0,
        }
        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        return rank_weighted_mae(y_val, preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Report diagnostics at best params
    best_model = XGBRegressor(
        **{k: v for k, v in study.best_params.items()},
        random_state=42
    )
    best_model.fit(X_tr, y_tr)
    best_preds = best_model.predict(X_val)

    print(
        f"  {pos}: "
        f"RW-MAE={study.best_value:.3f}  |  "
        f"MAE={mean_absolute_error(y_val, best_preds):.3f}  |  "
        f"Top-11 Recall={topk_recall(y_val, best_preds, k=11):.3f}  |  "
        f"params={study.best_params}"
    )
    return study.best_params


print("\n" + "="*55)
print("HYPERPARAMETER TUNING (rank-weighted MAE objective)")
print("="*55)

tune_train_df      = df[df["season_x"].isin(season_order[:-1])]
best_params_by_pos = {}

for pos in ["FWD", "MID", "DEF", "GK"]:
    print(f"\nTuning {pos}...")
    best_params_by_pos[pos] = tune_hyperparams(
        tune_train_df, pos, features, n_trials=50
    )

# =============================================================================
# 8. ROLLING SEASON-BY-SEASON TRAINING
# =============================================================================
MIN_TRAIN_SEASONS = 2
all_predictions   = []

for i, pred_season in enumerate(season_order):
    if i < MIN_TRAIN_SEASONS:
        continue

    train_seasons = season_order[:i]
    train_df      = df[df["season_x"].isin(train_seasons)]
    test_df       = df[df["season_x"] == pred_season]

    print(f"\n{'='*65}")
    print(f"Predicting : {pred_season}")
    print(f"Training on: {train_seasons}")
    print(f"{'='*65}")

    season_preds = []

    for pos in ["FWD", "MID", "DEF", "GK"]:
        tr = (
            train_df[train_df["FPL_pos"] == pos]
            [features + ["total_points"]]
            .dropna()
        )
        tr = tr.loc[:, ~tr.columns.duplicated()].reset_index(drop=True)

        te = (
            test_df[test_df["FPL_pos"] == pos]
            [features + ["total_points", "player", "date", "season_x",
                          "FPL_pos", "price", "gameweek"]]
            .dropna(subset=features)
            .copy()
        )
        te = te.loc[:, ~te.columns.duplicated()].reset_index(drop=True)

        if len(tr) < 200 or len(te) == 0:
            print(f"  {pos}: insufficient data, skipping")
            continue

        params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbosity": 0,
            **best_params_by_pos.get(pos, {})
        }
        model = XGBRegressor(**params)
        model.fit(tr[features].values, tr["total_points"].values)

        te["xPts"] = model.predict(te[features].values).round(3)

        mae    = mean_absolute_error(te["total_points"], te["xPts"])
        rwmae  = rank_weighted_mae(te["total_points"].values, te["xPts"].values)
        corr   = np.corrcoef(te["xPts"], te["total_points"])[0, 1]
        recall = topk_recall_by_gameweek(te, k=11)

        print(
            f"  {pos:3s}  |  "
            f"MAE: {mae:.3f}  |  "
            f"RW-MAE: {rwmae:.3f}  |  "
            f"Corr: {corr:.3f}  |  "
            f"Recall@11: {recall:.3f}  |  "
            f"n={len(te):,}"
        )

        season_preds.append(
            te[["player", "FPL_pos", "date", "gameweek", "season_x",
                "xPts", "total_points", "price"]]
        )
        all_predictions.append(season_preds[-1])

    if season_preds:
        season_df   = pd.concat(season_preds, ignore_index=True)
        pos_recalls = topk_recall_by_position(season_df)
        print(f"\n  Position recall (top-k per GW): {pos_recalls}")

# =============================================================================
# 9. COMBINE & EXPORT PREDICTIONS
# =============================================================================
output = pd.concat(all_predictions, ignore_index=True)
output = (
    output
    .sort_values("xPts", ascending=False)
    .drop_duplicates(subset=["player", "gameweek", "season_x"], keep="first")
    .sort_values(["season_x", "gameweek", "player"])
    .reset_index(drop=True)
)

output.to_csv("Data/xpts_team_context_season_by_season.csv", index=False)
print(f"\nSaved {len(output):,} rows → Data/xpts_team_context_season_by_season.csv")

print("\nTop 20 xPts predictions across all seasons:")
print(output.sort_values("xPts", ascending=False).head(20).to_string(index=False))

print("\nPer-season summary:")
season_summary_rows = []
for season, s_df in output.groupby("season_x"):
    mae    = mean_absolute_error(s_df["total_points"], s_df["xPts"])
    rwmae  = rank_weighted_mae(s_df["total_points"].values, s_df["xPts"].values)
    corr   = np.corrcoef(s_df["xPts"], s_df["total_points"])[0, 1]
    recall = topk_recall_by_gameweek(s_df, k=11)
    season_summary_rows.append({
        "season"      : season,
        "MAE"         : round(mae,    3),
        "RW-MAE"      : round(rwmae,  3),
        "Corr"        : round(corr,   3),
        "Recall@11"   : round(recall, 3),
        "n"           : len(s_df),
    })
print(pd.DataFrame(season_summary_rows).to_string(index=False))

# =============================================================================
# 10. BEST XI SELECTOR (NO BUDGET)
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
        best_team.sort_values("xPts", ascending=False)
        [["player", "FPL_pos", "xPts"]].to_string(index=False)
    )
    print(f"\nCaptain   : {captain['player']} ({captain['xPts']:.2f} xPts)")
    print(f"Total xPts: {total:.2f}  (with captain double)")
    return best_team, best_formation, total

# =============================================================================
# 11. BEST XI SELECTOR WITH BUDGET CONSTRAINT (ILP)
# =============================================================================
def get_best_xi_budget(df, gameweek, season=None, budget=83.0):
    FORMATIONS = [(3,4,3),(3,5,2),(4,4,2),(4,3,3),(5,3,2),(5,4,1)]

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

    players = df_gw.index.tolist()
    xpts    = df_gw["xPts"].to_dict()
    prices  = df_gw["price"].to_dict()
    pos_map = df_gw["FPL_pos"].to_dict()

    best_team, best_score, best_formation = None, -np.inf, None

    for (d, m, f) in FORMATIONS:
        prob = LpProblem(f"BestXI_{d}{m}{f}", LpMaximize)
        x    = {i: LpVariable(f"x_{i}", cat="Binary") for i in players}

        prob += lpSum(xpts[i]   * x[i] for i in players)
        prob += lpSum(prices[i] * x[i] for i in players) <= budget
        prob += lpSum(x[i] for i in players) == 11

        prob += lpSum(x[i] for i in players if pos_map[i] == "GK")  == 1
        prob += lpSum(x[i] for i in players if pos_map[i] == "DEF") == d
        prob += lpSum(x[i] for i in players if pos_map[i] == "MID") == m
        prob += lpSum(x[i] for i in players if pos_map[i] == "FWD") == f

        prob.solve(PULP_CBC_CMD(msg=0))

        if prob.status != 1:
            continue

        selected = [i for i in players if value(x[i]) == 1]
        team     = df_gw.loc[selected]
        score    = team["xPts"].sum()

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
        best_team.sort_values("xPts", ascending=False)
        [["player", "FPL_pos", "xPts", "price"]].to_string(index=False)
    )
    print(f"\nCaptain    : {captain['player']} ({captain['xPts']:.2f} xPts)")
    print(f"Total xPts : {total_pts:.2f}  (with captain double)")
    print(f"Spent      : £{spent:.1f}m / £{budget:.1f}m  (£{budget - spent:.1f}m remaining)")
    return best_team, best_formation, total_pts

# =============================================================================
# 12. ACTUAL BEST XI (HINDSIGHT) WITH BUDGET CONSTRAINT
# =============================================================================
def get_actual_best_xi_budget(df, gameweek, season=None, budget=83.0):
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

    players = df_gw.index.tolist()
    points  = df_gw["total_points"].to_dict()
    prices  = df_gw["price"].to_dict()
    pos_map = df_gw["FPL_pos"].to_dict()

    best_team, best_score, best_formation = None, -np.inf, None

    for (d, m, f) in FORMATIONS:
        prob = LpProblem(f"ActualBestXI_{d}{m}{f}", LpMaximize)
        x    = {i: LpVariable(f"x_{i}", cat="Binary") for i in players}

        prob += lpSum(points[i]  * x[i] for i in players)
        prob += lpSum(prices[i]  * x[i] for i in players) <= budget
        prob += lpSum(x[i] for i in players) == 11

        prob += lpSum(x[i] for i in players if pos_map[i] == "GK")  == 1
        prob += lpSum(x[i] for i in players if pos_map[i] == "DEF") == d
        prob += lpSum(x[i] for i in players if pos_map[i] == "MID") == m
        prob += lpSum(x[i] for i in players if pos_map[i] == "FWD") == f

        prob.solve(PULP_CBC_CMD(msg=0))

        if prob.status != 1:
            continue

        selected = [i for i in players if value(x[i]) == 1]
        team     = df_gw.loc[selected]
        score    = team["total_points"].sum()

        if score > best_score:
            best_score, best_team, best_formation = score, team, (d, m, f)

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
        best_team.sort_values("total_points", ascending=False)
        [["player", "FPL_pos", "total_points", "xPts", "price"]].to_string(index=False)
    )
    print(f"\nCaptain    : {captain['player']} ({captain['total_points']} pts)")
    print(f"Total pts  : {total_pts:.0f}  (with captain double)")
    print(f"Spent      : £{spent:.1f}m / £{budget:.1f}m  (£{budget - spent:.1f}m remaining)")
    return best_team, best_formation, total_pts

# =============================================================================
# 13. COMPARE PREDICTED VS ACTUAL BEST XI (SINGLE GAMEWEEK)
# =============================================================================
def compare_actual_vs_predicted(df, gameweek, season=None, budget=83.0):
    print(f"\n{'='*55}")
    print("PREDICTED best XI (based on xPts)")
    print(f"{'='*55}")
    pred_result   = get_best_xi_budget(df, gameweek, season, budget)

    print(f"\n{'='*55}")
    print("ACTUAL best XI (based on total_points)")
    print(f"{'='*55}")
    actual_result = get_actual_best_xi_budget(df, gameweek, season, budget)

    if pred_result is None or actual_result is None:
        return

    pred_team,   _, _            = pred_result
    actual_team, _, actual_total = actual_result

    gw_actual = (
        df[(df["gameweek"] == gameweek) & (df["season_x"] == season)]
        [["player", "total_points"]].drop_duplicates("player")
        .set_index("player")["total_points"]
    )

    captain_player   = pred_team.loc[pred_team["xPts"].idxmax(), "player"]
    captain_pts      = gw_actual.get(captain_player, 0)
    predicted_actual = pred_team["player"].map(gw_actual).fillna(0).sum() + captain_pts

    print(f"\n{'='*55}")
    print(f"COMPARISON SUMMARY — GW{gameweek}" + (f" {season}" if season else ""))
    print(f"{'='*55}")
    print(f"Predicted XI actual pts scored : {predicted_actual:.0f}")
    print(f"Optimal XI actual pts scored   : {actual_total:.0f}")
    print(f"Points left on table           : {actual_total - predicted_actual:.0f}")

    missed = set(actual_team["player"]) - set(pred_team["player"])
    gained = set(pred_team["player"])   - set(actual_team["player"])
    print(f"\nIn optimal XI but NOT in predicted : {', '.join(missed) if missed else 'none'}")
    print(f"In predicted XI but NOT in optimal : {', '.join(gained) if gained else 'none'}")

# =============================================================================
# 14. SEASON PERFORMANCE COMPARISON
# =============================================================================
def compare_season_performance(df, start_gw, end_gw, season, budget=83.0, export=True):
    gameweeks = list(range(start_gw, end_gw + 1))
    records   = []
    df_season = df[df["season_x"] == season].copy()

    print(f"\n{'='*60}")
    print(f"SEASON COMPARISON — {season}  |  GW{start_gw}–GW{end_gw}  |  Budget: £{budget}m")
    print(f"{'='*60}")

    for gw in gameweeks:
        df_gw = (
            df_season[df_season["gameweek"] == gw]
            .dropna(subset=["xPts", "total_points", "price"])
            .reset_index(drop=True)
        )
        if df_gw.empty:
            print(f"  GW{gw:2d}: no data, skipping")
            continue

        pred_result = get_best_xi_budget(df_season, gameweek=gw, season=None, budget=budget)
        if pred_result is None:
            print(f"  GW{gw:2d}: no valid predicted team, skipping")
            continue
        pred_team, pred_formation, _ = pred_result

        gw_actuals = (
            df_gw[["player", "total_points"]].drop_duplicates("player")
            .set_index("player")["total_points"]
        )

        pred_captain     = pred_team.loc[pred_team["xPts"].idxmax(), "player"]
        pred_captain_pts = gw_actuals.get(pred_captain, 0)
        pred_actual_pts  = (
            pred_team["player"].map(gw_actuals).fillna(0).sum() + pred_captain_pts
        )

        actual_result = get_actual_best_xi_budget(
            df_season, gameweek=gw, season=None, budget=budget
        )
        if actual_result is None:
            print(f"  GW{gw:2d}: no valid optimal team, skipping")
            continue
        actual_team, actual_formation, actual_total_pts = actual_result

        missed = set(actual_team["player"]) - set(pred_team["player"])
        wrong  = set(pred_team["player"])   - set(actual_team["player"])
        gap    = actual_total_pts - pred_actual_pts

        gw_recall = topk_recall(
            df_gw["total_points"].values,
            df_gw["xPts"].values, k=11
        )

        records.append({
            "gameweek"            : gw,
            "predicted_actual_pts": round(pred_actual_pts,   1),
            "optimal_pts"         : round(actual_total_pts,  1),
            "gap"                 : round(gap,               1),
            "recall_at_11"        : round(gw_recall,         3),
            "pred_formation"      : f"{pred_formation[0]}-{pred_formation[1]}-{pred_formation[2]}",
            "optimal_formation"   : f"{actual_formation[0]}-{actual_formation[1]}-{actual_formation[2]}",
            "pred_captain"        : pred_captain,
            "pred_captain_pts"    : pred_captain_pts,
            "optimal_captain"     : actual_team.loc[actual_team["total_points"].idxmax(), "player"],
            "missed_players"      : ", ".join(sorted(missed)) if missed else "none",
            "wrong_players"       : ", ".join(sorted(wrong))  if wrong  else "none",
            "pred_spent"          : round(pred_team["price"].sum(),   1),
            "optimal_spent"       : round(actual_team["price"].sum(), 1),
        })

        print(
            f"  GW{gw:2d}  |  "
            f"Predicted: {pred_actual_pts:5.1f}pts  |  "
            f"Optimal: {actual_total_pts:5.1f}pts  |  "
            f"Gap: {gap:5.1f}  |  "
            f"Recall@11: {gw_recall:.2f}  |  "
            f"Captain: {pred_captain}"
        )

    if not records:
        print("No results to summarise.")
        return None

    results_df = pd.DataFrame(records)

    total_predicted = results_df["predicted_actual_pts"].sum()
    total_optimal   = results_df["optimal_pts"].sum()
    total_gap       = results_df["gap"].sum()
    avg_gap         = results_df["gap"].mean()
    avg_recall      = results_df["recall_at_11"].mean()
    best_gw         = results_df.loc[results_df["predicted_actual_pts"].idxmax()]
    worst_gw        = results_df.loc[results_df["gap"].idxmax()]
    captain_correct = (results_df["pred_captain"] == results_df["optimal_captain"]).sum()

    print(f"\n{'='*60}")
    print(f"SUMMARY — {season}  GW{start_gw}–GW{end_gw}")
    print(f"{'='*60}")
    print(f"Total predicted XI pts   : {total_predicted:.1f}")
    print(f"Total optimal XI pts     : {total_optimal:.1f}")
    print(f"Total pts left on table  : {total_gap:.1f}")
    print(f"Average gap per GW       : {avg_gap:.1f}")
    print(f"Average Recall@11        : {avg_recall:.3f}")
    print(f"Best GW  (predicted)     : GW{int(best_gw['gameweek'])}  ({best_gw['predicted_actual_pts']:.1f} pts)")
    print(f"Worst GW (biggest gap)   : GW{int(worst_gw['gameweek'])}  ({worst_gw['gap']:.1f} pts)")
    print(f"Captain correct          : {captain_correct} / {len(results_df)} gameweeks")
    print(f"Efficiency               : {100 * total_predicted / total_optimal:.1f}%")

    # --- Plot 1: Predicted vs Optimal ---
    plt.figure(figsize=(14, 4))
    plt.bar(results_df["gameweek"] - 0.2, results_df["predicted_actual_pts"],
            width=0.4, label="Predicted XI (actual pts)")
    plt.bar(results_df["gameweek"] + 0.2, results_df["optimal_pts"],
            width=0.4, label="Optimal XI")
    plt.ylabel("Points")
    plt.title(f"Predicted vs Optimal XI — {season} GW{start_gw}–{end_gw}")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"Data/plot1_pred_vs_opt_{season}.png", dpi=150)
    plt.show()

    # --- Plot 2: Gap ---
    plt.figure(figsize=(14, 4))
    plt.bar(results_df["gameweek"], results_df["gap"],
            label="Points gap")
    plt.axhline(avg_gap, linestyle="--",
                linewidth=1.2, label=f"Avg gap ({avg_gap:.1f})")
    plt.ylabel("Points left on table")
    plt.title("Gap between Optimal and Predicted XI")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"Data/plot2_gap_{season}.png", dpi=150)
    plt.show()

    # --- Plot 3: Recall ---
    plt.figure(figsize=(14, 4))
    plt.bar(results_df["gameweek"], results_df["recall_at_11"],
            alpha=0.8, label="Recall@11")
    plt.axhline(avg_recall, linestyle="--",
                linewidth=1.2, label=f"Avg ({avg_recall:.3f})")
    plt.ylim(0, 1.05)
    plt.xlabel("Gameweek")
    plt.ylabel("Recall@11")
    plt.title("Top-11 Recall per Gameweek")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"Data/plot3_recall_{season}.png", dpi=150)
    plt.show()

    if export:
        csv_path = f"Data/comparison_teamcontext_{season.replace('-','_')}_GW{start_gw}_GW{end_gw}.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"Saved CSV  → {csv_path}")

    return results_df

# =============================================================================
# 15. OPTIMAL TEAM PREDICTABILITY METRIC
# =============================================================================
def optimal_team_predictability(df, start_gw, end_gw, season, budget=83.0, export=True):
    def gini(values):
        v = np.sort(np.abs(values))
        n = len(v)
        if n == 0 or v.sum() == 0:
            return 0.0
        cumv = np.cumsum(v)
        return (2 * np.sum((np.arange(1, n+1)) * v) / (n * cumv[-1])) - (n+1)/n

    gameweeks = list(range(start_gw, end_gw + 1))
    records   = []
    df_season = df[df["season_x"] == season].copy()

    print(f"\n{'='*60}")
    print(f"OPTIMAL TEAM PREDICTABILITY — {season}  GW{start_gw}–GW{end_gw}")
    print(f"{'='*60}")

    for gw in gameweeks:
        df_gw = (
            df_season[df_season["gameweek"] == gw]
            .dropna(subset=["xPts", "total_points", "price"])
            .reset_index(drop=True)
        )
        if df_gw.empty:
            continue

        pred_result   = get_best_xi_budget(df_season, gameweek=gw, season=None, budget=budget)
        actual_result = get_actual_best_xi_budget(df_season, gameweek=gw, season=None, budget=budget)

        if pred_result is None or actual_result is None:
            continue

        pred_team,   _, _                = pred_result
        actual_team, _, actual_total_pts = actual_result

        gw_actuals = (
            df_gw[["player", "total_points"]].drop_duplicates("player")
            .set_index("player")["total_points"]
        )

        total_xpts_of_optimal = (
            df_gw[df_gw["player"].isin(actual_team["player"])]
            [["player", "xPts"]].drop_duplicates("player")
            .set_index("player")["xPts"].sum()
        )
        pred_xpts_total = (
            df_gw[df_gw["player"].isin(pred_team["player"])]
            [["player", "xPts"]].drop_duplicates("player")
            .set_index("player")["xPts"].sum()
        )

        percentile_ranks = []
        for _, row in actual_team.iterrows():
            pos_pool = df_gw[df_gw["FPL_pos"] == row["FPL_pos"]]["xPts"]
            if len(pos_pool) == 0:
                continue
            percentile_ranks.append((pos_pool <= row["xPts"]).mean() * 100)
        avg_rank_pct = np.mean(percentile_ranks) if percentile_ranks else np.nan

        overlap        = len(set(actual_team["player"]) & set(pred_team["player"]))
        surprise_score = 1 - (overlap / 11)
        point_conc     = gini(actual_team["player"].map(gw_actuals).fillna(0).values)

        pred_captain     = pred_team.loc[pred_team["xPts"].idxmax(), "player"]
        pred_captain_pts = gw_actuals.get(pred_captain, 0)
        pred_actual_pts  = (
            pred_team["player"].map(gw_actuals).fillna(0).sum() + pred_captain_pts
        )
        expected_gap = total_xpts_of_optimal - pred_xpts_total

        records.append({
            "gameweek"            : gw,
            "predicted_actual_pts": round(pred_actual_pts,       1),
            "optimal_pts"         : round(actual_total_pts,      1),
            "gap"                 : round(actual_total_pts - pred_actual_pts, 1),
            "xPts_of_optimal"     : round(total_xpts_of_optimal, 2),
            "xPts_of_predicted"   : round(pred_xpts_total,       2),
            "expected_gap"        : round(expected_gap,          2),
            "avg_rank_pct"        : round(avg_rank_pct,          1),
            "surprise_score"      : round(surprise_score,        2),
            "point_concentration" : round(point_conc,            2),
        })

        print(
            f"  GW{gw:2d}  |  "
            f"Gap: {actual_total_pts - pred_actual_pts:5.1f}  |  "
            f"Optimal xPts: {total_xpts_of_optimal:5.2f}  |  "
            f"Avg rank: {avg_rank_pct:5.1f}%  |  "
            f"Surprise: {surprise_score:.2f}  |  "
            f"Conc: {point_conc:.2f}"
        )

    if not records:
        print("No results.")
        return None

    results_df = pd.DataFrame(records)

    results_df["verdict"] = "mixed"
    results_df.loc[
        (results_df["surprise_score"] >= 0.5) &
        (results_df["point_concentration"] >= 0.35),
        "verdict"
    ] = "unpredictable"
    results_df.loc[
        (results_df["surprise_score"] < 0.3) &
        (results_df["gap"] > results_df["gap"].median()),
        "verdict"
    ] = "model underperformed"
    results_df.loc[
        (results_df["surprise_score"] < 0.3) &
        (results_df["gap"] <= results_df["gap"].median()),
        "verdict"
    ] = "good prediction"

    print(f"\n{'='*60}")
    print("PREDICTABILITY SUMMARY")
    print(f"{'='*60}")
    print(f"Avg xPts assigned to optimal team  : {results_df['xPts_of_optimal'].mean():.2f}")
    print(f"Avg xPts assigned to predicted team: {results_df['xPts_of_predicted'].mean():.2f}")
    print(f"Avg optimal player rank percentile : {results_df['avg_rank_pct'].mean():.1f}%")
    print(f"Avg surprise score                 : {results_df['surprise_score'].mean():.2f}")
    print(f"Avg point concentration (Gini)     : {results_df['point_concentration'].mean():.2f}")
    print(f"\nGameweek verdicts:")
    print(results_df["verdict"].value_counts().to_string())

    # --- Plot 1: Gap vs Expected Gap ---
    plt.figure(figsize=(14, 4))
    plt.bar(results_df["gameweek"] - 0.2, results_df["gap"],
            width=0.4, label="Actual gap")
    plt.bar(results_df["gameweek"] + 0.2, results_df["expected_gap"],
            width=0.4, label="Expected gap (xPts diff)")
    plt.axhline(0, linewidth=0.8)
    plt.ylabel("Points")
    plt.title(f"Actual Gap vs Expected Gap — {season}")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"Data/plot4_gap_vs_expected_{season}.png", dpi=150)
    plt.show()

    # --- Plot 2: Surprise + Concentration ---
    plt.figure(figsize=(14, 4))
    plt.plot(results_df["gameweek"], results_df["surprise_score"],
             marker="o", label="Surprise score")
    plt.plot(results_df["gameweek"], results_df["point_concentration"],
             marker="s", label="Point concentration (Gini)")
    plt.axhline(0.5, linestyle="--", linewidth=0.8, alpha=0.5)
    plt.axhline(0.35, linestyle="--", linewidth=0.8, alpha=0.5)
    plt.ylabel("Score (0–1)")
    plt.title("Surprise Score & Point Concentration")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"Data/plot5_surprise_{season}.png", dpi=150)
    plt.show()

    # --- Plot 3: Rank Percentile ---
    plt.figure(figsize=(14, 4))
    plt.bar(results_df["gameweek"], results_df["avg_rank_pct"],
            alpha=0.8)
    plt.axhline(results_df["avg_rank_pct"].mean(),
                linestyle="--",
                linewidth=1.2,
                label=f"Mean ({results_df['avg_rank_pct'].mean():.1f}%)")
    plt.xlabel("Gameweek")
    plt.ylabel("Avg rank percentile")
    plt.title("How Highly Ranked Were the Optimal Players?")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"Data/plot6_rank_{season}.png", dpi=150)
    plt.show()

    if export:
        csv_path = f"Data/predictability_teamcontext_{season.replace('-','_')}_GW{start_gw}_GW{end_gw}.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"Saved CSV  → {csv_path}")

    return results_df

# =============================================================================
# 16. EXAMPLE CALLS
# =============================================================================
get_best_xi(output, gameweek=10, season="2023-24")
get_best_xi(output, gameweek=38, season="2024-25")

get_best_xi_budget(output, gameweek=10, season="2023-24", budget=83.0)
get_best_xi_budget(output, gameweek=38, season="2024-25", budget=83.0)

get_actual_best_xi_budget(output, gameweek=10, season="2023-24", budget=83.0)

compare_actual_vs_predicted(output, gameweek=10, season="2023-24", budget=83.0)

results = compare_season_performance(
    output, start_gw=1, end_gw=38, season="2024-25", budget=83.0
)

pred_df = optimal_team_predictability(
    output, start_gw=1, end_gw=38, season="2024-25", budget=83.0
)