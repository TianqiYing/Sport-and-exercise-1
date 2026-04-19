import os
import warnings
import itertools
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# =========================================================
# 1. SETTINGS
# =========================================================
DATA_PATH = "active_perfect_understat_enhanced.csv"
OUTPUT_DIR = "model_outputs"
MIN_TRAIN_SEASONS = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)

RUN_TRAINING = False
RUN_BEST_XI_ANALYSIS = True

# =========================================================
# 2. METRIC FUNCTIONS
# =========================================================
def safe_corr(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) < 2:
        return np.nan
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan

    return float(np.corrcoef(y_true, y_pred)[0, 1])


def calc_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    corr = safe_corr(y_true, y_pred)
    return mae, rmse, corr


def print_position_breakdown(df_eval):
    for pos in ["FWD", "MID", "DEF", "GK"]:
        tmp = df_eval[df_eval["FPL_pos"] == pos]
        if len(tmp) == 0:
            continue

        mae, rmse, corr = calc_metrics(tmp["total_points"], tmp["xPts"])
        print(
            f"  {pos:3s} | MAE: {mae:.3f} | RMSE: {rmse:.3f} | "
            f"Corr: {corr:.3f} | n={len(tmp):,}"
        )


def summarize_predictions(output, model_name):
    print(f"\n==============================")
    print(f"{model_name.upper()} — PER-SEASON SUMMARY")
    print(f"==============================")

    season_summary = (
        output.groupby("season_x")
        .apply(
            lambda x: pd.Series({
                "MAE": mean_absolute_error(x["total_points"], x["xPts"]),
                "RMSE": mean_squared_error(x["total_points"], x["xPts"]) ** 0.5,
                "Corr": safe_corr(x["total_points"], x["xPts"]),
                "n": len(x),
            }),
            include_groups=False
        )
        .round(3)
    )

    print(season_summary.to_string())

    overall_mae, overall_rmse, overall_corr = calc_metrics(
        output["total_points"], output["xPts"]
    )

    print(f"\nOverall MAE : {overall_mae:.3f}")
    print(f"Overall RMSE: {overall_rmse:.3f}")
    print(f"Overall Corr: {overall_corr:.3f}")
    print(f"Rows        : {len(output):,}")

    return season_summary, {
        "Model": model_name,
        "MAE": overall_mae,
        "RMSE": overall_rmse,
        "Corr": overall_corr,
        "n": len(output),
    }


# =========================================================
# 3. MODEL FACTORIES
# =========================================================
def build_linear_model():
    return LinearRegression()


def build_xgb_model():
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )


# =========================================================
# 4. LOAD DATA
# =========================================================
df = pd.read_csv(DATA_PATH)
df = df.loc[:, ~df.columns.duplicated()].copy()

df = df.drop(columns=[
    "league_id", "player_id", "position_id", "minutes_y",
    "own_goals_y", "game_id", "season_id",
    "game", "venue",
    "ea_index", "loaned_in", "loaned_out",
    "kickoff_time_formatted",
    "id",
], errors="ignore")

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.sort_values(["player", "date"]).reset_index(drop=True)

print(f"Duplicate columns : {df.columns[df.columns.duplicated()].tolist()}")
print(f"Duplicate indices : {df.index.duplicated().sum()}")

# =========================================================
# 5. OPPONENT STRENGTH FEATURES
# =========================================================
team_level = (
    df.groupby(["team", "date", "opponent", "season_x"], as_index=False)
    .agg(
        goals_scored_team=("goals_scored", "sum"),
        xg_team=("xg", "sum"),
    )
)

opp_conceded = (
    team_level
    .rename(columns={
        "team": "opponent",
        "opponent": "team",
        "goals_scored_team": "opp_goals_conceded",
        "xg_team": "opp_xg_conceded",
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
    "Duplicate (team, date) rows found in team-level table."

opp_form_cols = [
    "opp_goals_conceded_form_3", "opp_xg_conceded_form_3",
    "opp_goals_conceded_form_5", "opp_xg_conceded_form_5",
]

df = df.merge(
    team_level[["team", "date"] + opp_form_cols],
    left_on=["opponent", "date"],
    right_on=["team", "date"],
    how="left",
    suffixes=("", "_drop"),
)

df = df.drop(columns=[c for c in df.columns if c.endswith("_drop")], errors="ignore")
df = df.loc[:, ~df.columns.duplicated()].reset_index(drop=True)

print(f"After opp merge — duplicate indices: {df.index.duplicated().sum()}")

# =========================================================
# 6. PREVIOUS-SEASON FEATURES
# =========================================================
season_order = sorted(df["season_x"].dropna().unique())
season_rank = {s: i for i, s in enumerate(season_order)}
df["season_rank"] = df["season_x"].map(season_rank)

season_summary = (
    df.groupby(["player", "season_x", "season_rank"], as_index=False)
    .agg(
        prev_season_pts=("total_points", "sum"),
        prev_season_goals=("goals_scored", "sum"),
        prev_season_assists=("assists_x", "sum"),
        prev_season_minutes=("minutes_x", "sum"),
        prev_season_xg=("xg", "mean"),
        prev_season_xa=("xa", "mean"),
        prev_season_games=("played", "sum"),
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
    how="left",
)

df["prev_season_complete"] = (df["prev_season_games"] >= 25).astype(float)
df = df.loc[:, ~df.columns.duplicated()].reset_index(drop=True)

print(f"After season merge — duplicate indices: {df.index.duplicated().sum()}")

# =========================================================
# 7. FEATURE SET
# =========================================================
base_features = [
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

base_features = list(dict.fromkeys(base_features))

missing = [f for f in base_features if f not in df.columns]
if missing:
    raise ValueError(f"Missing feature columns: {missing}")

print(f"Features : {len(base_features)}")
print(f"Rows     : {len(df):,}")
print(f"Seasons  : {season_order}")

# =========================================================
# 8. FEATURE MATRIX BUILDER
# =========================================================
def build_global_matrices(train_df, test_df, features, target):
    meta_cols = ["player", "FPL_pos", "date", "season_x", target]

    train_cols = list(dict.fromkeys(features + ["FPL_pos"] + meta_cols))
    test_cols = list(dict.fromkeys(features + ["FPL_pos"] + meta_cols))

    train_keep = train_df[train_cols].dropna(subset=features + [target]).copy()
    test_keep = test_df[test_cols].dropna(subset=features).copy()

    X_train = pd.get_dummies(
        train_keep[features + ["FPL_pos"]],
        columns=["FPL_pos"],
        drop_first=False,
        dtype=float
    )
    X_test = pd.get_dummies(
        test_keep[features + ["FPL_pos"]],
        columns=["FPL_pos"],
        drop_first=False,
        dtype=float
    )

    X_train, X_test = X_train.align(X_test, join="outer", axis=1, fill_value=0)

    X_train = X_train.loc[:, ~X_train.columns.duplicated()].copy()
    X_test = X_test.loc[:, ~X_test.columns.duplicated()].copy()

    X_train = X_train.apply(pd.to_numeric, errors="coerce").astype(float)
    X_test = X_test.apply(pd.to_numeric, errors="coerce").astype(float)

    y_train = train_keep[target].values
    y_test = test_keep[target].values

    meta_test = test_keep[["player", "FPL_pos", "date", "season_x", target]].copy()
    if "gameweek" in test_df.columns:
        meta_test["gameweek"] = test_keep["gameweek"].values

    return X_train, X_test, y_train, y_test, meta_test


# =========================================================
# 9. GLOBAL MODEL RUNNER
# =========================================================
def run_global_model(df, model_name):
    all_predictions = []

    for i, pred_season in enumerate(season_order):
        if i < MIN_TRAIN_SEASONS:
            continue

        train_seasons = season_order[:i]
        train_df = df[df["season_x"].isin(train_seasons)].copy()
        test_df = df[df["season_x"] == pred_season].copy()

        print(f"\n{'='*60}")
        print(f"{model_name} — Predicting: {pred_season}")
        print(f"Training on: {train_seasons}")
        print(f"{'='*60}")

        target = "total_points"

        X_train, X_test, y_train, y_test, meta_test = build_global_matrices(
            train_df, test_df, base_features, target
        )

        if model_name == "Linear Regression":
            model = build_linear_model()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        elif model_name == "Single XGBoost":
            model = build_xgb_model()
            model.fit(X_train.to_numpy(), y_train)
            y_pred = model.predict(X_test.to_numpy())

        else:
            raise ValueError(f"Unknown model name: {model_name}")

        season_output = meta_test.copy()
        season_output["xPts"] = np.round(y_pred, 3)

        season_mae, season_rmse, season_corr = calc_metrics(
            season_output["total_points"], season_output["xPts"]
        )

        print(
            f"Overall | MAE: {season_mae:.3f} | RMSE: {season_rmse:.3f} | "
            f"Corr: {season_corr:.3f} | n={len(season_output):,}"
        )
        print_position_breakdown(season_output)

        all_predictions.append(season_output)

    output = pd.concat(all_predictions, ignore_index=True)

    file_stub = model_name.lower().replace(" ", "_")
    output.to_csv(os.path.join(OUTPUT_DIR, f"{file_stub}_predictions.csv"), index=False)

    season_summary, overall_summary = summarize_predictions(output, model_name)
    season_summary.to_csv(os.path.join(OUTPUT_DIR, f"{file_stub}_season_summary.csv"))

    return output, season_summary, overall_summary


# =========================================================
# 10. POSITION-SPECIFIC XGBOOST RUNNER
# =========================================================
def run_position_specific_xgboost(df):
    all_predictions = []

    for i, pred_season in enumerate(season_order):
        if i < MIN_TRAIN_SEASONS:
            continue

        train_seasons = season_order[:i]
        train_df = df[df["season_x"].isin(train_seasons)].copy()
        test_df = df[df["season_x"] == pred_season].copy()

        print(f"\n{'='*60}")
        print(f"Position-specific XGBoost — Predicting: {pred_season}")
        print(f"Training on: {train_seasons}")
        print(f"{'='*60}")

        season_preds = []

        for pos in ["FWD", "MID", "DEF", "GK"]:
            tr = (
                train_df[train_df["FPL_pos"] == pos]
                [base_features + ["total_points"]]
                .dropna(subset=base_features + ["total_points"])
                .copy()
            )

            te_cols = list(dict.fromkeys(
                base_features + ["total_points", "player", "date", "season_x", "FPL_pos", "gameweek"]
            ))
            te = (
                test_df[test_df["FPL_pos"] == pos][te_cols]
                .dropna(subset=base_features)
                .copy()
            )

            if len(tr) < 200 or len(te) == 0:
                print(f"  {pos}: insufficient data, skipping")
                continue

            X_train = tr[base_features].apply(pd.to_numeric, errors="coerce").astype(float).to_numpy()
            y_train = tr["total_points"].values

            X_test = te[base_features].apply(pd.to_numeric, errors="coerce").astype(float).to_numpy()

            model = build_xgb_model()
            model.fit(X_train, y_train)

            te["xPts"] = np.round(model.predict(X_test), 3)

            mae, rmse, corr = calc_metrics(te["total_points"], te["xPts"])
            print(
                f"  {pos:3s} | MAE: {mae:.3f} | RMSE: {rmse:.3f} | "
                f"Corr: {corr:.3f} | n={len(te):,}"
            )

            season_preds.append(
                te[["player", "FPL_pos", "date", "gameweek", "season_x", "xPts", "total_points"]]
            )

        if len(season_preds) == 0:
            continue

        season_output = pd.concat(season_preds, ignore_index=True)

        season_mae, season_rmse, season_corr = calc_metrics(
            season_output["total_points"], season_output["xPts"]
        )
        print(
            f"Overall | MAE: {season_mae:.3f} | RMSE: {season_rmse:.3f} | "
            f"Corr: {season_corr:.3f} | n={len(season_output):,}"
        )

        all_predictions.append(season_output)

    output = pd.concat(all_predictions, ignore_index=True)
    output.to_csv(
        os.path.join(OUTPUT_DIR, "position_specific_xgboost_predictions.csv"),
        index=False
    )

    season_summary, overall_summary = summarize_predictions(output, "Position-specific XGBoost")
    season_summary.to_csv(
        os.path.join(OUTPUT_DIR, "position_specific_xgboost_season_summary.csv")
    )

    return output, season_summary, overall_summary


# =========================================================
# 11. RUN TRAINING AND MODEL EVALUATION
# =========================================================
if RUN_TRAINING:
    lr_output, lr_season_summary, lr_overall = run_global_model(df, "Linear Regression")
    xgb_output, xgb_season_summary, xgb_overall = run_global_model(df, "Single XGBoost")
    pos_output, pos_season_summary, pos_overall = run_position_specific_xgboost(df)

    final_comparison = pd.DataFrame([
        lr_overall,
        xgb_overall,
        pos_overall
    ]).round(3)

    print(f"\n==============================")
    print("FINAL MODEL COMPARISON")
    print("==============================")
    print(final_comparison.to_string(index=False))

    final_comparison.to_csv(
        os.path.join(OUTPUT_DIR, "final_model_comparison.csv"),
        index=False
    )

# =========================================================
# BEST XI SELECTOR
# =========================================================
FORMATIONS = [
    (3, 4, 3),
    (3, 5, 2),
    (4, 4, 2),
    (4, 3, 3),
    (5, 3, 2),
    (5, 4, 1)
]

MODEL_FILE_MAP = {
    "linear": "model_outputs/linear_regression_predictions.csv",
    "xgboost": "model_outputs/single_xgboost_predictions.csv",
    "position_xgboost": "model_outputs/position_specific_xgboost_predictions.csv"
}

def load_prediction_file(model_key):
    if model_key not in MODEL_FILE_MAP:
        raise ValueError(f"Unknown model: {model_key}")

    file_path = MODEL_FILE_MAP[model_key]
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prediction file not found: {file_path}")

    return pd.read_csv(file_path)


def parse_budget_input(budget_input):
    budget_input = budget_input.strip().lower()

    if budget_input in ["unlimited", "none", "inf", "no limit", "nolimit"]:
        return None

    try:
        return float(budget_input)
    except ValueError:
        raise ValueError("Budget must be a number like 83 or 83.0, or 'unlimited'.")


def prepare_best_xi_pool(pred_df, source_df, season, gameweek):
    pred_df = pred_df.copy()
    pred_df["date"] = pd.to_datetime(pred_df["date"], errors="coerce")

    source_df = source_df.copy()
    source_df["date"] = pd.to_datetime(source_df["date"], errors="coerce")

    gw_pred = pred_df[
        (pred_df["season_x"] == season) &
        (pred_df["gameweek"] == gameweek)
    ].copy()

    if len(gw_pred) == 0:
        print(f"No predictions found for season={season}, gameweek={gameweek}")
        return None

    price_ref = (
        source_df[
            (source_df["season_x"] == season) &
            (source_df["gameweek"] == gameweek)
        ][["player", "FPL_pos", "season_x", "gameweek", "date", "price"]]
        .drop_duplicates(subset=["player", "FPL_pos", "season_x", "gameweek", "date"], keep="last")
        .copy()
    )

    gw = gw_pred.merge(
        price_ref,
        on=["player", "FPL_pos", "season_x", "gameweek", "date"],
        how="left"
    )

    gw = gw.dropna(subset=["xPts", "FPL_pos", "price", "total_points"]).copy()

    if len(gw) == 0:
        print(f"No usable rows found for season={season}, gameweek={gameweek}")
        return None

    return gw


def get_best_xi(pred_df, source_df, season, gameweek, budget=None):
    gw = prepare_best_xi_pool(pred_df, source_df, season, gameweek)
    if gw is None:
        return None

    # unlimited budget: use a much simpler fast selector
    if budget is None:
        gk = gw[gw["FPL_pos"] == "GK"].nlargest(1, "xPts")

        best_team = None
        best_score = -np.inf
        best_formation = None

        for d, m, f in FORMATIONS:
            team = pd.concat([
                gk,
                gw[gw["FPL_pos"] == "DEF"].nlargest(d, "xPts"),
                gw[gw["FPL_pos"] == "MID"].nlargest(m, "xPts"),
                gw[gw["FPL_pos"] == "FWD"].nlargest(f, "xPts"),
            ])

            if len(team) != 11:
                continue

            score = team["xPts"].sum()
            if score > best_score:
                best_score = score
                best_team = team.copy()
                best_formation = (d, m, f)

    else:
        # smaller candidate pools for budget-constrained search
        gks = gw[gw["FPL_pos"] == "GK"].sort_values("xPts", ascending=False).head(2)
        defs = gw[gw["FPL_pos"] == "DEF"].sort_values("xPts", ascending=False).head(6)
        mids = gw[gw["FPL_pos"] == "MID"].sort_values("xPts", ascending=False).head(6)
        fwds = gw[gw["FPL_pos"] == "FWD"].sort_values("xPts", ascending=False).head(4)

        if len(gks) == 0:
            print("No goalkeeper available.")
            return None

        best_team = None
        best_score = -np.inf
        best_formation = None

        print(f"Searching best XI for {season} GW{gameweek} under budget {budget:.1f}...")

        for d, m, f in FORMATIONS:
            for gk_idx in itertools.combinations(gks.index, 1):
                for def_idx in itertools.combinations(defs.index, d):
                    for mid_idx in itertools.combinations(mids.index, m):
                        for fwd_idx in itertools.combinations(fwds.index, f):
                            idx = list(gk_idx) + list(def_idx) + list(mid_idx) + list(fwd_idx)
                            team = gw.loc[idx].copy()

                            total_price = team["price"].sum()
                            if total_price > budget:
                                continue

                            predicted_score = team["xPts"].sum()
                            if predicted_score > best_score:
                                best_score = predicted_score
                                best_team = team.copy()
                                best_formation = (d, m, f)

    if best_team is None:
        if budget is None:
            print("No valid XI could be formed.")
        else:
            print(f"No valid XI could be formed under budget {budget:.1f}.")
        return None

    captain = best_team.loc[best_team["xPts"].idxmax()]

    predicted_total = best_team["xPts"].sum()
    predicted_total_with_captain = predicted_total + captain["xPts"]

    actual_total = best_team["total_points"].sum()
    actual_total_with_captain = actual_total + captain["total_points"]

    budget_used = best_team["price"].sum()

    gw_label = int(gameweek) if pd.notna(gameweek) else gameweek
    print(f"\nBest XI — {season} GW{gw_label}")
    print(f"Formation : {best_formation[0]}-{best_formation[1]}-{best_formation[2]}")

    if budget is None:
        print(f"Budget    : {budget_used:.1f} / unlimited")
    else:
        print(f"Budget    : {budget_used:.1f} / {budget:.1f}")

    print(
        best_team.sort_values("xPts", ascending=False)[
            ["player", "FPL_pos", "price", "xPts", "total_points"]
        ].to_string(index=False)
    )

    print(f"\nCaptain              : {captain['player']} ({captain['xPts']:.2f} xPts)")
    print(f"Predicted team xPts  : {predicted_total_with_captain:.2f}")
    print(f"Actual team score    : {actual_total_with_captain:.2f}")

    return best_team, best_formation, predicted_total_with_captain, actual_total_with_captain

def run_best_xi_selector():
    print("\nAvailable models:")
    print("  linear")
    print("  xgboost")
    print("  position_xgboost")

    model_key = input("Enter model name: ").strip().lower()
    season = input("Enter season (e.g. 2024-25): ").strip()
    gameweek = int(input("Enter gameweek (e.g. 15): ").strip())
    budget_input = input("Enter budget (e.g. 83 or unlimited): ").strip()

    budget = parse_budget_input(budget_input)
    pred_df = load_prediction_file(model_key)

    return get_best_xi(
        pred_df=pred_df,
        source_df=df,
        season=season,
        gameweek=gameweek,
        budget=budget
    )

# =========================================================
# SEASON-WIDE BEST XI EVALUATION
# =========================================================
def evaluate_model_over_season(model_key, season, budget=None, save_csv=True):
    pred_df = load_prediction_file(model_key)

    season_gws = sorted(
        pred_df.loc[pred_df["season_x"] == season, "gameweek"].dropna().unique()
    )

    if len(season_gws) == 0:
        print(f"No gameweeks found for season {season} in model {model_key}.")
        return None

    results = []

    print(f"\nEvaluating {model_key} over season {season}...")

    for gw in season_gws:
        out = get_best_xi(
            pred_df=pred_df,
            source_df=df,
            season=season,
            gameweek=gw,
            budget=budget
        )

        if out is None:
            continue

        team, formation, predicted_total, actual_total = out

        results.append({
            "model": model_key,
            "season": season,
            "gameweek": int(gw),
            "formation": f"{formation[0]}-{formation[1]}-{formation[2]}",
            "budget_used": round(team["price"].sum(), 3),
            "predicted_team_xPts": round(predicted_total, 3),
            "actual_team_score": round(actual_total, 3),
            "captain": team.loc[team["xPts"].idxmax(), "player"]
        })

    if len(results) == 0:
        print(f"No valid weekly teams were generated for {model_key} in {season}.")
        return None

    result_df = pd.DataFrame(results).sort_values("gameweek").reset_index(drop=True)

    print(f"\n===== {model_key} — {season} season summary =====")
    print(result_df.to_string(index=False))

    gw_mae = mean_absolute_error(
        result_df["actual_team_score"],
        result_df["predicted_team_xPts"]
    )
    gw_rmse = mean_squared_error(
        result_df["actual_team_score"],
        result_df["predicted_team_xPts"]
    ) ** 0.5
    gw_corr = safe_corr(
        result_df["actual_team_score"],
        result_df["predicted_team_xPts"]
    )

    print("\nAverages:")
    print("Average predicted team xPts :", round(result_df["predicted_team_xPts"].mean(), 3))
    print("Average actual team score   :", round(result_df["actual_team_score"].mean(), 3))
    print("Total actual team score     :", round(result_df["actual_team_score"].sum(), 3))

    print("\nGW-level accuracy:")
    print("GW MAE                      :", round(gw_mae, 3))
    print("GW RMSE                     :", round(gw_rmse, 3))
    print("GW Corr                     :", round(gw_corr, 3))

    if save_csv:
        budget_tag = "unlimited" if budget is None else str(budget).replace(".", "_")
        save_path = os.path.join(
            OUTPUT_DIR,
            f"{model_key}_{season}_budget_{budget_tag}_gw_summary.csv"
        )
        result_df.to_csv(save_path, index=False)
        print(f"\nSaved season summary to: {save_path}")

    return result_df

# =========================================================
# OPTIMAL XI AND SEASON COMPARISON
# =========================================================
def get_optimal_xi(source_df, season, gameweek, budget=None):
    source_df = source_df.copy()
    source_df["date"] = pd.to_datetime(source_df["date"], errors="coerce")

    gw = source_df[
        (source_df["season_x"] == season) &
        (source_df["gameweek"] == gameweek)
    ].copy()

    if len(gw) == 0:
        return None

    gw = gw.dropna(subset=["FPL_pos", "price", "total_points"]).copy()

    gks = gw[gw["FPL_pos"] == "GK"].sort_values("total_points", ascending=False).head(2)
    defs = gw[gw["FPL_pos"] == "DEF"].sort_values("total_points", ascending=False).head(6)
    mids = gw[gw["FPL_pos"] == "MID"].sort_values("total_points", ascending=False).head(6)
    fwds = gw[gw["FPL_pos"] == "FWD"].sort_values("total_points", ascending=False).head(4)

    if len(gks) == 0:
        return None

    best_team = None
    best_score = -np.inf
    best_formation = None

    for d, m, f in FORMATIONS:
        for gk_idx in itertools.combinations(gks.index, 1):
            for def_idx in itertools.combinations(defs.index, d):
                for mid_idx in itertools.combinations(mids.index, m):
                    for fwd_idx in itertools.combinations(fwds.index, f):
                        idx = list(gk_idx) + list(def_idx) + list(mid_idx) + list(fwd_idx)
                        team = gw.loc[idx].copy()

                        total_price = team["price"].sum()
                        if budget is not None and total_price > budget:
                            continue

                        score = team["total_points"].sum()

                        if score > best_score:
                            best_score = score
                            best_team = team.copy()
                            best_formation = (d, m, f)

    if best_team is None:
        return None

    captain = best_team.loc[best_team["total_points"].idxmax()]
    total_with_captain = best_team["total_points"].sum() + captain["total_points"]

    return best_team, best_formation, total_with_captain, captain["player"]


def plot_gw_comparison(result_df, season, budget=None, save_dir=OUTPUT_DIR):
    import matplotlib.pyplot as plt

    if result_df is None or len(result_df) == 0:
        return None

    budget_tag = "unlimited" if budget is None else str(budget).replace(".", "_")
    save_path = os.path.join(save_dir, f"comparison_{season}_budget_{budget_tag}.png")

    plt.figure(figsize=(10, 5))
    plt.plot(result_df["gameweek"], result_df["predicted_team_score"], marker="o", label="Predicted XI pts")
    plt.plot(result_df["gameweek"], result_df["optimal_team_score"], marker="o", label="Optimal XI pts")
    plt.xlabel("Gameweek")
    plt.ylabel("Points")
    plt.title(f"Predicted XI vs Optimal XI — {season}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    return save_path


def compare_season_performance(model_key, season, start_gw=1, end_gw=38, budget=None, save_csv=True, save_plot=True):
    pred_df = load_prediction_file(model_key)

    season_gws = sorted(
        pred_df.loc[
            (pred_df["season_x"] == season) &
            (pred_df["gameweek"] >= start_gw) &
            (pred_df["gameweek"] <= end_gw),
            "gameweek"
        ].dropna().unique()
    )

    if len(season_gws) == 0:
        print(f"No gameweeks found for season {season} in model {model_key}.")
        return None

    rows = []

    for gw in season_gws:
        pred_out = get_best_xi(
            pred_df=pred_df,
            source_df=df,
            season=season,
            gameweek=gw,
            budget=budget
        )

        opt_out = get_optimal_xi(
            source_df=df,
            season=season,
            gameweek=gw,
            budget=budget
        )

        if pred_out is None or opt_out is None:
            continue

        pred_team, pred_formation, pred_score, pred_actual_score = pred_out
        opt_team, opt_formation, opt_score, opt_captain = opt_out

        pred_captain = pred_team.loc[pred_team["xPts"].idxmax(), "player"]
        budget_used = pred_team["price"].sum()
        gap = opt_score - pred_actual_score

        rows.append({
            "model": model_key,
            "season": season,
            "gameweek": int(gw),
            "predicted_formation": f"{pred_formation[0]}-{pred_formation[1]}-{pred_formation[2]}",
            "optimal_formation": f"{opt_formation[0]}-{opt_formation[1]}-{opt_formation[2]}",
            "budget_used": round(budget_used, 3),
            "predicted_team_xPts": round(pred_score, 3),
            "predicted_team_score": round(pred_actual_score, 3),
            "optimal_team_score": round(opt_score, 3),
            "pts_left_on_table": round(gap, 3),
            "predicted_captain": pred_captain,
            "optimal_captain": opt_captain,
            "captain_correct": int(pred_captain == opt_captain)
        })

    if len(rows) == 0:
        print(f"No valid rows generated for {model_key} in {season}.")
        return None

    result_df = pd.DataFrame(rows).sort_values("gameweek").reset_index(drop=True)

    total_predicted = result_df["predicted_team_score"].sum()
    total_optimal = result_df["optimal_team_score"].sum()
    total_left = result_df["pts_left_on_table"].sum()
    avg_gap = result_df["pts_left_on_table"].mean()

    best_idx = result_df["predicted_team_score"].idxmax()
    worst_idx = result_df["pts_left_on_table"].idxmax()

    best_gw = int(result_df.loc[best_idx, "gameweek"])
    best_gw_pts = result_df.loc[best_idx, "predicted_team_score"]

    worst_gw = int(result_df.loc[worst_idx, "gameweek"])
    worst_gap = result_df.loc[worst_idx, "pts_left_on_table"]

    captain_correct = int(result_df["captain_correct"].sum())
    total_weeks = len(result_df)
    efficiency = 100 * total_predicted / total_optimal if total_optimal != 0 else np.nan

    budget_tag = "unlimited" if budget is None else str(budget).replace(".", "_")
    csv_path = os.path.join(
        OUTPUT_DIR,
        f"comparison_{model_key}_{season}_GW{start_gw}_GW{end_gw}_budget_{budget_tag}.csv"
    )

    if save_csv:
        result_df.to_csv(csv_path, index=False)

    plot_path = None
    if save_plot:
        plot_path = plot_gw_comparison(result_df, season=f"{season}_GW{start_gw}_GW{end_gw}", budget=budget)

    print(f"\nSUMMARY — {season}  GW{start_gw}-GW{end_gw}")
    print("=" * 62)
    print(f"Total predicted XI pts   : {total_predicted:.1f}")
    print(f"Total optimal XI pts     : {total_optimal:.1f}")
    print(f"Total pts left on table  : {total_left:.1f}")
    print(f"Average gap per GW       : {avg_gap:.1f}")
    print(f"Best GW (predicted)      : GW{best_gw} ({best_gw_pts:.1f} pts)")
    print(f"Worst GW (biggest gap)   : GW{worst_gw} ({worst_gap:.1f} pts left on table)")
    print(f"Captain correct          : {captain_correct} / {total_weeks} gameweeks")
    print(f"Efficiency               : {efficiency:.1f}% (predicted / optimal)")

    if plot_path is not None:
        print(f"Saved plot               -> {plot_path}")
    if save_csv:
        print(f"Saved CSV                -> {csv_path}")

    return result_df


def compare_models_over_season(season, budget=None):
    all_results = []

    for model_key in ["linear", "xgboost", "position_xgboost"]:
        result_df = evaluate_model_over_season(model_key, season, budget=budget, save_csv=True)
        if result_df is None:
            continue

        gw_mae = mean_absolute_error(
            result_df["actual_team_score"],
            result_df["predicted_team_xPts"]
        )
        gw_rmse = mean_squared_error(
            result_df["actual_team_score"],
            result_df["predicted_team_xPts"]
        ) ** 0.5
        gw_corr = safe_corr(
            result_df["actual_team_score"],
            result_df["predicted_team_xPts"]
        )

        all_results.append({
            "model": model_key,
            "season": season,
            "budget": "unlimited" if budget is None else budget,
            "weeks_evaluated": len(result_df),
            "avg_predicted_team_xPts": round(result_df["predicted_team_xPts"].mean(), 3),
            "avg_actual_team_score": round(result_df["actual_team_score"].mean(), 3),
            "total_actual_team_score": round(result_df["actual_team_score"].sum(), 3),
            "gw_mae": round(gw_mae, 3),
            "gw_rmse": round(gw_rmse, 3),
            "gw_corr": round(gw_corr, 3)
        })

    if len(all_results) == 0:
        print("No model results available.")
        return None

    comparison_df = pd.DataFrame(all_results).sort_values(
        "total_actual_team_score", ascending=False
    ).reset_index(drop=True)

    print(f"\n==============================")
    print(f"BEST XI MODEL COMPARISON — {season}")
    print(f"==============================")
    print(comparison_df.to_string(index=False))

    budget_tag = "unlimited" if budget is None else str(budget).replace(".", "_")
    save_path = os.path.join(
        OUTPUT_DIR,
        f"best_xi_model_comparison_{season}_budget_{budget_tag}.csv"
    )
    comparison_df.to_csv(save_path, index=False)
    print(f"\nSaved comparison to: {save_path}")

    return comparison_df

# =========================================================
# Plot
# =========================================================

def plot_points_left_boxplot(season="2024-25", budget=83, save_dir=OUTPUT_DIR):
    import matplotlib.pyplot as plt

    lr_cmp = compare_season_performance(
        "linear", season=season, start_gw=1, end_gw=38,
        budget=budget, save_csv=False, save_plot=False
    )
    xgb_cmp = compare_season_performance(
        "xgboost", season=season, start_gw=1, end_gw=38,
        budget=budget, save_csv=False, save_plot=False
    )

    if lr_cmp is None or xgb_cmp is None:
        print("Missing comparison results.")
        return None

    data = [
        lr_cmp["pts_left_on_table"].dropna(),
        xgb_cmp["pts_left_on_table"].dropna()
    ]

    plt.figure(figsize=(6.5, 5.5))
    plt.boxplot(data, labels=["Linear Regression", "XGBoost"], showmeans=True)

    plt.ylabel("Points left on the table")
    plt.title(f"Weekly Gap to Optimal XI — {season}")
    plt.tight_layout()

    budget_tag = "unlimited" if budget is None else str(budget).replace(".", "_")
    save_path = os.path.join(save_dir, f"points_left_boxplot_{season}_budget_{budget_tag}.png")
    plt.savefig(save_path, dpi=220)
    plt.close()

    print(f"Saved boxplot to: {save_path}")
    return save_path

# =========================================================
# Comparision between XGBoost and Linear Regression
# =========================================================

def plot_team_error_boxplot_from_gw_summary(season="2024-25", budget=83, save_dir=OUTPUT_DIR):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    budget_tag = "unlimited" if budget is None else str(budget).replace(".", "_")

    lr_path = os.path.join(save_dir, f"linear_{season}_budget_{budget_tag}_gw_summary.csv")
    xgb_path = os.path.join(save_dir, f"xgboost_{season}_budget_{budget_tag}_gw_summary.csv")

    if not os.path.exists(lr_path):
        raise FileNotFoundError(f"Missing file: {lr_path}")
    if not os.path.exists(xgb_path):
        raise FileNotFoundError(f"Missing file: {xgb_path}")

    lr_df = pd.read_csv(lr_path)
    xgb_df = pd.read_csv(xgb_path)

    lr_err = (lr_df["actual_team_score"] - lr_df["predicted_team_xPts"]).abs().values
    xgb_err = (xgb_df["actual_team_score"] - xgb_df["predicted_team_xPts"]).abs().values

    data = [lr_err, xgb_err]
    labels = ["Linear Regression", "XGBoost"]

    fig, ax = plt.subplots(figsize=(9, 6.5))

    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        widths=0.5,
        showmeans=True,
        meanprops=dict(marker="^", markersize=9),
        medianprops=dict(linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markersize=6, markerfacecolor="white", markeredgewidth=1.5)
    )

    for box in bp["boxes"]:
        box.set_facecolor("#D9D9D9")
        box.set_linewidth(1.5)

    rng = np.random.default_rng(42)
    for i, values in enumerate(data, start=1):
        x_jitter = rng.normal(loc=i, scale=0.04, size=len(values))
        ax.scatter(x_jitter, values, alpha=0.45, s=28)

    for i, values in enumerate(data, start=1):
        mean_v = np.mean(values)
        median_v = np.median(values)
        ax.text(i + 0.18, mean_v, f"mean = {mean_v:.1f}", va="center", fontsize=10)
        ax.text(i + 0.18, median_v, f"median = {median_v:.1f}", va="center", fontsize=10)

    ax.set_title(f"Weekly Team Prediction Error — {season}", fontsize=18, pad=14)
    ax.set_ylabel("|Actual team score - Predicted team xPts|", fontsize=13)
    ax.text(
        0.5, 1.01,
        "Lower is better: smaller weekly prediction error",
        transform=ax.transAxes,
        ha="center",
        fontsize=11
    )

    legend_handles = [
        Line2D([0], [0], color="black", linewidth=2, label="Median"),
        Line2D([0], [0], marker="^", linestyle="None", markersize=9, color="black", label="Mean"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=6, markerfacecolor="white",
               markeredgecolor="black", label="Outlier"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=6, color="black", alpha=0.45,
               label="One gameweek"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True)

    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"team_error_boxplot_{season}_budget_{budget_tag}.png")
    plt.savefig(save_path, dpi=240, bbox_inches="tight")
    plt.close()

    print(f"Saved improved boxplot to: {save_path}")
    return save_path

def plot_error_ecdf_from_gw_summary(season="2024-25", budget=83, save_dir=OUTPUT_DIR):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    budget_tag = "unlimited" if budget is None else str(budget).replace(".", "_")

    lr_path = os.path.join(save_dir, f"linear_{season}_budget_{budget_tag}_gw_summary.csv")
    xgb_path = os.path.join(save_dir, f"xgboost_{season}_budget_{budget_tag}_gw_summary.csv")

    if not os.path.exists(lr_path):
        raise FileNotFoundError(f"Missing file: {lr_path}")
    if not os.path.exists(xgb_path):
        raise FileNotFoundError(f"Missing file: {xgb_path}")

    lr_df = pd.read_csv(lr_path)
    xgb_df = pd.read_csv(xgb_path)

    lr_err = (lr_df["actual_team_score"] - lr_df["predicted_team_xPts"]).abs().dropna().values
    xgb_err = (xgb_df["actual_team_score"] - xgb_df["predicted_team_xPts"]).abs().dropna().values

    def ecdf(values):
        x = np.sort(values)
        y = np.arange(1, len(x) + 1) / len(x)
        return x, y

    lr_x, lr_y = ecdf(lr_err)
    xgb_x, xgb_y = ecdf(xgb_err)

    plt.figure(figsize=(8.5, 5.8))
    plt.plot(lr_x, lr_y, marker="o", markevery=max(len(lr_x)//10, 1), linewidth=2, label="Linear Regression")
    plt.plot(xgb_x, xgb_y, marker="o", markevery=max(len(xgb_x)//10, 1), linewidth=2, label="XGBoost")

    plt.xlabel("Weekly absolute team prediction error")
    plt.ylabel("Proportion of gameweeks")
    plt.title(f"Cumulative Weekly Error Distribution — {season}")
    plt.figtext(
        0.5, 0.01,
        "Higher and further left is better.",
        ha="center",
        fontsize=10
    )

    # 右边放一个小 summary
    lr_mean = lr_err.mean()
    xgb_mean = xgb_err.mean()
    lr_median = np.median(lr_err)
    xgb_median = np.median(xgb_err)

    summary_text = (
        f"Linear: mean={lr_mean:.1f}, median={lr_median:.1f}\n"
        f"XGBoost: mean={xgb_mean:.1f}, median={xgb_median:.1f}"
    )
    plt.text(
        0.98, 0.08,
        summary_text,
        transform=plt.gca().transAxes,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85)
    )

    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"error_ecdf_{season}_budget_{budget_tag}.png")
    plt.savefig(save_path, dpi=240, bbox_inches="tight")
    plt.close()

    print(f"Saved ECDF plot to: {save_path}")
    return save_path

# =========================================================
# EXAMPLES
# =========================================================

# 1. Run model training and save prediction files
# RUN_TRAINING = True
# RUN_BEST_XI_ANALYSIS = False

# 2. Interactive single-week Best XI selector
# run_best_xi_selector()

# 3. Single model, whole-season weekly Best XI summary
# evaluate_model_over_season("linear", season="2024-25", budget=83)
# evaluate_model_over_season("xgboost", season="2024-25", budget=83)
# evaluate_model_over_season("position_xgboost", season="2024-25", budget=83)

# 4. Compare the three models across one season
# compare_models_over_season(season="2024-25", budget=83)
# compare_models_over_season(season="2024-25", budget=None)

# 5. Detailed predicted-vs-optimal comparison for one model
# compare_season_performance("linear", season="2024-25", start_gw=1, end_gw=38, budget=83)
# compare_season_performance("xgboost", season="2024-25", start_gw=1, end_gw=38, budget=83)
# compare_season_performance("position_xgboost", season="2024-25", start_gw=1, end_gw=38, budget=83)

# 6. Unlimited-budget version
# compare_season_performance("linear", season="2024-25", start_gw=1, end_gw=38, budget=None)

# 7. Shorter range example
# compare_season_performance("linear", season="2024-25", start_gw=1, end_gw=10, budget=83)

#if RUN_BEST_XI_ANALYSIS:
#   compare_models_over_season(season="2024-25", budget=83)

# Plot 3 models
# plot_points_left_boxplot(season="2024-25", budget=83)

#plot_team_error_boxplot_from_gw_summary(season="2024-25", budget=83)
plot_error_ecdf_from_gw_summary(season="2024-25", budget=83)
