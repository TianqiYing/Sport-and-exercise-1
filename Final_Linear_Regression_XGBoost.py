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
# 11. RUN ALL MODELS
# =========================================================
lr_output, lr_season_summary, lr_overall = run_global_model(df, "Linear Regression")
xgb_output, xgb_season_summary, xgb_overall = run_global_model(df, "Single XGBoost")
pos_output, pos_season_summary, pos_overall = run_position_specific_xgboost(df)

# =========================================================
# 12. FINAL COMPARISON
# =========================================================
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
        defs = gw[gw["FPL_pos"] == "DEF"].sort_values("xPts", ascending=False).head(8)
        mids = gw[gw["FPL_pos"] == "MID"].sort_values("xPts", ascending=False).head(8)
        fwds = gw[gw["FPL_pos"] == "FWD"].sort_values("xPts", ascending=False).head(6)

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

    print(f"\nBest XI — {season} GW{gameweek}")
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

    print("\nAverages:")
    print("Average predicted team xPts :", round(result_df["predicted_team_xPts"].mean(), 3))
    print("Average actual team score   :", round(result_df["actual_team_score"].mean(), 3))
    print("Total actual team score     :", round(result_df["actual_team_score"].sum(), 3))

    if save_csv:
        budget_tag = "unlimited" if budget is None else str(budget).replace(".", "_")
        save_path = os.path.join(
            OUTPUT_DIR,
            f"{model_key}_{season}_budget_{budget_tag}_gw_summary.csv"
        )
        result_df.to_csv(save_path, index=False)
        print(f"\nSaved season summary to: {save_path}")

    return result_df


def compare_models_over_season(season, budget=None):
    all_results = []

    for model_key in ["linear", "xgboost", "position_xgboost"]:
        result_df = evaluate_model_over_season(model_key, season, budget=budget, save_csv=True)
        if result_df is None:
            continue

        all_results.append({
            "model": model_key,
            "season": season,
            "budget": "unlimited" if budget is None else budget,
            "weeks_evaluated": len(result_df),
            "avg_predicted_team_xPts": round(result_df["predicted_team_xPts"].mean(), 3),
            "avg_actual_team_score": round(result_df["actual_team_score"].mean(), 3),
            "total_actual_team_score": round(result_df["actual_team_score"].sum(), 3)
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


# Example:
#run_best_xi_selector()
#evaluate_model_over_season("linear", season="2024-25", budget=83)
#compare_models_over_season(season="2024-25", budget=83)
