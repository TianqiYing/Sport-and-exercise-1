import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import scipy.stats as stats

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

SEASON = "2024-25"
TARGET_GW = 38
BUDGET = 830
FORM_WINDOW = 5
EWMA_ALPHA = 0.6
ROLLING_WINDOW = 5

constraints = {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2}
MAX_PER_TEAM = 3
MC_ITER = 10000

GRAPH_START_GW = 10
GRAPH_END_GW = 38

df = pd.read_csv("active_perfect_understat_enhanced.csv", encoding="latin1", low_memory=False)
df = df[df["season_x"] == SEASON].copy()
df = df.sort_values(["player", "gameweek"])

def map_position(pos):
    if pos == "GK":
        return "GK"
    elif pos in ["DC", "DL", "DR", "DMC", "DML", "DMR"]:
        return "DEF"
    elif pos in ["MC", "ML", "MR", "AMC", "AML", "AMR"]:
        return "MID"
    elif pos in ["FW", "FWL", "FWR"]:
        return "FWD"
    elif pos == "Sub":
        return np.nan
    return np.nan

df["position_mapped"] = df["position"].apply(map_position)

df["position_mapped"] = df.groupby("player")["position_mapped"].transform(
    lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else "MID")
)

df = df[df["minutes_x"] > 0]

df["points_ewma"] = df.groupby("player")["total_points"].transform(
    lambda x: x.ewm(alpha=EWMA_ALPHA).mean()
)
df["minutes_ewma"] = df.groupby("player")["minutes_x"].transform(
    lambda x: x.ewm(alpha=EWMA_ALPHA).mean()
)
df["creativity_ewma"] = df.groupby("player")["creativity"].transform(
    lambda x: x.ewm(alpha=EWMA_ALPHA).mean()
)
df["threat_ewma"] = df.groupby("player")["threat"].transform(
    lambda x: x.ewm(alpha=EWMA_ALPHA).mean()
)

df["recent_points"] = df["form_pts_5"]
df["recent_goals"] = df["form_goals_5"]
df["recent_assists"] = df["form_assists_5"]

df["std_points"] = df.groupby("player")["total_points"].transform(
    lambda x: x.rolling(FORM_WINDOW, min_periods=1).std()
)
df["std_points"] = df["std_points"].fillna(df["std_points"].mean()).replace(0, 0.1)

features = [
    "points_ewma",
    "minutes_ewma",
    "creativity_ewma",
    "threat_ewma",
    "recent_points",
    "recent_goals",
    "recent_assists",
    "form_xg_5",
    "form_xa_5",
    "form_shots_5",
    "form_key_passes_y_5",
    "clean_sheet_form"
]

def select_team(data, mode="consistent"):
    df_sim = data.copy()

    if mode == "consistent":
        df_sim["score"] = df_sim["sim_mean"] / (1 + df_sim["sim_std"])
    else:
        df_sim["score"] = df_sim["sim_mean"] * (1 + df_sim["sim_std"])

    df_sim["value_metric"] = df_sim["score"] / df_sim["price"]

    selected = []
    budget = BUDGET
    pos_count = {k: 0 for k in constraints}
    team_count = {}

    for _, row in df_sim.sort_values("value_metric", ascending=False).iterrows():
        pos = row["position_mapped"]

        if pos_count[pos] >= constraints[pos]:
            continue
        if team_count.get(row["team"], 0) >= MAX_PER_TEAM:
            continue
        if budget < row["price"]:
            continue

        selected.append(row)
        budget -= row["price"]
        pos_count[pos] += 1
        team_count[row["team"]] = team_count.get(row["team"], 0) + 1

        if sum(pos_count.values()) == 11:
            break

    return pd.DataFrame(selected)

def run_full_backtest(start_gw=5, end_gw=38):
    results = []

    def score(team):
        pred = team["sim_mean"].sum()
        act = team["total_points"].fillna(0).sum()
        err = pred - act
        return pred, act, err

    for gw in range(start_gw, end_gw + 1):

        train_df = df[df["gameweek"] < gw]
        test_df = df[df["gameweek"] == gw]

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(train_df[features].fillna(0), train_df["total_points"])

        test_df = test_df.copy()
        test_df["expected_points"] = model.predict(test_df[features].fillna(0))
        test_df["sim_mean"] = test_df["expected_points"]
        test_df["sim_std"] = test_df["std_points"]

        cons_team = select_team(test_df, "consistent")
        risk_team = select_team(test_df, "risky")

        cp, ca, ce = score(cons_team)
        rp, ra, re = score(risk_team)

        results.append({
            "GW": gw,
            "cons_pred": cp,
            "cons_actual": ca,
            "cons_error": ce,
            "risk_pred": rp,
            "risk_actual": ra,
            "risk_error": re
        })

    return pd.DataFrame(results)

backtest_df = run_full_backtest()

plot_df = backtest_df[
    (backtest_df["GW"] >= GRAPH_START_GW) &
    (backtest_df["GW"] <= GRAPH_END_GW)
].copy()

plt.figure()
plt.plot(plot_df["GW"], plot_df["cons_pred"], label="Cons Pred")
plt.plot(plot_df["GW"], plot_df["cons_actual"], label="Cons Actual")
plt.plot(plot_df["GW"], plot_df["risk_pred"], linestyle="--", label="Risk Pred")
plt.plot(plot_df["GW"], plot_df["risk_actual"], linestyle="--", label="Risk Actual")
plt.legend()
plt.show()

plt.figure()
plt.plot(plot_df["GW"], plot_df["cons_error"], label="Cons Error")
plt.plot(plot_df["GW"], plot_df["risk_error"], label="Risk Error")
plt.axhline(0)
plt.legend()
plt.show()

plt.figure()

plot_df["cons_mae"] = plot_df["cons_error"].abs().rolling(ROLLING_WINDOW).mean()
plot_df["risk_mae"] = plot_df["risk_error"].abs().rolling(ROLLING_WINDOW).mean()

plt.plot(plot_df["GW"], plot_df["cons_mae"], label="Cons MAE")
plt.plot(plot_df["GW"], plot_df["risk_mae"], label="Risk MAE")
plt.legend()
plt.show()

plt.figure()

plot_df["cons_bias"] = plot_df["cons_error"].rolling(ROLLING_WINDOW).mean()
plot_df["risk_bias"] = plot_df["risk_error"].rolling(ROLLING_WINDOW).mean()

plt.plot(plot_df["GW"], plot_df["cons_bias"], label="Cons Bias")
plt.plot(plot_df["GW"], plot_df["risk_bias"], label="Risk Bias")
plt.axhline(0)
plt.legend()
plt.show()