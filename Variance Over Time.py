import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import scipy.stats as stats

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

SEASON = "2023-24"
TARGET_GW = 38
BUDGET = 1000
FORM_WINDOW = 5
EWMA_ALPHA = 0.6
ROLLING_WINDOW = 5

constraints = {"GK":1, "DEF":4, "MID":4, "FWD":2}
MAX_PER_TEAM = 3
MC_ITER = 10000

GRAPH_START_GW = 10
GRAPH_END_GW = 38

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("cleaned_merged_seasons_team_aggregated.csv", encoding="latin1", low_memory=False)
df = df[df["season_x"] == SEASON].copy()
df = df.sort_values(["name", "GW"])

# -------------------------
# FEATURE ENGINEERING
# -------------------------
df["goals_ewma"] = df.groupby("name")["goals_scored"].transform(lambda x: x.ewm(alpha=EWMA_ALPHA).mean())
df["assists_ewma"] = df.groupby("name")["assists"].transform(lambda x: x.ewm(alpha=EWMA_ALPHA).mean())
df["points_ewma"] = df.groupby("name")["total_points"].transform(lambda x: x.ewm(alpha=EWMA_ALPHA).mean())
df["minutes_ewma"] = df.groupby("name")["minutes"].transform(lambda x: x.ewm(alpha=EWMA_ALPHA).mean())
df["creativity_ewma"] = df.groupby("name")["creativity"].transform(lambda x: x.ewm(alpha=EWMA_ALPHA).mean())
df["threat_ewma"] = df.groupby("name")["threat"].transform(lambda x: x.ewm(alpha=EWMA_ALPHA).mean())

df["recent_points"] = df.groupby("name")["total_points"].transform(lambda x: x.rolling(FORM_WINDOW, min_periods=1).mean())
df["recent_goals"] = df.groupby("name")["goals_scored"].transform(lambda x: x.rolling(FORM_WINDOW, min_periods=1).mean())
df["recent_assists"] = df.groupby("name")["assists"].transform(lambda x: x.rolling(FORM_WINDOW, min_periods=1).mean())

df["std_points"] = df.groupby("name")["total_points"].transform(lambda x: x.rolling(FORM_WINDOW, min_periods=1).std())
df["std_points"] = df["std_points"].fillna(df["std_points"].mean()).replace(0, 0.1)

features = [
    "goals_ewma","assists_ewma","points_ewma","minutes_ewma",
    "creativity_ewma","threat_ewma","recent_points","recent_goals","recent_assists"
]

# -------------------------
# TEAM SELECTION
# -------------------------
def select_team(data, mode="consistent"):
    df_sim = data.copy()

    if mode == "consistent":
        df_sim["score"] = df_sim["sim_mean"] / (1 + df_sim["sim_std"])
    else:
        df_sim["score"] = df_sim["sim_mean"] * (1 + df_sim["sim_std"])

    df_sim["value_metric"] = df_sim["score"] / df_sim["value"]

    selected = []
    budget = BUDGET
    pos_count = {k:0 for k in constraints}
    team_count = {}

    for _, row in df_sim.sort_values("value_metric", ascending=False).iterrows():
        if pos_count[row["position"]] >= constraints[row["position"]]:
            continue
        if team_count.get(row["team_x"],0) >= MAX_PER_TEAM:
            continue
        if budget < row["value"]:
            continue

        selected.append(row)
        budget -= row["value"]
        pos_count[row["position"]] += 1
        team_count[row["team_x"]] = team_count.get(row["team_x"],0) + 1

        if sum(pos_count.values()) == 11:
            break

    return pd.DataFrame(selected)

# -------------------------
# BACKTEST LOOP
# -------------------------
def run_full_backtest(start_gw=5, end_gw=38):
    results = []

    for gw in range(start_gw, end_gw + 1):

        train_df = df[df["GW"] < gw]
        test_df = df[df["GW"] == gw]

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        X_train = train_df[features].fillna(0)
        y_train = train_df["total_points"]

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        test_df = test_df.copy()
        test_df["expected_points"] = model.predict(test_df[features].fillna(0))
        test_df["sim_mean"] = test_df["expected_points"]
        test_df["sim_std"] = test_df["std_points"]

        cons_team = select_team(test_df, "consistent")
        risk_team = select_team(test_df, "risky")

        def score(team):
            merged = team.merge(
                test_df.copy(),
                on="name",
                how="left",
                suffixes=("", "_y")
            )

            if "total_points" in merged.columns:
                merged["actual_points"] = merged["total_points"]
            elif "points" in merged.columns:
                merged["actual_points"] = merged["points"]
            else:
                raise ValueError("No actual points column found")

            pred = merged["sim_mean"].sum()
            act = merged["actual_points"].fillna(0).sum()
            err = pred - act

            return pred, act, err

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

# -------------------------
# FILTER FOR GRAPHING
# -------------------------
plot_df = backtest_df[
    (backtest_df["GW"] >= GRAPH_START_GW) &
    (backtest_df["GW"] <= GRAPH_END_GW)
].copy()

# -------------------------
# PLOTS
# -------------------------
plt.figure()
plt.plot(plot_df["GW"], plot_df["cons_pred"], label="Cons Pred")
plt.plot(plot_df["GW"], plot_df["cons_actual"], label="Cons Actual")
plt.plot(plot_df["GW"], plot_df["risk_pred"], linestyle="--", label="Risk Pred")
plt.plot(plot_df["GW"], plot_df["risk_actual"], linestyle="--", label="Risk Actual")
plt.legend()
plt.title("Predicted vs Actual Over Season")
plt.show()

plt.figure()
plt.plot(plot_df["GW"], plot_df["cons_error"], label="Cons Error")
plt.plot(plot_df["GW"], plot_df["risk_error"], label="Risk Error")
plt.axhline(0)
plt.legend()
plt.title("Error Over Season")
plt.show()

# -------------------------
# ROLLING MAE + BIAS
# -------------------------
plt.figure()

plot_df["cons_mae"] = plot_df["cons_error"].abs().rolling(ROLLING_WINDOW).mean()
plot_df["risk_mae"] = plot_df["risk_error"].abs().rolling(ROLLING_WINDOW).mean()

plt.plot(plot_df["GW"], plot_df["cons_mae"], label="Consistent MAE")
plt.plot(plot_df["GW"], plot_df["risk_mae"], label="Risky MAE")

plt.xlabel("Gameweek")
plt.ylabel("MAE")
plt.title("Rolling Prediction Error (MAE)")

plt.legend()
plt.show()

plt.figure()

plot_df["cons_bias"] = plot_df["cons_error"].rolling(ROLLING_WINDOW).mean()
plot_df["risk_bias"] = plot_df["risk_error"].rolling(ROLLING_WINDOW).mean()

plt.plot(plot_df["GW"], plot_df["cons_bias"], label="Consistent Bias")
plt.plot(plot_df["GW"], plot_df["risk_bias"], label="Risky Bias")

plt.axhline(0)

plt.xlabel("Gameweek")
plt.ylabel("Bias")
plt.title("Rolling Prediction Bias")

plt.legend()
plt.show()