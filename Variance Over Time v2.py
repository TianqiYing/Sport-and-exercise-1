import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import scipy.stats as stats
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

SEASON = "2024-25"
PREV_SEASON = "2023-24"
TARGET_GW = 38
BUDGET = 830
FORM_WINDOW = 5
EWMA_ALPHA = 0.6
ROLLING_WINDOW = 5

constraints = {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2}
MAX_PER_TEAM = 3
MC_ITER = 10000

GRAPH_START_GW = 1
GRAPH_END_GW = 38

df = pd.read_csv("active_perfect_understat_enhanced.csv", encoding="latin1", low_memory=False)

df = df[
    (df["season_x"] == SEASON) |
    (df["season_x"] == PREV_SEASON)
].copy()

df = df.sort_values(["player", "season_x", "gameweek"])

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

# -------------------------
# FEATURE ENGINEERING (NO DATA LEAKAGE)
# -------------------------

df["points_ewma"] = df.groupby("player")["total_points"].transform(
    lambda x: x.shift(1).ewm(alpha=EWMA_ALPHA).mean()
)

df["minutes_ewma"] = df.groupby("player")["minutes_x"].transform(
    lambda x: x.shift(1).ewm(alpha=EWMA_ALPHA).mean()
)

df["creativity_ewma"] = df.groupby("player")["creativity"].transform(
    lambda x: x.shift(1).ewm(alpha=EWMA_ALPHA).mean()
)

df["threat_ewma"] = df.groupby("player")["threat"].transform(
    lambda x: x.shift(1).ewm(alpha=EWMA_ALPHA).mean()
)

df["recent_points"] = df.groupby("player")["total_points"].transform(
    lambda x: x.shift(1).rolling(FORM_WINDOW, min_periods=1).mean()
)

df["recent_goals"] = df.groupby("player")["goals_scored"].transform(
    lambda x: x.shift(1).rolling(FORM_WINDOW, min_periods=1).mean()
)

df["recent_assists"] = df.groupby("player")["assists_x"].transform(
    lambda x: x.shift(1).rolling(FORM_WINDOW, min_periods=1).mean()
)

# -------------------------
# STD (ALSO SHIFTED)
# -------------------------

df["std_rolling"] = df.groupby("player")["total_points"].transform(
    lambda x: x.shift(1).rolling(FORM_WINDOW, min_periods=2).std()
)

df["std_expanding"] = df.groupby("player")["total_points"].transform(
    lambda x: x.shift(1).expanding().std()
)

USE_EXPANDING_STD = True
df["std_points"] = df["std_expanding"] if USE_EXPANDING_STD else df["std_rolling"]

df["std_points"] = df.groupby("player")["std_points"].transform(
    lambda x: x.fillna(x.expanding().mean())
)

df["std_points"] = df["std_points"].fillna(df["std_points"].median()).replace(0, 0.1)

features = [
    "points_ewma",
    "minutes_ewma",
    "creativity_ewma",
    "threat_ewma",
    "recent_points",
    "recent_goals",
    "recent_assists",
]

train_df = df[
    (df["season_x"] == PREV_SEASON) |
    ((df["season_x"] == SEASON) & (df["gameweek"] < TARGET_GW))
].copy()

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(train_df[features].fillna(0), train_df["total_points"])

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

def write_team_sheet(wb, df_team, sheet_name):
    ws = wb.create_sheet(title=str(sheet_name))

    cols = [
        "player", "position_mapped", "team", "opponent", "is_home",
        "price", "sim_mean", "sim_std", "total_points"
    ]

    df_out = df_team[cols].copy()

    for r in dataframe_to_rows(df_out, index=False, header=True):
        ws.append(r)

    ws.append([])
    ws.append(["Total Price", df_team["price"].sum()])
    ws.append(["Total Expected Points", df_team["sim_mean"].sum()])
    ws.append(["Total Actual Points", df_team["total_points"].fillna(0).sum()])

def run_full_backtest(start_gw=1, end_gw=38):

    wb = Workbook()
    wb.remove(wb.active)

    results = []

    def score(team):
        pred = team["sim_mean"].sum()
        act = team["total_points"].fillna(0).sum()
        err = pred - act
        return pred, act, err

    base_df = df.copy()

    for gw in range(start_gw, end_gw + 1):

        train = base_df[
            (base_df["season_x"] == PREV_SEASON) |
            ((base_df["season_x"] == SEASON) & (base_df["gameweek"] < gw))
        ].copy()

        test_raw = base_df[
            (base_df["season_x"] == SEASON) &
            (base_df["gameweek"] == gw)
        ].copy()

        test = test_raw.sort_values(["player", "gameweek"]).groupby("player").tail(1).copy()

        if len(train) == 0 or len(test) == 0:
            continue

        model.fit(train[features].fillna(0), train["total_points"])

        test = test.copy()
        test["expected_points"] = model.predict(test[features].fillna(0))
        test["sim_mean"] = test["expected_points"]
        test["sim_std"] = test["std_points"]

        cons_team = select_team(test, "consistent")
        risk_team = select_team(test, "risky")

        print("\nGW", gw)

        print("\nCONSISTENT TEAM")
        print(cons_team[[
            "player", "position_mapped", "team", "price", "sim_mean", "sim_std", "total_points"
        ]].to_string(index=False))

        print("\nRISKY TEAM")
        print(risk_team[[
            "player", "position_mapped", "team", "price", "sim_mean", "sim_std", "total_points"
        ]].to_string(index=False))

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

        write_team_sheet(wb, cons_team, f"GW{gw}_Consistent")
        write_team_sheet(wb, risk_team, f"GW{gw}_Risky")

    wb.save("FPL_All_Gameweek_Teams.xlsx")

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