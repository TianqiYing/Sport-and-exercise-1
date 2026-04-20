import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.inspection import permutation_importance

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

SEASON = "2024-25"
PREV_SEASON = "2023-24"
TARGET_GW = 38
BUDGET = 830
FORM_WINDOW = 3
EWMA_ALPHA = 0.1

constraints = {"GK":1, "DEF":4, "MID":4, "FWD":2}
MAX_PER_TEAM = 3
MC_ITER = 20000

# -------------------------
# LOAD DATA (WITH PREVIOUS SEASON)
# -------------------------
df = pd.read_csv("active_perfect_understat_enhanced.csv", encoding="latin1", low_memory=False)

df = df[
    (df["season_x"] == SEASON) |
    (df["season_x"] == PREV_SEASON)
].copy()

df = df.sort_values(["player", "season_x", "gameweek"])

# -------------------------
# POSITION MAPPING + SUB FIX
# -------------------------
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
    else:
        return np.nan

df["position_mapped"] = df["position"].apply(map_position)

df["position_mapped"] = df.groupby("player")["position_mapped"]\
    .transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else "MID"))

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
    "form_xg_5",
    "form_xa_5",
    "form_shots_5",
    "form_key_passes_y_5",
    "clean_sheet_form"
]

# -------------------------
# TRAIN MODEL (PREV SEASON + CURRENT)
# -------------------------
train_df = df[
    (df["season_x"] == PREV_SEASON) |
    ((df["season_x"] == SEASON) & (df["gameweek"] < TARGET_GW))
].copy()

X = train_df[features].fillna(0)
y = train_df["total_points"]

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X, y)

# -------------------------
# PREDICT TARGET GW
# -------------------------
gw_df = df[
    (df["season_x"] == SEASON) &
    (df["gameweek"] == TARGET_GW)
].copy()

gw_df["expected_points"] = model.predict(gw_df[features].fillna(0))

gw_df["sim_mean"] = gw_df["expected_points"]
gw_df["sim_std"] = gw_df["std_points"]

# -------------------------
# TEAM SELECTION
# -------------------------
def select_team(data, mode="consistent"):
    df_sim = data.copy()

    if mode == "consistent":
        df_sim["score"] = 2* df_sim["sim_mean"] / (1 + df_sim["sim_std"])
    else:
        df_sim["score"] = 2* df_sim["sim_mean"] * (1 + df_sim["sim_std"])

    df_sim["value_metric"] = df_sim["score"] / 0.5 * df_sim["price"]

    selected = []
    budget = BUDGET
    pos_count = {k:0 for k in constraints}
    team_count = {}

    for _, row in df_sim.sort_values("value_metric", ascending=False).iterrows():
        pos = row["position_mapped"]

        if pos_count[pos] >= constraints[pos]:
            continue
        if team_count.get(row["team"],0) >= MAX_PER_TEAM:
            continue
        if budget < row["price"]:
            continue

        selected.append(row)
        budget -= row["price"]
        pos_count[pos] += 1
        team_count[row["team"]] = team_count.get(row["team"],0) + 1

        if sum(pos_count.values()) == 11:
            break

    return pd.DataFrame(selected)

consistent_team = select_team(gw_df, "consistent")
risky_team = select_team(gw_df, "risky")

# -------------------------
# PRINT RESULTS
# -------------------------
def print_team(team, title):
    print("\n" + title)
    print("-" * len(title))

    print(team[[
        "player","position_mapped","team","opponent","is_home",
        "price","sim_mean","sim_std"
    ]].rename(columns={
        "player":"name",
        "position_mapped":"position",
        "is_home":"home",
        "sim_mean":"expected_points",
        "sim_std":"std_points"
    }))

    print("\nTotal Price:", team["price"].sum())
    print("Total Expected Points:", team["sim_mean"].sum())
    print("Actual Points", team["total_points"].sum())

print_team(consistent_team, "Consistent Team")
print_team(risky_team, "Risky Team")


# -------------------------
# MODEL EVALUATION
# -------------------------
eval_df = gw_df[["player","sim_mean","total_points"]].copy()
eval_df.rename(columns={
    "player":"name",
    "sim_mean":"predicted_points",
    "total_points":"actual_points"
}, inplace=True)

eval_df["error"] = eval_df["predicted_points"] - eval_df["actual_points"]

print("\nMODEL PERFORMANCE")
print("MAE:", eval_df["error"].abs().mean())
print("RMSE:", np.sqrt((eval_df["error"]**2).mean()))
print("Bias:", eval_df["error"].mean())

# -------------------------
# RESIDUAL PLOT
# -------------------------
plt.figure()
plt.scatter(eval_df["predicted_points"], eval_df["error"])
plt.axhline(0)
plt.xlabel("Predicted Points")
plt.ylabel("Error")
plt.title("Residual Plot")
plt.show()

# -------------------------
# PLAYER DISTRIBUTIONS
# -------------------------
plt.figure()

x_vals = np.linspace(
    0,
    max(consistent_team["sim_mean"].max(), risky_team["sim_mean"].max()) + 5,
    500
)

def plot_team(team, color):
    for _, row in team.iterrows():
        mean = row["sim_mean"]
        std = max(row["sim_std"], 0.4)
        pdf = stats.norm.pdf(x_vals, mean, std)
        plt.plot(x_vals, pdf, linestyle="--", alpha=0.5, color=color)

    for _, row in team.iterrows():
        mean = row["sim_mean"]
        std = max(row["sim_std"], 0.4)
        peak_y = stats.norm.pdf(mean, mean, std)
        plt.text(mean, peak_y, row["player"], rotation=45, ha="left", va="bottom", fontsize=8, color=color)

plot_team(consistent_team, "blue")
plot_team(risky_team, "red")

plt.title("Player Distributions")
plt.xlabel("Points")
plt.ylabel("Density")
plt.show()

# -------------------------
# TEAM DISTRIBUTIONS
# -------------------------
plt.figure()

def team_dist(team, color, label):
    sims = []
    for _ in range(MC_ITER):
        sims.append(np.random.normal(team["sim_mean"], team["sim_std"]).sum())
    kde = stats.gaussian_kde(sims)
    x = np.linspace(min(sims), max(sims), 500)
    plt.plot(x, kde(x), color=color, label=label)

team_dist(consistent_team, "blue", "Consistent")
team_dist(risky_team, "red", "Risky")

plt.legend()
plt.title("Team Distribution")
plt.show()