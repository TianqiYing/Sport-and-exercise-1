import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import scipy.stats as stats

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

# -------------------------
# Parameters
# -------------------------
SEASON = "2023-24"
TARGET_GW = 38
BUDGET = 1000
FORM_WINDOW = 5
EWMA_ALPHA = 0.6

constraints = {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2}
MAX_PER_TEAM = 3
MC_ITER = 10000

# -------------------------
# Load data
# -------------------------
df = pd.read_csv("cleaned_merged_seasons_team_aggregated.csv", encoding="latin1", low_memory=False)
df = df[df["season_x"] == SEASON].copy()
df = df.sort_values(["name", "GW"])

# -------------------------
# Recency features
# -------------------------
df["goals_ewma"] = df.groupby("name")["goals_scored"].transform(lambda x: x.ewm(alpha=EWMA_ALPHA).mean())
df["assists_ewma"] = df.groupby("name")["assists"].transform(lambda x: x.ewm(alpha=EWMA_ALPHA).mean())
df["points_ewma"] = df.groupby("name")["total_points"].transform(lambda x: x.ewm(alpha=EWMA_ALPHA).mean())
df["minutes_ewma"] = df.groupby("name")["minutes"].transform(lambda x: x.ewm(alpha=EWMA_ALPHA).mean())
df["creativity_ewma"] = df.groupby("name")["creativity"].transform(lambda x: x.ewm(alpha=EWMA_ALPHA).mean())
df["threat_ewma"] = df.groupby("name")["threat"].transform(lambda x: x.ewm(alpha=EWMA_ALPHA).mean())


def rolling_stats(x):
    return x.rolling(FORM_WINDOW, min_periods=1).mean()


df["recent_points"] = df.groupby("name")["total_points"].transform(rolling_stats)
df["recent_goals"] = df.groupby("name")["goals_scored"].transform(rolling_stats)
df["recent_assists"] = df.groupby("name")["assists"].transform(rolling_stats)

df["std_points"] = df.groupby("name")["total_points"].transform(lambda x: x.rolling(FORM_WINDOW, min_periods=1).std())
df["std_points"] = df["std_points"].fillna(df["std_points"].mean())
df["std_points"] = df["std_points"].replace(0, 0.1)

# -------------------------
# Train model
# -------------------------
train_df = df[df["GW"] < TARGET_GW].copy()

features = ["goals_ewma", "assists_ewma", "points_ewma", "minutes_ewma",
            "creativity_ewma", "threat_ewma", "recent_points", "recent_goals", "recent_assists"]

X = train_df[features].fillna(0)
y = train_df["total_points"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# -------------------------
# Predict GW
# -------------------------
gw_df = df[df["GW"] == TARGET_GW].copy()
X_pred = gw_df[features].fillna(0)
gw_df["expected_points"] = model.predict(X_pred)

team_def = train_df.groupby("team_x")["goals_conceded"].mean().to_dict()
avg_def = np.mean(list(team_def.values()))

gw_df["opp_def_strength"] = gw_df["opp_team_name"].map(team_def).fillna(avg_def)
gw_df["fixture_factor"] = gw_df["opp_def_strength"] / avg_def
gw_df["home_factor"] = gw_df["was_home"].apply(lambda x: 1.1 if x else 0.9)

gw_df["expected_points"] *= gw_df["fixture_factor"] * gw_df["home_factor"]

gw_df["sim_mean"] = gw_df["expected_points"]
gw_df["sim_std"] = gw_df["std_points"]


# -------------------------
# Team selection
# -------------------------
def select_team(df, mode="consistent"):
    df_sim = df.copy()

    if mode == "consistent":
        df_sim["score"] = df_sim["sim_mean"] / (1 + df_sim["sim_std"])
    else:
        df_sim["score"] = df_sim["sim_mean"] * (1 + df_sim["sim_std"])

    df_sim["value_metric"] = df_sim["score"] / df_sim["value"]

    selected = []
    budget = BUDGET
    pos_count = {k: 0 for k in constraints}
    team_count = {}

    for _, row in df_sim.sort_values("value_metric", ascending=False).iterrows():
        if pos_count[row["position"]] >= constraints[row["position"]]:
            continue
        if team_count.get(row["team_x"], 0) >= MAX_PER_TEAM:
            continue
        if budget < row["value"]:
            continue

        selected.append(row)
        budget -= row["value"]
        pos_count[row["position"]] += 1
        team_count[row["team_x"]] = team_count.get(row["team_x"], 0) + 1

        if sum(pos_count.values()) == 11:
            break

    return pd.DataFrame(selected)


consistent_team = select_team(gw_df, "consistent")
risky_team = select_team(gw_df, "risky")


# -------------------------
# Print Teams with Fixtures
# -------------------------
def print_team(team, title):
    print("\n" + title)
    print("-" * len(title))

    print(team[[
        "name",
        "position",
        "team_x",
        "opp_team_name",
        "was_home",
        "value",
        "sim_mean",
        "sim_std"
    ]].rename(columns={
        "team_x": "team",
        "opp_team_name": "opponent",
        "was_home": "home",
        "value": "price",
        "sim_mean": "expected_points",
        "sim_std": "std_points"
    }))

    print("\nTotal Price:", team["value"].sum())
    print("Total Expected Points:", team["sim_mean"].sum())


print_team(consistent_team, "Consistent Team")
print_team(risky_team, "Risky Team")

# -------------------------
# Player Distributions
# -------------------------
plt.figure(figsize=(16, 6))
x_vals = np.linspace(0, max(consistent_team["sim_mean"].max(), risky_team["sim_mean"].max()) + 5, 500)


def plot_team(team, color):
    for _, row in team.iterrows():
        mean = row["sim_mean"]
        std = max(row["sim_std"], 0.1)
        pdf = stats.norm.pdf(x_vals, mean, std)
        plt.plot(x_vals, pdf, linestyle="--", alpha=0.5, color=color)

    for _, row in team.iterrows():
        mean = row["sim_mean"]
        std = max(row["sim_std"], 0.1)
        peak_y = stats.norm.pdf(mean, mean, std)
        plt.text(mean, peak_y, row["name"], rotation=45, fontsize=8, ha='left', va='bottom', color=color)


plot_team(consistent_team, "blue")
plot_team(risky_team, "red")

plt.title("Player Distributions – Consistent vs Risky")
plt.xlabel("Points")
plt.ylabel("Probability Density")
plt.show()

# -------------------------
# Team Total Distributions
# -------------------------
plt.figure(figsize=(12, 6))


def team_dist(team, color, label):
    sims = []
    for _ in range(MC_ITER):
        sims.append(np.random.normal(team["sim_mean"], team["sim_std"]).sum())

    kde = stats.gaussian_kde(sims)
    x = np.linspace(min(sims), max(sims), 500)
    plt.plot(x, kde(x), color=color, label=label)


team_dist(consistent_team, "blue", "Consistent")
team_dist(risky_team, "red", "Risky")

plt.title("Team Total Points Distribution")
plt.xlabel("Total Points")
plt.ylabel("Density")
plt.legend()
plt.show()