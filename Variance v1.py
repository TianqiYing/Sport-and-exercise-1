import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import scipy.stats as stats

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
MC_ITER = 10000  # Monte Carlo iterations for team distribution

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


# Rolling recent stats
def rolling_stats(x):
    return x.rolling(FORM_WINDOW, min_periods=1).mean()


df["recent_points"] = df.groupby("name")["total_points"].transform(rolling_stats)
df["recent_goals"] = df.groupby("name")["goals_scored"].transform(rolling_stats)
df["recent_assists"] = df.groupby("name")["assists"].transform(rolling_stats)

# Standard deviation
df["std_points"] = df.groupby("name")["total_points"].transform(lambda x: x.rolling(FORM_WINDOW, min_periods=1).std())
df["std_points"] = df["std_points"].fillna(df["std_points"].mean())
df["std_points"] = df["std_points"].replace(0, 0.1)  # avoid zero std

# -------------------------
# Train on all rows before TARGET_GW
# -------------------------
train_df = df[df["GW"] < TARGET_GW].copy()
features = ["goals_ewma", "assists_ewma", "points_ewma", "minutes_ewma",
            "creativity_ewma", "threat_ewma", "recent_points", "recent_goals", "recent_assists"]

X = train_df[features].fillna(0)
y = train_df["total_points"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# -------------------------
# Predict for TARGET_GW
# -------------------------
gw_df = df[df["GW"] == TARGET_GW].copy()
X_pred = gw_df[features].fillna(0)
gw_df["expected_points"] = model.predict(X_pred)

# Fixture difficulty adjustment
team_def = train_df.groupby("team_x")["goals_conceded"].mean().to_dict()
avg_def = np.mean(list(team_def.values()))
gw_df["opp_def_strength"] = gw_df["opp_team_name"].map(team_def).fillna(avg_def)
gw_df["fixture_factor"] = gw_df["opp_def_strength"] / avg_def
gw_df["home_factor"] = gw_df["was_home"].apply(lambda x: 1.1 if x else 0.9)
gw_df["expected_points"] *= gw_df["fixture_factor"] * gw_df["home_factor"]

# Monte Carlo simulation for risk
np.random.seed(42)
gw_df["sim_mean"] = gw_df["expected_points"]
gw_df["sim_std"] = gw_df["std_points"]


# -------------------------
# Team selection function
# -------------------------
def select_team(df, mode="consistent"):
    df_sim = df.copy()
    if mode == "consistent":
        df_sim["score"] = df_sim["sim_mean"] / (1 + df_sim["sim_std"])
    elif mode == "risky":
        df_sim["score"] = df_sim["sim_mean"] * (1 + df_sim["sim_std"])
    else:
        df_sim["score"] = df_sim["sim_mean"]

    df_sim["value_metric"] = df_sim["score"] / df_sim["value"]

    selected = []
    budget = BUDGET
    pos_count = {k: 0 for k in constraints}
    team_count = {}

    df_sorted = df_sim.sort_values("value_metric", ascending=False)

    for _, row in df_sorted.iterrows():
        pos = row["position"]
        team = row["team_x"]
        price = row["value"]

        if pos_count[pos] >= constraints[pos]:
            continue
        if team_count.get(team, 0) >= MAX_PER_TEAM:
            continue
        if budget < price:
            continue

        selected.append(row)
        budget -= price

        pos_count[pos] += 1
        team_count[team] = team_count.get(team, 0) + 1

        if sum(pos_count.values()) == 11:
            break

    return pd.DataFrame(selected)


# -------------------------
# Generate teams
# -------------------------
consistent_team = select_team(gw_df, "consistent")
risky_team = select_team(gw_df, "risky")

# -------------------------
# Combined Player Distributions with Names on Peaks
# -------------------------
plt.figure(figsize=(16, 6))
x_vals = np.linspace(
    0,
    max(consistent_team["sim_mean"].max(), risky_team["sim_mean"].max()) + 5,
    500
)


def plot_team_players_fixed_labels(team, color):
    for _, row in team.sort_values("sim_mean").iterrows():
        mean = row["sim_mean"]
        std = max(row["sim_std"], 0.1)
        pdf = stats.norm.pdf(x_vals, mean, std)
        plt.plot(x_vals, pdf, linestyle="--", alpha=0.5, color=color)

        peak_y = stats.norm.pdf(mean, mean, std)
        plt.text(mean, peak_y, row["name"], rotation=45, fontsize=8, ha='left', va='bottom', color=color)


plot_team_players_fixed_labels(consistent_team, "blue")
plot_team_players_fixed_labels(risky_team, "red")

plt.title("Player Distributions – Consistent vs Risky")
plt.xlabel("Points")
plt.ylabel("Probability Density")
plt.show()

# -------------------------
# Combined Team Total Distributions
# -------------------------
plt.figure(figsize=(12, 6))


def team_total_distribution_mc(team, color, label):
    mc_sums = []
    for _ in range(MC_ITER):
        sample = np.random.normal(team["sim_mean"], team["sim_std"])
        mc_sums.append(sample.sum())
    mc_sums = np.array(mc_sums)

    kde = stats.gaussian_kde(mc_sums)
    x = np.linspace(mc_sums.min(), mc_sums.max(), 500)
    y = kde(x)

    plt.plot(x, y, color=color, linewidth=2.5, label=label)


team_total_distribution_mc(consistent_team, "blue", "Consistent Team")
team_total_distribution_mc(risky_team, "red", "Risky Team")

plt.title("Team Total Points Distribution – Consistent vs Risky")
plt.xlabel("Total Points")
plt.ylabel("Probability Density")
plt.legend()
plt.show()