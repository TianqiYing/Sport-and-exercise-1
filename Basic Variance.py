import pandas as pd
import numpy as np

SEASON = "2023-24"
TARGET_GW = 30
BUDGET = 1000
FORM_WINDOW = 5
EWMA_ALPHA = 0.6
N_SIM = 1000

constraints = {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2}
MAX_PER_TEAM = 3

# Load data
df = pd.read_csv("cleaned_merged_seasons_team_aggregated.csv", encoding="latin1")
df = df[df["season_x"] == SEASON]

train_df = df[df["GW"] < TARGET_GW].copy()
gw_df = df[df["GW"] == TARGET_GW].copy()
train_df = train_df.sort_values(["name", "GW"])

# -------------------------
# HYBRID FORM
# -------------------------
train_df["ewma_points"] = train_df.groupby("name")["total_points"] \
    .transform(lambda x: x.ewm(alpha=EWMA_ALPHA).mean())

train_df["gw_rank"] = train_df.groupby("name")["GW"] \
    .rank(ascending=False, method="first")

recent_avg = train_df[train_df["gw_rank"] <= FORM_WINDOW] \
    .groupby("name")["total_points"].mean().reset_index().rename(columns={"total_points": "recent_avg_points"})

ewma_latest = train_df.loc[train_df.groupby("name")["GW"].idxmax()][["name", "ewma_points"]]

form_df = ewma_latest.merge(recent_avg, on="name", how="left")
form_df["recent_avg_points"].fillna(form_df["ewma_points"], inplace=True)
form_df["expected_points"] = 0.6 * form_df["ewma_points"] + 0.4 * form_df["recent_avg_points"]

# -------------------------
# Variance for last N games
# -------------------------
recent_points = train_df[train_df["gw_rank"] <= FORM_WINDOW]
var_df = recent_points.groupby("name")["total_points"].std().reset_index().rename(
    columns={"total_points": "std_points"})
var_df["std_points"].fillna(var_df["std_points"].mean(), inplace=True)

agg_df = form_df.merge(var_df, on="name")

# -------------------------
# Latest price
# -------------------------
latest_price = train_df.loc[train_df.groupby("name")["GW"].idxmax()][["name", "value"]].rename(
    columns={"value": "latest_value"})
agg_df = agg_df.merge(latest_price, on="name")

# -------------------------
# GW38 fixtures
# -------------------------
fixtures = gw_df[["name", "team_x", "position", "opp_team_name", "was_home"]].rename(columns={"team_x": "team"})
agg_df = agg_df.merge(fixtures, on="name")

# -------------------------
# Fixture difficulty & home factor
# -------------------------
team_def = train_df.groupby("team_x")["goals_conceded"].mean().reset_index()
team_def_dict = dict(zip(team_def["team_x"], team_def["goals_conceded"]))
avg_def = team_def["goals_conceded"].mean()

agg_df["opp_def_strength"] = agg_df["opp_team_name"].map(team_def_dict).fillna(avg_def)
agg_df["fixture_factor"] = agg_df["opp_def_strength"] / avg_def
agg_df["home_factor"] = agg_df["was_home"].apply(lambda x: 1.1 if x else 0.9)

agg_df["expected_points"] = agg_df["expected_points"] * agg_df["fixture_factor"] * agg_df["home_factor"]

# -------------------------
# Monte Carlo simulations
# -------------------------
np.random.seed(42)


def simulate_team(df, mode="consistent"):
    """
    mode: "consistent" -> prefer low variance
          "risky" -> prefer high variance
    """
    df_sim = df.copy()

    # Score metric: expected points adjusted by variance
    if mode == "consistent":
        df_sim["score_metric"] = df_sim["expected_points"] / (1 + df_sim["std_points"])
    elif mode == "risky":
        df_sim["score_metric"] = df_sim["expected_points"] * (1 + df_sim["std_points"])
    else:
        df_sim["score_metric"] = df_sim["expected_points"]

    selected = []
    budget = BUDGET
    pos_count = {k: 0 for k in constraints}
    team_count = {}

    df_sorted = df_sim.sort_values("score_metric", ascending=False)

    for _, row in df_sorted.iterrows():
        pos = row["position"]
        team = row["team"]
        price = row["latest_value"]
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
# Select teams
# -------------------------
consistent_team = simulate_team(agg_df, mode="consistent")
risky_team = simulate_team(agg_df, mode="risky")

print("CONSISTENT TEAM\n",
      consistent_team[["name", "position", "team", "latest_value", "expected_points", "std_points"]])
print("\nRISKY TEAM\n", risky_team[["name", "position", "team", "latest_value", "expected_points", "std_points"]])