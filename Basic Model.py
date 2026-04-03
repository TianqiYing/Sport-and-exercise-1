import pandas as pd

# -------------------------
# Parameters
# -------------------------
SEASON = "2023-24"
TARGET_GW = 38
BUDGET = 1000

FORM_WINDOW = 5
EWMA_ALPHA = 0.6

constraints = {
    "GK": 1,
    "DEF": 4,
    "MID": 4,
    "FWD": 2
}

MAX_PER_TEAM = 3

# -------------------------
# Load data
# -------------------------
df = pd.read_csv("cleaned_merged_seasons_team_aggregated.csv", encoding="latin1")
df = df[df["season_x"] == SEASON]

# -------------------------
# Split data
# -------------------------
train_df = df[df["GW"] < TARGET_GW].copy()
gw_df = df[df["GW"] == TARGET_GW].copy()

train_df = train_df.sort_values(["name", "GW"])

# -------------------------
# HYBRID FORM MODEL
# -------------------------

# EWMA using all GWs
train_df["ewma_points"] = train_df.groupby("name")["total_points"]\
    .transform(lambda x: x.ewm(alpha=EWMA_ALPHA).mean())

ewma_latest = train_df.loc[
    train_df.groupby("name")["GW"].idxmax()
][["name", "ewma_points"]]

# Last N GW average
train_df["gw_rank"] = train_df.groupby("name")["GW"]\
    .rank(ascending=False, method="first")

recent_avg = train_df[train_df["gw_rank"] <= FORM_WINDOW]\
    .groupby("name")["total_points"]\
    .mean()\
    .reset_index()

recent_avg = recent_avg.rename(columns={"total_points": "recent_avg_points"})

# Combine both
form_df = ewma_latest.merge(recent_avg, on="name", how="left")

# Fill players with fewer than N games
form_df["recent_avg_points"] = form_df["recent_avg_points"].fillna(form_df["ewma_points"])

# Final expected base points
form_df["expected_points"] = (
    0.6 * form_df["ewma_points"] +
    0.4 * form_df["recent_avg_points"]
)

form_df = form_df[["name", "expected_points"]]

# -------------------------
# Latest price (GW ≤ 37)
# -------------------------
latest_price = train_df.loc[
    train_df.groupby("name")["GW"].idxmax()
][["name", "value"]]

latest_price = latest_price.rename(columns={"value": "latest_value"})

# -------------------------
# GW38 fixtures
# -------------------------
fixtures = gw_df[["name", "team_x", "position", "opp_team_name", "was_home"]]
fixtures = fixtures.rename(columns={"team_x": "team"})

# -------------------------
# Merge
# -------------------------
agg_df = form_df.merge(latest_price, on="name")
agg_df = agg_df.merge(fixtures, on="name")

# -------------------------
# Fixture difficulty
# -------------------------
team_def = train_df.groupby("team_x")["goals_conceded"].mean().reset_index()
team_def_dict = dict(zip(team_def["team_x"], team_def["goals_conceded"]))

agg_df["opp_def_strength"] = agg_df["opp_team_name"].map(team_def_dict)

avg_def = team_def["goals_conceded"].mean()
agg_df["opp_def_strength"] = agg_df["opp_def_strength"].fillna(avg_def)

agg_df["fixture_factor"] = agg_df["opp_def_strength"] / avg_def

# Home advantage
agg_df["home_factor"] = agg_df["was_home"].apply(lambda x: 1.1 if x else 0.9)

# -------------------------
# Final expected points
# -------------------------
agg_df["expected_points"] = (
    agg_df["expected_points"] *
    agg_df["fixture_factor"] *
    agg_df["home_factor"]
)

# -------------------------
# Value metric
# -------------------------
agg_df["value_metric"] = agg_df["expected_points"] / agg_df["latest_value"]

# -------------------------
# Team selection
# -------------------------
selected = []
budget = BUDGET

pos_count = {k: 0 for k in constraints}
team_count = {}

df_sorted = agg_df.sort_values("value_metric", ascending=False)

for _, row in df_sorted.iterrows():
    pos = row["position"]
    team = row["team"]
    price = row["latest_value"]

    if pos not in constraints:
        continue

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

team_df = pd.DataFrame(selected)

# -------------------------
# Output
# -------------------------
print(team_df[["name", "position", "team", "latest_value", "expected_points"]])
print("\nTotal Expected Points:", team_df["expected_points"].sum())
print("Budget Remaining:", budget)
print("Position Count:", pos_count)
print("Team Count:", team_count)