import soccerdata as sd
import pandas as pd

# ── LOAD DATA ──────────────────────────────────────────────────────────────────
understat = sd.Understat(
    leagues="ENG-Premier League",
    seasons=["2016-2017","2017-2018","2018-2019","2019-2020","2020-2021", "2021-22", "2022-23", "2023-24", "2024-25", "2025-2026"]
)
schedule = understat.read_schedule()
team_match_stats = understat.read_team_match_stats()
player_match_stats = understat.read_player_match_stats()

# ── BUILD GAME -> HOME/AWAY TEAM LOOKUP FROM SCHEDULE ─────────────────────────
sch = schedule.reset_index()[["game", "season", "date", "home_team", "away_team"]].drop_duplicates()
sch["date"] = pd.to_datetime(sch["date"])

# ── TEAM DATA ──────────────────────────────────────────────────────────────────
tms = team_match_stats.reset_index()

home_cols = [c for c in tms.columns if c.startswith("home_")]
away_cols = [c for c in tms.columns if c.startswith("away_")]

home_rows = tms[["league", "season", "game"] + home_cols].copy()
home_rows.rename(columns=lambda c: c.replace("home_", ""), inplace=True)
home_rows["venue"] = "home"

away_rows = tms[["league", "season", "game"] + away_cols].copy()
away_rows.rename(columns=lambda c: c.replace("away_", ""), inplace=True)
away_rows["venue"] = "away"

team_long = pd.concat([home_rows, away_rows], ignore_index=True)

# Merge team names, season and date from schedule
team_long = team_long.merge(sch, on=["game", "season"], how="left")
team_long["team"] = team_long.apply(
    lambda r: r["home_team"] if r["venue"] == "home" else r["away_team"], axis=1
)
team_long["opponent"] = team_long.apply(
    lambda r: r["away_team"] if r["venue"] == "home" else r["home_team"], axis=1
)

# Sort and derive gameweek as match number within season per team
team_long = team_long.sort_values(["team", "season", "date"]).reset_index(drop=True)
team_long["gameweek"] = team_long.groupby(["team", "season"]).cumcount() + 1

# Drop helper columns
team_long.drop(columns=["home_team", "away_team", "league"], inplace=True)

# Merge to get goals conceded and xGA
team_long = team_long.merge(
    team_long[["game", "team", "goals", "xg"]].rename(columns={
        "team": "opponent",
        "goals": "goals_conceded",
        "xg": "xga"
    }),
    on=["game", "opponent"],
    how="left"
)

# Final column order
front_cols = ["team", "season", "gameweek", "date", "venue", "opponent",
              "goals", "goals_conceded", "xg", "xga",
              "np_xg", "np_xg_difference", "points", "expected_points",
              "ppda", "deep_completions"]
other_cols = [c for c in team_long.columns if c not in front_cols]
team_long = team_long[front_cols + other_cols]

print("TEAM DATA")
print(team_long.shape)

# ── PLAYER DATA ────────────────────────────────────────────────────────────────
pms = player_match_stats.reset_index()

# Merge team names, season and date from schedule
pms = pms.merge(sch, on=["game", "season"], how="left")

pms["venue"] = pms.apply(
    lambda r: "home" if r["team"] == r["home_team"] else "away", axis=1
)
pms["opponent"] = pms.apply(
    lambda r: r["away_team"] if r["venue"] == "home" else r["home_team"], axis=1
)

pms.drop(columns=["home_team", "away_team", "league"], inplace=True)

# Replace player-derived gameweek with match-based gameweek from team data
game_gw = team_long[["game", "season", "gameweek"]].drop_duplicates()
pms = pms.merge(game_gw, on=["game", "season"], how="left")

pms = pms.sort_values(["player", "season", "date"]).reset_index(drop=True)

front_cols = ["player", "team", "season", "gameweek", "date", "venue", "opponent"]
other_cols = [c for c in pms.columns if c not in front_cols]
player_long = pms[front_cols + other_cols]

print("\nPLAYER DATA")
print(player_long.shape)

# ── PLAYER PRESENCE MATRIX ────────────────────────────────────────────────────
presence = (
    player_long
    .groupby(["player", "season", "gameweek"])
    .size()
    .gt(0)
    .astype(int)
    .reset_index(name="played")
    .pivot_table(index=["player", "season"], columns="gameweek", values="played", fill_value=0)
)
presence.columns = [f"GW{int(c)}" for c in presence.columns]
presence = presence.reset_index()

# Sanity check: count of players who played each GW (should be ~500-600 consistently)
gw_cols = [c for c in presence.columns if c.startswith("GW")]
print("\nPLAYERS PER GAMEWEEK")
print(presence[gw_cols].sum())

# ── QUICK CHECKS ───────────────────────────────────────────────────────────────
arsenal = team_long[team_long["team"] == "Arsenal"].reset_index(drop=True)
print("\nARSENAL - MATCH BY MATCH")
print(arsenal[["season", "gameweek", "date", "venue", "opponent", "goals", "goals_conceded", "xg", "xga", "points"]].to_string())

saka = player_long[player_long["player"] == "Bukayo Saka"].reset_index(drop=True)
print("\nBUKAYO SAKA - MATCH BY MATCH")
print(saka.to_string())

saka_presence = presence[presence["player"] == "Bukayo Saka"]
print("\nBUKAYO SAKA - PRESENCE MATRIX")
print(saka_presence.to_string())

# ── EXPORT ─────────────────────────────────────────────────────────────────────
output = "Data/player_data2016-present.csv"
player_long.to_csv(output, index=False)
print(50*"=")
print("Player dataset exported")
print(f"shape: {player_long.shape}")
print(50*"=")

output = "Data/team_data2016-present.csv"
team_long.to_csv(output, index=False)
print(50*"=")
print("Team dataset exported")
print(f"shape: {team_long.shape}")
print(50*"=")

output = "Data/player_presence2016-present.csv"
presence.to_csv(output, index=False)
print(50*"=")
print("Player presence matrix exported")
print(f"shape: {presence.shape}")
print(50*"=")
