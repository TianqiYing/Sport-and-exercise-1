import pandas as pd
import numpy as np
import glob

path = "Sport-and-exercise-1-main/Sport-and-exercise-1-main/data"

# Find all gw.csv files recursively
all_files = glob.glob(f"{path}/**/gw.csv", recursive=True)
print(f"Found {len(all_files)} files")

# Load and concatenate, extracting season/player from the path
dfs = []
for f in all_files:
    parts = f.replace("\\", "/").split("/")
    season = parts[-4]
    player_name = parts[-2]
    df = pd.read_csv(f, low_memory=False)
    df["season_x"] = season
    df["player"] = player_name
    dfs.append(df)

kosuke = pd.concat(dfs, ignore_index=True)
kosuke.to_csv("Data/kosuke.csv")
print("CSV exported")
print(kosuke.shape)
print(kosuke.head)
kosuke["kickoff_time"] = pd.to_datetime(kosuke["kickoff_time"])
kosuke["date"] = kosuke["kickoff_time"].dt.date
kosuke["player"] = kosuke["player"].str.replace("_", " ").str.replace(r"\s+\d+$", "", regex=True)

understat = pd.read_csv("Data/player_data2016-present.csv")
understat["date"] = pd.to_datetime(understat["date"])
understat["date"] = understat["date"].dt.date
print(understat["season_id"].value_counts())

merged = pd.merge(
    kosuke,
    understat,
    on=["date", "player"],
    how="left"
)
print(kosuke.shape)
print(understat.shape)
print(merged.shape)
print(merged.columns.tolist())
print(merged.head())

merged = merged.sort_values(["player", "date"]).reset_index(drop=True)
merged["played"] = (merged["minutes_x"] > 0).astype(int)

# ── 0. Filter to players with any game time ──────────────────────────────────
active = merged[merged["minutes_x"] > 0].copy()

# ── 1. Overall shape ─────────────────────────────────────────────────────────
print("=== DATASET OVERVIEW ===")
print(f"Full merged rows : {len(merged):,}")
print(f"Active rows (min > 0): {len(active):,}")
print(f"Dropped (0 min) : {len(merged) - len(active):,}  ({(1 - len(active)/len(merged))*100:.1f}%)")
print(f"Unique players (active): {active['player'].nunique():,}")
print(f"Seasons covered : {sorted(active['season_x'].unique())}")
print()

# ── 2. Column-level completeness ─────────────────────────────────────────────
def completeness_report(df, label=""):
    total = len(df)
    report = pd.DataFrame({
        "non_null": df.notna().sum(),
        "null": df.isna().sum(),
        "pct_complete": (df.notna().sum() / total * 100).round(1),
        "dtype": df.dtypes
    }).sort_values("pct_complete")
    print(f"=== COLUMN COMPLETENESS ({label}, n={total:,}) ===")
    print(report.to_string())
    print()
    return report

full_report   = completeness_report(merged, "full merge")
active_report = completeness_report(active, "active players only")

# ── 3. Source-split: which columns came from each dataset? ───────────────────
kosuke_cols    = [c for c in merged.columns if c.endswith("_x")]
understat_cols = [c for c in merged.columns if c.endswith("_y")]
shared_cols    = [c for c in merged.columns if not c.endswith(("_x", "_y"))]

print("=== UNDERSTAT MATCH RATE (left join fill) ===")
sample_us_col = understat_cols[0] if understat_cols else None
if sample_us_col:
    matched = active[sample_us_col].notna().sum()
    print(f"Rows with understat data : {matched:,} / {len(active):,}  ({matched/len(active)*100:.1f}%)")
    print(f"Rows WITHOUT understat   : {len(active)-matched:,}")
print()

# ── 4. Per-season completeness ───────────────────────────────────────────────
print("=== PER-SEASON ROW COUNTS & UNDERSTAT MATCH RATE ===")
if sample_us_col:
    season_stats = active.groupby("season_x").agg(
        rows=("player", "count"),
        understat_matched=(sample_us_col, lambda x: x.notna().sum())
    )
    season_stats["match_pct"] = (season_stats["understat_matched"] / season_stats["rows"] * 100).round(1)
    print(season_stats.to_string())
print()

# ── 5. Per-player understat completeness ─────────────────────────────────────
if understat_cols:
    player_us_null = active.groupby("player")[understat_cols].apply(
        lambda df: df.isna().mean().mean()
    ).sort_values(ascending=False)

    pct_missing = (player_us_null * 100).round(1)

    perfect  = (pct_missing == 0).sum()
    good     = ((pct_missing > 0)  & (pct_missing <= 25)).sum()
    partial  = ((pct_missing > 25) & (pct_missing < 100)).sum()
    no_match = (pct_missing == 100).sum()

    print("=== UNDERSTAT COMPLETENESS PER PLAYER ===")
    print(f"  100% complete (no missing)  : {perfect:4d} players  ({perfect / len(pct_missing) * 100:.1f}%)")
    print(f"  1–25% missing               : {good:4d} players  ({good / len(pct_missing) * 100:.1f}%)")
    print(f"  25–99% missing              : {partial:4d} players  ({partial / len(pct_missing) * 100:.1f}%)")
    print(f"  0% matched (100% missing)   : {no_match:4d} players  ({no_match / len(pct_missing) * 100:.1f}%)")
    print()
    print(f"--- Players with PERFECT understat coverage : {(pct_missing == 0).sum()} players")
    print(f"--- Players with ZERO understat coverage    : {(pct_missing == 100).sum()} players")
print()

# ── 6. Critical column presence check ────────────────────────────────────────
CRITICAL = ["player", "date", "season_x", "minutes_x", "total_points",
            "goals_scored", "assists_x", "xg", "xa"]
print("=== CRITICAL COLUMN STATUS ===")
for col in CRITICAL:
    if col in active.columns:
        pct = active[col].notna().mean() * 100
        print(f"  {col:25s} {pct:6.1f}% complete")
    else:
        print(f"  {col:25s} *** COLUMN MISSING ***")
print()

# ── 7. Duplicate rows check ───────────────────────────────────────────────────
dupe_key = ["player", "date"]
dupes = active[active.duplicated(subset=dupe_key, keep=False)]
print(f"=== DUPLICATE (player, date) ROWS: {len(dupes):,} ===")
if len(dupes):
    print(dupes[["player", "date", "season_x", "minutes_x"]].sort_values(dupe_key).head(20))
print()

# ── 8. Quick summary ──────────────────────────────────────────────────────────
print("=== SUMMARY ===")
fully_complete = active.dropna()
print(f"Rows with ZERO nulls : {len(fully_complete):,} / {len(active):,}  ({len(fully_complete)/len(active)*100:.1f}%)")
print(f"Columns ≥ 95% complete : {(active_report['pct_complete'] >= 95).sum()} / {len(active_report)}")
print(f"Columns < 50% complete : {(active_report['pct_complete'] < 50).sum()} / {len(active_report)}")
print()

# ── 9. Full-understat-coverage subset ────────────────────────────────────────
perfect_players = pct_missing[pct_missing == 0].index
active_perfect  = active[active["player"].isin(perfect_players)].copy()

print("=== FULL UNDERSTAT COVERAGE SUBSET ===")
print(f"Players  : {active_perfect['player'].nunique():,} / {active['player'].nunique():,}  ({active_perfect['player'].nunique()/active['player'].nunique()*100:.1f}%)")
print(f"Rows     : {len(active_perfect):,} / {len(active):,}  ({len(active_perfect)/len(active)*100:.1f}%)")
print()

season_breakdown = active_perfect.groupby("season_x").agg(
    players=("player", "nunique"),
    rows=("player", "count")
).reset_index()
print("--- Per-season breakdown ---")
print(season_breakdown.to_string(index=False))
print()

if "position" in active_perfect.columns:
    pos_breakdown = active_perfect.groupby("position").agg(
        players=("player", "nunique"),
        rows=("player", "count"),
        avg_xg=("xg", "mean"),
        avg_xa=("xa", "mean"),
        avg_pts=("total_points", "mean")
    ).round(3)
    print("--- Per-position breakdown ---")
    print(pos_breakdown.to_string())
    print()

print("--- Key stat summary (clean subset) ---")
key_cols = ["total_points", "minutes_x", "goals_scored", "assists_x",
            "xg", "xa", "xg_chain", "xg_buildup", "shots", "key_passes_y"]
key_cols = [c for c in key_cols if c in active_perfect.columns]
print(active_perfect[key_cols].describe().round(3).to_string())
print()
active_perfect = active_perfect.drop(columns=["yellow_cards_y","red_cards_y","assists_y","expected_assists", "expected_goals", "expected_goal_involvements", "starts", "mng_clean_sheets","mng_draw","mng_loss","mng_underdog_draw","mng_underdog_win","mng_win", "modified", "defensive_contribution"])
active_perfect["FPL_pos"] = active_perfect["position"].map({
    "D":   "DEF",
    "DL":  "DEF",
    "DR":  "DEF",
    "DC":  "DEF",
    "M":   "MID",
    "ML":  "MID",
    "MR":  "MID",
    "MC":  "MID",
    "FW":   "FWD",
    "FWL": "MID",
    "FWR": "MID",
    "GK":  "GK",
})
active_perfect.to_csv("Data/active_perfect_understat.csv", index=False)
print(f"Exported → Data/active_perfect_understat.csv  ({len(active_perfect):,} rows, {active_perfect.shape[1]} cols)")

for window in [3, 5, 10]:
    active_perfect[f"form_pts_{window}"] = (
        active_perfect.groupby("player")["total_points"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    active_perfect[f"form_goals_{window}"] = (
        active_perfect.groupby("player")["goals_scored"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    active_perfect[f"form_assists_{window}"] = (
        active_perfect.groupby("player")["assists_x"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    active_perfect[f"form_minutes_{window}"] = (
        active_perfect.groupby("player")["minutes_x"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    active_perfect[f"form_xg_{window}"] = (
        active_perfect.groupby("player")["xg"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    active_perfect[f"form_xa_{window}"] = (
        active_perfect.groupby("player")["xa"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    active_perfect[f"form_bps_{window}"] = (
        active_perfect.groupby("player")["bps"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    active_perfect[f"form_shots_{window}"] = (
        active_perfect.groupby("player")["shots"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    active_perfect[f"form_key_passes_y_{window}"] = (
        active_perfect.groupby("player")["key_passes_y"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    active_perfect[f"form_influence_{window}"] = (
        active_perfect.groupby("player")["influence"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    active_perfect[f"form_creativity_{window}"] = (
        active_perfect.groupby("player")["creativity"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

# ── 11. Cumulative season stats ───────────────────────────────────────────────
active_perfect["season_goals"] = (
    active_perfect.groupby(["player", "season_x"])["goals_scored"]
    .transform(lambda x: x.shift(1).cumsum())
)
active_perfect["season_assists"] = (
    active_perfect.groupby(["player", "season_x"])["assists_x"]
    .transform(lambda x: x.shift(1).cumsum())
)
active_perfect["season_minutes"] = (
    active_perfect.groupby(["player", "season_x"])["minutes_x"]
    .transform(lambda x: x.shift(1).cumsum())
)
active_perfect["games_played_season"] = (
    active_perfect.groupby(["player", "season_x"])["played"]
    .transform(lambda x: x.shift(1).cumsum())
)
active_perfect["clean_sheets_season"] = (
    active_perfect.groupby(["player", "season_x"])["clean_sheets"]
    .transform(lambda x: x.shift(1).cumsum())
)

active_perfect["clean_sheet_form"] = (
    active_perfect.groupby(["player", "season_x"])["clean_sheets"]
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)

active_perfect["is_home"] = active_perfect["was_home"].astype(int)
active_perfect["price"]   = active_perfect["value"] / 10
print(list(active_perfect.columns))
active_perfect.to_csv("Data/active_perfect_understat_enhanced.csv", index=False)
print(f"Exported → Data/active_perfect_understat_enhanced.csv  ({len(active_perfect):,} rows, {active_perfect.shape[1]} cols)")


# ── 10. Rolling form features ─────────────────────────────────────────────────
for window in [3, 5, 10]:
    merged[f"form_pts_{window}"] = (
        merged.groupby("player")["total_points"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    merged[f"form_goals_{window}"] = (
        merged.groupby("player")["goals_scored"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    merged[f"form_assists_{window}"] = (
        merged.groupby("player")["assists_x"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    merged[f"form_minutes_{window}"] = (
        merged.groupby("player")["minutes_x"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    merged[f"form_xg_{window}"] = (
        merged.groupby("player")["xg"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    merged[f"form_xa_{window}"] = (
        merged.groupby("player")["xa"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    merged[f"form_bps_{window}"] = (
        merged.groupby("player")["bps"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

for window in [3, 5, 10]:
    # existing ones ...

    merged[f"form_shots_{window}"] = (
        merged.groupby("player")["shots"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    merged[f"form_key_passes_y_{window}"] = (
        merged.groupby("player")["key_passes_y"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    merged[f"form_influence_{window}"] = (
        merged.groupby("player")["influence"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    merged[f"form_creativity_{window}"] = (
        merged.groupby("player")["creativity"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

# ── 11. Cumulative season stats ───────────────────────────────────────────────
merged["season_goals"] = (
    merged.groupby(["player", "season_x"])["goals_scored"]
    .transform(lambda x: x.shift(1).cumsum())
)
merged["season_assists"] = (
    merged.groupby(["player", "season_x"])["assists_x"]
    .transform(lambda x: x.shift(1).cumsum())
)
merged["season_minutes"] = (
    merged.groupby(["player", "season_x"])["minutes_x"]
    .transform(lambda x: x.shift(1).cumsum())
)
merged["games_played_season"] = (
    merged.groupby(["player", "season_x"])["played"]
    .transform(lambda x: x.shift(1).cumsum())
)

merged["is_home"] = merged["was_home"].astype(int)
merged["price"]   = merged["value"] / 10

# ── 12. Export ────────────────────────────────────────────────────────────────
merged.to_csv("Data/merged.csv")
n          = len(merged)
split_size = n // 6

for i in range(6):
    start = i * split_size
    end   = (i + 1) * split_size if i < 4 else n
    merged.iloc[start:end].to_csv(f"Data/merged{i+1}.csv", index=False)

print("merged out")

bukayo = merged[merged["player"] == "Bukayo Saka"].sort_values("date")
bukayo.to_csv("Data/bukayo.csv", index=False)
print(50 * "=")
print("Bukayo Saka exported")
print(50 * "=")
print(bukayo.head())

