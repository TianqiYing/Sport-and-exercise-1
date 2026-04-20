import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# =============================================================================
# 1. LOAD DATA
# =============================================================================
df = pd.read_csv("Data/active_perfect_understat_enhanced.csv")
df = df.loc[:, ~df.columns.duplicated()]
df = df.drop(columns=[
    "league_id", "player_id", "position_id", "minutes_y",
    "own_goals_y", "game_id", "season_id",
    "game", "venue",
    "ea_index", "loaned_in", "loaned_out",
    "kickoff_time_formatted",
    "id",
], errors="ignore")
df = df.sort_values(["player", "date"]).reset_index(drop=True)

# Colour palette
COLOURS = {
    "blue"   : "#1f77b4",
    "orange" : "#ff7f0e",
    "green"  : "#2ca02c",
    "red"    : "#d62728",
    "purple" : "#9467bd",
    "grey"   : "#7f7f7f",
}
POS_COLOURS = {"GK": "#9467bd", "DEF": "#2ca02c", "MID": "#1f77b4", "FWD": "#d62728"}

# =============================================================================
# PLOT 1 — Points Distribution
# Shows how skewed and noisy FPL points are
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("FPL Points Distribution — Evidence of Noise & Skew", fontsize=14, fontweight="bold")

# Left: overall histogram
ax = axes[0]
pts = df["total_points"].dropna()
ax.hist(pts, bins=range(int(pts.min()), int(pts.max()) + 2), color=COLOURS["blue"],
        edgecolor="white", linewidth=0.3, alpha=0.85)
ax.axvline(pts.mean(),   color=COLOURS["red"],    linestyle="--", linewidth=1.5,
           label=f"Mean ({pts.mean():.1f})")
ax.axvline(pts.median(), color=COLOURS["orange"], linestyle="--", linewidth=1.5,
           label=f"Median ({pts.median():.1f})")
ax.set_xlabel("Total Points per Gameweek")
ax.set_ylabel("Frequency")
ax.set_title("Overall Points Distribution")
ax.legend()
ax.grid(axis="y", alpha=0.3)
skew = stats.skew(pts.dropna())
ax.text(0.97, 0.95, f"Skewness: {skew:.2f}", transform=ax.transAxes,
        ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

# Right: by position
ax = axes[1]
for pos in ["GK", "DEF", "MID", "FWD"]:
    pos_pts = df[df["FPL_pos"] == pos]["total_points"].dropna()
    ax.hist(pos_pts, bins=range(0, 25), alpha=0.55, label=pos,
            color=POS_COLOURS[pos], edgecolor="white", linewidth=0.2)
ax.set_xlabel("Total Points per Gameweek")
ax.set_ylabel("Frequency")
ax.set_title("Points Distribution by Position")
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("Data/plot1_points_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved plot1_points_distribution.png")

# =============================================================================
# PLOT 2 — Standard Deviation & Variance
# Shows how unpredictable individual players are
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Points Variability — How Unpredictable Are Players?", fontsize=14, fontweight="bold")

# Left: per-position std dev box plot
ax = axes[0]
pos_data = [df[df["FPL_pos"] == pos]["total_points"].dropna().values
            for pos in ["GK", "DEF", "MID", "FWD"]]
bp = ax.boxplot(pos_data, patch_artist=True, notch=False,
                medianprops=dict(color="black", linewidth=2))
for patch, pos in zip(bp["boxes"], ["GK", "DEF", "MID", "FWD"]):
    patch.set_facecolor(POS_COLOURS[pos])
    patch.set_alpha(0.7)
ax.set_xticklabels(["GK", "DEF", "MID", "FWD"])
ax.set_ylabel("Total Points")
ax.set_title("Points Spread by Position")
ax.grid(axis="y", alpha=0.3)

# Middle: distribution of player-level std dev
ax = axes[1]
player_std = (
    df.groupby("player")["total_points"]
    .std()
    .dropna()
)
ax.hist(player_std, bins=30, color=COLOURS["orange"], edgecolor="white",
        linewidth=0.3, alpha=0.85)
ax.axvline(player_std.mean(), color=COLOURS["red"], linestyle="--", linewidth=1.5,
           label=f"Mean std ({player_std.mean():.2f})")
ax.set_xlabel("Standard Deviation of Points")
ax.set_ylabel("Number of Players")
ax.set_title("Distribution of Player-Level Std Dev")
ax.legend()
ax.grid(axis="y", alpha=0.3)

# Right: coefficient of variation per position
ax = axes[2]
cv_data = {}
for pos in ["GK", "DEF", "MID", "FWD"]:
    pos_df   = df[df["FPL_pos"] == pos]
    mean_pts = pos_df["total_points"].mean()
    std_pts  = pos_df["total_points"].std()
    cv_data[pos] = (std_pts / mean_pts) * 100  # as percentage

bars = ax.bar(cv_data.keys(), cv_data.values(),
              color=[POS_COLOURS[p] for p in cv_data.keys()],
              edgecolor="white", alpha=0.85)
ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=10)
ax.set_ylabel("Coefficient of Variation (%)")
ax.set_title("Coefficient of Variation by Position\n(higher = more unpredictable)")
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(cv_data.values()) * 1.2)

plt.tight_layout()
plt.savefig("Data/plot2_standard_deviation.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved plot2_standard_deviation.png")

# =============================================================================
# PLOT 3 — Week-to-Week Consistency
# Same player, wildly different scores across gameweeks
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Week-to-Week Inconsistency — The Core Prediction Challenge", fontsize=14, fontweight="bold")

# Left: autocorrelation of points (how well does last week predict this week?)
ax = axes[0]
lags = range(1, 9)
autocorrs = []
for lag in lags:
    shifted = df.groupby("player")["total_points"].shift(lag)
    corr    = df["total_points"].corr(shifted)
    autocorrs.append(corr)

bars = ax.bar(lags, autocorrs, color=COLOURS["blue"], edgecolor="white", alpha=0.85)
ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Lag (gameweeks)")
ax.set_ylabel("Autocorrelation")
ax.set_title("Autocorrelation of Points\n(how well does week N predict week N+k?)")
ax.set_xticks(list(lags))
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(min(autocorrs) - 0.05, max(autocorrs) + 0.08)

# Right: haul rate — how often do players score 9+ pts?
ax = axes[1]
haul_threshold = 9
haul_rates = {}
for pos in ["GK", "DEF", "MID", "FWD"]:
    pos_pts       = df[df["FPL_pos"] == pos]["total_points"].dropna()
    haul_rates[pos] = (pos_pts >= haul_threshold).mean() * 100

bars = ax.bar(haul_rates.keys(), haul_rates.values(),
              color=[POS_COLOURS[p] for p in haul_rates.keys()],
              edgecolor="white", alpha=0.85)
ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=10)
ax.set_ylabel(f"% of Gameweeks with ≥{haul_threshold} pts")
ax.set_title(f"Haul Rate by Position (≥{haul_threshold} pts)\n"
             f"(rare events are hardest to predict)")
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(haul_rates.values()) * 1.3)

plt.tight_layout()
plt.savefig("Data/plot3_weekly_consistency.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved plot3_weekly_consistency.png")

# =============================================================================
# PLOT 4 — Season-Level Noise
# Even across a full season, variance is enormous
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Season-Level Variance — Noise Persists Even Over 38 Gameweeks", fontsize=14, fontweight="bold")

# Left: distribution of total season points per player
season_totals = (
    df.groupby(["player", "season_x"])["total_points"]
    .sum()
    .reset_index()
)
ax = axes[0]
ax.hist(season_totals["total_points"], bins=40, color=COLOURS["purple"],
        edgecolor="white", linewidth=0.3, alpha=0.85)
ax.axvline(season_totals["total_points"].mean(), color=COLOURS["red"],
           linestyle="--", linewidth=1.5,
           label=f"Mean ({season_totals['total_points'].mean():.0f})")
ax.set_xlabel("Total Season Points")
ax.set_ylabel("Player-Season Count")
ax.set_title("Distribution of Total Season Points per Player")
ax.legend()
ax.grid(axis="y", alpha=0.3)

# Right: year-on-year correlation of season totals
# Does last season's total predict this season's?
ax = axes[1]
pivot = season_totals.pivot(index="player", columns="season_x", values="total_points")
seasons = sorted(pivot.columns)
yoy_corrs = []
yoy_labels = []
for i in range(len(seasons) - 1):
    s1, s2 = seasons[i], seasons[i + 1]
    paired  = pivot[[s1, s2]].dropna()
    if len(paired) > 20:
        corr = paired[s1].corr(paired[s2])
        yoy_corrs.append(corr)
        yoy_labels.append(f"{s1[-2:]}\n→{s2[-2:]}")

bars = ax.bar(yoy_labels, yoy_corrs, color=COLOURS["green"],
              edgecolor="white", alpha=0.85)
ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Pearson Correlation")
ax.set_title("Year-on-Year Season Total Correlation\n(how predictive is last season?)")
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(yoy_corrs) * 1.2)

plt.tight_layout()
plt.savefig("Data/plot4_season_variance.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved plot4_season_variance.png")

# =============================================================================
# PLOT 5 — Blank Rate & Score Concentration
# Most gameweeks most players score 1-2pts — the signal is buried in noise
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Blank Rate & Score Concentration — Most Players Contribute Nothing Most Weeks",
             fontsize=14, fontweight="bold")

# Left: blank rate by position (scored ≤ 2 pts)
ax = axes[0]
blank_threshold = 2
blank_rates = {}
for pos in ["GK", "DEF", "MID", "FWD"]:
    pos_pts            = df[df["FPL_pos"] == pos]["total_points"].dropna()
    blank_rates[pos]   = (pos_pts <= blank_threshold).mean() * 100

bars = ax.bar(blank_rates.keys(), blank_rates.values(),
              color=[POS_COLOURS[p] for p in blank_rates.keys()],
              edgecolor="white", alpha=0.85)
ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=10)
ax.set_ylabel(f"% of Gameweeks with ≤{blank_threshold} pts (blank)")
ax.set_title("Blank Rate by Position\n(fraction of gameweeks player contributes ≤2 pts)")
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, 100)

# Right: points brackets stacked bar — where do points actually come from?
ax = axes[1]
brackets     = ["0-2\n(blank)", "3-5\n(poor)", "6-8\n(ok)", "9-12\n(good)", "13+\n(elite)"]
bracket_bins = [(-1, 2), (3, 5), (6, 8), (9, 12), (13, 100)]
pos_list     = ["GK", "DEF", "MID", "FWD"]
bracket_data = {pos: [] for pos in pos_list}

for pos in pos_list:
    pos_pts = df[df["FPL_pos"] == pos]["total_points"].dropna()
    for lo, hi in bracket_bins:
        pct = ((pos_pts > lo) & (pos_pts <= hi)).mean() * 100
        bracket_data[pos].append(pct)

x      = np.arange(len(brackets))
width  = 0.2
for idx, pos in enumerate(pos_list):
    ax.bar(x + idx * width, bracket_data[pos], width,
           label=pos, color=POS_COLOURS[pos], edgecolor="white", alpha=0.85)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(brackets)
ax.set_ylabel("% of Gameweeks")
ax.set_title("Points Bracket Distribution by Position\n(most gameweeks are blanks or poor returns)")
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("Data/plot5_blank_rates.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved plot5_blank_rates.png")

# =============================================================================
# PLOT 6 — xG vs Actual Goals Variance
# Even chance quality doesn't reliably predict outcomes
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Expected vs Actual — Randomness in Outcomes", fontsize=14, fontweight="bold")

# Left: xG vs goals scatter with noise highlighted
ax = axes[0]
sample = df[df["xg"].notna() & df["goals_scored"].notna()].sample(
    min(3000, len(df)), random_state=42
)
ax.scatter(sample["xg"], sample["goals_scored"],
           alpha=0.15, s=8, color=COLOURS["blue"])
max_val = max(sample["xg"].max(), sample["goals_scored"].max())
ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Perfect prediction")
ax.set_xlabel("Expected Goals (xG)")
ax.set_ylabel("Actual Goals Scored")
ax.set_title("xG vs Actual Goals\n(scatter shows inherent randomness)")
ax.legend()
ax.grid(alpha=0.3)

xg_corr = sample["xg"].corr(sample["goals_scored"])
ax.text(0.97, 0.05, f"r = {xg_corr:.3f}", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

# Right: distribution of (actual - expected) points residuals
ax = axes[1]
df["pts_residual"] = df["total_points"] - df["xg"] * 5  # rough xPts proxy
residuals = df["pts_residual"].dropna()
residuals = residuals[np.abs(residuals) < residuals.quantile(0.99)]  # clip outliers

ax.hist(residuals, bins=50, color=COLOURS["orange"],
        edgecolor="white", linewidth=0.3, alpha=0.85)
ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero residual")
ax.axvline(residuals.std(),  color="grey", linestyle=":", linewidth=1.2,
           label=f"±1 std ({residuals.std():.1f})")
ax.axvline(-residuals.std(), color="grey", linestyle=":", linewidth=1.2)
ax.set_xlabel("Points Residual (Actual − xG×5)")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Prediction Residuals\n(wide spread = high noise)")
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("Data/plot6_xg_variance.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved plot6_xg_variance.png")

# =============================================================================
# SUMMARY STATS — Print key noise metrics
# =============================================================================
print("\n" + "="*60)
print("DATASET NOISE SUMMARY")
print("="*60)
print(f"Total observations        : {len(df):,}")
print(f"Overall mean points/GW   : {df['total_points'].mean():.2f}")
print(f"Overall std points/GW    : {df['total_points'].std():.2f}")
print(f"Overall skewness         : {stats.skew(df['total_points'].dropna()):.2f}")
print(f"Blank rate (≤2 pts)      : {(df['total_points'] <= 2).mean()*100:.1f}%")
print(f"Haul rate  (≥9 pts)      : {(df['total_points'] >= 9).mean()*100:.1f}%")
print(f"Lag-1 autocorrelation    : {autocorrs[0]:.3f}")
print(f"Mean player-level std    : {player_std.mean():.2f}")
print(f"\nBy position:")
for pos in ["GK", "DEF", "MID", "FWD"]:
    pos_pts = df[df["FPL_pos"] == pos]["total_points"].dropna()
    print(
        f"  {pos}  mean={pos_pts.mean():.2f}  "
        f"std={pos_pts.std():.2f}  "
        f"blank%={(pos_pts <= 2).mean()*100:.1f}%  "
        f"haul%={(pos_pts >= 9).mean()*100:.1f}%"
    )