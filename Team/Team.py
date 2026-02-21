import requests
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

BASE_URL = "https://www.football-data.co.uk/mmz4281"
CACHE_DIR = "data/cache"

SEASON_CODES = {
    "2024-25": "2425",
    "2023-24": "2324",
    "2022-23": "2223",
    "2021-22": "2122",
    "2020-21": "2021",
    "2019-20": "1920",
    "2018-19": "1819",
    "2017-18": "1718",
    "2016-17": "1617",
    "2015-16": "1516",
}

# Column name glossary
COLUMN_GLOSSARY = {
    # Basic match info
    "Div":   "League (E0=Premier League)",
    "Date":  "Match date",
    "Time":  "Kick-off time",
    "HomeTeam": "Home team",
    "AwayTeam": "Away team",
    # Match results
    "FTHG":  "Full Time Home Goals",
    "FTAG":  "Full Time Away Goals",
    "FTR":   "Full Time Result (H=Home Win, D=Draw, A=Away Win)",
    "HTHG":  "Half Time Home Goals",
    "HTAG":  "Half Time Away Goals",
    "HTR":   "Half Time Result",
    # Match statistics
    "HS":    "Home Shots",
    "AS":    "Away Shots",
    "HST":   "Home Shots on Target",
    "AST":   "Away Shots on Target",
    "HC":    "Home Corners",
    "AC":    "Away Corners",
    "HF":    "Home Fouls",
    "AF":    "Away Fouls",
    "HY":    "Home Yellow Cards",
    "AY":    "Away Yellow Cards",
    "HR":    "Home Red Cards",
    "AR":    "Away Red Cards",
    # Betting odds (using Bet365 as example)
    "B365H": "Bet365 Home Win Odds",
    "B365D": "Bet365 Draw Odds",
    "B365A": "Bet365 Away Win Odds",
}

def download_season(season_label, season_code):
    """Download CSV for a single season"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"E0_{season_code}.csv")

    if os.path.exists(cache_path):
        print(f"  [CACHE] {season_label}")
        return pd.read_csv(cache_path, encoding="utf-8-sig")

    url = f"{BASE_URL}/{season_code}/E0.csv"
    print(f"  [DOWNLOAD] {season_label}  ← {url}")
    try:
        resp = requests.get(url, timeout=30,
                            headers={"User-Agent": "Academic-Project/1.0"})
        resp.raise_for_status()
        with open(cache_path, "wb") as f:
            f.write(resp.content)
        time.sleep(0.5)
        return pd.read_csv(cache_path, encoding="utf-8-sig")
    except Exception as e:
        print(f"  [FAILED] {season_label}: {e}")
        return None

def download_all(n_seasons=1):
    """Download data for the most recent n seasons, return merged"""
    dfs = []
    for label, code in list(SEASON_CODES.items())[:n_seasons]:
        df = download_season(label, code)
        if df is not None and not df.empty:
            df["season"] = label
            dfs.append(df)
    if not dfs:
        raise RuntimeError("Could not download any data!")
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n  ✓ Total {len(combined)} matches, {len(dfs)} seasons\n")
    return combined

def clean(raw_df):
    """Clean: parse dates, select core columns, handle missing values"""
    df = raw_df.copy()

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")

    # Select core columns
    core = ["Date", "season", "HomeTeam", "AwayTeam",
            "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR", "Referee"]
    stats = ["HS", "AS", "HST", "AST", "HC", "AC",
             "HF", "AF", "HY", "AY", "HR", "AR"]
    odds = ["B365H", "B365D", "B365A"]

    keep = [c for c in core + stats + odds if c in df.columns]
    df = df[keep].copy()

    # Drop incomplete matches
    df = df.dropna(subset=["FTHG", "FTAG"])
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)

    # Derived columns
    df["total_goals"] = df["FTHG"] + df["FTAG"]
    df["goal_diff"] = df["FTHG"] - df["FTAG"]  # positive = home advantage

    df = df.sort_values("Date").reset_index(drop=True)
    return df

def analyze_basics(df):
    """Print basic statistics"""
    print("BASIC STATISTICS")

    n = len(df)
    h_win = (df["FTR"] == "H").sum()
    draw  = (df["FTR"] == "D").sum()
    a_win = (df["FTR"] == "A").sum()

    print(f"  Total matches:   {n}")
    print(f"  Home wins / Draws / Away wins:  {h_win}({h_win/n*100:.1f}%) / "
          f"{draw}({draw/n*100:.1f}%) / {a_win}({a_win/n*100:.1f}%)")
    print(f"  Avg total goals:  {df['total_goals'].mean():.2f}")
    print(f"  Avg home goals:   {df['FTHG'].mean():.2f}")
    print(f"  Avg away goals:   {df['FTAG'].mean():.2f}")
    print(f"  Home advantage:   avg goal diff {df['goal_diff'].mean():.2f}")

    if "HS" in df.columns:
        print(f"\n  Average match stats")
        print(f"  Shots:   Home {df['HS'].mean():.1f}  vs  Away {df['AS'].mean():.1f}")
        print(f"  Shots on target:   Home {df['HST'].mean():.1f}  vs  Away {df['AST'].mean():.1f}")
        print(f"  Corners:   Home {df['HC'].mean():.1f}  vs  Away {df['AC'].mean():.1f}")
        print(f"  Fouls:   Home {df['HF'].mean():.1f}  vs  Away {df['AF'].mean():.1f}")
        print(f"  Yellow cards:   Home {df['HY'].mean():.1f}  vs  Away {df['AY'].mean():.1f}")

    print()

def analyze_teams(df):
    """Aggregate performance per team"""
    print("TEAM PERFORMANCE RANKING")

    records = []
    all_teams = set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique())

    for team in all_teams:
        home = df[df["HomeTeam"] == team]
        away = df[df["AwayTeam"] == team]

        played = len(home) + len(away)
        wins = (home["FTR"] == "H").sum() + (away["FTR"] == "A").sum()
        draws = (home["FTR"] == "D").sum() + (away["FTR"] == "D").sum()
        losses = played - wins - draws
        gf = home["FTHG"].sum() + away["FTAG"].sum()
        ga = home["FTAG"].sum() + away["FTHG"].sum()
        pts = wins * 3 + draws

        record = {
            "Team": team,
            "P": played, "W": wins, "D": draws, "L": losses,
            "GF": gf, "GA": ga, "GD": gf - ga,
            "Pts": pts,
            "PPG": pts / played if played > 0 else 0,
        }

        # Shooting stats
        if "HS" in df.columns:
            shots_f = home["HS"].sum() + away["AS"].sum()
            shots_a = home["AS"].sum() + away["HS"].sum()
            record["Shots/G"] = shots_f / played if played > 0 else 0
            record["ShotsA/G"] = shots_a / played if played > 0 else 0

        records.append(record)

    table = pd.DataFrame(records).sort_values("PPG", ascending=False)
    table = table.reset_index(drop=True)
    table.index += 1  # Start ranking from 1

    show_cols = ["Team", "P", "W", "D", "L", "GF", "GA", "GD", "Pts", "PPG"]
    if "Shots/G" in table.columns:
        show_cols += ["Shots/G"]

    print(table[show_cols].to_string())
    print()
    return table

def analyze_odds_accuracy(df):
    """Check betting odds prediction accuracy"""
    if "B365H" not in df.columns:
        return

    print("BETTING ODDS ANALYSIS (Bet365)")

    temp = df[["FTR", "B365H", "B365D", "B365A"]].dropna().copy()

    # Odds → implied probability
    temp["prob_H"] = 1 / temp["B365H"]
    temp["prob_D"] = 1 / temp["B365D"]
    temp["prob_A"] = 1 / temp["B365A"]
    overround = temp["prob_H"] + temp["prob_D"] + temp["prob_A"]
    temp["prob_H"] /= overround
    temp["prob_D"] /= overround
    temp["prob_A"] /= overround

    # Most favoured outcome by odds
    temp["predicted"] = temp[["prob_H", "prob_D", "prob_A"]].idxmax(axis=1)
    temp["predicted"] = temp["predicted"].map(
        {"prob_H": "H", "prob_D": "D", "prob_A": "A"}
    )

    accuracy = (temp["predicted"] == temp["FTR"]).mean()

    print(f"  Odds prediction accuracy (highest probability): {accuracy*100:.1f}%")
    print(f"  Average overround: {overround.mean()*100:.1f}% "
          f"(>100% is bookmaker margin)")

    # Actual win rate by odds range
    temp["odds_bin"] = pd.cut(temp["B365H"], bins=[0, 1.5, 2.0, 3.0, 5.0, 50],
                              labels=["<1.5", "1.5-2.0", "2.0-3.0", "3.0-5.0", ">5.0"])
    calibration = temp.groupby("odds_bin", observed=True).agg(
        n_matches=("FTR", "count"),
        actual_home_win_pct=("FTR", lambda x: (x == "H").mean()),
        implied_prob=("prob_H", "mean"),
    ).round(3)
    print(f"\n  Odds calibration table (home win):")
    print(f"  {'Odds Range':<12} {'Matches':>6} {'Implied Prob':>10} {'Actual Win%':>10} {'Diff':>8}")
    print(f"  {'-'*48}")
    for idx, row in calibration.iterrows():
        diff = row["actual_home_win_pct"] - row["implied_prob"]
        print(f"  {str(idx):<12} {int(row['n_matches']):>6} "
              f"{row['implied_prob']:>10.1%} {row['actual_home_win_pct']:>10.1%} "
              f"{diff:>+8.1%}")

    print()

def plot_all(df, save=False, output_dir="output"):
    """Generate a few core charts"""
    if save:
        os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    # Figure 1: Goal distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].hist(df["FTHG"], bins=range(0, 10), alpha=0.7,
                 color="#2196F3", edgecolor="white", label="Home")
    axes[0].hist(df["FTAG"], bins=range(0, 10), alpha=0.5,
                 color="#F44336", edgecolor="white", label="Away")
    axes[0].set_xlabel("Goals")
    axes[0].set_ylabel("Matches")
    axes[0].set_title("Goal Distribution")
    axes[0].legend()

    # Figure 2: Home/away win rates
    results = df["FTR"].value_counts()
    labels = {"H": f"Home Win\n{results.get('H',0)}",
              "D": f"Draw\n{results.get('D',0)}",
              "A": f"Away Win\n{results.get('A',0)}"}
    colors = ["#4CAF50", "#FFC107", "#F44336"]
    vals = [results.get("H", 0), results.get("D", 0), results.get("A", 0)]
    axes[1].pie(vals, labels=[labels["H"], labels["D"], labels["A"]],
                colors=colors, autopct="%1.1f%%", startangle=90,
                textprops={"fontsize": 10})
    axes[1].set_title("Match Results")

    # Figure 3: Monthly avg goals trend
    monthly = df.set_index("Date").resample("ME")["total_goals"].mean()
    axes[2].plot(monthly.index, monthly.values, "-o",
                 color="#9C27B0", markersize=3)
    axes[2].set_ylabel("Avg Goals/Match")
    axes[2].set_title("Monthly Avg Goals Trend")
    axes[2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(output_dir, "fd_goals_overview.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Figure 4: Team shots vs goals scatter
    if "HS" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 7))

        teams_data = []
        all_teams = set(df["HomeTeam"]) | set(df["AwayTeam"])
        for team in all_teams:
            home = df[df["HomeTeam"] == team]
            away = df[df["AwayTeam"] == team]
            p = len(home) + len(away)
            if p == 0:
                continue
            gf = (home["FTHG"].sum() + away["FTAG"].sum()) / p
            sf = (home["HS"].sum() + away["AS"].sum()) / p
            teams_data.append({"team": team, "goals_pg": gf, "shots_pg": sf})

        td = pd.DataFrame(teams_data)
        ax.scatter(td["shots_pg"], td["goals_pg"],
                   s=100, c="#2196F3", edgecolors="white", zorder=5)
        for _, row in td.iterrows():
            ax.annotate(row["team"], (row["shots_pg"], row["goals_pg"]),
                        fontsize=8, ha="center", va="bottom",
                        xytext=(0, 6), textcoords="offset points")

        # Regression line
        z = np.polyfit(td["shots_pg"], td["goals_pg"], 1)
        x_line = np.linspace(td["shots_pg"].min(), td["shots_pg"].max(), 50)
        ax.plot(x_line, np.polyval(z, x_line), "--", color="gray", alpha=0.5)

        ax.set_xlabel("Shots per Game")
        ax.set_ylabel("Goals per Game")
        ax.set_title("Team: Shots vs Goals (per game)")
        plt.tight_layout()
        if save:
            fig.savefig(os.path.join(output_dir, "fd_shots_vs_goals.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

    # Figure 5: Odds calibration plot
    if "B365H" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        temp = df[["FTR", "B365H"]].dropna().copy()
        temp["implied_prob"] = 1 / temp["B365H"]
        temp["home_win"] = (temp["FTR"] == "H").astype(int)
        temp["prob_bin"] = pd.cut(temp["implied_prob"],
                                  bins=np.arange(0, 1.05, 0.1))
        cal = temp.groupby("prob_bin", observed=True).agg(
            implied=("implied_prob", "mean"),
            actual=("home_win", "mean"),
            n=("home_win", "count"),
        )
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
        ax.scatter(cal["implied"], cal["actual"],
                   s=cal["n"] * 2, c="#FF5722", edgecolors="white", zorder=5)
        ax.set_xlabel("Implied Probability (from odds)")
        ax.set_ylabel("Actual Win Rate")
        ax.set_title("Betting Odds Calibration (Home Win)")
        ax.legend()
        plt.tight_layout()
        if save:
            fig.savefig(os.path.join(output_dir, "fd_odds_calibration.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

    # If not saving, display all figures
    if not save:
        plt.show()

    if save:
        print(f"  ✓ Charts saved to {output_dir}/")

def main(n_seasons=4, save_plots=False):
    print("football-data.co.uk Premier League Data Collection & Analysis")

    # 1. Download
    print(f"\n[1] Downloading data (most recent {n_seasons} seasons)...")
    raw = download_all(n_seasons)

    # 2. Clean
    print("[2] Cleaning data...")
    df = clean(raw)
    print(f"  ✓ {len(df)} matches, {len(df['HomeTeam'].unique())} teams\n")

    # 3. Analysis
    print("[3] Data analysis...\n")
    analyze_basics(df)
    table = analyze_teams(df)
    analyze_odds_accuracy(df)

    # 4. Visualization
    print("[4] Generating charts...")
    plot_all(df, save=save_plots)

    # 5. Save
    out_path = os.path.join(CACHE_DIR, "epl_matches_clean.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  ✓ Cleaned data saved: {out_path}")

    # Print column name glossary
    print("COLUMN NAME GLOSSARY")
    for col, desc in COLUMN_GLOSSARY.items():
        if col in df.columns:
            print(f"  {col:<12s} {desc}")

    return df, table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="football-data.co.uk Premier League Data Collection"
    )

    parser.add_argument("--seasons", "-s", type=int, default=4,
                        help="Number of seasons to download (default 4)")

    parser.add_argument("--save-plots", action="store_true",
                        help="Save charts to output/")
    args = parser.parse_args()

    main(n_seasons=args.seasons, save_plots=args.save_plots)
