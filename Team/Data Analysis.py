from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def implied_probs_from_odds(b365h: pd.Series, b365d: pd.Series, b365a: pd.Series) -> pd.DataFrame:
    """Convert 1X2 odds into normalised implied probabilities (removing overround)."""
    inv_h = 1.0 / b365h
    inv_d = 1.0 / b365d
    inv_a = 1.0 / b365a
    s = inv_h + inv_d + inv_a
    return pd.DataFrame({"pH": inv_h / s, "pD": inv_d / s, "pA": inv_a / s})


def make_long_team_view(df: pd.DataFrame) -> pd.DataFrame:
    """Split each match into two rows: team-perspective (home + away)."""
    probs = implied_probs_from_odds(df["B365H"], df["B365D"], df["B365A"])
    df = pd.concat([df.copy(), probs], axis=1)

    home = df[
        [
            "season",
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
            "HS",
            "HST",
            "HC",
            "HF",
            "HY",
            "HR",
            "pH",
            "pD",
            "pA",
        ]
    ].copy()
    home.rename(
        columns={
            "HomeTeam": "Team",
            "AwayTeam": "Opp",
            "FTHG": "GF",
            "FTAG": "GA",
            "HS": "S",
            "HST": "SoT",
            "HC": "C",
            "HF": "F",
            "HY": "Y",
            "HR": "R",
        },
        inplace=True,
    )
    home["is_home"] = 1
    home["pWin"], home["pDraw"], home["pLose"] = home["pH"], home["pD"], home["pA"]

    away = df[
        [
            "season",
            "Date",
            "AwayTeam",
            "HomeTeam",
            "FTAG",
            "FTHG",
            "AS",
            "AST",
            "AC",
            "AF",
            "AY",
            "AR",
            "pH",
            "pD",
            "pA",
        ]
    ].copy()
    away.rename(
        columns={
            "AwayTeam": "Team",
            "HomeTeam": "Opp",
            "FTAG": "GF",
            "FTHG": "GA",
            "AS": "S",
            "AST": "SoT",
            "AC": "C",
            "AF": "F",
            "AY": "Y",
            "AR": "R",
        },
        inplace=True,
    )
    away["is_home"] = 0
    away["pWin"], away["pDraw"], away["pLose"] = away["pA"], away["pD"], away["pH"]

    def res(gf, ga):
        if gf > ga:
            return "W"
        if gf < ga:
            return "L"
        return "D"

    home["Res"] = [res(gf, ga) for gf, ga in zip(home["GF"], home["GA"])]
    away["Res"] = [res(gf, ga) for gf, ga in zip(away["GF"], away["GA"])]

    long = pd.concat([home, away], ignore_index=True)
    long["Pts"] = long["Res"].map({"W": 3, "D": 1, "L": 0})
    return long


def lin_r2_in_sample(y: pd.Series, x: pd.Series) -> float:
    m = ~(y.isna() | x.isna())
    X = x[m].values.reshape(-1, 1)
    model = LinearRegression().fit(X, y[m].values)
    return float(model.score(X, y[m].values))


def lin_r2_time_split(y: pd.Series, x: pd.Series, dates: pd.Series, test_frac: float = 0.2) -> float:
    m = ~(y.isna() | x.isna() | dates.isna())
    y, x, dates = y[m], x[m], dates[m]

    idx = np.argsort(dates.values)
    y = y.iloc[idx]
    x = x.iloc[idx]

    n = len(y)
    if n < 50:
        return float("nan")

    cut = int(np.floor((1.0 - test_frac) * n))
    X_train = x.iloc[:cut].values.reshape(-1, 1)
    y_train = y.iloc[:cut].values
    X_test = x.iloc[cut:].values.reshape(-1, 1)
    y_test = y.iloc[cut:].values

    model = LinearRegression().fit(X_train, y_train)
    return float(model.score(X_test, y_test))


@dataclass
class RollingConfig:
    windows: Tuple[int, ...] = (5, 8)
    compute_partial: bool = True  # min_periods=1 (runtime-friendly)
    compute_full: bool = True  # min_periods=window (evaluation-friendly)


def add_rolling_features(long_df: pd.DataFrame, cfg: RollingConfig) -> pd.DataFrame:
    """Add both partial and full rolling means (shifted to prevent leakage)."""
    long_sorted = long_df.sort_values(["season", "Team", "Date"]).copy()
    metrics = [("GF", "GF"), ("GA", "GA"), ("S", "S"), ("SoT", "SoT"), ("Pts", "Pts"), ("pWin", "pWin")]

    for w in cfg.windows:
        for col, base in metrics:
            if cfg.compute_partial:
                long_sorted[f"{base}_roll{w}_partial"] = long_sorted.groupby(["season", "Team"])[col].transform(
                    lambda s: s.shift(1).rolling(w, min_periods=1).mean()
                )
            if cfg.compute_full:
                long_sorted[f"{base}_roll{w}_full"] = long_sorted.groupby(["season", "Team"])[col].transform(
                    lambda s: s.shift(1).rolling(w, min_periods=w).mean()
                )
    return long_sorted


def main(csv_path: str, outdir: str, windows: Iterable[int]) -> None:
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])

    n_matches = len(df)
    seasons = sorted(df["season"].unique().tolist())
    missing_total = int(df.isna().sum().sum())

    # League summaries
    home_win = float((df["FTR"] == "H").mean())
    draw = float((df["FTR"] == "D").mean())
    away_win = float((df["FTR"] == "A").mean())

    if "total_goals" in df.columns:
        avg_total_goals = float(df["total_goals"].mean())
    else:
        avg_total_goals = float((df["FTHG"] + df["FTAG"]).mean())

    avg_home_goals = float(df["FTHG"].mean())
    avg_away_goals = float(df["FTAG"].mean())

    # Long table (team perspective)
    long = make_long_team_view(df)

    # Team strength summary
    team_agg = long.groupby("Team").agg(
        P=("Res", "size"),
        W=("Res", lambda x: int((x == "W").sum())),
        D=("Res", lambda x: int((x == "D").sum())),
        L=("Res", lambda x: int((x == "L").sum())),
        GF=("GF", "sum"),
        GA=("GA", "sum"),
        Pts=("Pts", "sum"),
        GFpg=("GF", "mean"),
        GApg=("GA", "mean"),
        S_pg=("S", "mean"),
        SoT_pg=("SoT", "mean"),
        pWin_mean=("pWin", "mean"),
    ).reset_index()
    team_agg["GD"] = team_agg["GF"] - team_agg["GA"]
    team_agg["PPG"] = team_agg["Pts"] / team_agg["P"]

    # Efficiency residuals: goals per game vs shots per game
    X = team_agg[["S_pg"]].values
    y = team_agg["GFpg"].values
    eff_model = LinearRegression().fit(X, y)
    team_agg["GFpg_pred"] = eff_model.predict(X)
    team_agg["eff_resid"] = team_agg["GFpg"] - team_agg["GFpg_pred"]
    team_agg.sort_values("PPG", ascending=False).to_csv(os.path.join(outdir, "team_strength_summary.csv"), index=False)

    # Odds implied strength + calibration
    probs = implied_probs_from_odds(df["B365H"], df["B365D"], df["B365A"])
    df2 = pd.concat([df.copy(), probs], axis=1)
    pred = np.where(
        (df2["pH"] >= df2["pD"]) & (df2["pH"] >= df2["pA"]),
        "H",
        np.where(df2["pD"] >= df2["pA"], "D", "A"),
    )
    df2["pred"] = pred
    acc_all = float((df2["pred"] == df2["FTR"]).mean())
    acc_by_season = df2.groupby("season").apply(lambda g: float((g["pred"] == g["FTR"]).mean()))

    bins = np.linspace(0, 1, 11)
    df2["pH_bin"] = pd.cut(df2["pH"], bins=bins, include_lowest=True)
    calib = df2.groupby("pH_bin").agg(
        n=("FTR", "size"),
        implied=("pH", "mean"),
        actual=("FTR", lambda x: float((x == "H").mean())),
    ).reset_index()
    calib.to_csv(os.path.join(outdir, "odds_calibration_homewin.csv"), index=False)

    # Correlation: season-specific PPG gap vs odds gap
    team_season_ppg = long.groupby(["season", "Team"]).agg(PPG=("Pts", "mean")).reset_index()
    m = df2[["season", "HomeTeam", "AwayTeam", "pH", "pA"]].merge(
        team_season_ppg, left_on=["season", "HomeTeam"], right_on=["season", "Team"], how="left"
    ).rename(columns={"PPG": "homePPG"}).drop(columns=["Team"])
    m = m.merge(
        team_season_ppg, left_on=["season", "AwayTeam"], right_on=["season", "Team"], how="left"
    ).rename(columns={"PPG": "awayPPG"}).drop(columns=["Team"])
    corr_gap = float(np.corrcoef((m["homePPG"] - m["awayPPG"]), (m["pH"] - m["pA"]))[0, 1])

    # Rolling feasibility (improved): compute partial+full but EVALUATE using full only
    cfg = RollingConfig(windows=tuple(windows), compute_partial=True, compute_full=True)
    long_roll = add_rolling_features(long, cfg)

    evidence_rows = []
    for w in cfg.windows:
        # Full window only: avoids early-season incomplete-window bias
        s_roll_full = long_roll.get(f"S_roll{w}_full")
        pwin_roll_full = long_roll.get(f"pWin_roll{w}_full")

        if s_roll_full is not None:
            evidence_rows.append(
                {
                    "feature": f"S_roll{w}_full",
                    "target": "GF",
                    "r2_in_sample": lin_r2_in_sample(long_roll["GF"], s_roll_full),
                    "r2_time_split": lin_r2_time_split(long_roll["GF"], s_roll_full, long_roll["Date"]),
                    "n_used": int((~(long_roll["GF"].isna() | s_roll_full.isna())).sum()),
                    "note": "EVALUATION uses full windows only (min_periods=window).",
                }
            )

        if pwin_roll_full is not None:
            evidence_rows.append(
                {
                    "feature": f"pWin_roll{w}_full",
                    "target": "Pts",
                    "r2_in_sample": lin_r2_in_sample(long_roll["Pts"], pwin_roll_full),
                    "r2_time_split": lin_r2_time_split(long_roll["Pts"], pwin_roll_full, long_roll["Date"]),
                    "n_used": int((~(long_roll["Pts"].isna() | pwin_roll_full.isna())).sum()),
                    "note": "EVALUATION uses full windows only (min_periods=window).",
                }
            )

    # Current fixture odds baseline (often strong)
    evidence_rows += [
        {
            "feature": "pWin_current",
            "target": "Pts",
            "r2_in_sample": lin_r2_in_sample(long_roll["Pts"], long_roll["pWin"]),
            "r2_time_split": lin_r2_time_split(long_roll["Pts"], long_roll["pWin"], long_roll["Date"]),
            "n_used": int((~(long_roll["Pts"].isna() | long_roll["pWin"].isna())).sum()),
            "note": "Baseline using current fixture odds only.",
        },
        {
            "feature": "pWin_current",
            "target": "GF",
            "r2_in_sample": lin_r2_in_sample(long_roll["GF"], long_roll["pWin"]),
            "r2_time_split": lin_r2_time_split(long_roll["GF"], long_roll["pWin"], long_roll["Date"]),
            "n_used": int((~(long_roll["GF"].isna() | long_roll["pWin"].isna())).sum()),
            "note": "Baseline using current fixture odds only.",
        },
    ]

    evidence = pd.DataFrame(evidence_rows)
    evidence.to_csv(os.path.join(outdir, "rolling_feasibility_evidence.csv"), index=False)

    # Save a sample table for downstream merging/debugging
    cols_core = ["season", "Date", "Team", "Opp", "is_home", "GF", "GA", "Pts", "pWin"]
    cols_roll = []
    for w in cfg.windows:
        for base in ["GF", "GA", "S", "SoT", "Pts", "pWin"]:
            cols_roll.append(f"{base}_roll{w}_partial")
            cols_roll.append(f"{base}_roll{w}_full")
    cols_sample = [c for c in cols_core + cols_roll if c in long_roll.columns]
    long_roll[cols_sample].head(500).to_csv(os.path.join(outdir, "team_rolling_features_sample.csv"), index=False)

    # Print evidence summary
    print("Evidence Summary")
    print(f"Seasons: {seasons} | Matches: {n_matches} | Missing cells: {missing_total}")
    print("[League totals]")
    print(f"Home/Draw/Away: {home_win:.3f} / {draw:.3f} / {away_win:.3f}")
    print(f"Avg goals: total {avg_total_goals:.3f} | home {avg_home_goals:.3f} | away {avg_away_goals:.3f}")
    print("[Odds]")
    print(f"3-class result accuracy (argmax pH/pD/pA): {acc_all:.3f}")
    for s, a in acc_by_season.items():
        print(f"  {s}: {a:.3f}")
    print(f"Corr(PPG gap, odds gap pH-pA): {corr_gap:.3f}")
    print("Rolling feasibility")
    with pd.option_context("display.max_rows", 60, "display.width", 140):
        print(evidence)
    print("Saved outputs in:", outdir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="epl_matches_clean.csv", help="Path to cleaned EPL match CSV (multi-season supported)")
    ap.add_argument("--outdir", default="outputs_v2", help="Directory for outputs")
    ap.add_argument("--windows", default="5,8", help="Rolling windows (comma-separated), e.g. 5,8,10")
    args = ap.parse_args()

    windows = tuple(int(x.strip()) for x in args.windows.split(",") if x.strip())
    main(args.csv, args.outdir, windows)
