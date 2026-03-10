import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


def implied_probs_from_odds(b365h: pd.Series, b365d: pd.Series, b365a: pd.Series) -> pd.DataFrame:
    """Convert 1X2 odds into normalised implied probabilities (remove overround)."""
    inv_h = 1.0 / b365h
    inv_d = 1.0 / b365d
    inv_a = 1.0 / b365a
    s = inv_h + inv_d + inv_a
    return pd.DataFrame({"pH": inv_h / s, "pD": inv_d / s, "pA": inv_a / s})


def to_team_match_long(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    probs = implied_probs_from_odds(df["B365H"], df["B365D"], df["B365A"])
    df = pd.concat([df, probs], axis=1)

    home = df[["season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "pH", "pD", "pA"]].copy()
    home.rename(columns={"HomeTeam": "Team", "AwayTeam": "Opp", "FTHG": "GF", "FTAG": "GA"}, inplace=True)
    home["is_home"] = 1
    home["pWin"] = home["pH"]

    away = df[["season", "Date", "AwayTeam", "HomeTeam", "FTAG", "FTHG", "pH", "pD", "pA"]].copy()
    away.rename(columns={"AwayTeam": "Team", "HomeTeam": "Opp", "FTAG": "GF", "FTHG": "GA"}, inplace=True)
    away["is_home"] = 0
    away["pWin"] = away["pA"]

    long = pd.concat([home, away], ignore_index=True)

    # Points from result
    long["Pts"] = np.where(long["GF"] > long["GA"], 3, np.where(long["GF"] == long["GA"], 1, 0))
    return long[["season", "Date", "Team", "Opp", "is_home", "GF", "GA", "Pts", "pWin"]]


def rolling_distribution_features(
    s: pd.Series,
    window: int,
    full_window_only_for_eval: bool = True,
    prefix: str = "",
) -> pd.DataFrame:
    s_shift = s.shift(1)

    # partial
    roll_p = s_shift.rolling(window, min_periods=1)
    out = pd.DataFrame(
        {
            f"{prefix}mean_roll{window}_partial": roll_p.mean(),
            f"{prefix}var_roll{window}_partial": roll_p.var(ddof=0),
            f"{prefix}std_roll{window}_partial": roll_p.std(ddof=0),
            f"{prefix}q10_roll{window}_partial": roll_p.quantile(0.10),
            f"{prefix}q50_roll{window}_partial": roll_p.quantile(0.50),
            f"{prefix}q90_roll{window}_partial": roll_p.quantile(0.90),
        }
    )

    # full
    roll_f = s_shift.rolling(window, min_periods=window)
    out[f"{prefix}mean_roll{window}_full"] = roll_f.mean()
    out[f"{prefix}var_roll{window}_full"] = roll_f.var(ddof=0)
    out[f"{prefix}std_roll{window}_full"] = roll_f.std(ddof=0)
    out[f"{prefix}q10_roll{window}_full"] = roll_f.quantile(0.10)
    out[f"{prefix}q50_roll{window}_full"] = roll_f.quantile(0.50)
    out[f"{prefix}q90_roll{window}_full"] = roll_f.quantile(0.90)

    return out


def conditional_distribution_by_bin(
    df_long: pd.DataFrame,
    value_col: str,
    condition_col: str,
    bins: List[float],
) -> pd.DataFrame:
    d = df_long.copy()
    d["bin"] = pd.cut(d[condition_col], bins=bins, include_lowest=True)

    def q(x, p):
        return float(np.quantile(x, p)) if len(x) else np.nan

    out = (
        d.groupby("bin")[value_col]
        .apply(
            lambda x: pd.Series(
                {
                    "n": int(x.shape[0]),
                    "mean": float(x.mean()),
                    "std": float(x.std(ddof=0)),
                    "q10": q(x, 0.10),
                    "q50": q(x, 0.50),
                    "q90": q(x, 0.90),
                }
            )
        )
        .reset_index()
    )
    return out


@dataclass
class Config:
    csv_path: str = "epl_matches_clean.csv"
    outdir: str = "outputs_variance"
    windows: Tuple[int, ...] = (5, 8)
    pwin_bins: Tuple[float, ...] = (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)


def run(cfg: Config) -> None:
    os.makedirs(cfg.outdir, exist_ok=True)
    df = pd.read_csv(cfg.csv_path)

    # Basic checks
    required = ["season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "B365H", "B365D", "B365A"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Long table (team perspective)
    long = to_team_match_long(df)
    long = long.sort_values(["season", "Team", "Date"]).reset_index(drop=True)

    # Rolling distribution features per team (Pts + GF + GA)
    feats = [long]
    for w in cfg.windows:
        feats.append(
            long.groupby(["season", "Team"], group_keys=False)["Pts"]
            .apply(lambda s: rolling_distribution_features(s, w, prefix="Pts_"))
        )
        feats.append(
            long.groupby(["season", "Team"], group_keys=False)["GF"]
            .apply(lambda s: rolling_distribution_features(s, w, prefix="GF_"))
        )
        feats.append(
            long.groupby(["season", "Team"], group_keys=False)["GA"]
            .apply(lambda s: rolling_distribution_features(s, w, prefix="GA_"))
        )
        feats.append(
            long.groupby(["season", "Team"], group_keys=False)["pWin"]
            .apply(lambda s: rolling_distribution_features(s, w, prefix="pWin_"))
        )

    long_with_roll = pd.concat(feats, axis=1)

    # Save a sample + full dataset
    long_with_roll.head(800).to_csv(os.path.join(cfg.outdir, "team_match_with_rolling_sample.csv"), index=False)
    long_with_roll.to_csv(os.path.join(cfg.outdir, "team_match_with_rolling_full.csv"), index=False)

    # Conditional distribution tables
    bins = list(cfg.pwin_bins)

    cond_pts = conditional_distribution_by_bin(long, value_col="Pts", condition_col="pWin", bins=bins)
    cond_pts.to_csv(os.path.join(cfg.outdir, "conditional_pts_given_pwin.csv"), index=False)

    cond_gf = conditional_distribution_by_bin(long, value_col="GF", condition_col="pWin", bins=bins)
    cond_gf.to_csv(os.path.join(cfg.outdir, "conditional_gf_given_pwin.csv"), index=False)

    print("Done.")
    print(f"Saved: {os.path.join(cfg.outdir, 'team_match_with_rolling_full.csv')}")
    print(f"Saved: {os.path.join(cfg.outdir, 'conditional_pts_given_pwin.csv')}")
    print(f"Saved: {os.path.join(cfg.outdir, 'conditional_gf_given_pwin.csv')}")
    print("Tip: use *_full columns for any evaluation/claims; *_partial is for early-season runtime.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="epl_matches_clean.csv", help="Input match CSV (multi-season supported)")
    ap.add_argument("--outdir", default="outputs_variance", help="Output directory")
    ap.add_argument("--windows", default="5,8", help="Comma-separated rolling windows (e.g., 5,8,10)")
    args = ap.parse_args()

    windows = tuple(int(x.strip()) for x in args.windows.split(",") if x.strip())
    cfg = Config(csv_path=args.csv, outdir=args.outdir, windows=windows)
    run(cfg)
