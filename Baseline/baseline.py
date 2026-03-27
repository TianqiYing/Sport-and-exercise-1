import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests

# OR-Tools CP-SAT (integer programming)
from ortools.sat.python import cp_model

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# FPL API client (no key)
FPL_BASE = "https://fantasy.premierleague.com/api"


class FPLClient:
    def __init__(self, cache_dir="cache_fpl", timeout=30, sleep=0.2):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": "FPL-Optimize/1.0"})
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.sleep = sleep
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, key: str) -> str:
        safe = key.replace("/", "_").replace(":", "_")
        return os.path.join(self.cache_dir, safe + ".json")

    def get_json(self, path: str, cache_key: str, refresh: bool = False):
        p = self._cache_path(cache_key)
        if os.path.exists(p) and not refresh:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)

        url = f"{FPL_BASE}/{path.lstrip('/')}"
        r = self.s.get(url, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f)
        time.sleep(self.sleep)
        return data

    def bootstrap_static(self, refresh=False):
        return self.get_json("bootstrap-static/", "bootstrap-static", refresh)

    def fixtures(self, refresh=False):
        return self.get_json("fixtures/", "fixtures", refresh)

    def element_summary(self, element_id: int, refresh=False):
        return self.get_json(f"element-summary/{element_id}/", f"element-summary-{element_id}", refresh)


# Data structures
@dataclass(frozen=True)
class Player:
    pid: int
    name: str
    pos: str          # GK/DEF/MID/FWD
    team_id: int
    team_name: str
    price: float      # now_cost / 10


POS_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


FEATURE_COLS = [
    "pts_wma5", "mins_wma5", "p0_5", "p60_5", "spread_5",
    "opp_fdr", "is_home", "home_x_easiness",
    "chance_play", "points_per_game", "form"
]


def build_players(client: FPLClient) -> Tuple[List[Player], pd.DataFrame, Dict[int, str]]:
    bs = client.bootstrap_static(refresh=False)
    teams = {t["id"]: t["name"] for t in bs["teams"]}
    players = []
    rows = []
    for e in bs["elements"]:
        pid = int(e["id"])
        name = f'{e["first_name"]} {e["second_name"]}'.strip()
        pos = POS_MAP[int(e["element_type"])]
        team_id = int(e["team"])
        price = float(e["now_cost"]) / 10.0
        players.append(Player(pid, name, pos, team_id, teams[team_id], price))
        rows.append({
            "pid": pid,
            "name": name,
            "pos": pos,
            "team_id": team_id,
            "team_name": teams[team_id],
            "price": price,
            "status": e.get("status"),
            "chance_of_playing_next_round": e.get("chance_of_playing_next_round"),
            "selected_by_percent": float(e.get("selected_by_percent") or 0.0),
            "form": float(e.get("form") or 0.0),
            "points_per_game": float(e.get("points_per_game") or 0.0),
        })
    df_players = pd.DataFrame(rows)
    return players, df_players, teams


def get_fixture_context(fixtures: list, team_id: int, gw: int) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Return (opp_team_id, is_home, difficulty) for team_id at GW.
    FPL fixtures endpoint provides difficulty from team perspective.
    """
    for fx in fixtures:
        if fx.get("event") != gw:
            continue
        h = fx["team_h"]
        a = fx["team_a"]
        if h == team_id:
            return a, 1, fx.get("team_h_difficulty")
        if a == team_id:
            return h, 0, fx.get("team_a_difficulty")
    return None, None, None


def load_history(client: FPLClient, pid: int) -> pd.DataFrame:
    js = client.element_summary(pid, refresh=False)
    hist = js.get("history", [])
    if not hist:
        return pd.DataFrame(columns=["gw", "minutes", "points"])
    df = pd.DataFrame({
        "gw": [int(h["round"]) for h in hist],
        "minutes": [int(h["minutes"]) for h in hist],
        "points": [int(h["total_points"]) for h in hist],
    })
    return df


# Feature engineering (baseline)
def weighted_moving_average(values: np.ndarray, alpha: float = 0.7) -> float:
    """
    Exponentially weighted average with weights 1, alpha, alpha^2...
    Assume values ordered from most recent to older.
    """
    if len(values) == 0:
        return 0.0
    w = np.array([alpha ** i for i in range(len(values))], dtype=float)
    return float(np.sum(w * values) / np.sum(w))


def build_features_for_gw(
    client: FPLClient,
    df_players: pd.DataFrame,
    fixtures: list,
    gw: int,
    window: int = 5,
    alpha: float = 0.7,
) -> pd.DataFrame:
    """
    Build a modeling table for one GW:
    - rolling weighted averages of points/minutes
    - p0/p60
    - spread (q90-q10) from last window points where minutes>0
    - fixture difficulty (FDR-like) + home + interaction
    Target y is not included here
    """
    rows = []
    for _, r in df_players.iterrows():
        pid = int(r["pid"])
        team_id = int(r["team_id"])
        opp_id, is_home, diff = get_fixture_context(fixtures, team_id, gw)
        if diff is None:
            # blank GW or no fixture
            continue

        hist = load_history(client, pid)
        past = hist[hist["gw"] < gw].sort_values("gw", ascending=False).head(window)

        pts = past["points"].to_numpy(dtype=float)
        mins = past["minutes"].to_numpy(dtype=float)

        pts_wma5 = weighted_moving_average(pts, alpha=alpha)
        mins_wma5 = weighted_moving_average(mins, alpha=alpha)

        p0 = float(np.mean(mins == 0)) if len(mins) else 0.0
        p60 = float(np.mean(mins >= 60)) if len(mins) else 0.0

        # spread using points when played (>0 mins)
        played_pts = past.loc[past["minutes"] > 0, "points"].to_numpy(dtype=float)
        if len(played_pts) >= 2:
            q10 = float(np.quantile(played_pts, 0.10))
            q90 = float(np.quantile(played_pts, 0.90))
            spread = q90 - q10
        else:
            spread = 0.0

        # basic availability flags from bootstrap-static
        chance = r.get("chance_of_playing_next_round")
        chance = float(chance) / 100.0 if chance is not None else 1.0

        rows.append({
            "pid": pid,
            "name": r["name"],
            "pos": r["pos"],
            "team_id": team_id,
            "team_name": r["team_name"],
            "price": float(r["price"]),
            "selected_by_percent": float(r.get("selected_by_percent", 0.0)),
            "form": float(r.get("form", 0.0)),
            "points_per_game": float(r.get("points_per_game", 0.0)),
            # recent form
            "pts_wma5": pts_wma5,
            "mins_wma5": mins_wma5,
            "p0_5": p0,
            "p60_5": p60,
            "spread_5": spread,
            "chance_play": chance,
            # fixture context
            "opp_team_id": opp_id,
            "is_home": int(is_home),
            "opp_fdr": int(diff),            # 1 (easy) .. 5 (hard)
            "home_x_easiness": int(is_home) * (6 - int(diff)),  # interaction
        })

    df = pd.DataFrame(rows)
    return df


# Baseline predictor for next GW points
def predict_baseline(df: pd.DataFrame) -> np.ndarray:
    """
    Simple baseline: recent weighted points adjusted by difficulty + availability.
    """
    # normalize difficulty: easy -> +, hard -> -
    easiness = (6 - df["opp_fdr"].astype(float))  # 1..5 => 5..1
    # coefficients (reasonable starting values)
    pred = (
        df["pts_wma5"].astype(float)
        + 0.25 * df["home_x_easiness"].astype(float)
        + 0.15 * easiness
        - 1.2 * df["p0_5"].astype(float)
        - 0.4 * (1.0 - df["chance_play"].astype(float))
    )
    # don't go negative
    return np.maximum(pred.to_numpy(), 0.0)


def predict_official(df: pd.DataFrame) -> np.ndarray:
    easiness = (6 - df["opp_fdr"].astype(float))

    # Blend official signals
    base = 0.75 * df["points_per_game"].astype(float) + 0.25 * df["form"].astype(float)

    pred = (
        base
        + 0.10 * easiness
        + 0.15 * df["home_x_easiness"].astype(float)
    )
    pred = pred * df["chance_play"].astype(float)
    return np.maximum(pred.to_numpy(), 0.0)


# Evaluation: RMSE + Spearman per GW
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    a = pd.Series(y_true).rank(method="average").to_numpy(dtype=float)
    b = pd.Series(y_pred).rank(method="average").to_numpy(dtype=float)
    # np.corrcoef returns 2x2 matrix
    corr = np.corrcoef(a, b)[0, 1]
    return float(corr)


def topk_recall(y_true: np.ndarray, y_pred: np.ndarray, k: int = 20) -> float:
    if len(y_true) == 0:
        return float("nan")
    k = min(k, len(y_true))
    idx_true = set(np.argsort(-y_true)[:k])
    idx_pred = set(np.argsort(-y_pred)[:k])
    return float(len(idx_true & idx_pred) / k)


def clean_pred_series(x):
    """
    Returns a numpy array.
    """
    arr = np.asarray(x, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def build_training_set(
    client: FPLClient,
    df_players: pd.DataFrame,
    fixtures: list,
    gw_start: int,
    gw_end: int,
    window: int = 5,
) -> pd.DataFrame:
    """
    Build a dataset across multiple GWs:
    features at GW t -> target y = points at GW t (or t+1).
    """
    all_rows = []
    # Preload histories once for speed
    hist_cache: Dict[int, pd.DataFrame] = {}

    for t in range(gw_start, gw_end + 1):
        feat = build_features_for_gw(client, df_players, fixtures, t, window=window)
        if feat.empty:
            continue

        # attach y = actual points at gw t from element_summary history
        ys = []
        for pid in feat["pid"].astype(int).tolist():
            if pid not in hist_cache:
                hist_cache[pid] = load_history(client, pid)
            h = hist_cache[pid]
            row = h[h["gw"] == t]
            y = float(row["points"].iloc[0]) if len(row) else 0.0
            ys.append(y)
        feat["y_points"] = ys
        feat["gw"] = t
        all_rows.append(feat)

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


# Integer Programming (CP-SAT) for selecting team
def select_best_starting_xi(
    df: pd.DataFrame,
    pred_col: str,
    budget: float = 100.0,
    max_per_team: int = 3,
    with_captain: bool = True,
) -> Dict[str, object]:
    """
    Select best starting XI for one GW:
    - 11 players
    - Formation constraints
    - Budget constraint
    - Max 3 per team
    - Optional captain selection (captain doubles; model as +pred for captain)
    """
    df = df.reset_index(drop=True)
    model = cp_model.CpModel()
    n = len(df)

    x = [model.NewBoolVar(f"x_{i}") for i in range(n)]
    c = [model.NewBoolVar(f"c_{i}") for i in range(n)] if with_captain else None

    # basic counts
    model.Add(sum(x) == 11)

    # formation constraints
    pos = df["pos"].tolist()
    def idxs(p): return [i for i, pp in enumerate(pos) if pp == p]

    model.Add(sum(x[i] for i in idxs("GK")) == 1)
    model.Add(sum(x[i] for i in idxs("DEF")) >= 3)
    model.Add(sum(x[i] for i in idxs("DEF")) <= 5)
    model.Add(sum(x[i] for i in idxs("MID")) >= 2)
    model.Add(sum(x[i] for i in idxs("MID")) <= 5)
    model.Add(sum(x[i] for i in idxs("FWD")) >= 1)
    model.Add(sum(x[i] for i in idxs("FWD")) <= 3)

    # team limit
    for team_id, g in df.groupby("team_id"):
        idx = g.index.to_list()
        model.Add(sum(x[i] for i in idx) <= max_per_team)

    # budget
    costs = (df["price"] * 10).round().astype(int).tolist()  # scale to int
    budget_int = int(round(budget * 10))
    model.Add(sum(costs[i] * x[i] for i in range(n)) <= budget_int)

    # captain constraints
    if with_captain:
        model.Add(sum(c) == 1)
        for i in range(n):
            model.Add(c[i] <= x[i])

    # objective
    preds = df[pred_col].astype(float).to_numpy()
    preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
    # scale objective to int
    scale = 1000
    obj_terms = [int(round(preds[i] * scale)) * x[i] for i in range(n)]
    if with_captain:
        obj_terms += [int(round(preds[i] * scale)) * c[i] for i in range(n)]  # add one more time for captain
    model.Maximize(sum(obj_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible solution found.")

    chosen_idx = [i for i in range(n) if solver.Value(x[i]) == 1]
    xi = df.iloc[chosen_idx].copy().sort_values(pred_col, ascending=False)

    cap_pid = None
    if with_captain:
        cap_idx = [i for i in range(n) if solver.Value(c[i]) == 1][0]
        cap_pid = int(df.iloc[cap_idx]["pid"])

    return {"xi": xi, "captain_pid": cap_pid}


def select_best_15_squad(
    df: pd.DataFrame,
    pred_col: str,
    budget: float = 100.0,
    max_per_team: int = 3,
) -> pd.DataFrame:
    """
    Select best 15-player squad (standard composition):
      GK 2, DEF 5, MID 5, FWD 3
    Objective: sum of predicted points (simple proxy; true optimal would consider best XI each GW)
    """
    df = df.reset_index(drop=True)
    model = cp_model.CpModel()
    n = len(df)
    x = [model.NewBoolVar(f"x_{i}") for i in range(n)]

    model.Add(sum(x) == 15)
    pos = df["pos"].tolist()
    def idxs(p): return [i for i, pp in enumerate(pos) if pp == p]
    model.Add(sum(x[i] for i in idxs("GK")) == 2)
    model.Add(sum(x[i] for i in idxs("DEF")) == 5)
    model.Add(sum(x[i] for i in idxs("MID")) == 5)
    model.Add(sum(x[i] for i in idxs("FWD")) == 3)

    for team_id, g in df.groupby("team_id"):
        idx = g.index.to_list()
        model.Add(sum(x[i] for i in idx) <= max_per_team)

    costs = (df["price"] * 10).round().astype(int).tolist()
    budget_int = int(round(budget * 10))
    model.Add(sum(costs[i] * x[i] for i in range(n)) <= budget_int)

    preds = df[pred_col].astype(float).to_numpy()
    scale = 1000
    model.Maximize(sum(int(round(preds[i] * scale)) * x[i] for i in range(n)))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible solution found.")

    chosen_idx = [i for i in range(n) if solver.Value(x[i]) == 1]
    return df.iloc[chosen_idx].copy().sort_values(pred_col, ascending=False)


# Main run
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gw", type=int, required=True, help="Target gameweek to optimize for")
    ap.add_argument("--mode", choices=["xi", "squad15"], default="xi", help="Optimize starting XI or 15-player squad")
    ap.add_argument(
        "--predictor",
        choices=["baseline", "official", "ridge"],
        default="official",
        help="Which predictor to use for the single-GW optimisation output.",
    )
    ap.add_argument("--budget", type=float, default=100.0)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--train_eval", action="store_true", help="Also build training set and report RMSE+Spearman baseline")
    ap.add_argument("--train_gw_start", type=int, default=2)
    ap.add_argument("--train_gw_end", type=int, default=10)
    ap.add_argument(
        "--eval_scheme",
        choices=["insample", "walkforward"],
        default="walkforward",
        help="Evaluation scheme for Ridge in --train_eval.",
    )
    ap.add_argument(
        "--min_train_gws",
        type=int,
        default=3,
        help="For walkforward eval: minimum distinct past GWs required before predicting a GW.",
    )
    args = ap.parse_args()

    client = FPLClient()
    players, df_players, teams = build_players(client)
    fixtures = client.fixtures(refresh=False)

    # Build features for target GW
    feat = build_features_for_gw(client, df_players, fixtures, args.gw, window=args.window, alpha=args.alpha)
    if feat.empty:
        raise RuntimeError("No fixtures/features for this GW. Check GW number.")

    # Baseline prediction
    feat["pred_baseline"] = predict_baseline(feat)
    feat["pred_baseline"] = feat["pred_baseline"].fillna(0.0)

    feat["pred_official"] = predict_official(feat)
    feat["pred_official"] = feat["pred_official"].fillna(0.0)

    # Ridge prediction for the single target GW (needed only if --predictor ridge)
    feat["pred_ridge"] = 0.0
    if args.predictor == "ridge":
        ds_for_ridge = build_training_set(
            client, df_players, fixtures, args.train_gw_start, args.train_gw_end, window=args.window
        )
        if not ds_for_ridge.empty:
            Xtr = ds_for_ridge[FEATURE_COLS].fillna(0.0).to_numpy(dtype=float)
            ytr = ds_for_ridge["y_points"].to_numpy(dtype=float)
            ridge_single = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25), fit_intercept=True)),
                ]
            )
            ridge_single.fit(Xtr, ytr)
            feat["pred_ridge"] = ridge_single.predict(feat[FEATURE_COLS].fillna(0.0).to_numpy(dtype=float))
            feat["pred_ridge"] = clean_pred_series(pd.Series(feat["pred_ridge"]))

    # Optimize
    if args.mode == "xi":
        pred_col_used = {
            "baseline": "pred_baseline",
            "official": "pred_official",
            "ridge": "pred_ridge",
        }[args.predictor]
        out = select_best_starting_xi(feat, pred_col_used, budget=args.budget, with_captain=True)
        xi = out["xi"]
        cap_pid = out["captain_pid"]
        print(f"\nBest Starting XI ({pred_col_used})")
        print(xi[["name", "pos", "team_name", "price", "opp_fdr", "is_home", pred_col_used]].to_string(index=False))
        print("Captain:", cap_pid, xi[xi["pid"] == cap_pid]["name"].iloc[0] if cap_pid in xi["pid"].values else cap_pid)

    else:
        squad15 = select_best_15_squad(feat, "pred_baseline", budget=args.budget)
        print("\nBest 15-player Squad (baseline proxy objective)")
        print(squad15[["name", "pos", "team_name", "price", "opp_fdr", "is_home", "pred_baseline"]].to_string(index=False))

    # Evaluation
    if args.train_eval:
        ds = build_training_set(
            client, df_players, fixtures, args.train_gw_start, args.train_gw_end, window=args.window
        )
        if ds.empty:
            print("No training set built (check GW range).")
            return

        # Predictions for multiple predictors
        ds["pred_baseline"] = clean_pred_series(predict_baseline(ds).astype(float))
        ds["pred_official"] = clean_pred_series(predict_official(ds).astype(float))

        # Ridge prediction: insample or walkforward
        alphas = np.logspace(-3, 3, 25)
        ds["pred_ridge"] = np.nan
        if args.eval_scheme == "insample":
            X = ds[FEATURE_COLS].fillna(0.0).to_numpy(dtype=float)
            y = ds["y_points"].to_numpy(dtype=float)
            ridge = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", RidgeCV(alphas=alphas, fit_intercept=True))
            ])
            ridge.fit(X, y)
            ds["pred_ridge"] = ridge.predict(X)
            ds["pred_ridge"] = clean_pred_series(pd.Series(ds["pred_ridge"]))
        else:
            # walk-forward by GW (out-of-sample)
            gws_sorted = sorted(ds["gw"].unique())
            for t in gws_sorted:
                train = ds[ds["gw"] < t]
                test = ds[ds["gw"] == t]
                if train["gw"].nunique() < args.min_train_gws or len(test) == 0:
                    continue
                Xtr = train[FEATURE_COLS].fillna(0.0).to_numpy(dtype=float)
                ytr = train["y_points"].to_numpy(dtype=float)
                Xte = test[FEATURE_COLS].fillna(0.0).to_numpy(dtype=float)
                ridge = Pipeline([
                    ("scaler", StandardScaler()),
                    ("ridge", RidgeCV(alphas=alphas, fit_intercept=True))
                ])
                ridge.fit(Xtr, ytr)
                ds.loc[test.index, "pred_ridge"] = ridge.predict(Xte)
            ds["pred_ridge"] = clean_pred_series(ds["pred_ridge"])  # fills early NaNs with 0

        # Unified evaluation helper
        y_true_all = ds["y_points"].to_numpy(dtype=float)

        def eval_predictor(pred_col: str, k: int = 20):
            yhat_all = np.nan_to_num(ds[pred_col].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            print(f"\nEvaluation: {pred_col}")
            print("Overall RMSE:", rmse(y_true_all, yhat_all))

            sp_list, topk_list, xi_scores = [], [], []
            for gw, g in ds.groupby("gw"):
                yt = g["y_points"].to_numpy(dtype=float)
                yp = g[pred_col].to_numpy(dtype=float)

                rho = spearman(yt, yp)
                if not math.isnan(rho):
                    sp_list.append(rho)
                topk_list.append(topk_recall(yt, yp, k=k))

                g = g.reset_index(drop=True)
                chosen = select_best_starting_xi(g, pred_col, budget=args.budget, with_captain=False)["xi"]
                xi_scores.append(float(chosen["y_points"].sum()))

            print("Mean Spearman (per-GW):", float(np.mean(sp_list)) if sp_list else float("nan"))
            print(f"Mean Recall@{k} (per-GW):", float(np.mean(topk_list)) if topk_list else float("nan"))
            print("Avg true XI points (chosen by pred):", float(np.mean(xi_scores)) if xi_scores else float("nan"))

        eval_predictor("pred_baseline", k=20)
        eval_predictor("pred_official", k=20)
        eval_predictor("pred_ridge", k=20)


if __name__ == "__main__":
    main()
