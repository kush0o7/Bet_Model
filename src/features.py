# src/features.py
import pandas as pd
import numpy as np

def _build_team_history_rows(df):
    """
    Convert match-level df to long format with one row per team per match,
    keeping info needed to compute rolling form.
    Expects columns: date, home, away, and either home_goals/away_goals or FTHG/FTAG.
    Returns a DataFrame with a deterministic match_id (0..N-1) matching df.reset_index(drop=True).
    """
    # Work on a reset index so match_id is 0..N-1
    df = df.reset_index(drop=True).copy()
    # unify goal column names
    if "home_goals" in df.columns and "away_goals" in df.columns:
        hcol = "home_goals"; acol = "away_goals"
    elif "FTHG" in df.columns and "FTAG" in df.columns:
        hcol = "FTHG"; acol = "FTAG"
    else:
        raise RuntimeError("Cannot find goal columns (need home_goals/away_goals or FTHG/FTAG)")

    rows = []
    for match_id, r in df.iterrows():
        d = r.get("date", None)
        # home row
        rows.append({
            "match_id": match_id,
            "date": d,
            "team": r["home"],
            "is_home": 1,
            "goals_for": r[hcol],
            "goals_against": r[acol],
            "opponent": r["away"],
            "result": r.get("result", None)
        })
        # away row
        rows.append({
            "match_id": match_id,
            "date": d,
            "team": r["away"],
            "is_home": 0,
            "goals_for": r[acol],
            "goals_against": r[hcol],
            "opponent": r["home"],
            "result": r.get("result", None)
        })

    long = pd.DataFrame(rows)
    # compute points per row from result (FTR style H/D/A)
    def points_for(row):
        res = row["result"]
        if pd.isna(res):
            return 0.0
        if res == "D":
            return 1.0
        if res == "H":
            return 3.0 if row["is_home"] == 1 else 0.0
        if res == "A":
            return 3.0 if row["is_home"] == 0 else 0.0
        return 0.0

    long["points"] = long.apply(points_for, axis=1).astype(float)
    # sort for stable rolling
    long = long.sort_values(["team", "date", "match_id"]).reset_index(drop=True)
    return long

def add_last_n_form(df, n=5):
    """
    Add last-n rolling features (excluding the current match) to match-level df.
    Returns a copy of df with these new columns:
      home_form_pts, away_form_pts, home_form_gf, away_form_gf, form_pts_diff, gf_diff
    """
    if "date" not in df.columns:
        raise RuntimeError("df must have 'date' column")

    # operate on a reset-index copy so match_id is 0..N-1
    df2 = df.reset_index(drop=True).copy()

    # Create match_id in the match-level DF so merge works
    df2["match_id"] = df2.index

    long = _build_team_history_rows(df2)

    # For each group, compute shift(1) then rolling mean; assign as numpy array to avoid index alignment issues
    pts_series = long.groupby("team")["points"].apply(lambda s: s.shift(1).rolling(window=n, min_periods=1).mean())
    gf_series = long.groupby("team")["goals_for"].apply(lambda s: s.shift(1).rolling(window=n, min_periods=1).mean())
    ga_series = long.groupby("team")["goals_against"].apply(lambda s: s.shift(1).rolling(window=n, min_periods=1).mean())

    # assign back using .values (same length & order as long)
    long["rolling_pts_mean"] = pts_series.values
    long["rolling_gf_mean"] = gf_series.values
    long["rolling_ga_mean"] = ga_series.values

    # pivot back to one row per match_id, picking home/away rows
    home = long[long["is_home"] == 1][["match_id", "rolling_pts_mean", "rolling_gf_mean", "rolling_ga_mean"]].rename(
        columns={
            "rolling_pts_mean": "home_form_pts",
            "rolling_gf_mean": "home_form_gf",
            "rolling_ga_mean": "home_form_ga"
        }
    )
    away = long[long["is_home"] == 0][["match_id", "rolling_pts_mean", "rolling_gf_mean", "rolling_ga_mean"]].rename(
        columns={
            "rolling_pts_mean": "away_form_pts",
            "rolling_gf_mean": "away_form_gf",
            "rolling_ga_mean": "away_form_ga"
        }
    )

    # merge features into df2 by match_id
    df2 = df2.merge(home, on="match_id", how="left")
    df2 = df2.merge(away, on="match_id", how="left")

    # derived features
    df2["form_pts_diff"] = df2["home_form_pts"].fillna(0.0) - df2["away_form_pts"].fillna(0.0)
    df2["gf_diff"] = df2["home_form_gf"].fillna(0.0) - df2["away_form_gf"].fillna(0.0)

    # fill remaining NaNs with zeros (early-season)
    for c in ["home_form_pts", "away_form_pts", "home_form_gf", "away_form_gf", "form_pts_diff", "gf_diff"]:
        if c in df2.columns:
            df2[c] = df2[c].fillna(0.0)

    # return in same order/shape as input (but with new columns)
    return df2
