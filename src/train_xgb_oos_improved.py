# src/train_xgb_oos_improved.py
import sys
from pathlib import Path

# --- make project root importable so `from src...` works ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ------------------------------------------------------------

import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# local project imports (assumes your repo has these)
from src.load_data import load_matches
from src.odds import odds_to_probs

# ---------------------------
# Small robust ELO implementation (adds elo_home_pre, elo_away_pre, elo_diff)
# ---------------------------
def compute_simple_elo(df, k=20, home_adv=100, init_rating=1500):
    """
    df must contain: date, home, away, home_goals, away_goals (or result)
    Produces columns: elo_home_pre, elo_away_pre, elo_diff (home - away)
    """
    # ensure sorted by date so ratings progress correctly
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    ratings = {}  # team -> rating
    elo_home_list = []
    elo_away_list = []

    def get_rating(team):
        return ratings.get(team, init_rating)

    for _, row in df.iterrows():
        home = row["home"]
        away = row["away"]

        Rh = get_rating(home)
        Ra = get_rating(away)

        # store pre-match ratings
        elo_home_list.append(Rh + home_adv)  # include home advantage for pre-match home elo
        elo_away_list.append(Ra)

        # compute expected score (using home advantage)
        exp_home = 1.0 / (1.0 + 10 ** (((Ra) - (Rh + home_adv)) / 400.0))
        exp_away = 1.0 - exp_home

        # derive actual score from available columns
        sh = sa = None
        if ("home_goals" in row and not pd.isna(row["home_goals"])) and ("away_goals" in row and not pd.isna(row["away_goals"])):
            hg = row["home_goals"]
            ag = row["away_goals"]
            if hg > ag:
                sh, sa = 1.0, 0.0
            elif hg == ag:
                sh, sa = 0.5, 0.5
            else:
                sh, sa = 0.0, 1.0
        elif "result" in row and not pd.isna(row["result"]):
            r = row["result"]
            if r in ("H", "h", "Home", "home"):
                sh, sa = 1.0, 0.0
            elif r in ("D", "d", "Draw", "draw"):
                sh, sa = 0.5, 0.5
            else:
                sh, sa = 0.0, 1.0

        if sh is not None:
            # update ratings (home_adv used only in expectation, not stored)
            new_Rh = Rh + k * (sh - exp_home)
            new_Ra = Ra + k * (sa - exp_away)
            ratings[home] = new_Rh
            ratings[away] = new_Ra
        else:
            # no result -> keep ratings unchanged
            ratings[home] = Rh
            ratings[away] = Ra

    out = df.copy()
    out["elo_home_pre"] = elo_home_list
    out["elo_away_pre"] = elo_away_list
    out["elo_diff"] = out["elo_home_pre"] - out["elo_away_pre"]
    out["elo_home_base"] = out["elo_home_pre"] - home_adv
    return out

# ---------------------------
# Prepare dataframe used for training
# ---------------------------
def prepare_df():
    df = load_matches()
    print(f"Loaded season files and {len(df)} rows (raw).")

    # compute market probabilities from odds (if not present)
    if not {"p_home","p_draw","p_away"}.issubset(df.columns):
        probs = df.apply(
            lambda r: odds_to_probs(r.get("odds_home"), r.get("odds_draw"), r.get("odds_away")),
            axis=1, result_type="expand"
        )
        if probs.shape[1] == 3:
            probs.columns = ["p_home","p_draw","p_away"]
            df = pd.concat([df.reset_index(drop=True), probs.reset_index(drop=True)], axis=1)

    # compute elo_diff if missing
    if "elo_diff" not in df.columns:
        df = compute_simple_elo(df, k=20, home_adv=100, init_rating=1500)
        print("Computed simple elo => columns elo_home_pre, elo_away_pre, elo_diff added.")

    # ensure we have result column in H/D/A
    if "result" not in df.columns and "FTR" in df.columns:
        df = df.rename(columns={"FTR":"result"})

    # filter to rows that have what we need
    required = ["p_home","p_draw","p_away","elo_diff","result"]
    df = df.dropna(subset=required).reset_index(drop=True)
    # keep only expected result codes
    df = df[df["result"].isin(["H","D","A"])].reset_index(drop=True)

    print("Rows used for training after dropna:", len(df))
    return df

# ---------------------------
# Training pipeline
# ---------------------------
def main():
    df = prepare_df()

    features = ["p_home","p_draw","p_away","elo_diff"]
    X = df[features].values
    y = df["result"].map({"H":0,"D":1,"A":2}).astype(int).values

    # time-series split for calibration
    tss = TimeSeriesSplit(n_splits=3)

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", xgb)
    ])

    print("Starting cross-validated training + calibration...")

    # NOTE: some sklearn versions accept `base_estimator=...` but many expect the estimator
    #       as the first (positional) argument. Use positional arg to be compatible.
    calib = CalibratedClassifierCV(pipe, method="isotonic", cv=tss)

    # fit the calibrated estimator (this will fit internal clones using the provided cv)
    calib.fit(X, y)

    # save
    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    joblib.dump(calib, out_dir / "xgb_oos_improved_calibrated.joblib")
    pd.DataFrame({"feature": features}).to_csv(out_dir / "xgb_oos_improved_feature_names.csv", index=False)

    summary = {
        "n_rows": int(len(df)),
        "features": features,
        "model": "XGBClassifier (calibrated isotonic)"
    }
    (out_dir / "xgb_oos_improved_summary.json").write_text(json.dumps(summary, indent=2))

    print("✅ Saved calibrated model ->", out_dir / "xgb_oos_improved_calibrated.joblib")
    print("Saved features ->", out_dir / "xgb_oos_improved_feature_names.csv")
    print("Done.")

if __name__ == "__main__":
    main()
