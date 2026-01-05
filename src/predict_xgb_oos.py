# src/predict_xgb_oos.py
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import joblib

from src.load_data import load_matches
from src.elo import add_elo
from src.odds import odds_to_probs
from src.features import add_last_n_form

MODEL = Path("models/xgb_oos_with_form.joblib")
OUT = Path("data/predictions_oos.csv")

def main():
    df = load_matches()
    df = add_elo(df)
    # select only OOS season matches
    df = df[df["date"] >= "2025-07-01"].copy()
    # market probs
    probs = df.apply(lambda r: odds_to_probs(r.get("odds_home"), r.get("odds_draw"), r.get("odds_away")), axis=1, result_type="expand")
    probs.columns = ["p_home","p_draw","p_away"]
    df = pd.concat([df.reset_index(drop=True), probs.reset_index(drop=True)], axis=1)
    df = df.dropna(subset=["p_home","p_draw","p_away","elo_diff","date"]).reset_index(drop=True)
    # compute features
    df = add_last_n_form(df, n=5)
    # load model and feature names
    model, FEATURES = joblib.load(MODEL)
    X = df[FEATURES].fillna(0).values
    preds = model.predict_proba(X)
    df["model_p_home"] = preds[:,0]
    df["model_p_draw"] = preds[:,1]
    df["model_p_away"] = preds[:,2]
    df.to_csv(OUT, index=False)
    print("✅ OOS predictions saved to", OUT)
    print(df[["date","home","away","p_home","model_p_home"]].head(10))

if __name__ == "__main__":
    main()
