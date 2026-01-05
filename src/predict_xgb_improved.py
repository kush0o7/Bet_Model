# src/predict_xgb_improved.py
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from src.load_data import load_matches
from src.features import add_last_n_form
from src.odds import odds_to_probs

MODEL = Path("models/xgb_oos_improved_calibrated.joblib")
FEATURES = Path("models/xgb_oos_improved_feature_names.joblib")

def kelly_fraction(p, b):
    # b = decimal odds - 1
    return max(0.0, (p*(b) - 1) / b) if b > 0 else 0.0

def main(out_csv="data/predictions_improved.csv", bankroll=1000.0):
    df = load_matches()
    df = add_last_n_form(df, n=5)

    # make sure p_home exists
    if not {"p_home","p_draw","p_away"}.issubset(df.columns):
        probs = df.apply(lambda r: odds_to_probs(r.get("odds_home"), r.get("odds_draw"), r.get("odds_away")), axis=1, result_type="expand")
        probs.columns = ["p_home","p_draw","p_away"]
        df = pd.concat([df.reset_index(drop=True), probs.reset_index(drop=True)], axis=1)

    model = joblib.load(MODEL)
    features = joblib.load(FEATURES)
    X = df[features].fillna(0.0).values

    proba = model.predict_proba(X)  # shape (n, 3)
    # model prob for home
    df["model_p_home"] = proba[:,0]
    df["model_p_draw"] = proba[:,1]
    df["model_p_away"] = proba[:,2]

    # compute market prob (p_home already present)
    # compute edge (model_p_home - p_home)
    df["edge_home"] = df["model_p_home"] - df["p_home"]

    # convert odds to decimal if needed: odds_home is assumed decimal
    df["dec_odds_home"] = df["odds_home"].astype(float)
    df["kelly_frac_home"] = df.apply(lambda r: kelly_fraction(r["model_p_home"], r["dec_odds_home"] - 1), axis=1)
    df["kelly_stake_home"] = (df["kelly_frac_home"] * bankroll).round(2)

    df.to_csv(out_csv, index=False)
    print("Saved", out_csv)
    print(df[["date","home","away","p_home","model_p_home","edge_home","dec_odds_home","kelly_stake_home"]].tail(10))

if __name__ == "__main__":
    main()
