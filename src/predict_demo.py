# src/predict_demo.py
# Ensure project root is on sys.path so "from src.*" imports work
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

MODEL_PATH = Path("models/epl_model.joblib")

def main():
    df = load_matches()
    df = add_elo(df)

    probs = df.apply(
        lambda r: odds_to_probs(r.get("odds_home"), r.get("odds_draw"), r.get("odds_away")),
        axis=1,
        result_type="expand"
    )
    probs.columns = ["p_home", "p_draw", "p_away"]
    df = pd.concat([df.reset_index(drop=True), probs.reset_index(drop=True)], axis=1)

    model = joblib.load(MODEL_PATH)
    X = df[["p_home", "p_draw", "p_away", "elo_diff"]].fillna(0).values
    preds = model.predict_proba(X)

    df["model_p_home"] = preds[:, 0]
    df["model_p_draw"] = preds[:, 1]
    df["model_p_away"] = preds[:, 2]

    out = Path("data/predictions.csv")
    df.to_csv(out, index=False)
    print("✅ Predictions saved to data/predictions.csv")
    print(df[["date","home","away","p_home","model_p_home"]].tail(10))

if __name__ == "__main__":
    main()
