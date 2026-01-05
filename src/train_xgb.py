# src/train_xgb.py
# Train XGBoost + calibration and save the calibrated model
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline

from src.load_data import load_matches
from src.elo import add_elo
from src.odds import odds_to_probs

OUT = Path("models")
OUT.mkdir(exist_ok=True)

def prepare_df():
    df = load_matches()
    df = add_elo(df)
    probs = df.apply(lambda r: odds_to_probs(r.get("odds_home"), r.get("odds_draw"), r.get("odds_away")),
                     axis=1, result_type="expand")
    probs.columns = ["p_home", "p_draw", "p_away"]
    df = pd.concat([df.reset_index(drop=True), probs.reset_index(drop=True)], axis=1)
    df = df.dropna(subset=["p_home","p_draw","p_away","elo_diff","result"]).reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = prepare_df()
    print("Rows for training:", len(df))

    X = df[["p_home","p_draw","p_away","elo_diff"]].values
    y = df["result"].map({"H":0,"D":1,"A":2}).astype(int).values

    # Use TimeSeriesSplit to respect chronological ordering
    tss = TimeSeriesSplit(n_splits=4)

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        use_label_encoder=False,
        objective="multi:softprob",
        eval_metric="mlogloss",
        verbosity=0,
        n_jobs=-1,
        random_state=42
    )

    # Calibrate using isotonic (more flexible) with cross-validation
    calib = CalibratedClassifierCV(estimator=xgb, method="isotonic", cv=tss)

    print("Training XGBoost + isotonic calibration (may take a short while)...")
    calib.fit(X, y)

    joblib.dump(calib, OUT / "xgb_calibrated.joblib")
    print("Saved calibrated model -> models/xgb_calibrated.joblib")
