# src/train_xgb_oos.py
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

from src.load_data import load_matches
from src.elo import add_elo
from src.odds import odds_to_probs
from src.features import add_last_n_form

OUT = Path("models")
OUT.mkdir(exist_ok=True)

def prepare_df():
    df = load_matches()
    df = add_elo(df)
    # compute market probs
    probs = df.apply(lambda r: odds_to_probs(r.get("odds_home"), r.get("odds_draw"), r.get("odds_away")), axis=1, result_type="expand")
    probs.columns = ["p_home","p_draw","p_away"]
    df = pd.concat([df.reset_index(drop=True), probs.reset_index(drop=True)], axis=1)
    # keep rows with essentials
    df = df.dropna(subset=["p_home","p_draw","p_away","elo_diff","result","date"]).reset_index(drop=True)
    # add last-5 form features
    df = add_last_n_form(df, n=5)
    return df

if __name__ == "__main__":
    df = prepare_df()
    # split seasons: train on dates before 2025-07-01, test after
    train_df = df[df["date"] < "2025-07-01"].copy()
    test_df  = df[df["date"] >= "2025-07-01"].copy()

    print("Train matches:", len(train_df))
    print("Test matches :", len(test_df))

    # feature list: market probs, elo_diff, plus form features
    FEATURES = ["p_home","p_draw","p_away","elo_diff","form_pts_diff","gf_diff"]

    X_train = train_df[FEATURES].fillna(0).values
    y_train = train_df["result"].map({"H":0,"D":1,"A":2}).astype(int).values

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
        use_label_encoder=False,
        verbosity=0
    )

    model = CalibratedClassifierCV(estimator=xgb, method="sigmoid", cv=3)

    print("Training OOS XGBoost model with form features...")
    model.fit(X_train, y_train)

    joblib.dump((model, FEATURES), OUT / "xgb_oos_with_form.joblib")
    print("✅ Saved OOS model → models/xgb_oos_with_form.joblib")
