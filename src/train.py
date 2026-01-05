# src/train.py
# Ensure project root is on sys.path so "from src.*" imports work
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

from src.load_data import load_matches
from src.elo import add_elo
from src.odds import odds_to_probs


def prepare_df():
    df = load_matches()
    df = add_elo(df)

    probs = df.apply(
        lambda r: odds_to_probs(
            r.get("odds_home"),
            r.get("odds_draw"),
            r.get("odds_away")
        ),
        axis=1,
        result_type="expand"
    )
    probs.columns = ["p_home", "p_draw", "p_away"]

    df = pd.concat([df.reset_index(drop=True), probs.reset_index(drop=True)], axis=1)
    df = df.dropna(
        subset=["p_home", "p_draw", "p_away", "elo_diff", "result"]
    ).reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = prepare_df()
    print("Rows used for training:", len(df))

    X = df[["p_home", "p_draw", "p_away", "elo_diff"]].values
    y = df["result"].map({"H": 0, "D": 1, "A": 2}).astype(int).values

    model = LogisticRegression(
    solver="lbfgs",
    max_iter=2000
    )

    model.fit(X, y)

    out = Path("models")
    out.mkdir(exist_ok=True)
    joblib.dump(model, out / "epl_model.joblib")

    pd.DataFrame(
        {"feature": ["p_home", "p_draw", "p_away", "elo_diff"]}
    ).to_csv(out / "feature_names.csv", index=False)

    print("✅ Model saved to models/epl_model.joblib")
