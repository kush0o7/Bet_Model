# src/predict_xgb_oos_improved.py
import sys
from pathlib import Path

# --- make project root importable ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# -----------------------------------

import pandas as pd
import joblib
import numpy as np

from src.load_data import load_matches
from src.odds import odds_to_probs

# Try to import add_elo if available; we'll use it to compute elo columns when needed
try:
    from src.elo import add_elo
except Exception:
    add_elo = None

MODEL_PATH = Path("models/xgb_oos_improved_calibrated.joblib")
FEATURES_PATH = Path("models/xgb_oos_improved_feature_names.csv")
OUT = Path("data/predictions_xgb_oos_improved.csv")


def ensure_elo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has elo-related features required by many models:
    - If elo_diff missing but elo_home_pre & elo_away_pre present -> compute elo_diff
    - If none present and add_elo available -> call add_elo(df)
    """
    if "elo_diff" in df.columns:
        return df

    # if home/away pre elo available, derive elo_diff
    if {"elo_home_pre", "elo_away_pre"}.issubset(df.columns):
        df["elo_diff"] = df["elo_home_pre"].astype(float) - df["elo_away_pre"].astype(float)
        return df

    # try add_elo function if available
    if add_elo is not None:
        print("Computing Elo features with src.elo.add_elo(...)")
        try:
            df2 = add_elo(df.copy())
            if "elo_diff" in df2.columns:
                return df2
            # if add_elo produced elo_home_pre/elo_away_pre, compute elo_diff
            if {"elo_home_pre", "elo_away_pre"}.issubset(df2.columns):
                df2["elo_diff"] = df2["elo_home_pre"] - df2["elo_away_pre"]
                return df2
            # otherwise fallthrough to error
            df = df2
        except Exception as e:
            print("add_elo failed:", e)

    # nothing we can auto-compute
    raise KeyError(
        "Missing Elo features. Need one of: 'elo_diff' or both 'elo_home_pre' and 'elo_away_pre'."
    )


def main():
    print("Loading matches...")
    df = load_matches()
    print(f"Loaded season files and {len(df)} rows (raw).")

    # compute market probabilities if missing
    if not {"p_home", "p_draw", "p_away"}.issubset(df.columns):
        print("Computing market probabilities from odds...")
        probs = df.apply(
            lambda r: odds_to_probs(r.get("odds_home"), r.get("odds_draw"), r.get("odds_away")),
            axis=1,
            result_type="expand",
        )
        probs.columns = ["p_home", "p_draw", "p_away"]
        df = pd.concat([df.reset_index(drop=True), probs.reset_index(drop=True)], axis=1)

    # load model + feature list
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature list not found at {FEATURES_PATH}")

    print("Loading model and feature list...")
    model = joblib.load(MODEL_PATH)
    features = pd.read_csv(FEATURES_PATH)["feature"].tolist()
    print("Model expects features:", features)

    # ensure Elo / features that can be derived are present
    try:
        df = ensure_elo(df)
    except KeyError as e:
        # if model doesn't need elo_diff, continue — but re-raise if model needs it
        if "elo_diff" in features:
            raise
        else:
            print("Warning: elo features missing but not required by model:", e)

    # Ensure all features exist; if some are missing, show a clear message and exit
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise KeyError(
            f"The model requires these missing columns: {missing}. "
            "Either compute them (features script) or update the feature list."
        )

    # drop rows missing required features
    df = df.dropna(subset=features).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No rows left after dropping rows with missing feature values.")

    X = df[features].values
    print("Predicting probabilities...")
    probs = model.predict_proba(X)

    # model classes assumed to be in order [home, draw, away]
    df["model_p_home"] = probs[:, 0]
    df["model_p_draw"] = probs[:, 1]
    df["model_p_away"] = probs[:, 2]

    # edges (model - market)
    df["edge_home"] = df["model_p_home"] - df["p_home"]
    df["edge_draw"] = df["model_p_draw"] - df["p_draw"]
    df["edge_away"] = df["model_p_away"] - df["p_away"]

    out_dir = OUT.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"✅ Predictions saved to {OUT}")
    print(df[["date", "home", "away", "p_home", "model_p_home", "edge_home"]].tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
