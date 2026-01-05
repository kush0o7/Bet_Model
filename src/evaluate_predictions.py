# src/evaluate_predictions.py
"""
Evaluate predictions CSV for the improved model.
Usage:
  python src/evaluate_predictions.py --pred data/predictions_xgb_oos_improved.csv --threshold 0.05
"""

import argparse
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.calibration import calibration_curve

def load_predictions(path):
    df = pd.read_csv(path)
    print("Loaded:", path, "rows:", len(df))
    # try to canonicalize column names
    # expect columns: 'result' or 'FTR' or 'home_goals' 'away_goals'
    if "result" not in df.columns:
        # try FTR
        if "FTR" in df.columns:
            df = df.rename(columns={"FTR":"result"})
        elif "home_goals" in df.columns and "away_goals" in df.columns:
            df["result"] = df.apply(lambda r: "H" if r["home_goals"]>r["away_goals"] else ("A" if r["home_goals"]<r["away_goals"] else "D"), axis=1)
        else:
            print("Warning: 'result' not found in predictions. Some metrics need ground-truth result.")
    # ensure needed columns exist
    for c in ["p_home","model_p_home","odds_home"]:
        if c not in df.columns:
            raise SystemExit(f"Column {c} not found in predictions CSV.")
    # convert result to binary home_win
    if "result" in df.columns:
        df["home_win"] = df["result"].apply(lambda x: 1 if str(x).strip().upper() in ("H","HOME","1") else 0)
    else:
        df["home_win"] = np.nan
    return df

def scoring(df):
    metrics = {}
    if df["home_win"].notna().all():
        y = df["home_win"]
        p = df["model_p_home"].clip(1e-6, 1-1e-6)
        metrics["brier"] = brier_score_loss(y, p)
        try:
            metrics["logloss"] = log_loss(y, p)
        except Exception as e:
            metrics["logloss"] = None
        try:
            metrics["roc_auc"] = roc_auc_score(y, p)
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["brier"] = metrics["logloss"] = metrics["roc_auc"] = None
    return metrics

def calibration_plot(df, out="calibration.png"):
    y = df["home_win"]
    p = df["model_p_home"]
    if y.isna().any():
        print("Skipping calibration plot: missing results.")
        return None
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10)
    plt.figure(figsize=(6,6))
    plt.plot(mean_pred, frac_pos, marker="o", label="model")
    plt.plot([0,1],[0,1], linestyle="--", color="k", label="perfect")
    plt.xlabel("mean predicted prob")
    plt.ylabel("empirical prob")
    plt.title("Calibration (home win)")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out)
    print("Saved calibration plot ->", out)
    return out

def simulate_bets(df, threshold=0.05, stake_mode="fixed", fixed_stake=1.0, cap_kelly=0.5):
    """
    stake_mode: 'fixed' or 'kelly'
    For fixed: bet fixed_stake whenever model_p_home - p_home >= threshold.
    For kelly: compute kelly fraction = (p*(b) - (1-p)) / b where b = odds - 1. Cap between 0 and cap_kelly.
    Returns bankroll series and summary.
    """
    df = df.copy()
    df["edge"] = df["model_p_home"] - df["p_home"]
    picks = df[df["edge"]>=threshold].copy()
    picks = picks.reset_index(drop=True)
    bankroll = 1000.0
    history = []
    for _, r in picks.iterrows():
        odds = r["odds_home"]
        p = r["model_p_home"]
        if math.isnan(odds) or odds <= 1:
            continue
        if stake_mode=="fixed":
            bet = fixed_stake
        else:
            b = odds - 1.0
            k = (p*b - (1-p)) / b if b>0 else 0
            k = max(0.0, k)
            k = min(k, cap_kelly)
            bet = bankroll * k
        # simulate result: need actual outcome
        if "home_win" in r.index and not math.isnan(r["home_win"]):
            won = bool(r["home_win"])
            profit = bet*(odds-1) if won else -bet
        else:
            profit = 0.0
        bankroll += profit
        history.append({"date": r.get("date"), "home": r.get("home"), "away": r.get("away"),
                        "edge": r["edge"], "odds": odds, "bet": bet, "profit": profit, "bankroll": bankroll})
    hist_df = pd.DataFrame(history)
    summary = {
        "bets": len(hist_df),
        "total_profit": float(hist_df["profit"].sum()) if not hist_df.empty else 0.0,
        "final_bankroll": float(bankroll),
        "roi": (hist_df["profit"].sum()/ (fixed_stake*len(hist_df)) ) if (stake_mode=="fixed" and len(hist_df)>0) else None
    }
    return hist_df, summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--outdir", default="data")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_predictions(args.pred)
    metrics = scoring(df)
    print("Metrics:", metrics)
    calibration_plot(df, out=os.path.join(args.outdir, "calibration_home.png"))

    # simulate fixed stake
    hist_fixed, summary_fixed = simulate_bets(df, threshold=args.threshold, stake_mode="fixed", fixed_stake=1.0)
    print("Fixed stake summary:", summary_fixed)
    hist_fixed.to_csv(os.path.join(args.outdir, "sim_fixed.csv"), index=False)

    # simulate kelly
    hist_k, summary_k = simulate_bets(df, threshold=args.threshold, stake_mode="kelly", cap_kelly=0.2)
    print("Kelly summary (cap 0.2):", summary_k)
    hist_k.to_csv(os.path.join(args.outdir, "sim_kelly.csv"), index=False)

    print("Saved sim CSVs to", args.outdir)

if __name__ == "__main__":
    main()
