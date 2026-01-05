# src/backtest_kelly.py
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PRED_CSV = Path("data/predictions_xgb.csv")
OUT_BETS = Path("data/bets_log_kelly.csv")

# Parameters (tweak if needed)
EDGE_THRESHOLD = 0.02       # require at least 2% edge
KELLY_FRACTION = 0.2        # use 20% of full Kelly
MAX_FRACTION = 0.02         # cap stake at 2% of bankroll
START_BANKROLL = 1000.0
MIN_STAKE = 1.0             # minimum stake in currency units

def kelly_fraction(p, o):
    """
    Full Kelly fraction for decimal odds o and win prob p:
    f* = (o*p - 1) / (o - 1)
    returns 0 if negative or invalid
    """
    if p <= 0 or p >= 1:
        return 0.0
    b = o - 1.0
    if b <= 0:
        return 0.0
    f = (o * p - 1.0) / b
    return max(0.0, f)

def main():
    df = pd.read_csv(PRED_CSV, parse_dates=["date"])
    bankroll = START_BANKROLL
    peak = bankroll
    max_dd = 0.0
    bets = []

    for _, r in df.iterrows():
        market_p = r.get("p_home")
        model_p = r.get("model_p_home")
        odds = r.get("odds_home")

        if pd.isna(market_p) or pd.isna(model_p) or pd.isna(odds):
            continue

        edge = model_p - market_p
        if edge < EDGE_THRESHOLD:
            continue

        # compute full Kelly (for this binary bet)
        f_full = kelly_fraction(model_p, odds)
        f_use = f_full * KELLY_FRACTION
        # cap fraction to MAX_FRACTION
        f_use = min(f_use, MAX_FRACTION)
        stake = max(MIN_STAKE, f_use * bankroll) if f_use > 0 else 0.0
        if stake <= 0 or stake > bankroll:
            continue

        # place the bet on Home
        if r["result"] == "H":
            profit = stake * (odds - 1.0)
        else:
            profit = -stake

        bankroll += profit
        peak = max(peak, bankroll)
        max_dd = max(max_dd, peak - bankroll)

        bets.append({
            "date": r["date"],
            "home": r["home"],
            "away": r["away"],
            "edge": edge,
            "odds": odds,
            "stake": round(stake,2),
            "profit": round(profit,2),
            "bankroll": round(bankroll,2)
        })

    bets_df = pd.DataFrame(bets)
    bets_df.to_csv(OUT_BETS, index=False)

    total_profit = bets_df["profit"].sum() if not bets_df.empty else 0.0
    total_staked = bets_df["stake"].sum() if not bets_df.empty else 0.0
    roi = total_profit / total_staked if total_staked > 0 else 0.0

    print("📊 KELLY BACKTEST RESULTS")
    print("Bets placed:", len(bets_df))
    print("Total profit:", round(total_profit,2))
    print("ROI:", round(roi,4))
    print("Max drawdown:", round(max_dd,2))
    print("Final bankroll:", round(bankroll,2))
    if not bets_df.empty:
        plt.figure(figsize=(8,4))
        plt.plot(bets_df["bankroll"].values)
        plt.title("Bankroll (Kelly)")
        plt.xlabel("Bet #")
        plt.ylabel("Bankroll")
        plt.tight_layout()
        plt.savefig("data/bankroll_kelly.png")
        print("Saved bankroll plot to data/bankroll_kelly.png")
    else:
        print("No bets placed (try lowering EDGE_THRESHOLD)")

if __name__ == "__main__":
    main()
