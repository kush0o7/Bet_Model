# src/backtest.py
# Ensure project root is on sys.path so "from src.*" imports work
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

START_BANKROLL = 1000.0
FLAT_STAKE = 10.0
EDGE_THRESHOLD = 0.05  # model - market

def main():
    df = pd.read_csv("data/predictions.csv", parse_dates=["date"])

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
        if edge >= EDGE_THRESHOLD:
            stake = FLAT_STAKE
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
                "stake": stake,
                "profit": profit,
                "bankroll": bankroll
            })

    bets_df = pd.DataFrame(bets)
    bets_df.to_csv("data/bets_log.csv", index=False)

    total_profit = bets_df["profit"].sum() if not bets_df.empty else 0.0
    total_staked = bets_df["stake"].sum() if not bets_df.empty else 1.0
    roi = total_profit / total_staked if total_staked > 0 else 0.0

    print("📊 BACKTEST RESULTS")
    print("Bets placed:", len(bets_df))
    print("Total profit:", round(total_profit,2))
    print("ROI:", round(roi,4))
    print("Max drawdown:", round(max_dd,2))

    if not bets_df.empty:
        plt.figure(figsize=(8,4))
        plt.plot(bets_df["bankroll"].values)
        plt.title("Bankroll over time")
        plt.xlabel("Bet index")
        plt.ylabel("Bankroll")
        plt.tight_layout()
        plt.savefig("data/bankroll.png")
        print("📈 Bankroll chart saved to data/bankroll.png")
    else:
        print("No bets were placed (try lowering EDGE_THRESHOLD)")

if __name__ == "__main__":
    main()
