# src/backtest_kelly_oos.py
# Kelly backtest on OUT-OF-SAMPLE matches only

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path("data/predictions_oos.csv")

START_BANKROLL = 1000.0
EDGE_THRESHOLD = 0.02
KELLY_FRACTION = 0.2
MAX_FRACTION = 0.02

def kelly_fraction(p, o):
    b = o - 1
    if b <= 0 or p <= 0 or p >= 1:
        return 0
    return max(0, (o * p - 1) / b)

def main():
    df = pd.read_csv(DATA, parse_dates=["date"])

    bankroll = START_BANKROLL
    peak = bankroll
    max_dd = 0
    bets = []

    for _, r in df.iterrows():
        if pd.isna(r["p_home"]) or pd.isna(r["model_p_home"]):
            continue

        edge = r["model_p_home"] - r["p_home"]
        if edge < EDGE_THRESHOLD:
            continue

        f = kelly_fraction(r["model_p_home"], r["odds_home"])
        f = min(f * KELLY_FRACTION, MAX_FRACTION)

        stake = f * bankroll
        if stake <= 0:
            continue

        profit = stake * (r["odds_home"] - 1) if r["result"] == "H" else -stake
        bankroll += profit
        peak = max(peak, bankroll)
        max_dd = max(max_dd, peak - bankroll)

        bets.append(profit)

    total_profit = sum(bets)
    roi = total_profit / START_BANKROLL

    print("📊 OOS KELLY RESULTS")
    print("Bets placed:", len(bets))
    print("Total profit:", round(total_profit,2))
    print("ROI:", round(roi,4))
    print("Max drawdown:", round(max_dd,2))
    print("Final bankroll:", round(bankroll,2))

if __name__ == "__main__":
    main()
