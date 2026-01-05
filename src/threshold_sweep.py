# src/threshold_sweep.py
import pandas as pd
import numpy as np
from pathlib import Path

PRED = Path("data/predictions.csv")

def simulate(df, threshold, flat_stake=10.0, start_bankroll=1000.0):
    bankroll = start_bankroll
    peak = bankroll
    max_dd = 0.0
    bets = []
    for _, r in df.iterrows():
        market_p = r["p_home"]
        model_p = r["model_p_home"]
        odds = r["odds_home"]
        if pd.isna(market_p) or pd.isna(model_p) or pd.isna(odds):
            continue
        edge = model_p - market_p
        if edge >= threshold:
            stake = flat_stake
            if r["result"] == "H":
                profit = stake * (odds - 1.0)
            else:
                profit = -stake
            bankroll += profit
            peak = max(peak, bankroll)
            max_dd = max(max_dd, peak - bankroll)
            bets.append(profit)
    total_profit = sum(bets)
    n = len(bets)
    roi = total_profit / (n * flat_stake) if n > 0 else 0.0
    win_rate = sum(1 for p in bets if p > 0) / n if n>0 else 0.0
    return {"threshold": threshold, "bets": n, "total_profit": total_profit, "roi": roi, "win_rate": win_rate, "max_drawdown": max_dd}

def main():
    df = pd.read_csv(PRED, parse_dates=["date"])
    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]
    rows = []
    for t in thresholds:
        stats = simulate(df, t)
        rows.append(stats)
        print(stats)
    out = pd.DataFrame(rows)
    out.to_csv("data/threshold_sweep_results.csv", index=False)
    print("\nSaved sweep results to data/threshold_sweep_results.csv")

if __name__ == "__main__":
    main()
