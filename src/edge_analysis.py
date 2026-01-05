# src/edge_analysis.py
# Inspect distribution of model - market edges for Home bets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PRED = Path("data/predictions.csv")

def main():
    df = pd.read_csv(PRED, parse_dates=["date"])
    # compute home edge
    df["edge_home"] = df["model_p_home"] - df["p_home"]
    # basic stats
    print("Matches:", len(df))
    print("Edge_home mean:", df["edge_home"].mean())
    print("Edge_home std:", df["edge_home"].std())
    print("Edge_home min/max:", df["edge_home"].min(), df["edge_home"].max())
    # percentiles
    for q in [0.99, 0.95, 0.9, 0.75, 0.5, 0.25]:
        print(f"{int(q*100)}th percentile:", df["edge_home"].quantile(q))
    # how many exceed common thresholds
    for t in [0.01, 0.02, 0.03, 0.04, 0.05]:
        print(f"edge >= {t:.2f} : {(df['edge_home'] >= t).sum()} matches")
    # show top edges
    print("\nTop 10 positive edges (home):")
    print(df.sort_values("edge_home", ascending=False)[["date","home","away","p_home","model_p_home","edge_home"]].head(10).to_string(index=False))

    # histogram
    plt.figure(figsize=(6,3))
    plt.hist(df["edge_home"].dropna(), bins=80)
    plt.title("Distribution of model - market (home edge)")
    plt.xlabel("edge")
    plt.ylabel("count")
    plt.tight_layout()
    out = Path("data/edge_hist.png")
    plt.savefig(out)
    print(f"\nSaved histogram to {out}")

if __name__ == "__main__":
    main()
