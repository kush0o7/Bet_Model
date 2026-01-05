# src/clean_and_predict_oos.py
import pandas as pd
from pathlib import Path

IN = Path("data/predictions_oos.csv")   # or predictions_xgb.csv if you used that
OUT = Path("data/predictions_oos_clean.csv")

df = pd.read_csv(IN, parse_dates=["date"])
print("Raw rows:", len(df))

# keep only rows with pre-match odds
df = df.dropna(subset=["odds_home", "p_home"], how="any")
print("After dropping rows with missing odds/p_home:", len(df))

# normalize team name whitespace
df["home"] = df["home"].astype(str).str.strip()
df["away"] = df["away"].astype(str).str.strip()

# sort so that for duplicate fixtures we prefer rows with largest model_p_home (or last one)
df = df.sort_values(["date","home","away","model_p_home"], ascending=[True,True,True,False])

# drop exact duplicates on date/home/away keeping the first (best scored) row
df_clean = df.drop_duplicates(subset=["date","home","away"], keep="first").reset_index(drop=True)
print("After dedupe:", len(df_clean))

df_clean.to_csv(OUT, index=False)
print("Saved clean predictions to", OUT)
