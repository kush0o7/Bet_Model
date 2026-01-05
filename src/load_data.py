# safer load_matches() — only reads season files named PL-*.csv
import pandas as pd
from pandas.errors import EmptyDataError
from pathlib import Path
import os
DATA_DIR = Path("data")

def _choose_column(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None

def load_matches():
    # Only consider official season files that start with PL-
    csvs = sorted(DATA_DIR.glob("PL-*.csv"))
    if not csvs:
        raise FileNotFoundError("No PL-*.csv season files found in data/. Place season CSVs named like 'PL-2024-2025.csv' there.")

    dfs = []
    for f in csvs:
        try:
            if os.path.getsize(f) == 0:
                print(f"Skipping empty file: {f.name}")
                continue
            tmp = pd.read_csv(f)
        except EmptyDataError:
            print(f"Skipping unreadable/empty CSV: {f.name}")
            continue
        except Exception as e:
            print(f"Skipping {f.name} (read error: {e})")
            continue

        cols = tmp.columns.tolist()
        home_col = _choose_column(cols, ["HomeTeam", "Home", "home"])
        away_col = _choose_column(cols, ["AwayTeam", "Away", "away"])
        date_col = _choose_column(cols, ["Date", "date"])
        result_col = _choose_column(cols, ["FTR", "Result", "result", "Res"])

        if not (home_col and away_col and result_col and date_col):
            print(f"Skipping {f.name} — not a match file (missing home/away/date/result).")
            continue

        tmp = tmp.rename(columns={
            date_col: "date",
            home_col: "home",
            away_col: "away",
            result_col: "result",
        })

        # Map common odds columns if present
        odds_home = _choose_column(cols, ["odds_home", "B365H", "AvgH", "MaxH", "PSH", "WHH"])
        odds_draw = _choose_column(cols, ["odds_draw", "B365D", "AvgD", "MaxD", "PSD", "WHD"])
        odds_away = _choose_column(cols, ["odds_away", "B365A", "AvgA", "MaxA", "PSA", "WHA"])
        if odds_home:
            tmp = tmp.rename(columns={odds_home: "odds_home"})
        if odds_draw:
            tmp = tmp.rename(columns={odds_draw: "odds_draw"})
        if odds_away:
            tmp = tmp.rename(columns={odds_away: "odds_away"})

        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce", dayfirst=True)
        tmp = tmp.dropna(subset=["home","away","result"])
        dfs.append(tmp)

    if not dfs:
        raise FileNotFoundError("No valid PL-season CSVs found in data/. Check filenames and columns.")
    all_matches = pd.concat(dfs, ignore_index=True, sort=False)
    print("Loaded season files:", [p.name for p in csvs])
    print("Columns found:", all_matches.columns.tolist())
    return all_matches
