# src/elo.py
import pandas as pd

def add_elo(df, k=20, home_adv=50):
    """
    Compute a simple pre-match Elo difference (home Elo - away Elo).
    Expects df to have 'home', 'away', 'date', and 'result' columns.
    Returns the same DataFrame with a new 'elo_diff' column.
    """
    df = df.sort_values("date").reset_index(drop=True)
    teams = pd.unique(df[["home","away"]].values.ravel('K'))
    elo = {t:1500 for t in teams}
    elo_diff = []
    for _, row in df.iterrows():
        h = row["home"]
        a = row["away"]
        elo_diff.append(elo.get(h,1500) - elo.get(a,1500))

        Rh = 10 ** ((elo.get(h,1500) + home_adv) / 400.0)
        Ra = 10 ** (elo.get(a,1500) / 400.0)
        Eh = Rh / (Rh + Ra)

        res = row.get("result")
        if res == "H":
            Sh = 1.0
        elif res == "D":
            Sh = 0.5
        else:
            Sh = 0.0

        elo[h] = elo.get(h,1500) + k * (Sh - Eh)
        elo[a] = elo.get(a,1500) + k * ((1 - Sh) - (1 - Eh))

    df["elo_diff"] = elo_diff
    return df
