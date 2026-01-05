# src/app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

st.set_page_config(layout="wide", page_title="Bet_Model Dashboard")

# ---- data / paths ----
DATA_DIR = "data"
PRED_XGB = os.path.join(DATA_DIR, "predictions_xgb_oos_improved.csv")
PRED_CLEAN = os.path.join(DATA_DIR, "predictions_oos_clean.csv")
PRED_SAMPLE = os.path.join(DATA_DIR, "sample_predictions.csv")  # keep a tiny sample in repo
BANKROLL_IMG = os.path.join(DATA_DIR, "sample_bankroll.png")

st.title("Bet_Model — Model vs Market dashboard")
st.markdown(
    """
Interactive dashboard for the XGBoost model (oos improved).

- Compares market probability vs model probability
- Highlights candidate bets by edge threshold
- Shows historical bankroll plot from Kelly backtest
"""
)

# -------------------------
# Helpers
# -------------------------
@st.cache_data
def load_predictions(path: str) -> Optional[pd.DataFrame]:
    """Load CSV if present and normalize basic columns."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # ensure we have a date column parsed (if available)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

def to_float_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Safe conversion to float series, creating missing column as NaNs if needed."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")

def compute_ev(odds: float, p_model: float) -> Optional[float]:
    """Expected value (profit) for 1 unit stake given decimal odds and model probability."""
    if pd.isna(odds) or pd.isna(p_model):
        return np.nan
    b = odds - 1.0
    return b * p_model - (1.0 - p_model)

def compute_kelly(odds: float, p_model: float) -> float:
    """Binary Kelly fraction: (b*p - q) / b, clipped at 0."""
    if pd.isna(odds) or pd.isna(p_model):
        return 0.0
    b = odds - 1.0
    q = 1.0 - p_model
    if b <= 0:
        return 0.0
    f = (b * p_model - q) / b
    return max(0.0, float(f))

# -------------------------
# Load data (prefer XGB predictions, fallback to clean predictions, fallback to sample)
# -------------------------
pred_xgb = load_predictions(PRED_XGB)
pred_clean = load_predictions(PRED_CLEAN)
pred_sample = load_predictions(PRED_SAMPLE)

pred = pred_xgb if pred_xgb is not None else (pred_clean if pred_clean is not None else pred_sample)

if pred is None:
    st.error("No prediction data found. Add `data/sample_predictions.csv` or real predictions and re-run.")
    st.stop()

# -------------------------
# Normalize numeric columns
# -------------------------
pred = pred.copy()
# create safe numeric columns (keep originals intact)
pred["odds_home_f"] = to_float_series(pred, "odds_home")
pred["p_home_f"] = to_float_series(pred, "p_home")         # market implied probability
pred["model_p_home_f"] = to_float_series(pred, "model_p_home")  # model probability

# compute edge
pred["edge_home"] = pred["model_p_home_f"] - pred["p_home_f"]

# compute EV and Kelly fractions
pred["ev_home"] = pred.apply(lambda r: compute_ev(r["odds_home_f"], r["model_p_home_f"]), axis=1)
pred["kelly_frac"] = pred.apply(lambda r: compute_kelly(r["odds_home_f"], r["model_p_home_f"]), axis=1)

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Controls")
edge_threshold = st.sidebar.slider("Edge threshold (model_p_home - p_home)", -0.05, 0.20, 0.01, 0.001)
min_prob = st.sidebar.slider("Min market p_home", 0.0, 1.0, 0.0, 0.01)

# date bounds: fallback safe defaults if date missing
date_min = pred["date"].min() if "date" in pred.columns and not pred["date"].isna().all() else pd.Timestamp("2000-01-01")
date_max = pred["date"].max() if "date" in pred.columns and not pred["date"].isna().all() else pd.Timestamp("2100-01-01")

date_from = st.sidebar.date_input("From date", value=date_min.date())
date_to = st.sidebar.date_input("To date", value=date_max.date())

sidebar_bankroll = st.sidebar.number_input("Bankroll ($)", min_value=10.0, value=1000.0, step=10.0)
kelly_cap = st.sidebar.slider("Kelly cap (max fraction of bankroll)", 0.0, 1.0, 0.2, 0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("Project: Bet_Model — demo dashboard")

# -------------------------
# Filter data
# -------------------------
df = pred.copy()
# Ensure date exists for filtering
if "date" not in df.columns:
    df["date"] = pd.NaT

mask = (
    (df["date"].dt.date >= date_from)
    & (df["date"].dt.date <= date_to)
    & (df["p_home_f"].fillna(0.0) >= min_prob)
)
df = df.loc[mask].sort_values(["date"]).reset_index(drop=True)

# Prepare additional columns for display
df["kelly_frac_capped"] = df["kelly_frac"].clip(upper=kelly_cap)
df["stake_usd"] = df["kelly_frac_capped"] * float(sidebar_bankroll)

# -------------------------
# Summary section
# -------------------------
st.header("Summary")
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    st.metric("Matches (rows)", int(len(pred)))
with c2:
    pos_matches = int((pred["edge_home"].fillna(0.0) >= 0.01).sum())
    st.metric("Matches edge ≥ 0.01", pos_matches)
with c3:
    st.write("Quick distributions")
    # hist of model - market
    diff = (pred["model_p_home_f"] - pred["p_home_f"]).dropna()
    fig, ax = plt.subplots(figsize=(6, 2))
    if len(diff) > 0:
        sns.histplot(diff, bins=40, ax=ax)
        ax.set_xlabel("model - market (home)")
    else:
        ax.text(0.5, 0.5, "No data for distribution", ha="center")
    plt.tight_layout()
    st.pyplot(fig)

# -------------------------
# Candidate bets
# -------------------------
st.header("Candidate bets")
c_left, c_right = st.columns([2, 1])
with c_left:
    bets = df.loc[df["edge_home"].fillna(-999) >= edge_threshold].copy()

    # choose display columns that exist
    display_cols = []
    for col in ["date", "home", "away", "p_home_f", "model_p_home_f", "edge_home", "odds_home_f", "ev_home", "kelly_frac_capped", "stake_usd"]:
        if col in bets.columns:
            display_cols.append(col)

    # user-friendly renames
    rename_map = {
        "p_home_f": "market_p_home",
        "model_p_home_f": "model_p_home",
        "odds_home_f": "odds_home",
        "kelly_frac_capped": "kelly_frac",
        "stake_usd": "stake_usd"
    }

    if bets.empty:
        st.info("No matches exceed the selected threshold.")
    else:
        if "date" in bets.columns:
            bets_display = bets[display_cols].copy()
            bets_display["date"] = bets_display["date"].dt.date
        else:
            bets_display = bets[display_cols].copy()
        bets_display = bets_display.rename(columns=rename_map)

        # pretty formatting
        st.dataframe(bets_display.reset_index(drop=True).style.format({
            "market_p_home": "{:.3f}",
            "model_p_home": "{:.3f}",
            "edge_home": "{:.3f}",
            "odds_home": "{:.3f}",
            "ev_home": "{:.3f}",
            "kelly_frac": "{:.3f}",
            "stake_usd": "${:,.2f}"
        }), height=420)

with c_right:
    st.metric("Candidates", len(bets))
    if len(bets) > 0:
        st.write("Top edges")
        st.write(bets.nlargest(5, "edge_home")[["home", "away", "edge_home"]].assign(edge_home=lambda d: d["edge_home"].round(3)))

# Top recommended bets (by EV)
st.header("Top recommended bets (by EV)")
top_bets = bets.sort_values("ev_home", ascending=False).head(3)
if top_bets.empty:
    st.info("No recommended bets under current filters.")
else:
    for _, r in top_bets.iterrows():
        date_str = r["date"].date() if pd.notna(r["date"]) else "unknown date"
        ev = r.get("ev_home", np.nan)
        kf = r.get("kelly_frac_capped", 0.0)
        stake = r.get("stake_usd", 0.0)
        st.markdown(f"**{r.get('home','?')} vs {r.get('away','?')} ({date_str})** — EV ${ev:.2f}, Kelly {kf:.3f}, Stake ${stake:.2f}")

# -------------------------
# Match explorer
# -------------------------
st.header("Match explorer (filtered matches)")
if df.empty:
    st.info("No matches available with current filters.")
else:
    indices = list(range(len(df)))
    def fmt(i):
        row = df.iloc[i]
        try:
            date_part = row["date"].date() if pd.notna(row["date"]) else "unknown"
        except Exception:
            date_part = "unknown"
        home = row.get("home", "")
        away = row.get("away", "")
        return f"{i} — {home} vs {away} ({date_part})"
    row = st.selectbox("Pick match (index)", indices, format_func=fmt)
    r = df.iloc[row]
    st.subheader(f"{r.get('home','?')} vs {r.get('away','?')} ({r.get('date','')})")
    st.write({
        "Market p_home": r.get("p_home_f"),
        "Model p_home": r.get("model_p_home_f"),
        "Edge (model - market)": r.get("edge_home"),
        "Odds_home": r.get("odds_home_f"),
        "EV (per unit stake)": r.get("ev_home"),
        "Kelly fraction (capped)": r.get("kelly_frac_capped"),
        "Suggested stake ($)": r.get("stake_usd")
    })

# -------------------------
# Bankroll / Backtest plot
# -------------------------
st.header("Backtest / Bankroll")
if os.path.exists(BANKROLL_IMG):
    st.image(BANKROLL_IMG, caption="Bankroll over time (Kelly backtest)", use_column_width=True)
else:
    st.info("bankroll_kelly.png not found in data/. Run backtests and save plot to data/")

# -------------------------
# Download CSV of candidates
# -------------------------
st.header("Download")
if not bets.empty:
    csv = bets_display.to_csv(index=False)
    st.download_button("Download candidate bets CSV", csv, file_name="candidate_bets.csv", mime="text/csv")
else:
    st.write("No candidate bets to download under current filters.")

# footer note
st.markdown("---")
st.markdown("**Notes:** This dashboard shows model-derived suggestions and examples for demonstration only. Do **not** treat these as financial advice.")
