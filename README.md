# Bet_Model — EPL Model vs Market Analysis

This project builds an end-to-end machine learning pipeline to analyze English Premier League (EPL) matches by comparing **model-predicted probabilities** with **bookmaker-implied probabilities**.  
The focus is on **probabilistic modeling, evaluation, and risk-aware decision analysis**, not guaranteed betting profit.

---

## Project Goals

- Train a probabilistic ML model to predict match outcomes
- Compare model probabilities against market odds
- Evaluate whether the model identifies systematic differences (“edge”)
- Simulate betting strategies under realistic risk constraints
- Present results in an interactive dashboard

---

## Data Ingestion & Processing

- Historical EPL match data is loaded from season CSVs
- Column formats are normalized across seasons
- Match dates are parsed and strictly ordered to avoid data leakage
- Bookmaker odds are converted into **market-implied probabilities**
- Raw data and generated artifacts are excluded from the repo for size/licensing reasons

---

## Feature Engineering

- **ELO ratings** are computed and updated match-by-match
- Home/away ELO difference is used as a key signal
- Market probabilities derived from odds are included as model inputs
- All features are constructed using **only pre-match information**

---

## Model Training

- An **XGBoost classifier** is trained to predict match outcome probabilities
- Time-aware (out-of-sample) train/test split is used
- No random shuffling to preserve real-world temporal structure
- Model outputs probabilities rather than hard classifications

---

## Probability Calibration

- Raw model probabilities are calibrated using **isotonic regression**
- Calibration quality is evaluated with:
  - Brier score
  - Log loss
  - Calibration plots
- Ensures predicted probabilities are meaningful for decision analysis

---

## Model Evaluation

- ROC AUC shows strong discrimination ability
- Brier score and log loss indicate reasonable probabilistic accuracy
- Evaluation is strictly out-of-sample to prevent overfitting

---

## Model vs Market Analysis

- **Edge** is defined as:
edge = model_probability − market_probability

- Edge distributions are analyzed across matches
- Most edges are small, reflecting market efficiency
- Large edges are rare and unstable, reinforcing realistic expectations

---

## Betting Strategy Simulation

Two strategies are evaluated for educational purposes:

### Fixed Stake
- Same stake on every qualifying bet
- Used to evaluate raw signal quality

### Kelly Criterion (Capped)
- Bet sizing based on estimated edge
- Capped to prevent over-exposure
- Demonstrates how improper risk management can dominate outcomes

**Key insight:**  
Risk management has a larger impact on results than model accuracy alone.

---

## Interactive Streamlit Dashboard

The Streamlit app allows users to:

- Compare model vs market probabilities
- Filter matches by edge threshold and date
- View expected value (EV) and capped Kelly stake suggestions
- Explore individual matches in detail
- Download candidate bets as CSV
- View historical bankroll/backtest plots

Run locally:
```bash
streamlit run src/app.py

