"""
predict.py
----------
Fetches upcoming AFL fixtures from the Squiggle API and predicts winners.

Run with:
    python predict.py
    python predict.py --round 5       # predict a specific round
    python predict.py --year 2025     # predict a specific year
"""

import argparse
import requests
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from datetime import datetime
from dotenv import load_dotenv

from config import MODEL_PATH, FEATURES_PATH
from data_loader import load_all
from email_service import send_predictions_email
from feature_engineering import build_features

load_dotenv()

# Must match KEEP_FEATURES in train.py
KEEP_FEATURES = [
    "home_rolling_win_rate", "away_rolling_win_rate",
    "home_rolling_score_for", "away_rolling_score_for",
    "home_rolling_score_against", "away_rolling_score_against",
    "home_rolling_margin", "away_rolling_margin",
    "h2h_home_win_rate", "h2h_total_games",
    "home_venue_win_rate", "away_venue_win_rate",
    "home_streak", "away_streak", "streak_diff",
    "home_days_rest", "away_days_rest", "rest_diff",
    "venue_encoded",
    "home_ladder_position", "away_ladder_position", "ladder_position_diff",
    "away_interstate", "home_interstate",
    "home_elo", "away_elo", "elo_diff", "elo_home_win_prob",
]

SQUIGGLE_API = "https://api.squiggle.com.au/"


# ── Fetch fixture from Squiggle ───────────────────────────────────────────────

def fetch_upcoming_games(year: int, round_num: int = 0) -> pd.DataFrame:
    params = f"?q=games;year={year}"
    if round_num is not None:
        params += f";round={round_num}"

    url = SQUIGGLE_API + params
    print(f"Fetching fixture from Squiggle API: {url}")

    response = requests.get(url, headers={"User-Agent": "afl-predictor/1.0"})
    response.raise_for_status()

    games = response.json().get("games", [])
    if not games:
        print("No games found.")
        return pd.DataFrame()

    df = pd.DataFrame(games)

    # Filter to incomplete games only
    upcoming = df[df["complete"] < 100].copy()

    # Filter out finals where teams aren't determined yet
    upcoming = upcoming[upcoming["hteam"].notna() & upcoming["ateam"].notna()]

    # If no round specified, only show the next 7 days
    if round_num is None:
        upcoming["date_dt"] = pd.to_datetime(upcoming["date"])
        today = pd.Timestamp.now()
        upcoming = upcoming[upcoming["date_dt"] <= today + pd.Timedelta(days=7)]

    if upcoming.empty:
        print("No upcoming games found.")
        return pd.DataFrame()

    upcoming = upcoming.rename(columns={"hteam": "hometeam", "ateam": "awayteam", "id": "gameid"})
    upcoming["date"] = pd.to_datetime(upcoming["date"]).dt.strftime("%Y-%m-%d")
    upcoming["gameid"] = upcoming["gameid"].astype(str)

    print(f"Found {len(upcoming)} upcoming games")
    return upcoming[["gameid", "date", "year", "round", "venue", "hometeam", "awayteam"]]


# ── Load saved model artifacts ────────────────────────────────────────────────

def load_model_artifacts():
    """Load the trained model, imputer, and feature column list."""
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    imputer = joblib.load("imputer.pkl")
    with open(FEATURES_PATH) as f:
        feature_cols = [line.strip() for line in f.readlines()]
    print(f"Loaded model with {len(feature_cols)} features")
    return model, imputer, feature_cols


# ── Build features for upcoming games ────────────────────────────────────────

def build_prediction_features(upcoming_df: pd.DataFrame, historical_games: pd.DataFrame, historical_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Append upcoming games to historical data so rolling features
    are computed correctly, then extract just the upcoming rows.
    """
    upcoming = upcoming_df.copy()

    # Use a clearly fake gameid format that won't clash with the DB format
    upcoming["gameid"] = ["PREDICT_" + str(i) for i in range(len(upcoming))]

    # Placeholder scores so the pipeline doesn't break
    upcoming["hometeamscore"] = 0
    upcoming["awayteamscore"] = 0

    # Combine and sort
    combined = pd.concat([historical_games, upcoming], ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)

    # Build features on full dataset
    full_features = build_features(combined, historical_stats)

    # Extract only upcoming game rows by our fake gameid prefix
    pred_features = full_features[full_features["gameid"].str.startswith("PREDICT_")].copy()

    # Restore the original display info from upcoming_df
    pred_features = pred_features.reset_index(drop=True)
    pred_features["hometeam"] = upcoming_df["hometeam"].values
    pred_features["awayteam"] = upcoming_df["awayteam"].values
    pred_features["date"] = upcoming_df["date"].values

    print(pred_features[["hometeam", "awayteam", "home_elo", "away_elo", "elo_diff"]].to_string())

    return pred_features

# ── Print results table ───────────────────────────────────────────────────────

def print_predictions(results: pd.DataFrame):
    """Print a nicely formatted predictions table."""
    print("\n" + "=" * 75)
    print("  AFL PREDICTIONS")
    print("=" * 75)
    print(f"  {'DATE':<12} {'HOME':<25} {'AWAY':<25} {'TIP':<25} {'CONF'}")
    print("-" * 75)

    for _, row in results.iterrows():
        winner = row["predicted_winner"]
        conf = row["confidence"]

        conf_str = f"{conf:.0%}"
        if conf >= 0.70:
            conf_str += " **"
        elif conf >= 0.65:
            conf_str += " *"

        print(f"  {row['date']:<12} {row['hometeam']:<25} {row['awayteam']:<25} {winner:<25} {conf_str}")

    print("=" * 75)
    print("  ** Strong tip (70%+ confidence)   * Good tip (65%+ confidence)")
    print("=" * 75)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Predict AFL game winners")
    parser.add_argument("--year",  type=int, default=datetime.now().year, help="Season year (default: current year)")
    parser.add_argument("--round", type=int, default=None, help="Round number (default: all upcoming)")
    args = parser.parse_args()

    # 1. Fetch upcoming fixture
    upcoming = fetch_upcoming_games(year=args.year, round_num=args.round)
    if upcoming.empty:
        return

    # 2. Load historical data for feature computation
    print("Loading historical data...")
    historical_games, historical_stats = load_all()

    # 3. Load trained model
    model, imputer, feature_cols = load_model_artifacts()

    # 4. Build features
    print("Building features for upcoming games...")
    pred_features = build_prediction_features(upcoming, historical_games, historical_stats)

    if pred_features.empty:
        print("Could not build features for upcoming games.")
        return

    # 5. Select and align features
    X_pred = pred_features.reindex(columns=KEEP_FEATURES)
    X_pred = pd.DataFrame(imputer.transform(X_pred), columns=KEEP_FEATURES)

    # 6. Predict
    proba = model.predict_proba(X_pred)[:, 1]

    # 7. Build results table
    results = pred_features[["gameid", "date", "hometeam", "awayteam"]].copy().reset_index(drop=True)
    results["home_win_probability"] = proba.round(3)
    results["away_win_probability"] = (1 - proba).round(3)
    results["predicted_winner"] = np.where(proba >= 0.5, results["hometeam"], results["awayteam"])
    results["confidence"] = results[["home_win_probability", "away_win_probability"]].max(axis=1) * 100
    results = results.sort_values("date").reset_index(drop=True)

    print("\nElo ratings going into 2026:")
    print(pred_features[["hometeam", "awayteam", "home_elo", "away_elo", "elo_diff"]].to_string())

    # 8. Send predictions email
    send_predictions_email(results)

if __name__ == "__main__":
    main()