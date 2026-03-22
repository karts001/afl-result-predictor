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
import json
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import joblib
from datetime import datetime
from dotenv import load_dotenv

from utils.logger import logger
from utils.config import KEEP_FEATURES, MODEL_PATH, FEATURES_PATH
from services.data_loader import load_all
from services.email_service import send_predictions_email
from feature_engineering import build_features
from services.squiggle_api import fetch_upcoming_games

load_dotenv()

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

    logger.info("upcoming_df gameids:")
    logger.info(upcoming_df[["gameid", "hometeam", "awayteam"]].to_string())

    logger.info("pred_features gameids:")
    logger.info(pred_features[["gameid", "hometeam", "awayteam"]].to_string())

    # Restore the original display info from upcoming_df
    date_lookup = upcoming_df[["hometeam", "awayteam", "date"]].copy()
    date_lookup["gameid"] = ["PREDICT_" + str(i) for i in range(len(upcoming_df))]

    pred_features = pred_features.drop(columns=["date"], errors="ignore")
    pred_features = pred_features.merge(date_lookup[["gameid", "date"]], on="gameid", how="left")

    return pred_features

# ── Print results table ───────────────────────────────────────────────────────

def print_predictions(results: pd.DataFrame):
    """Print a nicely formatted predictions table."""
    logger.info("\n" + "=" * 75)
    logger.info("  AFL PREDICTIONS")
    logger.info("=" * 75)
    logger.info(f"  {'DATE':<12} {'HOME':<25} {'AWAY':<25} {'TIP':<25} {'CONF'}")
    logger.info("-" * 75)

    for _, row in results.iterrows():
        winner = row["predicted_winner"]
        conf = row["confidence"]

        conf_str = f"{conf:.0%}"
        if conf >= 0.70:
            conf_str += " **"
        elif conf >= 0.65:
            conf_str += " *"

        logger.info(f"  {row['date']:<12} {row['hometeam']:<25} {row['awayteam']:<25} {winner:<25} {conf_str}")

    logger.info("=" * 75)
    logger.info("  ** Strong tip (70%+ confidence)   * Good tip (65%+ confidence)")
    logger.info("=" * 75)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Predict AFL game winners")
    parser.add_argument("--year",  type=int, default=datetime.now().year, help="Season year (default: current year)")
    # parser.add_argument("--round", type=int, default=None, help="Round number (default: all upcoming)")
    args = parser.parse_args()

    # 1. Fetch upcoming fixture
    upcoming = fetch_upcoming_games(year=args.year)
    if upcoming.empty:
        return

    # 2. Load historical data for feature computation
    logger.info(("Loading historical data..."))
    historical_games, historical_stats = load_all()

    games_2026 = historical_games[historical_games["year"] == 2026].sort_values("date")
    logger.info(games_2026[["date", "round", "hometeam", "awayteam", "hometeamscore", "awayteamscore"]].to_string())
    logger.info(f"Total 2026 games: {len(games_2026)}")

    # 3. Load trained model
    model, imputer, feature_cols = load_model_artifacts()

    # 4. Build features
    logger.info("Building features for upcoming games...")
    pred_features = build_prediction_features(upcoming, historical_games, historical_stats)

    # Debug: Hawthorns vs Sydney ladder positions
    syd_haw = pred_features[
        (pred_features["hometeam"] == "Hawthorn") & 
        (pred_features["awayteam"] == "Sydney")
    ][["hometeam", "awayteam", "home_elo", "away_elo", "rolling_margin_diff", "rolling_win_rate_diff", "streak_diff"]].T

    logger.info(syd_haw)

    if pred_features.empty:
        logger.warning("Could not build features for upcoming games.")
        return

    # 5. Select and align features
    X_pred = pred_features.reindex(columns=KEEP_FEATURES)
    X_pred = pd.DataFrame(imputer.transform(X_pred), columns=KEEP_FEATURES)

    explainer = shap.TreeExplainer(model)

    idx = 2  # adjust based on output above
    row = X_pred.iloc[[idx]]
    shap_vals = explainer.shap_values(X_pred)

    shap_df = pd.DataFrame({
        'feature': KEEP_FEATURES,
        'value': row.iloc[0].values,
        'shap': shap_vals[idx]
    }).sort_values('shap', ascending=False)

    logger.info(f"\nGWS vs St Kilda")
    logger.info(shap_df.to_string())
    logger.info(f"\nBase value: {explainer.expected_value:.3f}")
    logger.info(f"Predicted probability: {model.predict_proba(row)[0][1]:.3f}")

    # 6. Predict
    proba = model.predict_proba(X_pred)[:, 1]

    # 7. Build results table
    results = pred_features[["gameid", "date", "hometeam", "awayteam"]].copy().reset_index(drop=True)
    results["home_win_probability"] = proba.round(3)
    results["away_win_probability"] = (1 - proba).round(3)
    results["predicted_winner"] = np.where(proba >= 0.5, results["hometeam"], results["awayteam"])
    results["confidence"] = results[["home_win_probability", "away_win_probability"]].max(axis=1) * 100
    results = results.sort_values("date").reset_index(drop=True)

    logger.info("\nElo ratings going into 2026:")
    logger.info(pred_features[["hometeam", "awayteam", "home_elo", "away_elo", "elo_diff"]].to_string())

    # 8. Send predictions email
    send_predictions_email(results)

if __name__ == "__main__":
    main()