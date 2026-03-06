"""
train.py
--------
Trains an XGBoost classifier to predict AFL game winners (home team wins = 1).

Run this with:
    python train.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
)
from sklearn.impute import SimpleImputer
import joblib

from config import TEST_SIZE, RECENCY_WEIGHTING, XGBOOST_PARAMS, MODEL_PATH, FEATURES_PATH, MIN_YEAR
from data_loader import load_all
from feature_engineering import build_features
from logger import logger


# ── Columns that are NOT features ─────────────────────────────────────────────
NON_FEATURE_COLS = ["gameid", "year", "date", "round", "hometeam", "awayteam", "home_win", "venue"]
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

def prepare_data(feature_df: pd.DataFrame):
    """
    Split the feature DataFrame into X (features) and y (label),
    then into train and test sets.
    """
    # Filter to recent years only
    before = len(feature_df)
    feature_df = feature_df[feature_df["year"] >= MIN_YEAR].copy()
    logger.info(f"   Filtered to {MIN_YEAR}+: {before} -> {len(feature_df)} games")

    # Sort chronologically — test set will be the most recent games
    feature_df = feature_df.sort_values("date").reset_index(drop=True)

    feature_cols = [c for c in feature_df.columns if c not in NON_FEATURE_COLS]

    X = feature_df[[c for c in KEEP_FEATURES if c in feature_df.columns]]
    feature_cols = X.columns.tolist()

    y = feature_df["home_win"]

    # Compute sample weights based on recency (oldest = 1, most recent = 5)
    if RECENCY_WEIGHTING:
        min_yr = feature_df["year"].min()
        max_yr = feature_df["year"].max()
        sample_weights = feature_df["year"].map(
            lambda yr: 1 + (yr - min_yr) / (max_yr - min_yr) * 4
        ).values
    else:
        sample_weights = None

    # Chronological train/test split
    split_idx = int(len(feature_df) * (1 - TEST_SIZE))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    weights_train = sample_weights[:split_idx] if sample_weights is not None else None

    logger.info(f"   Train: {len(X_train)} games | Test: {len(X_test)} games")
    logger.info(f"   Home win rate - Train: {y_train.mean():.1%} | Test: {y_test.mean():.1%}")

    # Impute missing values with column median
    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols)
    X_test  = pd.DataFrame(imputer.transform(X_test),      columns=feature_cols)

    return X_train, X_test, y_train, y_test, imputer, feature_cols, weights_train


def train_model(X_train, y_train, sample_weights=None):
    """Train an XGBoost classifier."""
    logger.info("\n Training XGBoost model...")
    model = xgb.XGBClassifier(**XGBOOST_PARAMS)
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_train, y_train)],
        verbose=False,
    )
    logger.info("Training complete")
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test, feature_cols):
    """logger.info evaluation metrics and plot useful charts."""

    # ── Accuracy ──────────────────────────────────────────────────────────────
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc  = accuracy_score(y_test,  model.predict(X_test))

    logger.info(f"\n Results:")
    logger.info(f"   Train accuracy : {train_acc:.1%}")
    logger.info(f"   Test accuracy  : {test_acc:.1%}")

    # ── ROC-AUC ───────────────────────────────────────────────────────────────
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    logger.info(f"   ROC-AUC        : {auc:.3f}  (0.5 = random, 1.0 = perfect)")

    # ── Classification report ─────────────────────────────────────────────────
    logger.info("\n Classification Report:")
    logger.debug(classification_report(y_test, model.predict(X_test),
                                 target_names=["Away Win", "Home Win"]))

    # ── Cross-validation (gives a more reliable accuracy estimate) ─────────────
    logger.info("5-fold cross-validation on training set...")
    cv = StratifiedKFold(n_splits=5, shuffle=False)  # no shuffle → respects time order
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    logger.info(f"   CV accuracy: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

    # ── Confusion matrix plot ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cm = confusion_matrix(y_test, model.predict(X_test))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Away Win", "Home Win"])
    disp.plot(ax=axes[0], colorbar=False)
    axes[0].set_title("Confusion Matrix (Test Set)")

    # ── Feature importance plot ───────────────────────────────────────────────
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top20 = importances.nlargest(20).sort_values()
    top20.plot(kind="barh", ax=axes[1], color="steelblue")
    axes[1].set_title("Top 20 Feature Importances")
    axes[1].set_xlabel("Importance Score")

    plt.tight_layout()
    plt.savefig("model_evaluation.png", dpi=150)
    logger.info("\n Saved chart → model_evaluation.png")

    return test_acc, auc


def save_artifacts(model, imputer, feature_cols):
    """Save model and feature list so we can load them for predictions later."""
    model.save_model(MODEL_PATH)
    joblib.dump(imputer, "imputer.pkl")
    with open(FEATURES_PATH, "w") as f:
        f.write("\n".join(feature_cols))
    logger.info(f"\n Saved model → {MODEL_PATH}")
    logger.info(f"   Saved feature list → {FEATURES_PATH}")
    logger.info(f"   Saved imputer → imputer.pkl")


def main():
    # 1. Load data
    games, stats = load_all()

    # 2. Build features
    feature_df = build_features(games, stats)

    # 3. Prepare train/test sets
    X_train, X_test, y_train, y_test, imputer, feature_cols, weights_train = prepare_data(feature_df)

    # 4. Train
    model = train_model(X_train, y_train, weights_train)

    # 5. Evaluate
    evaluate_model(model, X_train, X_test, y_train, y_test, feature_cols)

    # 6. Save
    save_artifacts(model, imputer, feature_cols)

    logger.info("\n Pipeline complete! Run predict.py to forecast upcoming games.")


if __name__ == "__main__":
    main()