"""
config.py
---------
Central configuration for the AFL ML pipeline.
Update DATABASE_URL with your NeonDB connection string.
"""

# ── Database ──────────────────────────────────────────────────────────────────
# Replace with your actual NeonDB connection string.
# Format: postgresql://user:password@host/dbname?sslmode=require
DATABASE_URL = "postgresql://USER:PASSWORD@HOST/DBNAME?sslmode=require"

# ── Feature engineering ───────────────────────────────────────────────────────
# How many recent games to use when calculating rolling averages
ROLLING_WINDOW = 10
# Only use games from this year onwards
MIN_YEAR=2019

# ── Model ─────────────────────────────────────────────────────────────────────
# Fraction of data to hold out for testing (0.2 = 20%)
TEST_SIZE = 0.2
# How much to weight recent seasons relative to older ones
# i.e. More recent seasons get a higher weighting
RECENCY_WEIGHTING = True

# XGBoost hyperparameters (good starting defaults for beginners)
XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 3,        # was 4 — shallower trees = less overfitting
    "learning_rate": 0.05,
    "subsample": 0.7,      # was 0.8
    "colsample_bytree": 0.7, # was 0.8
    "min_child_weight": 5,  # add this — prevents splits on small groups
    "reg_alpha": 0.1,       # add this — L1 regularisation
    "reg_lambda": 1.5,      # add this — L2 regularisation
    "eval_metric": "logloss",
}

# ── Output paths ──────────────────────────────────────────────────────────────
MODEL_PATH = "afl_model.json"          # Saved XGBoost model
FEATURES_PATH = "feature_columns.txt"  # List of feature names used in training