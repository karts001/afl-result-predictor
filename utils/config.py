"""
config.py
---------
Central configuration for the AFL ML pipeline.
Update DATABASE_URL with your NeonDB connection string.
"""

# ── Database ──────────────────────────────────────────────────────────────────

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
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.03,       # slower learning = better generalisation
    "subsample": 0.7,
    "colsample_bytree": 0.6,     # was 0.7 — see fewer features per tree
    "min_child_weight": 8,       # was 5 — require larger groups before splitting
    "reg_alpha": 0.5,            # was 0.1 — stronger L1
    "reg_lambda": 2.0,           # was 1.5 — stronger L2
    "gamma": 1.0,                # add this — minimum loss reduction to split
    "eval_metric": "logloss",
    "random_state": 42,
}

# ── Output paths ──────────────────────────────────────────────────────────────
MODEL_PATH = "afl_model.json"          # Saved XGBoost model
FEATURES_PATH = "feature_columns.txt"  # List of feature names used in training

# ── Features actually used in the ML prediction ──────────────────────────────────────────────────────────────

KEEP_FEATURES = [
    "elo_diff",
    "h2h_home_win_rate",
    "h2h_total_games",
    "home_venue_win_rate",
    "away_venue_win_rate",
    "rolling_margin_diff",        # replaces home_ and away_ individually
    "rolling_score_for_diff",     # replaces home_ and away_ individually
    "rolling_score_against_diff", # replaces home_ and away_ individually
    "rolling_win_rate_diff",      # replaces home_ and away_ individually
    "streak_diff",
    "rest_diff",
    "ladder_position_diff",
    "venue_encoded",
    "away_interstate",
    "home_interstate",
]

# ── Database columns which aren't used in the ML prediction ──────────────────────────────────────────────────────────────
NON_FEATURE_COLS = ["gameid", "year", "date", "round", "hometeam", "awayteam", "home_win", "venue"]
