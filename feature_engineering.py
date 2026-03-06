"""
  Transform raw game data into ML ready features

  Key ideas:
  - Build features from PAST games only
  - Rolling averages to capture recent form
  - H2H history to capture matchup dynamics
  - Venue and conditions to add context
"""

import pandas as pd
import numpy as np

from logger import logger
from config import ROLLING_WINDOW


# ── Step 1: Aggregate player stats to team level per game ─────────────────────

def aggregate_team_stats(stats_df: pd.DataFrame) -> pd.DataFrame:
  """
    Collapse individual player rows into one row per team per game.
    Sum counting stats (disposals, goals etc.) across all players
  """

  team_stats = (
    stats_df
    .groupby(["gameid", "team"])
    .agg(
      total_disposals=("disposals", "sum"),
      total_kicks=("kicks", "sum"),
      total_marks=("marks", "sum"),
      total_handballs=("handballs", "sum"),
      total_goals=("goals", "sum"),
      total_behinds=("behinds", "sum"),
      total_hitouts=("hitouts", "sum"),
      total_tackles=("tackles", "sum"),
      total_rebounds=("rebounds", "sum"),
      total_inside50s=("inside50s", "sum"),
      total_clearances=("clearances", "sum"),
      total_clangers=("clangers", "sum"),
      total_frees=("frees", "sum"),
      total_freesagainst=("freesagainst", "sum"),
      total_contestedpossessions=("contestedpossessions", "sum"),
      total_uncontestedpossessions=("uncontestedpossessions", "sum"),
      total_contestedmarks=("contestedmarks", "sum"),
      total_marksinside50=("marksinside50", "sum"),
      total_goalassists=("goalassists", "sum"),
      avg_percentplayed=("percentplayed", "mean"),
      num_players=("playerid", "nunique")
    )
    .reset_index()
  )

  return team_stats

# ── Step 2: Add win/loss label to games ───────────────────────────────────────

def add_labels(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a binary label: 1 = home team wins, 0 = home team loses.
    Draws are dropped (very rare in AFL).
    """
    df = games_df.copy()
    df["home_win"] = (df["hometeamscore"] > df["awayteamscore"]).astype(int)

    # Drop draws
    is_predict = df["gameid"].astype(str).str.startswith("PREDICT_")
    draws = (df["hometeamscore"] == df["awayteamscore"]) & ~is_predict
    
    n_draws = draws.sum()
    if n_draws > 0:
        logger.info(f"Dropping {n_draws} drawn games")
        df = df[~draws]

    # Score margin (useful context, but NOT used as a feature — that's leakage)
    df["score_margin"] = df["hometeamscore"] - df["awayteamscore"]

    return df


# ── Step 3: Rolling team form features ────────────────────────────────────────

def _rolling_team_features(games_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    For each game, compute rolling averages of the PREVIOUS `window` games
    for both the home and away team.

    IMPORTANT: We shift by 1 so we never include the current game's result.
    This prevents data leakage.
    """
    df = games_df.sort_values(["year", "date"]).copy()

    # We'll compute rolling stats from both perspectives
    records = []

    # Get all teams
    all_teams = set(df["hometeam"]).union(set(df["awayteam"]))

    team_history = {team: [] for team in all_teams}

    for _, row in df.iterrows():
        home = row["hometeam"]
        away = row["awayteam"]

        def rolling_avg(team, stat):
            """Average of last `window` values for a team."""
            history = team_history[team]
            if len(history) == 0:
                return np.nan
            recent = history[-window:]
            return np.mean([h[stat] for h in recent if stat in h])

        # Build feature dict for this game
        feat = {
            "gameid": row["gameid"],
            # Home team rolling stats
            "home_rolling_win_rate": rolling_avg(home, "win"),
            "home_rolling_score_for": rolling_avg(home, "score_for"),
            "home_rolling_score_against": rolling_avg(home, "score_against"),
            "home_rolling_margin": rolling_avg(home, "margin"),
            # Away team rolling stats
            "away_rolling_win_rate": rolling_avg(away, "win"),
            "away_rolling_score_for": rolling_avg(away, "score_for"),
            "away_rolling_score_against": rolling_avg(away, "score_against"),
            "away_rolling_margin": rolling_avg(away, "margin"),
        }
        records.append(feat)

        # Now update history AFTER building features (no leakage)
        home_win = int(row["hometeamscore"] > row["awayteamscore"])
        team_history[home].append({
            "win": home_win,
            "score_for": row["hometeamscore"],
            "score_against": row["awayteamscore"],
            "margin": row["hometeamscore"] - row["awayteamscore"],
        })
        team_history[away].append({
            "win": 1 - home_win,
            "score_for": row["awayteamscore"],
            "score_against": row["hometeamscore"],
            "margin": row["awayteamscore"] - row["hometeamscore"],
        })

    return pd.DataFrame(records)


# ── Step 4: Head-to-head features ────────────────────────────────────────────

def _head_to_head_features(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each game, compute the home team's historical win rate against the away team.
    Again, only uses games BEFORE the current one.
    """
    df = games_df.sort_values(["year", "date"]).copy()

    h2h = {}  # key: (home_team, away_team) → list of home wins
    records = []

    for _, row in df.iterrows():
        home, away = row["hometeam"], row["awayteam"]
        key = (home, away)
        reverse_key = (away, home)

        # Past matchups in either direction
        home_as_home = h2h.get(key, [])
        home_as_away = h2h.get(reverse_key, [])

        total_games = len(home_as_home) + len(home_as_away)
        home_wins = sum(home_as_home) + sum(1 - w for w in home_as_away)

        records.append({
            "gameid": row["gameid"],
            "h2h_home_win_rate": home_wins / total_games if total_games > 0 else np.nan,
            "h2h_total_games": total_games,
        })

        # Update h2h history after extracting features
        h2h.setdefault(key, []).append(int(row["hometeamscore"] > row["awayteamscore"]))

    return pd.DataFrame(records)


# ── Step 5: Team stats rolling averages ───────────────────────────────────────

def _rolling_team_stat_features(games_df: pd.DataFrame, team_stats_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Compute rolling averages of aggregated team stats (disposals, tackles etc.)
    for home and away teams.
    """
    df = games_df.sort_values(["year", "date"]).copy()
    stat_cols = [c for c in team_stats_df.columns if c not in ["gameid", "team"]]

    # Build a lookup: gameid → {team → stats}
    stats_lookup = (
        team_stats_df
        .set_index(["gameid", "team"])
        [stat_cols]
        .to_dict(orient="index")
    )

    team_stat_history = {}  # team → list of stat dicts
    records = []

    for _, row in df.iterrows():
        home, away = row["hometeam"], row["awayteam"]
        feat = {"gameid": row["gameid"]}

        for team, prefix in [(home, "home"), (away, "away")]:
            history = team_stat_history.get(team, [])
            recent = history[-window:] if history else []
            for col in stat_cols:
                vals = [h.get(col, np.nan) for h in recent]
                vals = [v for v in vals if not np.isnan(v)]
                feat[f"{prefix}_{col}_rolling"] = np.mean(vals) if vals else np.nan

        records.append(feat)

        # Update histories after extracting (no leakage)
        for team in [home, away]:
            key = (row["gameid"], team)
            if key in stats_lookup:
                team_stat_history.setdefault(team, []).append(stats_lookup[key])

    return pd.DataFrame(records)


# ── Step 6: Venue & conditions features ──────────────────────────────────────

def _venue_features(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode venue as a category and include weather conditions.
    """
    df = games_df[["gameid", "venue", "maxtemp", "mintemp", "rainfall"]].copy()
    df["venue_encoded"] = df["venue"].astype("category").cat.codes
    return df

# ── Step 7: Home advantage and win streak features ──────────────────────────────────────
def _home_ground_advantage(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute each team's historical win rate at each venue.
    Only uses past games to avoid leakage.
    """
    df = games_df.sort_values("date").copy()
    
    # track win rate per team per venue
    venue_history = {}  # (team, venue) -> list of wins
    records = []

    for _, row in df.iterrows():
        home, away, venue = row["hometeam"], row["awayteam"], row["venue"]

        home_key = (home, venue)
        away_key = (away, venue)

        home_venue_games = venue_history.get(home_key, [])
        away_venue_games = venue_history.get(away_key, [])

        records.append({
            "gameid": row["gameid"],
            "home_venue_win_rate": np.mean(home_venue_games) if home_venue_games else np.nan,
            "home_venue_games_played": len(home_venue_games),
            "away_venue_win_rate": np.mean(away_venue_games) if away_venue_games else np.nan,
            "away_venue_games_played": len(away_venue_games),
        })

        # Update after extracting features (no leakage)
        home_win = int(row["hometeamscore"] > row["awayteamscore"])
        venue_history.setdefault(home_key, []).append(home_win)
        venue_history.setdefault(away_key, []).append(1 - home_win)

    return pd.DataFrame(records)


def _streak_and_rest_features(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute win/loss streak and days rest for each team going into each game.
    """
    df = games_df.sort_values("date").copy()
    df["date_dt"] = pd.to_datetime(df["date"], format="mixed", dayfirst=True)

    team_history = {}  # team -> list of (date, win)
    records = []

    for _, row in df.iterrows():
        home, away = row["hometeam"], row["awayteam"]
        game_date = row["date_dt"]

        def get_streak(team):
            """Positive = win streak, negative = loss streak."""
            history = team_history.get(team, [])
            if not history:
                return 0
            streak = 0
            last_result = history[-1][1]
            for _, result in reversed(history):
                if result == last_result:
                    streak += 1
                else:
                    break
            return streak if last_result == 1 else -streak

        def get_days_rest(team):
            """Days since last game."""
            history = team_history.get(team, [])
            if not history:
                return np.nan
            last_date = history[-1][0]
            return (game_date - last_date).days

        records.append({
            "gameid": row["gameid"],
            "home_streak": get_streak(home),
            "away_streak": get_streak(away),
            "streak_diff": get_streak(home) - get_streak(away),
            "home_days_rest": get_days_rest(home),
            "away_days_rest": get_days_rest(away),
            "rest_diff": (get_days_rest(home) or 0) - (get_days_rest(away) or 0),
        })

        # Update after extracting (no leakage)
        home_win = int(row["hometeamscore"] > row["awayteamscore"])
        team_history.setdefault(home, []).append((game_date, home_win))
        team_history.setdefault(away, []).append((game_date, 1 - home_win))

    return pd.DataFrame(records)

# ──  Step 8 Ladder position and Interstate travel features ───────────────────────────────────────────────────────────

def _ladder_position_features(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute each team's current ladder position at the time of each game.
    Only uses results from the current season up to (but not including) the current game.
    """
    df = games_df.sort_values(["year", "date"]).copy()

    records = []

    for year, season_df in df.groupby("year"):
        # Track wins and percentage for each team within the season
        season_stats = {}  # team -> {"wins": 0, "points_for": 0, "points_against": 0}

        for _, row in season_df.iterrows():
            home, away = row["hometeam"], row["awayteam"]

            # Initialise teams if not seen yet
            for team in [home, away]:
                if team not in season_stats:
                    season_stats[team] = {"wins": 0, "points_for": 0, "points_against": 0, "games": 0}

            # Compute ladder BEFORE this game
            def get_position(team):
                stats = season_stats.get(team, {})
                if stats.get("games", 0) == 0:
                    return np.nan
                # AFL ladder is ordered by wins then percentage
                pf = stats["points_for"]
                pa = stats["points_against"]
                pct = (pf / pa * 100) if pa > 0 else 0
                return (stats["wins"], pct)

            all_teams = {t: get_position(t) for t in season_stats}
            sorted_teams = sorted(
                [t for t in all_teams if all_teams[t] is not np.nan],
                key=lambda t: all_teams[t],
                reverse=True
            )

            home_pos = sorted_teams.index(home) + 1 if home in sorted_teams else np.nan
            away_pos = sorted_teams.index(away) + 1 if away in sorted_teams else np.nan

            records.append({
                "gameid": row["gameid"],
                "home_ladder_position": home_pos,
                "away_ladder_position": away_pos,
                "ladder_position_diff": (away_pos - home_pos) if (
                    not np.isnan(home_pos) and not np.isnan(away_pos)
                ) else np.nan,
            })

            # Update season stats AFTER extracting features
            home_win = int(row["hometeamscore"] > row["awayteamscore"])
            season_stats[home]["wins"] += home_win
            season_stats[home]["points_for"] += row["hometeamscore"]
            season_stats[home]["points_against"] += row["awayteamscore"]
            season_stats[home]["games"] += 1

            season_stats[away]["wins"] += (1 - home_win)
            season_stats[away]["points_for"] += row["awayteamscore"]
            season_stats[away]["points_against"] += row["hometeamscore"]
            season_stats[away]["games"] += 1

    return pd.DataFrame(records)

def _interstate_travel_features(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag whether the away team is travelling interstate.
    Based on each team's known home state.
    """
    # Each AFL team's home state
    TEAM_STATE = {
        "Adelaide": "SA",
        "Port Adelaide": "SA",
        "Melbourne": "VIC",
        "Collingwood": "VIC",
        "Richmond": "VIC",
        "Carlton": "VIC",
        "Essendon": "VIC",
        "Hawthorn": "VIC",
        "Western Bulldogs": "VIC",
        "North Melbourne": "VIC",
        "St Kilda": "VIC",
        "Geelong": "VIC",
        "West Coast": "WA",
        "Fremantle": "WA",
        "Brisbane Lions": "QLD",
        "Gold Coast": "QLD",
        "Sydney": "NSW",
        "Greater Western Sydney": "NSW",
    }

    # Venue state lookup
    VENUE_STATE = {
        "M.C.G.": "VIC",
        "Docklands": "VIC",
        "Kardinia Park": "VIC",
        "Eureka Stadium": "VIC",
        "Adelaide Oval": "SA",
        "Football Park": "SA",
        "Norwood Oval": "SA",
        "Barossa Oval": "SA",
        "Perth Stadium": "WA",
        "Subiaco": "WA",
        "Gabba": "QLD",
        "Carrara": "QLD",
        "Cazaly's Stadium": "QLD",
        "Riverway Stadium": "QLD",
        "S.C.G.": "NSW",
        "Stadium Australia": "NSW",
        "Sydney Showground": "NSW",
        "Blacktown": "NSW",
        "Manuka Oval": "ACT",
        "Bellerive Oval": "TAS",
        "York Park": "TAS",
        "Hands Oval": "TAS",
        "Traeger Park": "NT",
        "Marrara Oval": "NT",
        "Jiangwan Stadium": "CHN",
        "Summit Sports Park": "CHN",
        "Wellington": "NZ",
        "Barossa Park": "SA",
        "Hands Oval": "TAS",
    }

    df = games_df.copy()

    def is_interstate(row):
        away_state = TEAM_STATE.get(row["awayteam"])
        venue_state = VENUE_STATE.get(row["venue"])
        if away_state is None or venue_state is None:
            return np.nan
        return int(away_state != venue_state)

    def is_home_interstate(row):
        home_state = TEAM_STATE.get(row["hometeam"])
        venue_state = VENUE_STATE.get(row["venue"])
        if home_state is None or venue_state is None:
            return np.nan
        return int(home_state != venue_state)

    df["away_interstate"] = df.apply(is_interstate, axis=1)
    df["home_interstate"] = df.apply(is_home_interstate, axis=1)

    return df[["gameid", "away_interstate", "home_interstate"]]

# ── Elo features ───────────────────────────────────────────────────────────

def _elo_features(games_df: pd.DataFrame, k: int = 32, carry_over: float = 0.75) -> pd.DataFrame:
    df = games_df.sort_values(["year", "date"]).copy()

    elo_ratings = {team: 1500 for team in set(df["hometeam"].dropna()).union(set(df["awayteam"].dropna()))}
    current_year = None
    records = []

    for _, row in df.iterrows():
        home, away = row["hometeam"], row["awayteam"]
        year = row["year"]

        # Regress ratings at start of each new season
        if year != current_year:
            current_year = year
            elo_ratings = {
                team: 1500 + carry_over * (rating - 1500)
                for team, rating in elo_ratings.items()
            }

        home_elo = elo_ratings.get(home, 1500)
        away_elo = elo_ratings.get(away, 1500)

        expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))

        records.append({
            "gameid": row["gameid"],
            "home_elo": round(home_elo, 1),
            "away_elo": round(away_elo, 1),
            "elo_diff": round(home_elo - away_elo, 1),
            "elo_home_win_prob": round(expected_home, 3),
        })

        # Only update ratings for real games (not placeholder upcoming games)
        is_real_game = not str(row["gameid"]).startswith("PREDICT_")
        if is_real_game:
            actual_home = 1 if row["hometeamscore"] > row["awayteamscore"] else 0
            change = k * (actual_home - expected_home)
            elo_ratings[home] += change
            elo_ratings[away] -= change

    return pd.DataFrame(records)

# ── Master function ───────────────────────────────────────────────────────────

def build_features(games_df: pd.DataFrame, stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates all feature engineering steps and returns a single
    ML-ready DataFrame with one row per game.

    Returns:
        feature_df  — all features + 'home_win' label
    """
    logger.info("Building features...")

    # Label
    games = add_labels(games_df)

    # Aggregate player stats to team level
    team_stats = aggregate_team_stats(stats_df)

    # Get rolling window from environment

    # Rolling form
    logger.info("Computing rolling form...")
    rolling = _rolling_team_features(games, window=ROLLING_WINDOW)

    # Head-to-head
    logger.info("Computing head-to-head history...")
    h2h = _head_to_head_features(games)

    # Rolling aggregated stats
    logger.info("Computing rolling team stats...")
    rolling_stats = _rolling_team_stat_features(games, team_stats, window=ROLLING_WINDOW)

    # Venue / conditions
    logger.info("Computing venue features")
    venue = _venue_features(games)

    logger.info("Computing home ground advantage...")
    hga = _home_ground_advantage(games)

    logger.info("Computing streak and rest feature")
    streak_and_rest = _streak_and_rest_features(games)

    logger.info("Computing ladder position feature")
    ladder_pos = _ladder_position_features(games)

    logger.info("Computing interstate travel feature")
    ist = _interstate_travel_features(games)

    logger.info("Computing elo features")
    elo = _elo_features(games)

    print("ELO sample:", elo.tail(10)[["gameid", "home_elo", "away_elo"]].to_string())
    print("Upcoming gameids in elo:", elo[elo["gameid"].str.startswith("PREDICT_")][["gameid", "home_elo", "away_elo"]].to_string())

    # Merge everything on gameid
    feature_df = games[["gameid", "year", "date", "round", "hometeam", "awayteam", "home_win"]].copy()
    for df_part in [rolling, h2h, rolling_stats, venue, hga, streak_and_rest, ladder_pos, ist, elo]:
        feature_df = feature_df.merge(df_part, on="gameid", how="left")

    # Drop rows where rolling features are all NaN (first few games of a team's history)
    before = len(feature_df)
    feature_df = feature_df.dropna(subset=["home_rolling_win_rate", "away_rolling_win_rate"])
    after = len(feature_df)
    logger.info(f"Dropped {before - after} rows with insufficient rolling history")

    logger.info(f"Feature matrix: {feature_df.shape[0]} games × {feature_df.shape[1]} columns")
    return feature_df

