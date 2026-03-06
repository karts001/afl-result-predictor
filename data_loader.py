"""
  Load data from database into pandas DataFrame
"""

import os
from typing import Tuple

import pandas as pd
from dotenv import load_dotenv
import sqlalchemy

from logger import logger

load_dotenv()

def get_engine():
  assert os.getenv("DB_CONNECTION_STRING") is not None, "DB_CONNECTION_STRING must be set in .env file"
  return sqlalchemy.create_engine(os.getenv("DB_CONNECTION_STRING")) # type: ignore

def load_games(engine) -> pd.DataFrame:
  """Load all game data from the games table. Each row contains data from a single game."""

  query = """
    SELECT
      gameid,
      year,
      round,
      date,
      venue,
      starttime,
      attendance,
      maxtemp,
      mintemp,
      rainfall,
      hometeam,
      hometeamscore,
      awayteam,
      awayteamscore
    FROM games
    ORDER BY year, date
  """

  df = pd.read_sql(query, engine)
  logger.info(f"Loaded {len(df)} rows of game data")

  return df

def load_stats(engine) -> pd.DataFrame:
  """Load all player stats from the stats table. Each row contains data from a single player's performance in a single game."""

  query = """
    SELECT
      gameid,
      team,
      year,
      round,
      playerid,
      disposals,
      kicks,
      marks,
      handballs,
      goals,
      behinds,
      hitouts,
      tackles,
      rebounds,
      inside50s,
      clearances,
      clangers,
      frees,
      freesagainst,
      brownlowvotes,
      contestedpossessions,
      uncontestedpossessions,
      contestedmarks,
      marksinside50,
      goalassists,
      percentplayed
    FROM stats
  """

  df = pd.read_sql(query, engine)
  logger.info(f"Loaded {len(df)} rows of player stats data")

  return df

def load_all() -> Tuple[pd.DataFrame, pd.DataFrame]:
  """Load all data from the database and return as a tuple of DataFrames: (games_df, stats_df)"""

  engine = get_engine()
  games_df = load_games(engine)
  stats_df = load_stats(engine)

  return games_df, stats_df
