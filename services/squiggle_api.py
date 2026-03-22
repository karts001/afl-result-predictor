import pandas as pd
import requests

from utils.logger import logger


base_url = "https://api.squiggle.com.au/"

def fetch_upcoming_games(year: int) -> pd.DataFrame:
  params = f"?q=games;year={year}"

  url = base_url + params
  logger.info(f"Fetching fixture from Squiggle API: {url}")

  response = requests.get(url, headers={"User-Agent": "afl-predictor/1.0"})
  response.raise_for_status()

  games = response.json().get("games", [])
  if not games:
      logger.warning("No games found.")
      return pd.DataFrame()

  df = pd.DataFrame(games)

  # Filter to incomplete games only
  upcoming = df[df["complete"] < 100].copy()

  # Filter out finals where teams aren't determined yet
  upcoming = upcoming[upcoming["hteam"].notna() & upcoming["ateam"].notna()]

  upcoming["date_dt"] = pd.to_datetime(upcoming["date"])
  today = pd.Timestamp.now()
  upcoming = upcoming[upcoming["date_dt"] <= today + pd.Timedelta(days=7)]

  if upcoming.empty:
      logger.warning("No upcoming games found.")
      return pd.DataFrame()

  upcoming = upcoming.rename(columns={"hteam": "hometeam", "ateam": "awayteam", "id": "gameid"})
  upcoming["date"] = pd.to_datetime(upcoming["date"]).dt.strftime("%Y-%m-%d")
  upcoming["gameid"] = upcoming["gameid"].astype(str)

  logger.info(f"Found {len(upcoming)} upcoming games")
  return upcoming[["gameid", "date", "year", "round", "venue", "hometeam", "awayteam"]]