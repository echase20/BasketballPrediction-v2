import pandas as pd
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import (
    PlayerGameLog,
    PlayerCareerStats,
    TeamDashboardByOpponent,
)
import time


def get_player_id(first_name: str, last_name: str) -> int:
    full_name = f"{first_name} {last_name}"
    matches = players.find_players_by_full_name(full_name)
    if not matches:
        raise ValueError(f"Player '{full_name}' not found.")
    return matches[0]["id"]


def get_game_log(player_id: int, season: str = "2023-24") -> pd.DataFrame:
    """Return a cleaned game log for a player in a given season."""
    log = PlayerGameLog(player_id=player_id, season=season)
    df = log.get_data_frames()[0]

    cols = ["GAME_DATE", "MATCHUP", "WL", "MIN", "PTS", "REB", "AST",
            "FGA", "FGM", "FG_PCT", "FG3A", "FG3M", "FTA", "FTM", "TOV"]
    df = df[cols].copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    # Rolling season average points
    df["ROLLING_AVG_PTS"] = df["PTS"].expanding().mean().round(2)

    # Extract opponent abbreviation from MATCHUP (e.g. "BOS vs. MIA" -> "MIA")
    df["OPP"] = df["MATCHUP"].apply(lambda x: x.split()[-1])

    return df


def get_career_avg(player_id: int) -> float:
    """Return a player's career points per game average."""
    career = PlayerCareerStats(player_id=player_id)
    df = career.get_data_frames()[0]
    # Last row is career totals
    totals = df[df["SEASON_ID"] == "Career"]
    if totals.empty:
        return df["PTS"].iloc[-1]
    return round(float(totals["PTS"].values[0]), 2)


def get_opponent_defense(team_abbreviation: str) -> dict:
    """
    Return opponent defensive stats (pts allowed per game) for a given team.
    Uses TeamDashboardByOpponent for the current season.
    """
    team_list = teams.get_teams()
    team = next((t for t in team_list if t["abbreviation"] == team_abbreviation), None)
    if team is None:
        raise ValueError(f"Team abbreviation '{team_abbreviation}' not found.")

    time.sleep(0.6)  # Respect NBA API rate limits
    dashboard = TeamDashboardByOpponent(team_id=team["id"], season="2023-24")
    df = dashboard.get_data_frames()[0]

    return {
        "OPP_PTS_ALLOWED": round(float(df["OPP_PTS"].mean()), 2),
        "OPP_FG_PCT": round(float(df["OPP_FG_PCT"].mean()), 4),
    }


def build_dataset(first_name: str, last_name: str, season: str = "2023-24") -> pd.DataFrame:
    """
    Build the full feature dataset for a player:
      - Game-by-game stats
      - Rolling season average
      - Opponent defensive ratings per game
    """
    player_id = get_player_id(first_name, last_name)
    print(f"Found player: {first_name} {last_name} (ID: {player_id})")

    game_log = get_game_log(player_id, season)
    career_avg = get_career_avg(player_id)
    print(f"Career PPG: {career_avg}")

    print("Fetching opponent defensive stats (this may take a moment)...")
    opp_defense = []
    seen = {}
    for opp in game_log["OPP"]:
        if opp not in seen:
            try:
                seen[opp] = get_opponent_defense(opp)
            except Exception:
                seen[opp] = {"OPP_PTS_ALLOWED": None, "OPP_FG_PCT": None}
        opp_defense.append(seen[opp])

    defense_df = pd.DataFrame(opp_defense)
    game_log = pd.concat([game_log.reset_index(drop=True), defense_df], axis=1)
    game_log["CAREER_AVG_PTS"] = career_avg

    return game_log


if __name__ == "__main__":
    df = build_dataset("Jayson", "Tatum", season="2023-24")
    print(df.to_string())
