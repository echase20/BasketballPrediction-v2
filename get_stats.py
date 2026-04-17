import pandas as pd
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import (
    PlayerGameLog,
    PlayerCareerStats,
    TeamDashboardByOpponent,
)
import time

LEAGUE_AVG_PTS_ALLOWED = 112.0


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

    # Extract opponent abbreviation from MATCHUP (e.g. "BOS vs. MIA" -> "MIA")
    df["OPP"] = df["MATCHUP"].apply(lambda x: x.split()[-1])

    return df


def get_career_avg(player_id: int) -> float:
    """Return a player's career points per game average."""
    career = PlayerCareerStats(player_id=player_id)
    df = career.get_data_frames()[0]
    totals = df[df["SEASON_ID"] == "Career"]
    if totals.empty:
        return round(float(df["PTS"].iloc[-1]), 2)
    return round(float(totals["PTS"].values[0]), 2)


def get_opponent_defense(team_abbreviation: str, season: str = "2023-24") -> dict:
    """Return defensive stats (pts allowed per game) for a given team."""
    team_list = teams.get_teams()
    team = next((t for t in team_list if t["abbreviation"] == team_abbreviation), None)
    if team is None:
        raise ValueError(f"Team abbreviation '{team_abbreviation}' not found.")

    time.sleep(0.6)  # Respect NBA API rate limits
    dashboard = TeamDashboardByOpponent(team_id=team["id"], season=season)
    df = dashboard.get_data_frames()[0]

    return {
        "OPP_PTS_ALLOWED": round(float(df["OPP_PTS"].mean()), 2),
        "OPP_FG_PCT": round(float(df["OPP_FG_PCT"].mean()), 4),
    }


def add_features(game_log: pd.DataFrame, career_avg: float,
                 opp_defense_map: dict) -> pd.DataFrame:
    """
    Add the 5 prediction features to the game log:
      1. ROLLING_AVG_PTS  — expanding season average up to (not including) this game
      2. CAREER_AVG_PTS   — career PPG (constant)
      3. RECENT_FORM      — average points over the last 5 games
      4. VS_OPP_AVG       — historical avg points vs this opponent in prior games
      5. OPP_ADJ          — season avg scaled by opponent defensive strength
    """
    df = game_log.copy()

    # 1. Rolling season average (shift by 1 so we only use prior games)
    df["ROLLING_AVG_PTS"] = df["PTS"].shift(1).expanding().mean().round(2)
    df["ROLLING_AVG_PTS"].fillna(career_avg, inplace=True)

    # 2. Career average
    df["CAREER_AVG_PTS"] = career_avg

    # 3. Recent form: avg of last 5 games (using prior games only)
    df["RECENT_FORM"] = df["PTS"].shift(1).rolling(5, min_periods=1).mean().round(2)
    df["RECENT_FORM"].fillna(career_avg, inplace=True)

    # 4. Historical avg vs this specific opponent (prior games only)
    vs_opp = []
    for i, row in df.iterrows():
        prior = df.loc[:i - 1]
        prior_vs_opp = prior[prior["OPP"] == row["OPP"]]["PTS"]
        vs_opp.append(round(prior_vs_opp.mean(), 2) if not prior_vs_opp.empty else career_avg)
    df["VS_OPP_AVG"] = vs_opp

    # 5. Opponent-adjusted expectation: rolling avg scaled by how good/bad the defense is
    df["OPP_PTS_ALLOWED"] = df["OPP"].map(
        lambda opp: opp_defense_map.get(opp, {}).get("OPP_PTS_ALLOWED", LEAGUE_AVG_PTS_ALLOWED)
    )
    df["OPP_ADJ"] = (df["ROLLING_AVG_PTS"] * (df["OPP_PTS_ALLOWED"] / LEAGUE_AVG_PTS_ALLOWED)).round(2)

    return df


def build_dataset(first_name: str, last_name: str, season: str = "2023-24") -> pd.DataFrame:
    """
    Build the full feature dataset for a player.
    Returns a DataFrame with all 5 prediction features and actual PTS for each game.
    """
    player_id = get_player_id(first_name, last_name)
    print(f"Found player: {first_name} {last_name} (ID: {player_id})")

    game_log = get_game_log(player_id, season)
    career_avg = get_career_avg(player_id)
    print(f"Career PPG: {career_avg}")

    print("Fetching opponent defensive stats (this may take a moment)...")
    opp_defense_map = {}
    for opp in game_log["OPP"].unique():
        try:
            opp_defense_map[opp] = get_opponent_defense(opp, season)
        except Exception:
            opp_defense_map[opp] = {"OPP_PTS_ALLOWED": LEAGUE_AVG_PTS_ALLOWED, "OPP_FG_PCT": None}

    dataset = add_features(game_log, career_avg, opp_defense_map)
    return dataset


def get_next_game_features(first_name: str, last_name: str,
                           next_opp: str, season: str = "2023-24") -> dict:
    """
    Return the 5 prediction features for a player's NEXT game against next_opp.
    Uses the most recent values from the current season log.
    """
    dataset = build_dataset(first_name, last_name, season)
    last = dataset.iloc[-1]

    opp_defense = get_opponent_defense(next_opp, season)
    rolling_avg = last["ROLLING_AVG_PTS"]
    opp_adj = round(rolling_avg * (opp_defense["OPP_PTS_ALLOWED"] / LEAGUE_AVG_PTS_ALLOWED), 2)

    prior_vs_opp = dataset[dataset["OPP"] == next_opp]["PTS"]
    vs_opp_avg = round(prior_vs_opp.mean(), 2) if not prior_vs_opp.empty else last["CAREER_AVG_PTS"]

    return {
        "ROLLING_AVG_PTS": rolling_avg,
        "CAREER_AVG_PTS": last["CAREER_AVG_PTS"],
        "RECENT_FORM": last["RECENT_FORM"],
        "VS_OPP_AVG": vs_opp_avg,
        "OPP_ADJ": opp_adj,
        "dataset": dataset,
    }


if __name__ == "__main__":
    df = build_dataset("Jayson", "Tatum", season="2023-24")
    features = ["GAME_DATE", "OPP", "PTS", "ROLLING_AVG_PTS", "CAREER_AVG_PTS",
                "RECENT_FORM", "VS_OPP_AVG", "OPP_ADJ"]
    print(df[features].to_string())
