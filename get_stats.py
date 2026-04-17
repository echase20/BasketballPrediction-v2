import time
import numpy as np
import pandas as pd
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import (
    PlayerGameLog,
    PlayerCareerStats,
    LeagueDashTeamStats,
)

LEAGUE_AVG_PTS_ALLOWED = 112.0
TEST_GAMES = 20


def get_player_id(first_name: str, last_name: str) -> int:
    full_name = f"{first_name} {last_name}"
    matches = players.find_players_by_full_name(full_name)
    if not matches:
        raise ValueError(f"Player '{full_name}' not found.")
    return matches[0]["id"]


def get_all_seasons(player_id: int) -> list[str]:
    """Return list of every season the player appeared in, oldest first."""
    career = PlayerCareerStats(player_id=player_id)
    df = career.get_data_frames()[0]
    return df[df["SEASON_ID"] != "Career"]["SEASON_ID"].tolist()


def get_season_game_log(player_id: int, season: str) -> pd.DataFrame:
    """Return raw game log for a single season, or empty DataFrame on failure."""
    time.sleep(0.5)
    try:
        log = PlayerGameLog(player_id=player_id, season=season)
        df = log.get_data_frames()[0]
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    cols = ["GAME_DATE", "MATCHUP", "WL", "MIN", "PTS", "REB", "AST",
            "FGA", "FGM", "FG_PCT", "FG3A", "FG3M", "FTA", "FTM", "TOV"]
    df = df[[c for c in cols if c in df.columns]].copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="mixed")
    df["SEASON"] = season
    df["OPP"] = df["MATCHUP"].apply(lambda x: x.split()[-1])
    return df


def get_league_defense(season: str) -> dict:
    """
    Return {team_abbreviation: pts_allowed_per_game} for all teams in a season.
    Uses LeagueDashTeamStats in Opponent measure mode (one API call per season).
    Falls back to league average on failure.
    """
    time.sleep(0.5)
    try:
        stats = LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense="Opponent",
            per_mode_simple="PerGame",
        )
        df = stats.get_data_frames()[0]
        # In Opponent mode, PTS = points the team allows per game
        return dict(zip(df["TEAM_ABBREVIATION"], df["PTS"]))
    except Exception:
        return {}


def get_full_career_log(player_id: int) -> pd.DataFrame:
    """Pull and concatenate game logs for every season in a player's career."""
    seasons = get_all_seasons(player_id)
    print(f"  Fetching game logs across {len(seasons)} seasons...")
    dfs = []
    for season in seasons:
        df = get_season_game_log(player_id, season)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        raise ValueError("No game log data found for this player.")

    full = pd.concat(dfs, ignore_index=True)
    full = full.sort_values("GAME_DATE").reset_index(drop=True)
    return full


def add_features(game_log: pd.DataFrame, fallback_avg: float,
                 league_defense_by_season: dict) -> pd.DataFrame:
    """
    Add the 5 prediction features to the career game log.
    All features use only data available BEFORE each game (no lookahead).

    Features:
        1. ROLLING_AVG_PTS  — expanding season average (resets each season)
        2. CAREER_AVG_PTS   — expanding career average across all seasons
        3. RECENT_FORM      — last 5 games average (crosses seasons)
        4. VS_OPP_AVG       — career average vs this specific opponent
        5. OPP_ADJ          — rolling season avg scaled by opponent defense that season
    """
    df = game_log.copy()

    # 1. Season rolling average (shift by 1 to exclude current game)
    df["ROLLING_AVG_PTS"] = (
        df.groupby("SEASON")["PTS"]
        .transform(lambda x: x.shift(1).expanding().mean())
        .fillna(fallback_avg)
        .round(2)
    )

    # 2. Career average up to (not including) each game
    df["CAREER_AVG_PTS"] = (
        df["PTS"].shift(1).expanding().mean()
        .fillna(fallback_avg)
        .round(2)
    )

    # 3. Recent form: last 5 games (cross-season, shift by 1)
    df["RECENT_FORM"] = (
        df["PTS"].shift(1).rolling(5, min_periods=1).mean()
        .fillna(fallback_avg)
        .round(2)
    )

    # 4. Historical avg vs this opponent (O(n), no lookahead)
    vs_opp_avgs = []
    opp_history: dict[str, list] = {}
    for _, row in df.iterrows():
        opp = row["OPP"]
        history = opp_history.get(opp, [])
        vs_opp_avgs.append(round(np.mean(history), 2) if history else fallback_avg)
        opp_history.setdefault(opp, []).append(row["PTS"])
    df["VS_OPP_AVG"] = vs_opp_avgs

    # 5. Opponent-adjusted expectation: season rolling avg * (opp defense / league avg)
    def opp_adj(row):
        season_defense = league_defense_by_season.get(row["SEASON"], {})
        pts_allowed = season_defense.get(row["OPP"], LEAGUE_AVG_PTS_ALLOWED)
        return round(row["ROLLING_AVG_PTS"] * (pts_allowed / LEAGUE_AVG_PTS_ALLOWED), 2)

    df["OPP_ADJ"] = df.apply(opp_adj, axis=1)

    return df


def build_career_dataset(first_name: str, last_name: str) -> dict:
    """
    Build the full career dataset and split into train/test.

    Returns
    -------
    dict with keys:
        train     — all games except the last TEST_GAMES
        test      — the last TEST_GAMES games
        full      — complete dataset
        fallback_avg — career PPG used as feature fallback
    """
    player_id = get_player_id(first_name, last_name)
    print(f"Found: {first_name} {last_name} (ID: {player_id})")

    game_log = get_full_career_log(player_id)
    fallback_avg = round(float(game_log["PTS"].mean()), 2)
    print(f"  Total games: {len(game_log)} | Career PPG: {fallback_avg}")

    # Fetch opponent defensive stats per season (one call per season)
    seasons = game_log["SEASON"].unique().tolist()
    print(f"  Fetching league defense for {len(seasons)} seasons...")
    league_defense_by_season = {}
    for season in seasons:
        league_defense_by_season[season] = get_league_defense(season)

    dataset = add_features(game_log, fallback_avg, league_defense_by_season)

    train = dataset.iloc[:-TEST_GAMES].copy()
    test = dataset.iloc[-TEST_GAMES:].copy()
    print(f"  Train: {len(train)} games | Test: {len(test)} games")

    return {
        "train": train,
        "test": test,
        "full": dataset,
        "fallback_avg": fallback_avg,
        "current_season": game_log["SEASON"].iloc[-1],
        "league_defense_by_season": league_defense_by_season,
    }


def get_next_game_features(data: dict, next_opp: str) -> dict:
    """
    Build the 5 prediction features for an upcoming game against next_opp.
    Uses the latest values from the end of the full career dataset.

    Parameters
    ----------
    data : dict
        Output of build_career_dataset().
    next_opp : str
        Opponent team abbreviation (e.g. "MIA").
    """
    full = data["full"]
    last = full.iloc[-1]
    current_season = data["current_season"]
    fallback = data["fallback_avg"]

    # Latest rolling and career averages
    rolling_avg = last["ROLLING_AVG_PTS"]
    career_avg = last["CAREER_AVG_PTS"]
    recent_form = last["RECENT_FORM"]

    # Historical avg vs next opponent across entire career
    prior_vs_opp = full[full["OPP"] == next_opp]["PTS"]
    vs_opp_avg = round(float(prior_vs_opp.mean()), 2) if not prior_vs_opp.empty else fallback

    # Opponent defense this season
    season_defense = data["league_defense_by_season"].get(current_season, {})
    pts_allowed = season_defense.get(next_opp, LEAGUE_AVG_PTS_ALLOWED)
    opp_adj = round(rolling_avg * (pts_allowed / LEAGUE_AVG_PTS_ALLOWED), 2)

    return {
        "ROLLING_AVG_PTS": rolling_avg,
        "CAREER_AVG_PTS":  career_avg,
        "RECENT_FORM":     recent_form,
        "VS_OPP_AVG":      vs_opp_avg,
        "OPP_ADJ":         opp_adj,
    }


if __name__ == "__main__":
    data = build_career_dataset("Jayson", "Tatum")
    cols = ["GAME_DATE", "SEASON", "OPP", "PTS",
            "ROLLING_AVG_PTS", "CAREER_AVG_PTS", "RECENT_FORM", "VS_OPP_AVG", "OPP_ADJ"]
    print("\n--- Last 5 rows of training set ---")
    print(data["train"][cols].tail().to_string())
    print("\n--- Test set (last 20 games) ---")
    print(data["test"][cols].to_string())
