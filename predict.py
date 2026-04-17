"""
Entry point: pulls player stats and runs the Monte Carlo predictor.

Usage:
    python predict.py --player "Jayson Tatum" --season 2023-24 --simulations 1000
"""

import argparse
import sys
from get_stats import build_dataset
from monte_carlo import MonteCarloPredictor


def main():
    parser = argparse.ArgumentParser(description="Predict NBA player points using MCST.")
    parser.add_argument("--player", required=True, help='Full player name, e.g. "Jayson Tatum"')
    parser.add_argument("--season", default="2023-24", help="NBA season string (default: 2023-24)")
    parser.add_argument("--simulations", type=int, default=1000,
                        help="Number of Monte Carlo simulations (default: 1000)")
    args = parser.parse_args()

    name_parts = args.player.strip().split()
    if len(name_parts) < 2:
        print("Error: please provide both first and last name.")
        sys.exit(1)

    first, last = name_parts[0], " ".join(name_parts[1:])

    print(f"\nFetching stats for {first} {last} ({args.season})...")
    game_log = build_dataset(first, last, season=args.season)

    print(f"\nRunning {args.simulations} Monte Carlo simulations...")
    predictor = MonteCarloPredictor(game_log, simulations=args.simulations)
    results = predictor.run()

    print(f"\n--- Prediction for {first} {last} ---")
    print(f"  Predicted points : {results['predicted_pts']}")
    print(f"  Std deviation    : {results['std']}")
    print(f"  80% range        : {results['low_80']} – {results['high_80']} pts")


if __name__ == "__main__":
    main()
