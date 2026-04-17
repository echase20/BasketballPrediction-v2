"""
Entry point: finds the optimal feature weighting via MCST, then predicts
how many points a player will score against a specific opponent.

Usage:
    python predict.py --player "Jayson Tatum" --opponent MIA --season 2023-24
"""

import argparse
import sys
from get_stats import get_next_game_features
from monte_carlo import MonteCarloPredictor, FEATURES


def main():
    parser = argparse.ArgumentParser(
        description="Predict NBA player points using MCST weight optimization."
    )
    parser.add_argument("--player", required=True, help='Full name, e.g. "Jayson Tatum"')
    parser.add_argument("--opponent", required=True, help="Opponent team abbreviation, e.g. MIA")
    parser.add_argument("--season", default="2023-24", help="NBA season (default: 2023-24)")
    parser.add_argument("--simulations", type=int, default=500,
                        help="MCST iterations for weight search (default: 500)")
    args = parser.parse_args()

    name_parts = args.player.strip().split()
    if len(name_parts) < 2:
        print("Error: provide both first and last name.")
        sys.exit(1)

    first, last = name_parts[0], " ".join(name_parts[1:])

    print(f"\nFetching stats for {first} {last} ({args.season})...")
    features = get_next_game_features(first, last, args.opponent.upper(), args.season)
    dataset = features.pop("dataset")

    print(f"\nRunning MCST weight optimization ({args.simulations} simulations)...")
    predictor = MonteCarloPredictor(dataset, simulations=args.simulations)
    results = predictor.run()

    print(f"\n--- Optimal weight equation (MAE: {results['mae']} pts on historical data) ---")
    for feat, w in results["weight_map"].items():
        print(f"  {feat}: {w}%")

    prediction = predictor.predict(features, results["optimal_weights"])

    print(f"\n--- Prediction: {first} {last} vs {args.opponent.upper()} ---")
    for feat in FEATURES:
        print(f"  {feat}: {features[feat]}")
    print(f"\n  Predicted points: {prediction}")


if __name__ == "__main__":
    main()
