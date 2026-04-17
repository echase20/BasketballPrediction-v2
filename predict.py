"""
Entry point: trains the MCST weight optimizer on a player's full career
(minus the last 20 games), then tests accuracy on those last 20 games.

Usage:
    python predict.py --player "Jayson Tatum"
    python predict.py --player "LeBron James" --simulations 1000
"""

import argparse
import sys
from get_stats import build_career_dataset, get_next_game_features
from monte_carlo import MonteCarloPredictor, FEATURES


def main():
    parser = argparse.ArgumentParser(
        description="Train MCST weight optimizer on career data and test on last 20 games."
    )
    parser.add_argument("--player", required=True, help='Full name, e.g. "Jayson Tatum"')
    parser.add_argument("--simulations", type=int, default=500,
                        help="MCST iterations for weight search (default: 500)")
    parser.add_argument("--next-opponent", default=None,
                        help="Opponent abbreviation for next game prediction, e.g. MIA")
    args = parser.parse_args()

    name_parts = args.player.strip().split()
    if len(name_parts) < 2:
        print("Error: provide both first and last name.")
        sys.exit(1)
    first, last = name_parts[0], " ".join(name_parts[1:])

    # ── 1. Build dataset ──────────────────────────────────────────────────────
    print(f"\nBuilding career dataset for {first} {last}...")
    data = build_career_dataset(first, last)
    train, test = data["train"], data["test"]

    # ── 2. Train: find optimal weights on career minus last 20 games ──────────
    print(f"\nRunning MCST weight optimization ({args.simulations} simulations)...")
    predictor = MonteCarloPredictor(train, simulations=args.simulations)
    train_results = predictor.train_weights()

    print(f"\n--- Optimal weight equation (train MAE: {train_results['train_mae']} pts) ---")
    for feat, w in train_results["weight_map"].items():
        print(f"  {feat}: {w}%")

    # ── 3. Test: evaluate on last 20 games ────────────────────────────────────
    print(f"\n--- Test results: last 20 games ---")
    test_results = predictor.evaluate_test(test)

    results_df = test_results["results"]
    results_df["GAME_DATE"] = results_df["GAME_DATE"].dt.strftime("%Y-%m-%d")
    print(results_df.to_string(index=False))

    print(f"\n--- Accuracy summary ---")
    print(f"  MAE:          {test_results['test_mae']} pts")
    print(f"  RMSE:         {test_results['test_rmse']} pts")
    print(f"  Within 5 pts: {test_results['within_5']}%")
    print(f"  Within 10 pts:{test_results['within_10']}%")

    # ── 4. Next game prediction (optional) ────────────────────────────────────
    if args.next_opponent:
        next_opp = args.next_opponent.upper()
        print(f"\n--- Next game prediction: {first} {last} vs {next_opp} ---")
        next_features = get_next_game_features(data, next_opp)
        for feat, val in next_features.items():
            print(f"  {feat}: {val}")
        prediction = predictor.predict(next_features, train_results["optimal_weights"])
        print(f"\n  Predicted points: {prediction}")


if __name__ == "__main__":
    main()
