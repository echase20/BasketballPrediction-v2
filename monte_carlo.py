"""
Monte Carlo Search Tree for NBA points prediction via weight optimization.

The tree searches over combinations of 5 feature weights (summing to 100)
to find the weighting equation that best predicts a player's historical
scoring. The winning weights are then applied to next-game features
to produce a prediction.

Features (5):
    1. ROLLING_AVG_PTS  — season rolling average
    2. CAREER_AVG_PTS   — career PPG
    3. RECENT_FORM      — last 5 games average
    4. VS_OPP_AVG       — historical avg vs next opponent
    5. OPP_ADJ          — rolling avg scaled by opponent defensive strength

Prediction equation:
    pts = (w1*ROLLING_AVG + w2*CAREER_AVG + w3*RECENT_FORM
           + w4*VS_OPP_AVG + w5*OPP_ADJ) / 100
"""

import math
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

FEATURES = ["ROLLING_AVG_PTS", "CAREER_AVG_PTS", "RECENT_FORM", "VS_OPP_AVG", "OPP_ADJ"]
N_FEATURES = len(FEATURES)
WEIGHT_STEP = 5      # weights move in increments of 5
WEIGHT_SUM = 100


def _random_weights() -> list[int]:
    """Generate a random valid weight combination summing to WEIGHT_SUM."""
    cuts = sorted(random.sample(range(0, WEIGHT_SUM + 1, WEIGHT_STEP),
                                N_FEATURES - 1))
    cuts = [0] + cuts + [WEIGHT_SUM]
    return [cuts[i + 1] - cuts[i] for i in range(N_FEATURES)]


def _perturb_weights(weights: list[int], n: int = 8) -> list[list[int]]:
    """
    Generate n neighboring weight combinations by transferring
    WEIGHT_STEP units from one feature to another.
    """
    neighbors = []
    attempts = 0
    while len(neighbors) < n and attempts < n * 10:
        attempts += 1
        w = weights.copy()
        i, j = random.sample(range(N_FEATURES), 2)
        amount = random.choice([WEIGHT_STEP, WEIGHT_STEP * 2])
        if w[i] >= amount:
            w[i] -= amount
            w[j] += amount
            if w not in neighbors:
                neighbors.append(w)
    return neighbors


def _evaluate(weights: list[int], dataset: pd.DataFrame) -> float:
    """
    Score a weight combination by computing the weighted prediction
    for each historical game and returning the Mean Absolute Error
    against actual points scored.

    Lower MAE = better weights.
    """
    df = dataset.dropna(subset=FEATURES + ["PTS"])
    if df.empty:
        return float("inf")

    feature_matrix = df[FEATURES].values          # shape (n_games, 5)
    w = np.array(weights) / WEIGHT_SUM            # normalize to sum=1
    predicted = feature_matrix @ w                 # dot product per game
    actual = df["PTS"].values
    return float(np.mean(np.abs(predicted - actual)))


@dataclass
class Node:
    weights: list
    parent: Optional["Node"] = None
    children: list = field(default_factory=list)
    visits: int = 0
    total_error: float = 0.0

    @property
    def avg_error(self) -> float:
        return self.total_error / self.visits if self.visits > 0 else float("inf")

    def uct_score(self, exploration: float = 1.41) -> float:
        """
        UCT adapted for minimization: lower error is better,
        so we negate avg_error for the exploitation term.
        """
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent else self.visits
        exploitation = -self.avg_error
        exploration_bonus = exploration * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration_bonus

    def best_child(self) -> "Node":
        return max(self.children, key=lambda c: c.uct_score())

    def is_leaf(self) -> bool:
        return len(self.children) == 0


class MonteCarloPredictor:
    """
    Monte Carlo Search Tree that finds the optimal feature weighting
    equation to predict NBA player points.

    Parameters
    ----------
    dataset : pd.DataFrame
        Historical game log with all 5 feature columns and actual PTS.
        Produced by get_stats.build_dataset().
    simulations : int
        Number of MCST iterations to run.
    branching_factor : int
        Number of weight perturbations to branch from each node.
    """

    def __init__(self, dataset: pd.DataFrame, simulations: int = 500,
                 branching_factor: int = 8):
        self.dataset = dataset
        self.simulations = simulations
        self.branching_factor = branching_factor

        equal = [WEIGHT_SUM // N_FEATURES] * N_FEATURES
        self.root = Node(weights=equal)

    def _select(self) -> "Node":
        node = self.root
        while not node.is_leaf():
            node = node.best_child()
        return node

    def _expand(self, node: Node):
        for w in _perturb_weights(node.weights, self.branching_factor):
            child = Node(weights=w, parent=node)
            node.children.append(child)

    def _simulate(self, node: Node) -> float:
        """Rollout: evaluate a random neighbor of this node's weights."""
        random_weights = _random_weights()
        return _evaluate(random_weights, self.dataset)

    def _backpropagate(self, node: Node, error: float):
        current = node
        while current is not None:
            current.visits += 1
            current.total_error += error
            current = current.parent

    def run(self) -> dict:
        """
        Run the MCST weight search.

        Returns
        -------
        dict with keys:
            optimal_weights  — list of 5 ints summing to 100
            feature_names    — the 5 feature labels
            mae              — historical MAE of the optimal weights
            weight_map       — {feature: weight} dict for readability
        """
        best_node = self.root
        best_error = float("inf")

        for _ in range(self.simulations):
            # Selection
            node = self._select()

            # Expansion
            if node.visits > 0 or node is self.root:
                self._expand(node)
                if node.children:
                    node = random.choice(node.children)

            # Simulation (evaluate this node's weights directly)
            error = _evaluate(node.weights, self.dataset)

            # Track global best
            if error < best_error:
                best_error = error
                best_node = node

            # Backpropagation
            self._backpropagate(node, error)

        optimal = best_node.weights
        return {
            "optimal_weights": optimal,
            "feature_names": FEATURES,
            "mae": round(best_error, 3),
            "weight_map": dict(zip(FEATURES, optimal)),
        }

    def predict(self, next_game_features: dict, optimal_weights: list[int]) -> float:
        """
        Apply the optimal weights to next-game features to produce a prediction.

        Parameters
        ----------
        next_game_features : dict
            Feature values for the upcoming game (from get_stats.get_next_game_features).
        optimal_weights : list[int]
            Weight combination found by run().
        """
        w = np.array(optimal_weights) / WEIGHT_SUM
        feature_values = np.array([next_game_features[f] for f in FEATURES])
        return round(float(w @ feature_values), 1)


if __name__ == "__main__":
    # Smoke test with synthetic data
    rng = np.random.default_rng(42)
    n = 60
    rolling = rng.normal(27, 3, n).clip(10)
    fake_data = pd.DataFrame({
        "PTS":             rng.normal(27, 7, n).clip(0).round(1),
        "ROLLING_AVG_PTS": rolling.round(2),
        "CAREER_AVG_PTS":  np.full(n, 26.5),
        "RECENT_FORM":     rng.normal(27, 5, n).clip(0).round(2),
        "VS_OPP_AVG":      rng.normal(25, 6, n).clip(0).round(2),
        "OPP_ADJ":         (rolling * rng.uniform(0.9, 1.1, n)).round(2),
    })

    predictor = MonteCarloPredictor(fake_data, simulations=300)
    results = predictor.run()

    print("Optimal weight equation:")
    for feat, w in results["weight_map"].items():
        print(f"  {feat}: {w}%")
    print(f"Historical MAE: {results['mae']} pts")

    # Simulate a next-game prediction
    next_game = {
        "ROLLING_AVG_PTS": 27.3,
        "CAREER_AVG_PTS":  26.5,
        "RECENT_FORM":     29.1,
        "VS_OPP_AVG":      24.8,
        "OPP_ADJ":         26.0,
    }
    pred = predictor.predict(next_game, results["optimal_weights"])
    print(f"\nPredicted points: {pred}")
