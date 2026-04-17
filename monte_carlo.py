"""
Monte Carlo Search Tree for NBA points prediction via weight optimization.

Training phase:
    The tree searches over combinations of 5 feature weights (summing to 100)
    to find the equation that minimizes MAE on the training set (full career
    minus last 20 games).

Test phase:
    The optimal weights are applied to each of the last 20 games to produce
    a predicted score, which is then compared against the actual score.

Prediction equation:
    pts = (w1*ROLLING_AVG_PTS + w2*CAREER_AVG_PTS + w3*RECENT_FORM
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
WEIGHT_STEP = 5
WEIGHT_SUM = 100


def _random_weights() -> list[int]:
    """Generate a random valid weight combination summing to WEIGHT_SUM."""
    cuts = sorted(random.sample(range(0, WEIGHT_SUM + 1, WEIGHT_STEP), N_FEATURES - 1))
    cuts = [0] + cuts + [WEIGHT_SUM]
    return [cuts[i + 1] - cuts[i] for i in range(N_FEATURES)]


def _perturb_weights(weights: list[int], n: int = 8) -> list[list[int]]:
    """Generate n neighboring weight combinations by moving WEIGHT_STEP units between features."""
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
    Score a weight combination against a dataset.
    Returns MAE between weighted prediction and actual PTS.
    Lower is better.
    """
    df = dataset.dropna(subset=FEATURES + ["PTS"])
    if df.empty:
        return float("inf")
    w = np.array(weights) / WEIGHT_SUM
    predicted = df[FEATURES].values @ w
    return float(np.mean(np.abs(predicted - df["PTS"].values)))


def _predict_single(weights: list[int], features: dict) -> float:
    """Apply weights to a single game's feature dict and return predicted points."""
    w = np.array(weights) / WEIGHT_SUM
    values = np.array([features[f] for f in FEATURES])
    return round(float(w @ values), 1)


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
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent else self.visits
        return -self.avg_error + exploration * math.sqrt(math.log(parent_visits) / self.visits)

    def best_child(self) -> "Node":
        return max(self.children, key=lambda c: c.uct_score())

    def is_leaf(self) -> bool:
        return len(self.children) == 0


class MonteCarloPredictor:
    """
    Monte Carlo Search Tree that finds the optimal feature weighting equation.

    Parameters
    ----------
    train : pd.DataFrame
        Training set — all career games except the last TEST_GAMES.
        Must contain the 5 FEATURES columns and PTS.
    simulations : int
        Number of MCST iterations for weight optimization.
    branching_factor : int
        Child nodes generated per expansion.
    """

    def __init__(self, train: pd.DataFrame, simulations: int = 500,
                 branching_factor: int = 8):
        self.train = train
        self.simulations = simulations
        self.branching_factor = branching_factor
        equal = [WEIGHT_SUM // N_FEATURES] * N_FEATURES
        self.root = Node(weights=equal)
        self.optimal_weights: list[int] | None = None

    def _select(self) -> Node:
        node = self.root
        while not node.is_leaf():
            node = node.best_child()
        return node

    def _expand(self, node: Node):
        for w in _perturb_weights(node.weights, self.branching_factor):
            node.children.append(Node(weights=w, parent=node))

    def _backpropagate(self, node: Node, error: float):
        current = node
        while current is not None:
            current.visits += 1
            current.total_error += error
            current = current.parent

    def train_weights(self) -> dict:
        """
        Run the MCST to find optimal feature weights on the training set.

        Returns
        -------
        dict with keys:
            optimal_weights  — list of 5 ints summing to 100
            weight_map       — {feature: weight}
            train_mae        — MAE of optimal weights on training data
        """
        best_node = self.root
        best_error = float("inf")

        for _ in range(self.simulations):
            node = self._select()

            if node.visits > 0 or node is self.root:
                self._expand(node)
                if node.children:
                    node = random.choice(node.children)

            error = _evaluate(node.weights, self.train)

            if error < best_error:
                best_error = error
                best_node = node

            self._backpropagate(node, error)

        self.optimal_weights = best_node.weights
        return {
            "optimal_weights": self.optimal_weights,
            "weight_map": dict(zip(FEATURES, self.optimal_weights)),
            "train_mae": round(best_error, 3),
        }

    def evaluate_test(self, test: pd.DataFrame) -> dict:
        """
        Apply the optimal weights to the test set (last 20 games) and
        report game-by-game predictions vs actuals.

        Parameters
        ----------
        test : pd.DataFrame
            The held-out test games from build_career_dataset().

        Returns
        -------
        dict with keys:
            results     — DataFrame with GAME_DATE, OPP, ACTUAL, PREDICTED, ERROR
            test_mae    — mean absolute error on the test set
            test_rmse   — root mean squared error on the test set
            within_5    — % of predictions within 5 points of actual
            within_10   — % of predictions within 10 points of actual
        """
        if self.optimal_weights is None:
            raise RuntimeError("Call train_weights() before evaluate_test().")

        df = test.dropna(subset=FEATURES + ["PTS"]).copy()
        w = np.array(self.optimal_weights) / WEIGHT_SUM
        df["PREDICTED"] = (df[FEATURES].values @ w).round(1)
        df["ACTUAL"] = df["PTS"]
        df["ERROR"] = (df["PREDICTED"] - df["ACTUAL"]).round(1)
        df["ABS_ERROR"] = df["ERROR"].abs()

        results = df[["GAME_DATE", "SEASON", "OPP", "ACTUAL", "PREDICTED", "ERROR"]].copy()
        errors = df["ABS_ERROR"].values

        return {
            "results": results.reset_index(drop=True),
            "test_mae": round(float(np.mean(errors)), 3),
            "test_rmse": round(float(np.sqrt(np.mean(errors ** 2))), 3),
            "within_5": round(float(np.mean(errors <= 5) * 100), 1),
            "within_10": round(float(np.mean(errors <= 10) * 100), 1),
        }


if __name__ == "__main__":
    # Smoke test with synthetic career-length data
    rng = np.random.default_rng(42)
    n = 800
    rolling = rng.normal(27, 3, n).clip(10)
    fake_data = pd.DataFrame({
        "PTS":             rng.normal(27, 7, n).clip(0).round(1),
        "ROLLING_AVG_PTS": rolling.round(2),
        "CAREER_AVG_PTS":  np.linspace(20, 27, n).round(2),
        "RECENT_FORM":     rng.normal(27, 5, n).clip(0).round(2),
        "VS_OPP_AVG":      rng.normal(25, 6, n).clip(0).round(2),
        "OPP_ADJ":         (rolling * rng.uniform(0.9, 1.1, n)).round(2),
        "GAME_DATE":       pd.date_range("2010-01-01", periods=n, freq="3D"),
        "SEASON":          ["2010-11"] * 400 + ["2011-12"] * 400,
        "OPP":             rng.choice(["MIA", "LAL", "GSW", "BOS", "CHI"], n),
    })

    train, test = fake_data.iloc[:-20], fake_data.iloc[-20:]

    predictor = MonteCarloPredictor(train, simulations=300)
    train_results = predictor.train_weights()
    test_results = predictor.evaluate_test(test)

    print("Optimal weights:")
    for feat, w in train_results["weight_map"].items():
        print(f"  {feat}: {w}%")
    print(f"Train MAE: {train_results['train_mae']} pts")
    print(f"\nTest MAE:      {test_results['test_mae']} pts")
    print(f"Test RMSE:     {test_results['test_rmse']} pts")
    print(f"Within 5 pts:  {test_results['within_5']}%")
    print(f"Within 10 pts: {test_results['within_10']}%")
    print(f"\n{test_results['results'].to_string()}")
