"""
Monte Carlo Search Tree for NBA player points prediction.

Each simulation samples a points outcome for an upcoming game based on:
  - Player's historical game log
  - Opponent defensive strength
  - Career and rolling season averages

After N simulations, the tree aggregates results to produce
a predicted score distribution and a single point estimate.
"""

import math
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Node:
    """A node in the Monte Carlo Search Tree."""
    points: float                        # Simulated points outcome for this node
    visits: int = 0
    total_score: float = 0.0
    parent: Optional["Node"] = None
    children: list = field(default_factory=list)

    @property
    def value(self) -> float:
        """Average simulated score at this node."""
        return self.total_score / self.visits if self.visits > 0 else 0.0

    def uct_score(self, exploration: float = 1.41) -> float:
        """Upper Confidence Bound score used for node selection."""
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent else self.visits
        return self.value + exploration * math.sqrt(math.log(parent_visits) / self.visits)

    def best_child(self) -> "Node":
        return max(self.children, key=lambda c: c.uct_score())

    def is_leaf(self) -> bool:
        return len(self.children) == 0


class MonteCarloPredictor:
    """
    Monte Carlo Search Tree that simulates possible scoring outcomes
    for a player in an upcoming game.

    Parameters
    ----------
    game_log : pd.DataFrame
        Historical game log produced by get_stats.build_dataset().
        Must contain a 'PTS' column and optionally 'OPP_PTS_ALLOWED'.
    simulations : int
        Number of Monte Carlo simulations to run.
    branching_factor : int
        Number of candidate outcomes to branch from each node.
    """

    def __init__(self, game_log: pd.DataFrame, simulations: int = 1000,
                 branching_factor: int = 10):
        self.game_log = game_log
        self.simulations = simulations
        self.branching_factor = branching_factor

        self.historical_pts = game_log["PTS"].dropna().tolist()
        self.mean_pts = float(np.mean(self.historical_pts))
        self.std_pts = float(np.std(self.historical_pts))

        # Defensive adjustment: if available, shift distribution by how the
        # opponent's defense compares to league average (~112 pts allowed/game)
        self.def_adjustment = 0.0
        if "OPP_PTS_ALLOWED" in game_log.columns:
            opp_avg = game_log["OPP_PTS_ALLOWED"].dropna().mean()
            league_avg = 112.0
            self.def_adjustment = (opp_avg - league_avg) * 0.05

        self.root = Node(points=self.mean_pts)

    def _sample_outcome(self) -> float:
        """Sample a plausible points outcome using a normal distribution
        fitted to the player's historical scoring."""
        raw = random.gauss(self.mean_pts + self.def_adjustment, self.std_pts)
        return max(0.0, round(raw, 1))

    def _expand(self, node: Node):
        """Add candidate child nodes branching from the given node."""
        for _ in range(self.branching_factor):
            child = Node(points=self._sample_outcome(), parent=node)
            node.children.append(child)

    def _simulate(self, node: Node) -> float:
        """Run a random rollout from a node and return the simulated score."""
        return self._sample_outcome()

    def _backpropagate(self, node: Node, score: float):
        """Propagate the simulation result back up the tree."""
        current = node
        while current is not None:
            current.visits += 1
            current.total_score += score
            current = current.parent

    def run(self) -> dict:
        """
        Execute the MCST search.

        Returns
        -------
        dict with keys:
            predicted_pts   — single point estimate (mean of all simulations)
            std             — standard deviation of simulated outcomes
            low_80          — 10th percentile outcome
            high_80         — 90th percentile outcome
            all_outcomes    — list of every simulated score
        """
        all_outcomes = []

        for _ in range(self.simulations):
            # Selection
            node = self.root
            while not node.is_leaf():
                node = node.best_child()

            # Expansion
            if node.visits > 0:
                self._expand(node)
                if node.children:
                    node = random.choice(node.children)

            # Simulation
            score = self._simulate(node)
            all_outcomes.append(score)

            # Backpropagation
            self._backpropagate(node, score)

        outcomes = np.array(all_outcomes)
        return {
            "predicted_pts": round(float(np.mean(outcomes)), 1),
            "std": round(float(np.std(outcomes)), 2),
            "low_80": round(float(np.percentile(outcomes, 10)), 1),
            "high_80": round(float(np.percentile(outcomes, 90)), 1),
            "all_outcomes": all_outcomes,
        }


if __name__ == "__main__":
    # Quick smoke test with synthetic data
    rng = np.random.default_rng(42)
    fake_log = pd.DataFrame({
        "PTS": rng.normal(27, 7, 60).clip(0).round(1),
        "OPP_PTS_ALLOWED": rng.normal(112, 4, 60),
    })

    predictor = MonteCarloPredictor(fake_log, simulations=500)
    results = predictor.run()
    print("Prediction results:")
    for k, v in results.items():
        if k != "all_outcomes":
            print(f"  {k}: {v}")
