import unittest

import numpy as np

from research.code.distopt.generators import path_adjacency, make_graph_from_adjacency, make_random_spd_problem
from research.code.distopt.metrics import default_metrics
from research.code.distopt.oracles import Counters
from research.code.distopt.runner import TargetAvgSqDistToXStarAllNodes


class _DummyState:
    def __init__(self, X: np.ndarray):
        self.X = np.asarray(X, dtype=np.float64)
        self.counters = Counters()
        self.t = 0

    @property
    def x_bar(self) -> np.ndarray:
        return self.X.mean(axis=0)


class _DummyAlg:
    pass


class TestMetricsAndStop(unittest.TestCase):
    def test_avg_sq_dist_metric_identity(self) -> None:
        adj = path_adjacency(5)
        graph = make_graph_from_adjacency(adj, lazy=0.5)
        problem = make_random_spd_problem(graph, d=3, mu=1.0, L=5.0, seed=0)

        rng = np.random.default_rng(0)
        X = rng.normal(size=(problem.n, problem.d))

        state = _DummyState(X)
        row = default_metrics(problem, _DummyAlg(), state)

        # Direct MATLAB-style residual: (1/n)||X - 1 x*^T||_F^2
        direct = float(np.linalg.norm(X - problem.x_star[None, :], ord="fro") ** 2 / float(problem.n))

        self.assertIn("avg_sq_dist_to_x_star_all_nodes", row)
        self.assertAlmostEqual(row["avg_sq_dist_to_x_star_all_nodes"], direct, places=10)

    def test_stop_condition_matches_metric(self) -> None:
        adj = path_adjacency(4)
        graph = make_graph_from_adjacency(adj, lazy=0.5)
        problem = make_random_spd_problem(graph, d=2, mu=1.0, L=3.0, seed=1)

        # Exact optimum at all nodes => residual 0.
        X_opt = np.repeat(problem.x_star[None, :], repeats=problem.n, axis=0)
        state = _DummyState(X_opt)

        stop = TargetAvgSqDistToXStarAllNodes(tol=0.0)
        self.assertTrue(stop.should_stop(problem, _DummyAlg(), state))

        stop2 = TargetAvgSqDistToXStarAllNodes(tol=1e-16)
        self.assertTrue(stop2.should_stop(problem, _DummyAlg(), state))


if __name__ == "__main__":
    unittest.main()
