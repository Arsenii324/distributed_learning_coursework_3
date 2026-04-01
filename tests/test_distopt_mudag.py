import unittest

import numpy as np

from research.code.distopt.algorithms import MUDAG
from research.code.distopt.generators import make_graph_from_adjacency, path_adjacency
from research.code.distopt.graphs import Graph
from research.code.distopt.problems import DistributedQuadraticProblem


def _make_identical_quadratic_problem(graph: Graph, *, d: int) -> DistributedQuadraticProblem:
    # Simple SPD quadratic shared across nodes.
    A0 = np.diag(np.linspace(1.0, 2.0, num=d))
    b0 = np.arange(1, d + 1, dtype=np.float64)

    A = np.repeat(A0[None, :, :], repeats=graph.n, axis=0)
    b = np.repeat(b0[None, :], repeats=graph.n, axis=0)

    return DistributedQuadraticProblem(graph=graph, A=A, b=b)


class TestMUDAG(unittest.TestCase):
    def test_requires_psd_W(self) -> None:
        # W with eigenvalues {1, -1} is symmetric + doubly-stochastic but not PSD.
        adj = np.array([[False, True], [True, False]], dtype=bool)
        W = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        graph = Graph(adj=adj, W=W, validate=True)
        problem = _make_identical_quadratic_problem(graph, d=2)

        alg = MUDAG(c_K=0.2)
        with self.assertRaises(ValueError):
            alg.check(problem)

    def test_consensus_init(self) -> None:
        adj = path_adjacency(5)
        graph = make_graph_from_adjacency(adj, lazy=0.5)
        problem = _make_identical_quadratic_problem(graph, d=3)

        rng = np.random.default_rng(0)
        X0 = rng.normal(size=(problem.n, problem.d))

        alg = MUDAG(c_K=0.2)
        alg.check(problem)
        state = alg.init_state(problem, X0=X0)

        # All rows should be identical.
        self.assertTrue(np.allclose(state.X, state.X[0][None, :], atol=1e-12, rtol=0.0))
        self.assertTrue(np.allclose(state.Y, state.X, atol=1e-12, rtol=0.0))
        self.assertTrue(np.allclose(state.Y_prev, state.X, atol=1e-12, rtol=0.0))

    def test_counters_increment_as_expected(self) -> None:
        adj = path_adjacency(6)
        graph = make_graph_from_adjacency(adj, lazy=0.5)
        problem = _make_identical_quadratic_problem(graph, d=2)

        alg = MUDAG(c_K=0.2)
        alg.check(problem)
        state0 = alg.init_state(problem)

        K = int(state0.K)
        self.assertGreaterEqual(K, 0)

        state1 = alg.step(problem, state0)
        self.assertEqual(int(state1.counters.mix_rounds), K + 1)
        self.assertEqual(int(state1.counters.grad_evals_per_node), 1)

        state2 = alg.step(problem, state1)
        self.assertEqual(int(state2.counters.mix_rounds), 2 * (K + 1))
        self.assertEqual(int(state2.counters.grad_evals_per_node), 2)

    def test_preserves_consensus_on_identical_objectives(self) -> None:
        adj = path_adjacency(7)
        graph = make_graph_from_adjacency(adj, lazy=0.5)
        problem = _make_identical_quadratic_problem(graph, d=4)

        alg = MUDAG(c_K=0.2)
        alg.check(problem)
        state = alg.init_state(problem)

        for _ in range(3):
            state = alg.step(problem, state)
            self.assertTrue(np.allclose(state.X, state.X[0][None, :], atol=1e-12, rtol=0.0))


if __name__ == "__main__":
    unittest.main()
