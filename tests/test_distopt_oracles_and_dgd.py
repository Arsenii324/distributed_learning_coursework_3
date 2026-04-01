import unittest

import numpy as np

from research.code.distopt.algorithms import DGD
from research.code.distopt.generators import make_graph_from_adjacency, path_adjacency
from research.code.distopt.problems import DistributedQuadraticProblem


class TestOraclesAndDGD(unittest.TestCase):
    def test_x_star_solves_global_system(self) -> None:
        adj = path_adjacency(4)
        graph = make_graph_from_adjacency(adj, lazy=0.2)

        # Identical quadratic across nodes.
        d = 3
        A0 = np.diag(np.array([1.0, 2.0, 4.0]))
        b0 = np.array([1.0, 0.0, -1.0])

        A = np.repeat(A0[None, :, :], repeats=graph.n, axis=0)
        b = np.repeat(b0[None, :], repeats=graph.n, axis=0)

        problem = DistributedQuadraticProblem(graph=graph, A=A, b=b)

        # x_star should solve A_bar x = b_bar exactly.
        r = problem.A_bar @ problem.x_star - problem.b_bar
        self.assertTrue(np.linalg.norm(r) <= 1e-10)

    def test_dgd_matches_centralized_step_on_consensus_identical_objective(self) -> None:
        adj = path_adjacency(5)
        graph = make_graph_from_adjacency(adj, lazy=0.0)

        d = 2
        A0 = np.diag(np.array([2.0, 5.0]))
        b0 = np.array([1.0, -2.0])

        A = np.repeat(A0[None, :, :], repeats=graph.n, axis=0)
        b = np.repeat(b0[None, :], repeats=graph.n, axis=0)
        problem = DistributedQuadraticProblem(graph=graph, A=A, b=b)

        alpha = 0.1
        alg = DGD(alpha=alpha)
        alg.check(problem)

        x = np.array([0.3, -0.7])
        X0 = np.repeat(x[None, :], repeats=problem.n, axis=0)

        state0 = alg.init_state(problem, X0=X0)
        state1 = alg.step(problem, state0)

        # Since X0 is consensus and objectives identical, WX0 = X0 and grads identical.
        g = A0 @ x - b0
        x_expected = x - alpha * g
        self.assertTrue(np.allclose(state1.X, x_expected[None, :], atol=1e-12, rtol=0.0))

        # And counters should increment by one mix + one grad per step.
        self.assertEqual(int(state1.counters.mix_rounds), 1)
        self.assertEqual(int(state1.counters.grad_evals_per_node), 1)


if __name__ == "__main__":
    unittest.main()
