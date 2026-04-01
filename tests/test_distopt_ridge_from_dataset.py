import unittest

import numpy as np

from research.code.distopt.generators import complete_adjacency, make_graph_from_adjacency


class TestRidgeFromDataset(unittest.TestCase):
    def test_make_problem_shapes_and_spd(self):
        try:
            from scipy import sparse
        except Exception:
            self.skipTest("SciPy not installed")

        from research.code.distopt.datasets.ridge import make_distributed_ridge_problem_from_dataset

        # Tiny dataset: 4 samples, 3 features.
        X = np.array(
            [
                [1.0, 0.0, 2.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        y = np.array([1.0, -1.0, 2.0, 0.5])
        X = sparse.csr_matrix(X)

        m = 2
        graph = make_graph_from_adjacency(complete_adjacency(m))

        problem = make_distributed_ridge_problem_from_dataset(
            graph,
            X,
            y,
            m_agents=m,
            lambda_reg=0.1,
            partition="contiguous",
            seed=0,
            dataset_name="tiny",
        )

        self.assertEqual(problem.A.shape, (m, 3, 3))
        self.assertEqual(problem.b.shape, (m, 3))

        stats = problem.ensure_stats()
        self.assertGreater(stats.mu_g, 0.0)
        self.assertGreater(stats.L_g, 0.0)

        # x_star should satisfy A_bar x = b_bar.
        x_star = problem.x_star
        self.assertTrue(np.allclose(problem.A_bar @ x_star, problem.b_bar))


if __name__ == "__main__":
    unittest.main()
