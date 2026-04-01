import unittest

import numpy as np

from research.code.distopt.accsonata_generators import RidgeSpec, make_accsonata_er_graph, make_accsonata_ridge_problem
from research.code.distopt.algorithms.acc_sonata import _chebyshev3_mix
from research.code.distopt.algorithms import AccSonataF, AccSonataL
from research.code.distopt.oracles import Counters, CostedOracles


class TestChebyshevCommCounting(unittest.TestCase):
    def test_chebyshev_counts_mix_rounds(self):
        n = 4
        from research.code.distopt.graphs import Graph
        from research.code.distopt.problems import DistributedQuadraticProblem

        W = np.ones((n, n)) / n
        adj = ~np.eye(n, dtype=bool)
        graph = Graph(adj=adj, W=W)
        d = 3
        A = np.stack([np.eye(d) for _ in range(n)], axis=0)
        b = np.zeros((n, d))
        problem = DistributedQuadraticProblem(graph=graph, A=A, b=b)

        counters = Counters()
        orc = CostedOracles(problem, counters)

        X = np.random.default_rng(0).standard_normal((n, d))
        steps = 7
        _ = _chebyshev3_mix(
            orc,
            X,
            steps,
            chi=float(problem.graph.ensure_stats().chi),
            lambda_max_L=float(problem.graph.ensure_stats().lambda_max_L),
        )

        self.assertEqual(counters.mix_rounds, steps)


class TestAccSonataSmoke(unittest.TestCase):
    def test_ridge_generator_no_spectral_scaling_changes_A(self):
        graph = make_accsonata_er_graph(6, 0.9, seed=7)
        spec = RidgeSpec(d=5, n_local=40, lambda_reg=0.1, mu0=1.0, L0=10.0, noise_std=0.1)

        prob_scaled = make_accsonata_ridge_problem(graph, spec, seed=11, no_spectral_scaling=False)
        prob_unscaled = make_accsonata_ridge_problem(graph, spec, seed=11, no_spectral_scaling=True)

        diff = float(np.linalg.norm(prob_scaled.A - prob_unscaled.A))
        self.assertGreater(diff, 0.0)

    def test_ridge_generator_paper_protocol_can_make_large_kappa(self):
        # Paper protocol should be able to generate globally ill-conditioned
        # problems when Sigma has eigenvalues spanning [mu0, L0] and lambda=0.
        graph = make_accsonata_er_graph(5, 0.9, seed=0)
        spec = RidgeSpec(d=10, n_local=4000, lambda_reg=0.0, mu0=1.0, L0=1000.0, noise_std=0.1)
        prob = make_accsonata_ridge_problem(
            graph,
            spec,
            seed=1,
            no_spectral_scaling=True,
            paper_protocol=True,
            x_true_mean=5.0,
        )
        stats = prob.ensure_stats()
        self.assertGreater(float(stats.kappa_g), 50.0)

    def test_accsonataF_smoke_step_counts(self):
        # Construct a tiny quadratic with guaranteed beta > mu_g so ACC-SONATA-F's
        # theory-driven defaults are valid.
        from research.code.distopt.graphs import Graph
        from research.code.distopt.problems import DistributedQuadraticProblem

        n, d = 5, 4
        W = np.ones((n, n)) / n
        adj = ~np.eye(n, dtype=bool)
        graph = Graph(adj=adj, W=W)

        A = np.zeros((n, d, d))
        for i in range(n):
            diag = np.ones(d)
            if i == 0:
                diag[-1] = 1.0
            else:
                diag[-1] = 1000.0
            A[i] = np.diag(diag)
        b = np.zeros((n, d))
        problem = DistributedQuadraticProblem(graph=graph, A=A, b=b)

        X0 = np.ones((n, d))
        alg = AccSonataF(inner_comm_budget=2, cheb_steps=2, eps_stop=1e-30)
        alg.check(problem)
        state = alg.init_state(problem, X0=X0, seed=0)

        prev_mix = state.counters.mix_rounds
        prev_grad = state.counters.grad_evals_per_node

        state2 = alg.step(problem, state)

        # NEXT_F: per inner loop iteration does 1 chebyshev mix for X and 1 for Y -> 2*cheb_steps comm.
        # With comm_budget=2 and cheb_steps=2, loop runs once (2 comm) for X, then once (2 comm) for Y.
        self.assertGreaterEqual(state2.counters.mix_rounds - prev_mix, 4)
        self.assertGreaterEqual(state2.counters.grad_evals_per_node - prev_grad, 2)

    def test_accsonataL_smoke_step_runs(self):
        graph = make_accsonata_er_graph(5, 0.9, seed=2)
        problem = make_accsonata_ridge_problem(graph, RidgeSpec(d=4, n_local=30, lambda_reg=0.1), seed=3)

        alg = AccSonataL(inner_comm_budget=2, cheb_steps=2)
        alg.check(problem)
        state = alg.init_state(problem, seed=0)
        state2 = alg.step(problem, state)

        self.assertEqual(state2.X.shape, state.X.shape)
        self.assertGreaterEqual(state2.counters.mix_rounds, 1)
        self.assertGreaterEqual(state2.counters.grad_evals_per_node, 1)

    def test_accsonataL_early_stop_updates_Y(self):
        # Regression: if the embedded early-stop shortcut triggers, ACC-SONATA
        # must still return a consistent (X, Y) pair. Previously we returned
        # (X1, Y_old), which can cause the outer acceleration to drift away.
        graph = make_accsonata_er_graph(5, 0.9, seed=2)
        problem = make_accsonata_ridge_problem(graph, RidgeSpec(d=4, n_local=30, lambda_reg=0.1), seed=3)

        alg = AccSonataL(inner_comm_budget=2, cheb_steps=1, eps_stop=1e6)
        alg.check(problem)
        state = alg.init_state(problem, seed=0)
        state2 = alg.step(problem, state)

        self.assertGreater(float(np.linalg.norm(state2.Y - state.Y)), 0.0)


if __name__ == "__main__":
    unittest.main()
