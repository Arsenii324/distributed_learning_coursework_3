from __future__ import annotations

"""Metric functions used by the experiment runner.

Metrics are intended to be:
- scalar (easy to store as a history table)
- derived from `(problem, algorithm, state)`

The runner logs:
1) `default_metrics`
2) user-provided `metric_fns`
3) `algorithm.diagnostics()`

This keeps static quantities (graph/problem stats) separate from dynamic traces
like objective gap or consensus error.
"""

from typing import Any, Callable

import numpy as np

from .problems import DistributedQuadraticProblem


MetricFn = Callable[[DistributedQuadraticProblem, Any, Any], dict[str, float]]


def default_metrics(problem: DistributedQuadraticProblem, algorithm: Any, state: Any) -> dict[str, float]:
    """Default dynamic metrics computed from (problem, state)."""

    _ = algorithm

    X = np.asarray(state.X)
    x_bar = np.asarray(state.x_bar)

    counters = state.counters

    dist_to_x_star = float(np.linalg.norm(x_bar - problem.x_star))

    f_x = problem.global_value(x_bar)
    f_star = problem.global_value(problem.x_star)
    global_gap = float(f_x - f_star)

    consensus_error = float(np.linalg.norm(X - x_bar[None, :], ord="fro"))

    # MATLAB-style residual used in archived MUDAG code:
    #   (1/n) ||X - 1 x*^T||_F^2
    # Using the identity:
    #   (1/n)||X-1x*^T||_F^2 = ||x̄-x*||_2^2 + (1/n)||X-1x̄^T||_F^2.
    avg_sq_dist_to_x_star_all_nodes = float(dist_to_x_star**2 + (consensus_error**2) / float(problem.n))

    return {
        "t": float(state.t),
        "mix_rounds": float(counters.mix_rounds),
        "grad_evals_per_node": float(counters.grad_evals_per_node),
        "dist_to_x_star": dist_to_x_star,
        "avg_sq_dist_to_x_star_all_nodes": avg_sq_dist_to_x_star_all_nodes,
        "objective_gap": global_gap,
        "consensus_error": consensus_error,
    }
