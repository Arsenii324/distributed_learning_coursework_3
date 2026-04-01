from __future__ import annotations

"""Experiment runner.

`run_experiment` executes an algorithm on a problem and logs a scalar history.

Design choices:
- Stops when *any* stop condition triggers (use a list for composite stopping).
- Logs at `t=0` and then every `log_every` iterations.
- Separates:
    - static stats (graph/problem) stored once
    - dynamic metrics (per-iteration)
    - algorithm diagnostics (per-iteration, provided by the algorithm)

The returned `ExperimentResult` is an in-memory object suitable for notebooks.
Persistence (JSON/Parquet) is intentionally left optional and can be added later
once the experiment schema is stabilized.
"""

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .graphs import GraphStats
from .metrics import MetricFn, default_metrics
from .problems import DistributedQuadraticProblem, QuadraticStats


class StopCondition(Protocol):
    def should_stop(self, problem: DistributedQuadraticProblem, algorithm: Any, state: Any) -> bool: ...

    def describe(self) -> str: ...


@dataclass(frozen=True)
class MaxIters:
    max_iters: int

    def should_stop(self, problem: DistributedQuadraticProblem, algorithm: Any, state: Any) -> bool:
        _ = problem, algorithm
        return int(state.t) >= int(self.max_iters)

    def describe(self) -> str:
        return f"MaxIters(max_iters={self.max_iters})"


@dataclass(frozen=True)
class MaxMixRounds:
    max_mix_rounds: int

    def should_stop(self, problem: DistributedQuadraticProblem, algorithm: Any, state: Any) -> bool:
        _ = problem, algorithm
        return int(state.counters.mix_rounds) >= int(self.max_mix_rounds)

    def describe(self) -> str:
        return f"MaxMixRounds(max_mix_rounds={self.max_mix_rounds})"


@dataclass(frozen=True)
class TargetXStarDist:
    tol: float

    def should_stop(self, problem: DistributedQuadraticProblem, algorithm: Any, state: Any) -> bool:
        _ = algorithm
        x_bar = np.asarray(state.x_bar)
        return float(np.linalg.norm(x_bar - problem.x_star)) <= float(self.tol)

    def describe(self) -> str:
        return f"TargetXStarDist(tol={self.tol})"


@dataclass(frozen=True)
class TargetObjectiveGap:
    tol: float

    def should_stop(self, problem: DistributedQuadraticProblem, algorithm: Any, state: Any) -> bool:
        _ = algorithm
        x_bar = np.asarray(state.x_bar)
        gap = problem.global_value(x_bar) - problem.global_value(problem.x_star)
        return float(gap) <= float(self.tol)

    def describe(self) -> str:
        return f"TargetObjectiveGap(tol={self.tol})"


@dataclass(frozen=True)
class TargetAvgSqDistToXStarAllNodes:
    """Stop when (1/n) ||X - 1 x*^T||_F^2 <= tol.

    This matches the archived MATLAB code’s `evaluate(x)` metric used in
    the MUDAG simulations.

    Note: `tol` is a *squared* error tolerance.
    """

    tol: float

    def should_stop(self, problem: DistributedQuadraticProblem, algorithm: Any, state: Any) -> bool:
        _ = algorithm
        X = np.asarray(state.X, dtype=np.float64)
        diff = X - problem.x_star[None, :]
        avg_sq = float(np.linalg.norm(diff, ord="fro") ** 2 / float(problem.n))
        return avg_sq <= float(self.tol)

    def describe(self) -> str:
        return f"TargetAvgSqDistToXStarAllNodes(tol={self.tol})"


@dataclass
class ExperimentResult:
    config: dict[str, Any]
    graph_stats: dict[str, Any]
    problem_stats: dict[str, Any]
    metadata: dict[str, Any]
    history: list[dict[str, float]]
    final: dict[str, Any]


def _summarize_graph_stats(stats: GraphStats) -> dict[str, Any]:
    return {
        "n": int(stats.degrees.shape[0]),
        "connected": bool(stats.connected),
        "degrees": stats.degrees.astype(int).tolist(),
        "lambda2_W": float(stats.lambda2_W),
        "gamma": float(stats.gamma),
        "chi": float(stats.chi),
        "lambda_min_pos_L": float(stats.lambda_min_pos_L),
        "lambda_max_L": float(stats.lambda_max_L),
    }


def _summarize_problem_stats(stats: QuadraticStats) -> dict[str, Any]:
    out: dict[str, Any] = {
        "L_i": stats.L_i.astype(float).tolist(),
        "mu_i": stats.mu_i.astype(float).tolist(),
        "L_l": float(stats.L_l),
        "mu_l": float(stats.mu_l),
        "L_g": float(stats.L_g),
        "mu_g": float(stats.mu_g),
        "kappa_l": float(stats.kappa_l),
        "kappa_g": float(stats.kappa_g),
        "beta": float(stats.beta),
    }
    if stats.delta is not None:
        out["delta"] = float(stats.delta)
    return out


def run_experiment(
    problem: DistributedQuadraticProblem,
    algorithm: Any,
    *,
    stop: StopCondition | list[StopCondition],
    X0: np.ndarray | None = None,
    seed: int | None = None,
    log_every: int = 1,
    metric_fns: list[MetricFn] | None = None,
) -> ExperimentResult:
    """Run an algorithm on a problem and return metrics history.

    Notes:
    - `stop` can be a single stop condition or a list; stopping occurs when *any*
      condition triggers.
    - Logging happens at t=0 and then every `log_every` iterations.
    """

    if log_every <= 0:
        raise ValueError("log_every must be a positive integer")

    stop_list = stop if isinstance(stop, list) else [stop]

    algorithm.check(problem)
    state = algorithm.init_state(problem, X0=X0, seed=seed)

    fns: list[MetricFn] = [default_metrics]
    if metric_fns:
        fns.extend(metric_fns)

    history: list[dict[str, float]] = []

    def log_row() -> dict[str, float]:
        row: dict[str, float] = {}
        for fn in fns:
            row.update(fn(problem, algorithm, state))
        row.update({k: float(v) for k, v in algorithm.diagnostics(problem, state).items()})
        history.append(row)
        return row

    # Static summaries
    graph_stats = _summarize_graph_stats(problem.graph.ensure_stats())
    problem_stats = _summarize_problem_stats(problem.ensure_stats())

    config = {
        "algorithm": getattr(algorithm, "name", algorithm.__class__.__name__),
        "stop": [s.describe() for s in stop_list],
        "log_every": int(log_every),
    }

    log_row()

    while True:
        if any(s.should_stop(problem, algorithm, state) for s in stop_list):
            break

        state = algorithm.step(problem, state)

        if int(state.t) % int(log_every) == 0:
            log_row()

    final = {
        "t": int(state.t),
        "mix_rounds": int(state.counters.mix_rounds),
        "grad_evals_per_node": int(state.counters.grad_evals_per_node),
        "x_bar": np.asarray(state.x_bar, dtype=np.float64),
    }

    # Add final dynamic metrics so callers don't need log_every=1 to reliably
    # know whether a target tolerance was reached.
    final_row = default_metrics(problem, algorithm, state)
    final.update(
        {
            "dist_to_x_star": float(final_row["dist_to_x_star"]),
            "avg_sq_dist_to_x_star_all_nodes": float(final_row["avg_sq_dist_to_x_star_all_nodes"]),
            "objective_gap": float(final_row["objective_gap"]),
            "consensus_error": float(final_row["consensus_error"]),
        }
    )

    return ExperimentResult(
        config=config,
        graph_stats=graph_stats,
        problem_stats=problem_stats,
        metadata=dict(problem.metadata),
        history=history,
        final=final,
    )
