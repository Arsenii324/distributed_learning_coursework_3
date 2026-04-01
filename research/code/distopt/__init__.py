"""Decentralized optimization experiment harness (quadratic focus)."""

from .graphs import Graph, GraphStats, metropolis_mixing_matrix
from .problems import DistributedQuadraticProblem, QuadraticStats
from .runner import (
    ExperimentResult,
    MaxIters,
    MaxMixRounds,
    StopCondition,
    TargetAvgSqDistToXStarAllNodes,
    TargetObjectiveGap,
    TargetXStarDist,
    run_experiment,
)

__all__ = [
    "Graph",
    "GraphStats",
    "metropolis_mixing_matrix",
    "DistributedQuadraticProblem",
    "QuadraticStats",
    "ExperimentResult",
    "StopCondition",
    "MaxIters",
    "MaxMixRounds",
    "TargetXStarDist",
    "TargetAvgSqDistToXStarAllNodes",
    "TargetObjectiveGap",
    "run_experiment",
]
