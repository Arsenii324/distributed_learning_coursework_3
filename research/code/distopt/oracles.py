from __future__ import annotations

"""Cost accounting utilities.

This repo’s experiments compare algorithms primarily by *logical costs*, not
wall-clock time:

- `mix_rounds`: number of applications of the mixing operator `W`
- `grad_evals_per_node`: number of local gradient evaluations per node

`CostedOracles` wraps the problem oracles and increments these counters.
Algorithms are expected to use `CostedOracles` rather than calling `W @ X` or
`problem.local_grad(X)` directly, to keep accounting consistent.
"""

from dataclasses import dataclass

import numpy as np

from .problems import DistributedQuadraticProblem


@dataclass
class Counters:
    mix_rounds: int = 0
    grad_evals_per_node: int = 0


class CostedOracles:
    """Wrap problem oracles while incrementing counters.

    Convention:
    - One call to `mix` counts as one mixing/communication round.
    - One call to `local_grad` counts as one local gradient evaluation per node.
    """

    def __init__(self, problem: DistributedQuadraticProblem, counters: Counters):
        self.problem = problem
        self.counters = counters

    def mix(self, X: np.ndarray) -> np.ndarray:
        self.counters.mix_rounds += 1
        return self.problem.graph.mix(X)

    def local_grad(self, X: np.ndarray) -> np.ndarray:
        self.counters.grad_evals_per_node += 1
        return self.problem.local_grad(X)
