from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ..oracles import Counters, CostedOracles
from ..problems import DistributedQuadraticProblem


class Algorithm(Protocol):
    name: str

    def check(self, problem: DistributedQuadraticProblem) -> None: ...

    def init_state(
        self,
        problem: DistributedQuadraticProblem,
        *,
        X0: np.ndarray | None = None,
        seed: int | None = None,
    ) -> Any: ...

    def step(self, problem: DistributedQuadraticProblem, state: Any) -> Any: ...

    def diagnostics(self, problem: DistributedQuadraticProblem, state: Any) -> dict[str, float]: ...


def init_X0(problem: DistributedQuadraticProblem, *, X0: np.ndarray | None, seed: int | None) -> np.ndarray:
    if X0 is None:
        return np.zeros((problem.n, problem.d), dtype=np.float64)
    X0 = np.asarray(X0, dtype=np.float64)
    if X0.shape != (problem.n, problem.d):
        raise ValueError(f"X0 must have shape (n,d)=({problem.n},{problem.d}); got {X0.shape}")
    return X0


@dataclass
class BaseState:
    t: int
    X: np.ndarray
    counters: Counters

    @property
    def x_bar(self) -> np.ndarray:
        return self.X.mean(axis=0)

    def oracles(self, problem: DistributedQuadraticProblem) -> CostedOracles:
        return CostedOracles(problem, self.counters)
