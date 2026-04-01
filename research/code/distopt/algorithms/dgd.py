from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..oracles import Counters
from ..problems import DistributedQuadraticProblem
from .base import BaseState, init_X0


@dataclass
class DGD:
    """Decentralized gradient descent: X^{k+1} = W X^k - α ∇F(X^k)."""

    alpha: float
    name: str = "DGD"

    def check(self, problem: DistributedQuadraticProblem) -> None:
        if not (self.alpha > 0):
            raise ValueError("alpha must be positive")
        problem.graph.ensure_stats()
        problem.ensure_stats()

    def init_state(
        self,
        problem: DistributedQuadraticProblem,
        *,
        X0: np.ndarray | None = None,
        seed: int | None = None,
    ) -> BaseState:
        _ = seed
        X_init = init_X0(problem, X0=X0, seed=seed)
        return BaseState(t=0, X=X_init, counters=Counters())

    def step(self, problem: DistributedQuadraticProblem, state: BaseState) -> BaseState:
        orc = state.oracles(problem)
        G = orc.local_grad(state.X)
        X_mix = orc.mix(state.X)
        X_next = X_mix - self.alpha * G
        return BaseState(t=state.t + 1, X=X_next, counters=state.counters)

    def diagnostics(self, problem: DistributedQuadraticProblem, state: BaseState) -> dict[str, float]:
        _ = problem, state
        return {"alpha": float(self.alpha)}
