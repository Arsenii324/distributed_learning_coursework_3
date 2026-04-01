from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..oracles import Counters, CostedOracles
from ..problems import DistributedQuadraticProblem
from .base import init_X0


@dataclass
class GradientTrackingState:
    t: int
    X: np.ndarray
    Y: np.ndarray
    G: np.ndarray
    counters: Counters

    @property
    def x_bar(self) -> np.ndarray:
        return self.X.mean(axis=0)

    def oracles(self, problem: DistributedQuadraticProblem) -> CostedOracles:
        return CostedOracles(problem, self.counters)


@dataclass
class GradientTracking:
    """Gradient tracking (DIGing-style).

    One common form:
        X^{k+1} = W X^k - α Y^k
        Y^{k+1} = W Y^k + ∇F(X^{k+1}) - ∇F(X^k)

    Costs per iteration in this implementation:
    - 2 mixing rounds (mix X and mix Y)
    - 1 local gradient evaluation per node (at X^{k+1})
    """

    alpha: float
    name: str = "GradientTracking"

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
    ) -> GradientTrackingState:
        _ = seed
        X_init = init_X0(problem, X0=X0, seed=seed)
        counters = Counters()
        orc = CostedOracles(problem, counters)
        G0 = orc.local_grad(X_init)
        Y0 = G0.copy()
        return GradientTrackingState(t=0, X=X_init, Y=Y0, G=G0, counters=counters)

    def step(
        self, problem: DistributedQuadraticProblem, state: GradientTrackingState
    ) -> GradientTrackingState:
        orc = state.oracles(problem)

        WX = orc.mix(state.X)
        WY = orc.mix(state.Y)

        X_next = WX - self.alpha * state.Y
        G_next = orc.local_grad(X_next)
        Y_next = WY + (G_next - state.G)

        return GradientTrackingState(
            t=state.t + 1,
            X=X_next,
            Y=Y_next,
            G=G_next,
            counters=state.counters,
        )

    def diagnostics(
        self, problem: DistributedQuadraticProblem, state: GradientTrackingState
    ) -> dict[str, float]:
        _ = problem, state
        return {"alpha": float(self.alpha)}
