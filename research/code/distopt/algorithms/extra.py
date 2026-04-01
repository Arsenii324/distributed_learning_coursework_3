from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..oracles import Counters, CostedOracles
from ..problems import DistributedQuadraticProblem
from .base import init_X0


@dataclass
class ExtraState:
    t: int
    X: np.ndarray
    counters: Counters

    X_prev: np.ndarray | None
    G_prev: np.ndarray | None
    WX_prev: np.ndarray | None

    @property
    def x_bar(self) -> np.ndarray:
        return self.X.mean(axis=0)

    def oracles(self, problem: DistributedQuadraticProblem) -> CostedOracles:
        return CostedOracles(problem, self.counters)


@dataclass
class EXTRA:
    """EXTRA (Shi et al.) for symmetric doubly-stochastic W.

    Update (from repo notes):
        X^{k+1} = X^k + W X^k - W~ X^{k-1} - α(∇F(X^k) - ∇F(X^{k-1}))
    with W~ = (I + W)/2.

    Implementation uses one new mixing per iteration by caching W X^{k-1}.
    """

    alpha: float
    name: str = "EXTRA"

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
    ) -> ExtraState:
        _ = seed
        X_init = init_X0(problem, X0=X0, seed=seed)
        return ExtraState(
            t=0,
            X=X_init,
            counters=Counters(),
            X_prev=None,
            G_prev=None,
            WX_prev=None,
        )

    def step(self, problem: DistributedQuadraticProblem, state: ExtraState) -> ExtraState:
        orc = state.oracles(problem)

        if state.t == 0:
            # First iterate: X^1 = W X^0 - α ∇F(X^0)
            G0 = orc.local_grad(state.X)
            WX0 = orc.mix(state.X)
            X1 = WX0 - self.alpha * G0
            return ExtraState(
                t=1,
                X=X1,
                counters=state.counters,
                X_prev=state.X,
                G_prev=G0,
                WX_prev=WX0,
            )

        if state.X_prev is None or state.G_prev is None or state.WX_prev is None:
            raise RuntimeError("EXTRA state missing history; did you call init_state + step?")

        G = orc.local_grad(state.X)
        WX = orc.mix(state.X)

        Wtilde_X_prev = 0.5 * (state.X_prev + state.WX_prev)
        X_next = state.X + WX - Wtilde_X_prev - self.alpha * (G - state.G_prev)

        return ExtraState(
            t=state.t + 1,
            X=X_next,
            counters=state.counters,
            X_prev=state.X,
            G_prev=G,
            WX_prev=WX,
        )

    def diagnostics(self, problem: DistributedQuadraticProblem, state: ExtraState) -> dict[str, float]:
        _ = problem, state
        return {"alpha": float(self.alpha)}
