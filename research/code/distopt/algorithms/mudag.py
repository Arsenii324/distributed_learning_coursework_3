from __future__ import annotations

"""MUDAG (Momentum-accelerated decentralized algorithm with FastMix).

This is a MATLAB-faithful implementation of the archived code in
`research/Acceleration-in-Distributed-Optimization-Under-Similarity-main/*/Mudag.m`.

Key conventions in this implementation
-------------------------------------
- Global objective is the average: f(x) = (1/n) * sum_i f_i(x).
- We mirror the MATLAB convention of scaling each local gradient by 1/n.
- FastMix (Chebyshev-style recursion) runs for k=0..K, i.e. K+1 mixing rounds.
- MUDAG assumes the mixing matrix W is PSD (eigenvalues in [0, 1]).
  In the MATLAB workflow this is enforced by lazification W <- (W + I)/2.

Cost model
----------
Per *outer* iteration:
- mixing rounds: K+1
- grad evals per node: 1 (the previous gradient is cached)

Note: `state.t` counts outer iterations; compare methods via `mix_rounds` and
`grad_evals_per_node`.
"""

from dataclasses import dataclass

import numpy as np

from ..oracles import Counters, CostedOracles
from ..problems import DistributedQuadraticProblem
from .base import init_X0


@dataclass
class MudagState:
    t: int
    X: np.ndarray
    Y: np.ndarray
    Y_prev: np.ndarray
    G_Y_prev: np.ndarray
    counters: Counters

    # Precomputed constants (problem + graph dependent, fixed across iterations).
    K: int
    eta: float
    al: float
    eta_w: float
    lambda2_W: float
    M_over_L: float
    kappa_g: float

    @property
    def x_bar(self) -> np.ndarray:
        return self.X.mean(axis=0)

    def oracles(self, problem: DistributedQuadraticProblem) -> CostedOracles:
        return CostedOracles(problem, self.counters)


@dataclass
class MUDAG:
    """MUDAG with Chebyshev/FastMix inner loop.

    Hyperparameter:
    - c_K: controls the number of inner mixing rounds K via the reference formula.

    Reference formulas (MATLAB):
    - K = ceil(c_K / sqrt(1-λ₂(W)) * log((M/L) * κ_g))
    - eta_w = (1 - sqrt(1 - λ₂(W)^2)) / (1 + sqrt(1 - λ₂(W)^2))
    - eta = 1/L,  al = sqrt(mu/L)
    """

    c_K: float
    fastmix_stop_eps_sq: float | None = None
    name: str = "MUDAG"

    def check(self, problem: DistributedQuadraticProblem) -> None:
        if not (self.c_K > 0):
            raise ValueError("c_K must be positive")

        gstats = problem.graph.ensure_stats()
        if not bool(gstats.connected):
            raise ValueError("MUDAG requires a connected graph (adjacency must be connected)")

        # MUDAG assumes PSD W (eigenvalues in [0,1]); MATLAB uses W_PSD=(W+I)/2.
        tol = float(problem.graph.tol)
        min_eig = float(np.min(gstats.eigvals_W))
        if min_eig < -tol:
            raise ValueError(
                "MUDAG requires a PSD mixing matrix W (nonnegative eigenvalues). "
                "In this repo, use lazification e.g. Graph.from_adjacency(..., lazy=0.5) "
                "or make_graph_from_adjacency(..., lazy=0.5), which corresponds to W <- (W+I)/2. "
                f"Got min eigenvalue={min_eig:.3e}."
            )

        # Cache quadratic constants.
        problem.ensure_stats()

    def init_state(
        self,
        problem: DistributedQuadraticProblem,
        *,
        X0: np.ndarray | None = None,
        seed: int | None = None,
    ) -> MudagState:
        _ = seed

        X_init = init_X0(problem, X0=X0, seed=seed)

        # MATLAB requires consensus initialization: x_init = 1 x_bar^T.
        x_bar0 = X_init.mean(axis=0)
        X_cons = np.repeat(x_bar0[None, :], repeats=problem.n, axis=0)

        pstats = problem.ensure_stats()
        gstats = problem.graph.ensure_stats()

        L = float(pstats.L_g)
        mu = float(pstats.mu_g)
        if not (L > 0 and mu > 0):
            raise ValueError("Require positive L_g and mu_g for MUDAG")

        al = float(np.sqrt(mu / L))
        eta = float(1.0 / L)

        eig2 = float(gstats.lambda2_W)
        if not np.isfinite(eig2):
            raise ValueError("GraphStats.lambda2_W is not finite; invalid graph for MUDAG")
        if eig2 >= 1.0:
            raise ValueError(
                f"Require lambda2_W < 1 for mixing; got lambda2_W={eig2}. Is the graph connected?"
            )

        gap = float(1.0 - eig2)
        if gap <= 0.0:
            raise ValueError(f"Require 1 - lambda2_W > 0; got {gap}")

        # Use M = max_i L_i (worst local smoothness), L = L_g (global smoothness).
        M = float(pstats.L_l)
        kappa_g = float(pstats.kappa_g)
        M_over_L = float(M / L)

        # MATLAB formula for K.
        arg = float(M_over_L * kappa_g)
        K_real = float(self.c_K * (1.0 / np.sqrt(gap)) * np.log(arg))
        K = int(np.ceil(K_real))
        K = max(K, 0)

        # MATLAB eta_w uses lambda2_W^2.
        s = float(1.0 - eig2**2)
        if s < 0.0 and abs(s) <= 1e-12:
            s = 0.0
        if s < 0.0:
            raise ValueError(f"Invalid value under sqrt for eta_w: 1 - lambda2_W^2 = {s}")

        root = float(np.sqrt(s))
        eta_w = float((1.0 - root) / (1.0 + root))

        counters = Counters()

        return MudagState(
            t=0,
            X=X_cons,
            Y=X_cons.copy(),
            Y_prev=X_cons.copy(),
            G_Y_prev=np.zeros_like(X_cons),
            counters=counters,
            K=int(K),
            eta=eta,
            al=al,
            eta_w=eta_w,
            lambda2_W=eig2,
            M_over_L=M_over_L,
            kappa_g=kappa_g,
        )

    @staticmethod
    def _avg_sq_dist_to_x_star_all_nodes(X: np.ndarray, x_star: np.ndarray) -> float:
        diff = X - x_star[None, :]
        return float(np.linalg.norm(diff, ord="fro") ** 2 / float(X.shape[0]))

    def step(self, problem: DistributedQuadraticProblem, state: MudagState) -> MudagState:
        orc = state.oracles(problem)

        # MATLAB convention: each local gradient is scaled by 1/n.
        inv_n = 1.0 / float(problem.n)
        G_y = orc.local_grad(state.Y) * inv_n
        if int(state.t) == 0:
            G_prev = np.zeros_like(G_y)
        else:
            G_prev = state.G_Y_prev

        X_para = state.Y + (state.X - state.Y_prev) - state.eta * (G_y - G_prev)

        # FastMix / Chebyshev recursion, k = 0..K (so K+1 mixing rounds).
        X0 = X_para
        X_m1 = X_para
        eps_sq = self.fastmix_stop_eps_sq
        for _ in range(int(state.K) + 1):
            WX0 = orc.mix(X0)
            X1 = (1.0 + state.eta_w) * WX0 - state.eta_w * X_m1
            X_m1 = X0
            X0 = X1

            # MATLAB archived code optionally stops the inner loop early once the
            # global residual reaches the target tolerance.
            if eps_sq is not None:
                if self._avg_sq_dist_to_x_star_all_nodes(X0, problem.x_star) <= float(eps_sq):
                    break
        X_next = X0

        coef = float((1.0 - state.al) / (1.0 + state.al))
        Y_next = X_next + coef * (X_next - state.X)

        return MudagState(
            t=int(state.t) + 1,
            X=X_next,
            Y=Y_next,
            Y_prev=state.Y,
            G_Y_prev=G_y,
            counters=state.counters,
            K=state.K,
            eta=state.eta,
            al=state.al,
            eta_w=state.eta_w,
            lambda2_W=state.lambda2_W,
            M_over_L=state.M_over_L,
            kappa_g=state.kappa_g,
        )

    def diagnostics(self, problem: DistributedQuadraticProblem, state: MudagState) -> dict[str, float]:
        _ = problem
        return {
            "mudag_c_K": float(self.c_K),
            "mudag_K": float(state.K),
            "mudag_eta": float(state.eta),
            "mudag_al": float(state.al),
            "mudag_eta_w": float(state.eta_w),
            "mudag_lambda2_W": float(state.lambda2_W),
            "mudag_M_over_L": float(state.M_over_L),
            "mudag_kappa_g": float(state.kappa_g),
        }
