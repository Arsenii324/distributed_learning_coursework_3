from __future__ import annotations

"""Distributed quadratic objectives.

We work with strongly convex quadratic local objectives:

    f_i(x) = 0.5 x^T A_i x - b_i^T x
    ∇f_i(x) = A_i x - b_i

The global objective is always the uniform average:

    f(x) = (1/n) * Σ_i f_i(x)

Shapes
------
- `A` has shape (n, d, d)
- `b` has shape (n, d)
- stacked iterates `X` have shape (n, d)
- consensus average `x̄ = mean(X, axis=0)` has shape (d,)

The quadratic setting allows an exact global optimum:

    Ā = mean(A_i),  b̄ = mean(b_i),  x* = Ā^{-1} b̄

`ensure_stats()` computes exact (dense) spectral quantities such as κ_g, κ_ℓ and
Hessian similarity β = max_i ||A_i - Ā||₂ (spectral norm).
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .graphs import Graph


def _symmetrize(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + np.swapaxes(A, -1, -2))


@dataclass(frozen=True)
class QuadraticStats:
    L_i: np.ndarray  # (n,)
    mu_i: np.ndarray  # (n,)
    L_l: float
    mu_l: float
    L_g: float
    mu_g: float
    kappa_l: float
    kappa_g: float
    beta: float
    delta: float | None


@dataclass
class DistributedQuadraticProblem:
    """Distributed quadratic objective on a fixed graph.

    Locals are:
        f_i(x) = 0.5 x^T A_i x - b_i^T x
    with gradient:
        grad f_i(x) = A_i x - b_i.

    The global objective is always:
        f(x) = (1/n) * sum_i f_i(x).
    """

    graph: Graph
    A: np.ndarray  # (n, d, d)
    b: np.ndarray  # (n, d)
    validate: bool = True
    tol: float = 1e-12
    metadata: dict[str, Any] = field(default_factory=dict)

    _A_bar: np.ndarray | None = None
    _b_bar: np.ndarray | None = None
    _x_star: np.ndarray | None = None
    _stats: QuadraticStats | None = None

    def __post_init__(self) -> None:
        self.A = np.asarray(self.A, dtype=np.float64)
        self.b = np.asarray(self.b, dtype=np.float64)

        if self.A.ndim != 3:
            raise ValueError(f"A must have shape (n,d,d); got shape={self.A.shape}")
        if self.b.ndim != 2:
            raise ValueError(f"b must have shape (n,d); got shape={self.b.shape}")
        if self.A.shape[0] != self.graph.n:
            raise ValueError("A first dimension must match graph.n")
        if self.b.shape[0] != self.graph.n:
            raise ValueError("b first dimension must match graph.n")
        if self.A.shape[1] != self.A.shape[2]:
            raise ValueError("A must have shape (n,d,d) with square last two dims")
        if self.b.shape[1] != self.A.shape[1]:
            raise ValueError("b must have shape (n,d) matching A's d")

        # Ensure exact symmetry (numerically).
        self.A = _symmetrize(self.A)

        if self.validate:
            # SPD checks are done in ensure_stats (eigendecomp), to avoid duplication.
            self.ensure_stats()

    @property
    def n(self) -> int:
        return int(self.A.shape[0])

    @property
    def d(self) -> int:
        return int(self.A.shape[1])

    @property
    def A_bar(self) -> np.ndarray:
        if self._A_bar is None:
            self._A_bar = _symmetrize(self.A.mean(axis=0))
        assert self._A_bar is not None
        return self._A_bar

    @property
    def b_bar(self) -> np.ndarray:
        if self._b_bar is None:
            self._b_bar = self.b.mean(axis=0)
        assert self._b_bar is not None
        return self._b_bar

    @property
    def x_star(self) -> np.ndarray:
        if self._x_star is None:
            self._x_star = np.linalg.solve(self.A_bar, self.b_bar)
        assert self._x_star is not None
        return self._x_star

    def local_grad(self, X: np.ndarray) -> np.ndarray:
        """Compute stacked local gradients at local points X.

        X has shape (n, d); returns shape (n, d).
        """

        X = np.asarray(X, dtype=np.float64)
        if X.shape != (self.n, self.d):
            raise ValueError(f"X must have shape (n,d)=({self.n},{self.d}); got {X.shape}")
        return np.einsum("ijk,ik->ij", self.A, X) - self.b

    def global_value(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.d,):
            raise ValueError(f"x must have shape (d,) with d={self.d}; got {x.shape}")
        return float(0.5 * x @ (self.A_bar @ x) - self.b_bar @ x)

    def global_grad(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.d,):
            raise ValueError(f"x must have shape (d,) with d={self.d}; got {x.shape}")
        return self.A_bar @ x - self.b_bar

    def x_bar(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.shape != (self.n, self.d):
            raise ValueError(f"X must have shape (n,d)=({self.n},{self.d}); got {X.shape}")
        return X.mean(axis=0)

    def ensure_stats(self, *, compute_delta: bool = False) -> QuadraticStats:
        if self._stats is not None and (self._stats.delta is not None or not compute_delta):
            return self._stats

        tol = self.tol

        L_i = np.empty(self.n, dtype=np.float64)
        mu_i = np.empty(self.n, dtype=np.float64)

        for i in range(self.n):
            eig = np.linalg.eigvalsh(self.A[i])
            L_i[i] = float(eig.max())
            mu_i[i] = float(eig.min())

        if self.validate:
            if not np.all(mu_i > tol):
                bad = np.where(mu_i <= tol)[0]
                raise ValueError(f"Some A_i are not SPD within tol={tol}; indices={bad.tolist()}")

        A_bar = self.A_bar
        eig_bar = np.linalg.eigvalsh(A_bar)
        L_g = float(eig_bar.max())
        mu_g = float(eig_bar.min())
        if self.validate and mu_g <= tol:
            raise ValueError(f"A_bar not SPD within tol={tol}; mu_g={mu_g}")

        L_l = float(L_i.max())
        mu_l = float(mu_i.min())

        kappa_l = float(L_l / mu_l) if mu_l > 0 else float("inf")
        kappa_g = float(L_g / mu_g) if mu_g > 0 else float("inf")

        # beta = max_i ||A_i - A_bar||_2 for symmetric matrices.
        beta = 0.0
        for i in range(self.n):
            diff = _symmetrize(self.A[i] - A_bar)
            ev = np.linalg.eigvalsh(diff)
            beta = max(beta, float(np.max(np.abs(ev))))

        delta: float | None
        if compute_delta:
            delta_val = 0.0
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    diff = _symmetrize(self.A[i] - self.A[j])
                    ev = np.linalg.eigvalsh(diff)
                    delta_val = max(delta_val, float(np.max(np.abs(ev))))
            delta = float(delta_val)
        else:
            delta = None

        stats = QuadraticStats(
            L_i=L_i,
            mu_i=mu_i,
            L_l=L_l,
            mu_l=mu_l,
            L_g=L_g,
            mu_g=mu_g,
            kappa_l=kappa_l,
            kappa_g=kappa_g,
            beta=float(beta),
            delta=delta,
        )

        self._stats = stats
        return stats
