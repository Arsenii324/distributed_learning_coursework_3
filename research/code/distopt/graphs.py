from __future__ import annotations

"""Graph primitives for decentralized optimization experiments.

Core idea
---------
We store both:
- `adj`: the *physical* undirected topology (simple graph, no self-loops)
- `W`: the *one-round communication/mixing* operator used by algorithms

Default assumptions (validated by default)
-----------------------------------------
- `adj` is symmetric boolean with a false diagonal
- `W` is symmetric and doubly-stochastic (rows and columns sum to 1)
- `W_{ij}=0` whenever `(i,j)` is not an edge (for i!=j)

The main operation is `Graph.mix(X)`, which performs one synchronous mixing step
`X <- W X` and is interpreted as one communication round in the cost model.

`Graph.ensure_stats()` caches common spectral quantities used in the repo:
- one-sided spectral gap: γ = 1 - λ₂(W)
- graph condition number: χ = λ_max(L) / λ_min⁺(L) where L = I - W
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np


WeightRule = Literal["metropolis"]


def _as_bool_square_matrix(adj: np.ndarray) -> np.ndarray:
    adj = np.asarray(adj)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adj must be square (n,n); got shape={adj.shape}")
    if adj.dtype != bool:
        adj = adj.astype(bool)
    return adj


def _check_no_self_loops(adj: np.ndarray) -> None:
    if np.any(np.diag(adj)):
        raise ValueError("adj must have no self-loops (diag must be False)")


def _check_symmetric(mat: np.ndarray, *, name: str, tol: float) -> None:
    if not np.allclose(mat, mat.T, atol=tol, rtol=0.0):
        raise ValueError(f"{name} must be symmetric within tol={tol}")


def metropolis_mixing_matrix(
    adj: np.ndarray,
    *,
    lazy: float = 0.0,
    validate: bool = True,
    tol: float = 1e-12,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    """Build a symmetric doubly-stochastic mixing matrix W via Metropolis weights.

    For an undirected unweighted simple graph, for i != j with (i,j) an edge:
        w_ij = 1 / (1 + max(deg(i), deg(j)))
    and
        w_ii = 1 - sum_{j != i} w_ij

    Lazification uses: W <- (1-lazy) W + lazy I.
    """

    adj = _as_bool_square_matrix(adj)
    _check_no_self_loops(adj)
    _check_symmetric(adj.astype(np.float64), name="adj", tol=0.0)

    n = adj.shape[0]
    deg = adj.sum(axis=1).astype(int)

    W = np.zeros((n, n), dtype=dtype)

    # Off-diagonal weights.
    for i in range(n):
        nbrs = np.flatnonzero(adj[i])
        for j in nbrs:
            if i == j:
                continue
            W[i, j] = 1.0 / (1.0 + max(deg[i], deg[j]))

    # Diagonal to make rows sum to 1.
    np.fill_diagonal(W, 1.0 - W.sum(axis=1))

    if lazy:
        if not (0.0 <= lazy < 1.0):
            raise ValueError("lazy must be in [0,1)")
        W = (1.0 - lazy) * W + lazy * np.eye(n, dtype=dtype)

    if validate:
        _validate_mixing_matrix(W, adj=adj, tol=tol, require_support=True)

    return W


def _validate_mixing_matrix(
    W: np.ndarray,
    *,
    adj: np.ndarray | None,
    tol: float,
    require_support: bool,
) -> None:
    W = np.asarray(W)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be square (n,n); got shape={W.shape}")

    _check_symmetric(W, name="W", tol=tol)

    row_sums = W.sum(axis=1)
    col_sums = W.sum(axis=0)
    if not np.allclose(row_sums, 1.0, atol=tol, rtol=0.0):
        raise ValueError("W must be row-stochastic (rows sum to 1)")
    if not np.allclose(col_sums, 1.0, atol=tol, rtol=0.0):
        raise ValueError("W must be column-stochastic (cols sum to 1)")

    if adj is not None and require_support:
        adj = _as_bool_square_matrix(adj)
        _check_no_self_loops(adj)
        # For i!=j with no edge, weights must be ~0.
        mask_forbidden = (~adj) & (~np.eye(adj.shape[0], dtype=bool))
        if np.any(np.abs(W[mask_forbidden]) > tol):
            raise ValueError("W has nonzero weights where adj has no edge")


def _is_connected(adj: np.ndarray) -> bool:
    adj = _as_bool_square_matrix(adj)
    n = adj.shape[0]
    if n == 0:
        return True

    seen = np.zeros(n, dtype=bool)
    stack: list[int] = [0]
    seen[0] = True

    while stack:
        i = stack.pop()
        for j in np.flatnonzero(adj[i]):
            if not seen[j]:
                seen[j] = True
                stack.append(int(j))

    return bool(seen.all())


@dataclass(frozen=True)
class GraphStats:
    degrees: np.ndarray  # (n,)
    connected: bool
    eigvals_W: np.ndarray  # (n,)
    eigvals_L: np.ndarray  # (n,)
    lambda2_W: float
    gamma: float
    chi: float
    lambda_min_pos_L: float
    lambda_max_L: float


@dataclass
class Graph:
    """Static undirected graph + symmetric doubly-stochastic mixing matrix."""

    adj: np.ndarray  # bool (n,n)
    W: np.ndarray  # float (n,n)
    validate: bool = True
    tol: float = 1e-12

    _stats: GraphStats | None = None

    def __post_init__(self) -> None:
        self.adj = _as_bool_square_matrix(self.adj)
        _check_no_self_loops(self.adj)
        _check_symmetric(self.adj.astype(np.float64), name="adj", tol=0.0)

        self.W = np.asarray(self.W, dtype=np.float64)
        if self.W.shape != self.adj.shape:
            raise ValueError(f"W shape must match adj shape; got W={self.W.shape}, adj={self.adj.shape}")

        if self.validate:
            _validate_mixing_matrix(self.W, adj=self.adj, tol=self.tol, require_support=True)

    @property
    def n(self) -> int:
        return int(self.adj.shape[0])

    @property
    def L(self) -> np.ndarray:
        return np.eye(self.n) - self.W

    @classmethod
    def from_adjacency(
        cls,
        adj: np.ndarray,
        *,
        weight_rule: WeightRule = "metropolis",
        lazy: float = 0.0,
        W_override: np.ndarray | None = None,
        validate: bool = True,
        tol: float = 1e-12,
    ) -> "Graph":
        adj_bool = _as_bool_square_matrix(adj)
        _check_no_self_loops(adj_bool)
        _check_symmetric(adj_bool.astype(np.float64), name="adj", tol=0.0)

        if W_override is None:
            if weight_rule != "metropolis":
                raise ValueError(f"Unsupported weight_rule={weight_rule!r}")
            W = metropolis_mixing_matrix(adj_bool, lazy=lazy, validate=validate, tol=tol)
        else:
            W = np.asarray(W_override, dtype=np.float64)

        return cls(adj=adj_bool, W=W, validate=validate, tol=tol)

    @classmethod
    def from_edge_list(
        cls,
        edges: list[tuple[int, int]],
        n: int,
        *,
        weight_rule: WeightRule = "metropolis",
        lazy: float = 0.0,
        W_override: np.ndarray | None = None,
        validate: bool = True,
        tol: float = 1e-12,
    ) -> "Graph":
        adj = np.zeros((n, n), dtype=bool)
        for i, j in edges:
            if i == j:
                raise ValueError("Self-loops are not allowed in edge list")
            if not (0 <= i < n and 0 <= j < n):
                raise ValueError("Edge contains node out of bounds")
            adj[i, j] = True
            adj[j, i] = True

        return cls.from_adjacency(
            adj,
            weight_rule=weight_rule,
            lazy=lazy,
            W_override=W_override,
            validate=validate,
            tol=tol,
        )

    def mix(self, X: np.ndarray) -> np.ndarray:
        """Apply one synchronous mixing step: X <- W X.

        X can have shape (n, d) or (n, ...); mixing is applied along the first axis.
        """

        X = np.asarray(X)
        if X.ndim < 1 or X.shape[0] != self.n:
            raise ValueError(f"X must have first dimension n={self.n}; got shape={X.shape}")

        return np.tensordot(self.W, X, axes=(1, 0))

    def ensure_stats(self) -> GraphStats:
        if self._stats is not None:
            return self._stats

        degrees = self.adj.sum(axis=1).astype(int)
        connected = _is_connected(self.adj)

        eigvals_W = np.linalg.eigvalsh(self.W)
        eigvals_L = np.linalg.eigvalsh(self.L)

        # For symmetric doubly-stochastic W, lambda_max = 1.
        if self.n >= 2:
            lambda2_W = float(eigvals_W[-2])
        else:
            lambda2_W = float("nan")

        gamma = float(1.0 - lambda2_W) if self.n >= 2 else float("nan")

        tol = self.tol
        pos = eigvals_L[eigvals_L > tol]
        if pos.size == 0:
            lambda_min_pos_L = float("nan")
            chi = float("inf")
        else:
            lambda_min_pos_L = float(pos.min())
            lambda_max_L = float(eigvals_L.max())
            chi = float(lambda_max_L / lambda_min_pos_L)

        lambda_max_L = float(eigvals_L.max())

        stats = GraphStats(
            degrees=degrees,
            connected=connected,
            eigvals_W=eigvals_W,
            eigvals_L=eigvals_L,
            lambda2_W=lambda2_W,
            gamma=gamma,
            chi=chi,
            lambda_min_pos_L=lambda_min_pos_L,
            lambda_max_L=lambda_max_L,
        )
        self._stats = stats
        return stats
