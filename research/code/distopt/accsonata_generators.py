from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .graphs import Graph
from .problems import DistributedQuadraticProblem


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _is_connected_from_adj(A: np.ndarray) -> bool:
    n = int(A.shape[0])
    if n == 0:
        return True
    seen = np.zeros(n, dtype=bool)
    stack = [0]
    seen[0] = True
    while stack:
        i = stack.pop()
        nbrs = np.flatnonzero(A[i])
        for j in nbrs:
            if not seen[j]:
                seen[j] = True
                stack.append(int(j))
    return bool(seen.all())


def _metropolis_weights_from_adj(A: np.ndarray) -> np.ndarray:
    A = (A > 0).astype(np.float64)
    n = int(A.shape[0])
    deg = A.sum(axis=1)
    W = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if A[i, j] > 0:
                W[i, j] = 1.0 / (1.0 + max(deg[i], deg[j]))
        W[i, i] = 1.0 - W[i].sum()
    return W


def make_accsonata_er_graph(
    n: int,
    p_edge: float,
    *,
    seed: int | None = None,
    max_tries: int = 200,
    lazy: float = 0.0,
) -> Graph:
    """ER graph + Metropolis weights with connectivity filter.

    Mirrors the MATLAB pipeline:
    - Sample undirected ER adjacency
    - Reject until connected
    - Metropolis weights -> symmetric doubly stochastic W

    Notes:
    - MATLAB uses a transformed probability for the generator.
      Here we accept `p_edge` directly as an edge probability.
    """

    if n <= 1:
        W = np.ones((n, n), dtype=np.float64)
        adj = np.zeros((n, n), dtype=bool)
        return Graph(adj=adj, W=W)

    if not (0.0 <= p_edge <= 1.0):
        raise ValueError("p_edge must be in [0, 1]")

    if not (0.0 <= lazy <= 1.0):
        raise ValueError("lazy must be in [0, 1]")

    rng = _rng(seed)

    for _ in range(int(max_tries)):
        U = rng.random((n, n))
        A = (U < p_edge).astype(np.int8)
        A = np.triu(A, 1)
        A = A + A.T
        np.fill_diagonal(A, 0)

        if not _is_connected_from_adj(A):
            continue

        W = _metropolis_weights_from_adj(A)
        if lazy:
            W = float(lazy) * np.eye(n, dtype=np.float64) + (1.0 - float(lazy)) * W
        return Graph(adj=(A > 0), W=W)

    raise RuntimeError("Failed to sample a connected ER graph within max_tries")


@dataclass(frozen=True)
class RidgeSpec:
    d: int
    n_local: int
    mu0: float = 1.0
    L0: float = 1000.0
    lambda_reg: float = 0.1
    noise_std: float = 0.1


def make_accsonata_ridge_problem(
    graph: Graph,
    spec: RidgeSpec,
    *,
    seed: int | None = None,
    no_spectral_scaling: bool = False,
    paper_protocol: bool = False,
    x_true_mean: float = 5.0,
) -> DistributedQuadraticProblem:
    """Synthetic ridge regression -> quadratic objective family.

    Each node i has:
      f_i(x) = (1/(2 n_local)) ||U_i x - v_i||^2 + (lambda_reg/2) ||x||^2

    So A_i = (U_i^T U_i) / n_local + lambda_reg I
       b_i = (U_i^T v_i) / n_local

    This matches the structure used in the MATLAB ACC-SONATA experiments.
    """

    rng = _rng(seed)

    d = int(spec.d)
    n_local = int(spec.n_local)

    if d <= 0 or n_local <= 0:
        raise ValueError("d and n_local must be positive")

    mu0 = float(spec.mu0)
    L0 = float(spec.L0)
    if not (mu0 > 0 and L0 >= mu0):
        raise ValueError("Require mu0 > 0 and L0 >= mu0")

    if bool(paper_protocol):
        # ACC-SONATA Exp.1 (paper): rows drawn iid from N(0, Sigma) where
        # Sigma has eigenvalues Uniform[mu0, L0] and eigenvectors from QR.
        G = rng.standard_normal((d, d))
        U_cov, _ = np.linalg.qr(G)
        if d == 1:
            eigs = np.array([rng.uniform(mu0, L0)], dtype=np.float64)
        else:
            # Ensure the spectrum spans [mu0, L0] so the global conditioning can
            # approach L0/mu0 when lambda_reg is small, matching the paper’s
            # intended kappa sweep.
            eigs = np.concatenate(
                [
                    np.array([mu0, L0], dtype=np.float64),
                    rng.uniform(mu0, L0, size=d - 2).astype(np.float64),
                ]
            )
            rng.shuffle(eigs)
        sqrt_eigs = np.sqrt(eigs)

        # Shared ground truth.
        x_true = rng.standard_normal(d) + float(x_true_mean) * np.ones(d)

        # Sample per-node designs and labels.
        # If Sigma = U diag(eigs) U^T then row = (z * sqrt_eigs) @ U^T.
        Z = rng.standard_normal((graph.n, n_local, d))
        U = (Z * sqrt_eigs[None, None, :]) @ U_cov.T
        noise = float(spec.noise_std) * rng.standard_normal((graph.n, n_local))
        v = np.einsum("nid,d->ni", U, x_true) + noise
    else:
        # Legacy: iid normal features + additive noise labels with x_true = 1.
        U = rng.standard_normal((graph.n, n_local, d))
        noise = float(spec.noise_std) * rng.standard_normal((graph.n, n_local))
        v = U.sum(axis=2) + noise

    A = np.empty((graph.n, d, d), dtype=np.float64)
    b = np.empty((graph.n, d), dtype=np.float64)

    I = np.eye(d, dtype=np.float64)
    for i in range(graph.n):
        Ui = U[i]
        vi = v[i]
        A[i] = (Ui.T @ Ui) / float(n_local) + float(spec.lambda_reg) * I
        b[i] = (Ui.T @ vi) / float(n_local)

    # Optional spectral scaling of each local Hessian into [mu0, L0].
    # Default is MATLAB-like scaling; can be disabled to preserve the natural
    # dependence of beta/kappa on n_local and lambda_reg in sweep experiments.
    if not bool(no_spectral_scaling):
        for i in range(graph.n):
            evals = np.linalg.eigvalsh(A[i])
            lam_min = float(evals.min())
            lam_max = float(evals.max())
            if lam_min <= 0:
                raise ValueError("Generated local Hessian is not SPD")

            # Scale to enforce local conditioning roughly like MATLAB.
            # Use a single scalar so we don't destroy ridge structure.
            scale = float(L0 / lam_max)
            A[i] = scale * A[i]
            b[i] = scale * b[i]

            # And shift if needed to ensure at least mu0.
            evals2 = np.linalg.eigvalsh(A[i])
            lam_min2 = float(evals2.min())
            if lam_min2 < mu0:
                A[i] = A[i] + (mu0 - lam_min2) * I

    return DistributedQuadraticProblem(graph=graph, A=A, b=b)
