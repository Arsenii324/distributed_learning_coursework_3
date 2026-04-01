from __future__ import annotations

"""Instance generators for `distopt` experiments.

This module provides:
- simple adjacency constructors (path, cycle, complete, Erdos–Renyi)
- a default graph constructor that builds `W` from adjacency
- quadratic problem families that return a `DistributedQuadraticProblem`

The families are meant to probe different structural hypotheses:
- random eigenbases vs shared eigenbasis (alignment)
- ridge ERM / Wishart-style Hessians (a simple stochastic-ERM surrogate)
"""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .graphs import Graph
from .problems import DistributedQuadraticProblem


Spectrum = Literal["uniform", "logspace"]


def path_adjacency(n: int) -> np.ndarray:
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n - 1):
        adj[i, i + 1] = True
        adj[i + 1, i] = True
    return adj


def cycle_adjacency(n: int) -> np.ndarray:
    adj = path_adjacency(n)
    if n >= 2:
        adj[0, n - 1] = True
        adj[n - 1, 0] = True
    return adj


def complete_adjacency(n: int) -> np.ndarray:
    adj = np.ones((n, n), dtype=bool)
    np.fill_diagonal(adj, False)
    return adj


def erdos_renyi_adjacency(n: int, p: float, *, seed: int | None = None) -> np.ndarray:
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0,1]")
    rng = np.random.default_rng(seed)
    A = rng.uniform(size=(n, n))
    upper = np.triu(A, k=1) < p
    adj = upper | upper.T
    np.fill_diagonal(adj, False)
    return adj


def make_graph_from_adjacency(
    adj: np.ndarray,
    *,
    lazy: float = 0.0,
    W_override: np.ndarray | None = None,
    validate: bool = True,
    tol: float = 1e-12,
) -> Graph:
    return Graph.from_adjacency(
        adj,
        weight_rule="metropolis",
        lazy=lazy,
        W_override=W_override,
        validate=validate,
        tol=tol,
    )


def _random_orthonormal(rng: np.random.Generator, d: int) -> np.ndarray:
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    # Ensure a proper orthonormal basis (QR sign ambiguity is fine here).
    return Q


def make_random_spd_problem(
    graph: Graph,
    *,
    d: int,
    mu: float = 1.0,
    L: float = 10.0,
    spectrum: Spectrum = "logspace",
    seed: int | None = None,
) -> DistributedQuadraticProblem:
    """Random SPD Hessians with per-node random eigenbases."""

    if not (mu > 0 and L >= mu):
        raise ValueError("Require 0 < mu <= L")

    rng = np.random.default_rng(seed)
    n = graph.n

    A = np.empty((n, d, d), dtype=np.float64)
    b = rng.normal(size=(n, d))

    if spectrum == "uniform":
        base_eigs = None
    elif spectrum == "logspace":
        base_eigs = np.logspace(np.log10(mu), np.log10(L), num=d)
    else:
        raise ValueError(f"Unknown spectrum={spectrum!r}")

    for i in range(n):
        Q = _random_orthonormal(rng, d)
        if base_eigs is None:
            eigs = rng.uniform(mu, L, size=d)
        else:
            # jitter the base spectrum slightly across nodes
            jitter = rng.uniform(0.9, 1.1, size=d)
            eigs = np.clip(base_eigs * jitter, mu, L)
        A[i] = Q @ np.diag(eigs) @ Q.T

    return DistributedQuadraticProblem(
        graph=graph,
        A=A,
        b=b,
        metadata={
            "family": "random_spd",
            "d": int(d),
            "mu": float(mu),
            "L": float(L),
            "spectrum": spectrum,
            "seed": seed,
        },
    )


def make_shared_eigenbasis_problem(
    graph: Graph,
    *,
    d: int,
    mu: float = 1.0,
    L: float = 10.0,
    spectrum: Spectrum = "logspace",
    seed: int | None = None,
) -> DistributedQuadraticProblem:
    """SPD Hessians that share a common eigenbasis across nodes."""

    if not (mu > 0 and L >= mu):
        raise ValueError("Require 0 < mu <= L")

    rng = np.random.default_rng(seed)
    n = graph.n

    Q = _random_orthonormal(rng, d)

    if spectrum == "uniform":
        base_eigs = None
    elif spectrum == "logspace":
        base_eigs = np.logspace(np.log10(mu), np.log10(L), num=d)
    else:
        raise ValueError(f"Unknown spectrum={spectrum!r}")

    A = np.empty((n, d, d), dtype=np.float64)
    b = rng.normal(size=(n, d))

    for i in range(n):
        if base_eigs is None:
            eigs = rng.uniform(mu, L, size=d)
        else:
            jitter = rng.uniform(0.9, 1.1, size=d)
            eigs = np.clip(base_eigs * jitter, mu, L)
        A[i] = Q @ np.diag(eigs) @ Q.T

    return DistributedQuadraticProblem(
        graph=graph,
        A=A,
        b=b,
        metadata={
            "family": "shared_eigenbasis",
            "d": int(d),
            "mu": float(mu),
            "L": float(L),
            "spectrum": spectrum,
            "seed": seed,
        },
    )


def make_wishart_ridge_problem(
    graph: Graph,
    *,
    d: int,
    m_per_node: int,
    lambda_reg: float = 1.0,
    noise_std: float = 0.0,
    seed: int | None = None,
) -> DistributedQuadraticProblem:
    """Quadratic induced by ridge regression with iid Gaussian features.

    Each node i has samples (Z_i, y_i), with Z_i shape (m,d).
    Local objective:
        f_i(x) = (1/(2m)) ||Z_i x - y_i||^2 + (lambda/2)||x||^2
    which corresponds to:
        A_i = (1/m) Z_i^T Z_i + lambda I
        b_i = (1/m) Z_i^T y_i
    (constant terms dropped).
    """

    if m_per_node <= 0:
        raise ValueError("m_per_node must be positive")
    if lambda_reg <= 0:
        raise ValueError("lambda_reg must be positive")
    if noise_std < 0:
        raise ValueError("noise_std must be nonnegative")

    rng = np.random.default_rng(seed)
    n = graph.n

    x_true = rng.normal(size=(d,))

    A = np.empty((n, d, d), dtype=np.float64)
    b = np.empty((n, d), dtype=np.float64)

    for i in range(n):
        Z = rng.normal(size=(m_per_node, d))
        y = Z @ x_true
        if noise_std:
            y = y + rng.normal(scale=noise_std, size=y.shape)

        A_i = (Z.T @ Z) / float(m_per_node) + float(lambda_reg) * np.eye(d)
        b_i = (Z.T @ y) / float(m_per_node)

        A[i] = A_i
        b[i] = b_i

    return DistributedQuadraticProblem(
        graph=graph,
        A=A,
        b=b,
        metadata={
            "family": "wishart_ridge",
            "d": int(d),
            "m_per_node": int(m_per_node),
            "lambda_reg": float(lambda_reg),
            "noise_std": float(noise_std),
            "seed": seed,
        },
    )
