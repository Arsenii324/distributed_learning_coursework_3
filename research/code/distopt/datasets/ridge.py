from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..graphs import Graph
from ..problems import DistributedQuadraticProblem


@dataclass(frozen=True)
class PartitionSpec:
    m_agents: int
    partition: str = "shuffle"  # 'shuffle' or 'contiguous'
    seed: int | None = None


def _partition_indices(n_samples: int, spec: PartitionSpec) -> list[np.ndarray]:
    if spec.m_agents <= 0:
        raise ValueError("m_agents must be positive")

    idx = np.arange(n_samples)
    if spec.partition == "shuffle":
        rng = np.random.default_rng(spec.seed)
        rng.shuffle(idx)
    elif spec.partition == "contiguous":
        pass
    else:
        raise ValueError("partition must be 'shuffle' or 'contiguous'")

    splits = np.array_split(idx, spec.m_agents)
    return [np.asarray(s, dtype=np.int64) for s in splits]


def make_distributed_ridge_problem_from_dataset(
    graph: Graph,
    X,
    y,
    *,
    m_agents: int | None = None,
    lambda_reg: float = 1.0,
    partition: str = "shuffle",
    seed: int | None = 0,
    dtype: np.dtype | type = np.float64,
    dataset_name: str | None = None,
) -> DistributedQuadraticProblem:
    """Convert a dataset (X, y) into a distributed ridge quadratic.

    Per agent i with shard (X_i, y_i) and n_i rows:
      f_i(x) = (1/(2 n_i)) ||X_i x - y_i||^2 + (lambda_reg/2)||x||^2

    This yields a quadratic with:
      A_i = (X_i^T X_i)/n_i + lambda_reg I
      b_i = (X_i^T y_i)/n_i

    Performance notes:
    - Works efficiently for sparse X via `scipy.sparse`.
    - Stores only per-agent dense `(d,d)` matrices and `(d,)` vectors.
    - Avoids any O(N^2) memory.
    """

    try:
        from scipy import sparse  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "make_distributed_ridge_problem_from_dataset requires SciPy (scipy.sparse). Install with `pip install scipy`."
        ) from e

    if m_agents is None:
        m_agents = graph.n

    if graph.n != int(m_agents):
        raise ValueError(f"graph.n must equal m_agents; got graph.n={graph.n}, m_agents={m_agents}")

    if not (lambda_reg > 0):
        raise ValueError("lambda_reg must be positive")

    X = X
    y = np.asarray(y, dtype=dtype)

    if sparse.issparse(X):
        X_csr = X.tocsr().astype(dtype)
        n_samples, d = X_csr.shape
    else:
        X_arr = np.asarray(X, dtype=dtype)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2D")
        n_samples, d = X_arr.shape
        X_csr = sparse.csr_matrix(X_arr)

    if y.shape != (n_samples,):
        raise ValueError(f"y must have shape ({n_samples},); got {y.shape}")

    shards = _partition_indices(n_samples, PartitionSpec(m_agents=int(m_agents), partition=partition, seed=seed))

    A = np.empty((graph.n, d, d), dtype=np.float64)
    b = np.empty((graph.n, d), dtype=np.float64)

    I = np.eye(d, dtype=np.float64)

    for i, rows in enumerate(shards):
        if rows.size == 0:
            raise ValueError("Empty shard encountered; reduce m_agents or change partitioning")

        Xi = X_csr[rows]
        yi = y[rows]
        ni = float(rows.size)

        # A_i = (X_i^T X_i)/n_i + lambda I
        Ai = (Xi.T @ Xi) / ni
        if sparse.issparse(Ai):
            Ai = Ai.toarray()
        Ai = np.asarray(Ai, dtype=np.float64)
        Ai = Ai + float(lambda_reg) * I

        # b_i = (X_i^T y_i)/n_i
        bi = (Xi.T @ yi) / ni
        bi = np.asarray(bi).reshape(-1).astype(np.float64)

        A[i] = 0.5 * (Ai + Ai.T)  # numerical symmetry
        b[i] = bi

    meta = {
        "family": "dataset_ridge",
        "dataset": dataset_name or "unknown",
        "lambda_reg": float(lambda_reg),
        "partition": str(partition),
        "seed": seed,
        "n_samples": int(n_samples),
        "d": int(d),
    }

    return DistributedQuadraticProblem(graph=graph, A=A, b=b, metadata=meta)
