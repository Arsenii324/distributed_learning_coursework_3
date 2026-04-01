from __future__ import annotations

from pathlib import Path

import numpy as np


def _open_maybe_compressed(path: Path):
    import bz2
    import gzip

    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".bz2":
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    if suf == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("rt", encoding="utf-8", errors="replace")


def load_svmlight(
    path: str | Path,
    *,
    n_features: int | None = None,
    dtype: np.dtype | type = np.float64,
):
    """Load an SVMLight / LIBSVM file.

    Returns `(X, y)` where:
    - `X` is a `scipy.sparse.csr_matrix` with shape `(n_samples, n_features)`.
    - `y` is a 1D `np.ndarray` of length `n_samples`.

    This is a minimal loader intended to avoid pulling in scikit-learn.

    Performance notes:
    - Parsing is one-pass and stores only CSR triplets (O(nnz)).
    - No O(N^2) operations.
    """

    try:
        from scipy import sparse  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "load_svmlight requires SciPy (scipy.sparse). Install with `pip install scipy`."
        ) from e

    path = Path(path)

    data: list[float] = []
    indices: list[int] = []
    indptr: list[int] = [0]
    y: list[float] = []

    max_idx = -1

    with _open_maybe_compressed(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # Strip comments.
            if "#" in line:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue

            parts = line.split()
            y.append(float(parts[0]))

            for feat in parts[1:]:
                if feat.startswith("qid:"):
                    continue
                if ":" not in feat:
                    continue
                k_str, v_str = feat.split(":", 1)
                k = int(k_str) - 1  # SVMLight is 1-based
                v = float(v_str)
                if k < 0:
                    raise ValueError(f"Invalid feature index in line: {raw!r}")
                indices.append(k)
                data.append(v)
                if k > max_idx:
                    max_idx = k

            indptr.append(len(indices))

    n_samples = len(y)
    if n_features is None:
        n_features = max_idx + 1 if max_idx >= 0 else 0

    X = sparse.csr_matrix(
        (np.asarray(data, dtype=dtype), np.asarray(indices, dtype=np.int32), np.asarray(indptr, dtype=np.int32)),
        shape=(n_samples, int(n_features)),
    )

    # Canonicalize representation (sum duplicates, sorted indices).
    X.sum_duplicates()
    X.sort_indices()

    return X, np.asarray(y, dtype=dtype)
