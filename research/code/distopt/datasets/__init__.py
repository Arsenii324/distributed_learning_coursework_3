"""Dataset utilities for converting real datasets into quadratic problems.

This package is intentionally minimal and keeps the main `distopt` framework
quadratic-only by converting datasets into ridge-regression quadratics:

    f_i(x) = (1/(2n_i)) ||X_i x - y_i||^2 + (lambda/2)||x||^2

so that each node i has (A_i, b_i) with:
    A_i = (X_i^T X_i)/n_i + lambda I
    b_i = (X_i^T y_i)/n_i
"""

from .download import get_data_dir, download_libsvm_dataset
from .ridge import make_distributed_ridge_problem_from_dataset
from .svmlight import load_svmlight

__all__ = [
    "get_data_dir",
    "download_libsvm_dataset",
    "load_svmlight",
    "make_distributed_ridge_problem_from_dataset",
]
