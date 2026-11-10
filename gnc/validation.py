"""
Shared input checks
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]

def vector(value: ArrayLike, size: int, name: str) -> Array:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector of shape ({size},)")
    return result.copy()

def positive(value: float, name: str, *, zero: bool = False) -> float:
    if not np.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return float(value)

def inertia_tensor(inertia_body_kgm2: ArrayLike) -> Array:
    tensor = np.asarray(inertia_body_kgm2, dtype=float)
    if tensor.shape != (3, 3) or not np.all(np.isfinite(tensor)):
        raise ValueError("inertia must be a finite 3x3 tensor")
    if not np.allclose(tensor, tensor.T, rtol=0, atol=1e-12):
        raise ValueError("inertia must be symmetric")
    if np.min(np.linalg.eigvalsh(tensor)) <= 0:
        raise ValueError("inertia must be positive definite")
    return tensor.copy()