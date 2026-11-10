"""
Quaternion algebra and conversions
"""

import numpy as np
from numpy.typing import NDArray

from gnc.validation import Array, vector

Quat = NDArray[np.float64]
Dcm = NDArray[np.float64]

def quat_multiply(q_lhs: Quat, q_rhs: Quat) -> Quat:
    lhs = vector(q_lhs, 4, "q_lhs")
    rhs = vector(q_rhs, 4, "q_rhs")
    return np.asarray(
        np.r_[
            lhs[3] * rhs[:3] + rhs[3] * lhs[:3] + np.cross(lhs[:3], rhs[:3]),
            lhs[3] * rhs[3] - np.dot(lhs[:3], rhs[:3]),
        ],
        dtype=float,
    )

def dcm_from_quat(q_b_from_a: Quat) -> Dcm:
    unit = normalize(q_b_from_a)
    x, y, z, w = unit
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.asarray(
        (w * w - np.dot(unit[:3], unit[:3])) * np.eye(3)
        + 2 * np.outer(unit[:3], unit[:3])
        + 2 * w * skew,
        dtype=float,
    )

def normalize(q_b_from_a: Quat) -> Quat:
    value = vector(q_b_from_a, 4, "q_b_from_a")
    scale = float(np.max(np.abs(value)))
    if scale == 0:
        raise ValueError("quaternion must be nonzero")
    value /= scale
    return value / np.linalg.norm(value)

def conjugate(q_b_from_a: Quat) -> Quat:
    unit = normalize(q_b_from_a)
    unit[:3] *= -1
    return unit

def from_rotvec(rotation_vector_rad: Array) -> Quat:
    rotvec = vector(rotation_vector_rad, 3, "rotation_vector_rad")
    angle = float(np.linalg.norm(rotvec))
    return np.asarray(
        np.r_[0.5 * np.sinc(angle / (2 * np.pi)) * rotvec, np.cos(angle / 2)], dtype=float
    )

def shortest(q_b_from_a: Quat) -> Quat:
    unit = normalize(q_b_from_a)
    pivot = unit[3] if abs(unit[3]) > 1e-15 else unit[int(np.argmax(np.abs(unit[:3])))]
    return unit if pivot >= 0 else -unit

def angle_rad(q_b_from_a: Quat) -> float:
    unit = normalize(q_b_from_a)
    return float(2 * np.arctan2(np.linalg.norm(unit[:3]), abs(unit[3])))