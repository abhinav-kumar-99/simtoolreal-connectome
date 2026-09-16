"""Small pure-PyTorch rotation subset used when PyTorch3D is unavailable.

The conventions match ``pytorch3d.transforms``: quaternions are real-first
``(w, x, y, z)`` and axis-angle vectors encode axis times angle in radians.
The conversion algorithms follow PyTorch3D's BSD-licensed
``transforms/rotation_conversions.py`` implementation.
"""

from __future__ import annotations

import torch
from torch import Tensor


def quaternion_to_matrix(quaternions: Tensor) -> Tensor:
    if quaternions.shape[-1] != 4:
        raise ValueError(f"Expected (..., 4) quaternions, got {quaternions.shape}")
    real, i, j, k = torch.unbind(quaternions, dim=-1)
    two_s = 2.0 / (quaternions * quaternions).sum(dim=-1)
    return torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * real),
            two_s * (i * k + j * real),
            two_s * (i * j + k * real),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * real),
            two_s * (i * k - j * real),
            two_s * (j * k + i * real),
            1 - two_s * (i * i + j * j),
        ),
        dim=-1,
    ).reshape(quaternions.shape[:-1] + (3, 3))


def quaternion_xyzw_to_rotation_6d(quaternions: Tensor) -> Tensor:
    """Convert Isaac-style ``(..., x, y, z, w)`` quaternions to 6D rotations.

    The result concatenates the first and second rotation-matrix columns. It is
    invariant to the quaternion double cover: ``q`` and ``-q`` map identically.
    """
    if quaternions.shape[-1] != 4:
        raise ValueError(f"Expected (..., 4) xyzw quaternions, got {quaternions.shape}")
    real_first = torch.cat((quaternions[..., 3:], quaternions[..., :3]), dim=-1)
    matrix = quaternion_to_matrix(real_first)
    return matrix[..., :, :2].transpose(-1, -2).reshape(
        quaternions.shape[:-1] + (6,)
    )


def axis_angle_to_matrix(axis_angle: Tensor) -> Tensor:
    if axis_angle.shape[-1] != 3:
        raise ValueError(f"Expected (..., 3) axis angles, got {axis_angle.shape}")
    angles = torch.linalg.vector_norm(axis_angle, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    small = angles.abs() < 1.0e-6
    sin_half_over_angle = torch.empty_like(angles)
    sin_half_over_angle[~small] = torch.sin(half_angles[~small]) / angles[~small]
    squared = angles[small] * angles[small]
    sin_half_over_angle[small] = 0.5 - squared / 48.0
    quaternion = torch.cat(
        (torch.cos(half_angles), axis_angle * sin_half_over_angle), dim=-1
    )
    return quaternion_to_matrix(quaternion)


def _sqrt_positive_part(values: Tensor) -> Tensor:
    result = torch.zeros_like(values)
    positive = values > 0
    result[positive] = torch.sqrt(values[positive])
    return result


def matrix_to_quaternion(matrix: Tensor) -> Tensor:
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (..., 3, 3) matrices, got {matrix.shape}")
    m00 = matrix[..., 0, 0]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m10 = matrix[..., 1, 0]
    m11 = matrix[..., 1, 1]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]
    m22 = matrix[..., 2, 2]
    q_abs = _sqrt_positive_part(
        torch.stack(
            (
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ),
            dim=-1,
        )
    )
    quaternion_candidates = torch.stack(
        (
            torch.stack((q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21), dim=-1),
            torch.stack((m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2), dim=-1),
        ),
        dim=-2,
    )
    denominator = 2.0 * q_abs[..., None].clamp_min(0.1)
    quaternion_candidates = quaternion_candidates / denominator
    best = q_abs.argmax(dim=-1)
    gather_index = best[..., None, None].expand(best.shape + (1, 4))
    quaternion = torch.gather(quaternion_candidates, -2, gather_index).squeeze(-2)
    return torch.where(quaternion[..., :1] < 0, -quaternion, quaternion)
