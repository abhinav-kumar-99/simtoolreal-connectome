from __future__ import annotations

import torch

from simtoolreal_shared.rotation_transforms import (
    axis_angle_to_matrix,
    matrix_to_quaternion,
    quaternion_xyzw_to_rotation_6d,
    quaternion_to_matrix,
)


def test_rotation_round_trip_and_gradients() -> None:
    torch.manual_seed(9)
    axis_angles = (torch.randn(32, 3, dtype=torch.float64) * 2.0).requires_grad_()
    matrices = axis_angle_to_matrix(axis_angles)
    quaternions = matrix_to_quaternion(matrices)
    reconstructed = quaternion_to_matrix(quaternions)
    torch.testing.assert_close(reconstructed, matrices, rtol=1.0e-7, atol=1.0e-7)
    reconstructed.square().sum().backward()
    assert axis_angles.grad is not None
    assert torch.isfinite(axis_angles.grad).all()


def test_zero_axis_angle_is_identity() -> None:
    matrices = axis_angle_to_matrix(torch.zeros(4, 3))
    torch.testing.assert_close(matrices, torch.eye(3).expand(4, 3, 3))
    quaternions = matrix_to_quaternion(matrices)
    torch.testing.assert_close(
        quaternions, torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(4, 4)
    )


def test_xyzw_rotation_6d_uses_matrix_columns_and_removes_double_cover() -> None:
    half = torch.tensor(0.5).sqrt()
    xyzw = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, half, half],
        ]
    )
    encoded = quaternion_xyzw_to_rotation_6d(xyzw)
    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
        ]
    )
    torch.testing.assert_close(encoded, expected, atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(
        quaternion_xyzw_to_rotation_6d(-xyzw), encoded, atol=0.0, rtol=0.0
    )
