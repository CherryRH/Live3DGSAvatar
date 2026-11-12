"""头姿角度提取：后训练视角补偿依赖这里。"""

from __future__ import annotations

import numpy as np

from live3dgsavatar.data.pose import head_pitch, head_yaw


def _blend_weight(R: np.ndarray) -> np.ndarray:
    w = np.zeros((1, 129), np.float32)
    w[0, 100:106] = R[:2].reshape(-1)
    return w


def test_head_yaw_and_pitch_from_rotation() -> None:
    cy, sy = np.cos(np.radians(30)), np.sin(np.radians(30))
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float64)
    assert abs(float(head_yaw(_blend_weight(Ry))[0]) - 30.0) < 1e-4

    cp, sp = np.cos(np.radians(10)), np.sin(np.radians(10))
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], np.float64)
    assert abs(float(head_pitch(_blend_weight(Rx))[0]) - 10.0) < 1e-4
