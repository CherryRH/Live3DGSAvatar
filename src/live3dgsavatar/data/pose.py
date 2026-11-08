"""数据集的姿态统计：从驱动参数里取头姿 yaw。

用途是**增强阶段挑帧**：微调时希望锚定与选点都偏向大角度，而不是沿用预训练的
切片方式（预训练怎么切分是别人的事，增强阶段不该依赖它）。

## 约定

`blend_weight[100:106]` 是头姿 rot6d（旋转矩阵前两行，与 `matrix_to_rot6d` 互逆）。
Gram-Schmidt 还原矩阵后取 `yaw = atan2(-R[2,0], R[2,2])` 度；第 0 帧约为 0，
与数据朝向一致。这个量只用于**排序/加权**，不参与训练。
"""

from __future__ import annotations

import numpy as np

__all__ = ["rot6d_to_matrix", "head_yaw", "angle_weights"]

#: 头姿在 blend_weight 中的切片
HEAD_ROT_SLICE = slice(100, 106)


def rot6d_to_matrix(d6):
    """`[..., 6]` → `[..., 3, 3]`（行向量为基）。"""
    d6 = np.asarray(d6, dtype=np.float64)
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-2)


def head_yaw(blend_weight) -> np.ndarray:
    """`[N, >=106]` 驱动参数 → `[N]` 头姿 yaw（度）。"""
    w = np.asarray(blend_weight, dtype=np.float64)
    if w.ndim != 2 or w.shape[1] < HEAD_ROT_SLICE.stop:
        raise ValueError(f"blend_weight 应为 [N,>=106]，实际 {w.shape}")
    R = rot6d_to_matrix(w[:, HEAD_ROT_SLICE])
    return np.degrees(np.arctan2(-R[..., 2, 0], R[..., 2, 2]))


def angle_weights(yaw, *, scale: float = 20.0, floor: float = 1.0) -> np.ndarray:
    """采样权重：`floor + |yaw| / scale`，偏向大角度。

    Args:
        yaw: `[N]` 角度（度）
        scale: 多大角度算"权重翻倍"
        floor: 基础权重，保证小角度帧仍可能被抽到

    Returns:
        `[N]` 非负权重（未归一化）
    """
    y = np.abs(np.asarray(yaw, dtype=np.float64))
    return float(floor) + y / max(1e-9, float(scale))
