"""相机姿态的纯几何工具：绕一点旋转、投影。

**本模块只做数学**（不读写文件、不知道数据集与模型），因此放在 `core/`。
评估、精修输入生成、脚本都从这里取同一套实现 —— 早前这条公式在脚本里各写一份，
其中一份绕世界原点旋转，导致头被转出画面。

约定与 `docs/CONVENTIONS.md` §2 一致：`w2c` 是 world→camera，相机沿 **+z** 看。
"""

from __future__ import annotations

import math

import torch

__all__ = ["rotation_x", "rotation_y", "rotation_z", "orbit_w2c",
           "roll_w2c", "project_point", "as_cpu"]


def as_cpu(x) -> torch.Tensor:
    """任意张量 / 数组 → CPU float32 张量。

    几何换算统一在 CPU 上做：相机、内参、旋转中心可能分别来自 CPU 与 CUDA
    （例如 `scene.K` 在 GPU 上），混用会在 matmul 处报设备不一致。
    """
    return torch.as_tensor(x, dtype=torch.float32).detach().cpu()


def _rot(axis: str, deg: float) -> torch.Tensor:
    r = math.radians(float(deg))
    c, s = math.cos(r), math.sin(r)
    m = torch.eye(4, dtype=torch.float32)
    if axis == "x":
        m[1, 1], m[1, 2], m[2, 1], m[2, 2] = c, -s, s, c
    elif axis == "y":
        m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    elif axis == "z":
        m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
    else:                                              # pragma: no cover
        raise ValueError(f"未知轴：{axis}")
    return m


def rotation_x(deg: float) -> torch.Tensor:
    """绕 x 轴旋转（角度制，4×4 齐次）。"""
    return _rot("x", deg)


def rotation_y(deg: float) -> torch.Tensor:
    """绕 y 轴旋转。"""
    return _rot("y", deg)


def rotation_z(deg: float) -> torch.Tensor:
    """绕 z 轴旋转。"""
    return _rot("z", deg)


def orbit_w2c(w2c, center, yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> torch.Tensor:
    """让相机**绕 `center` 转**，返回新的 `w2c`。

    世界绕 `center` 旋转 `R`，等价于相机反向绕它转：

        p' = center + R (p - center) = M p,   M = T(center) R T(-center)
        w2c_new = w2c @ M

    因此 `center` 的投影位置**完全不动**，只有观察角度改变；`yaw=pitch=0`
    时恒等返回。若把旋转直接乘在 `w2c` 左侧（`R @ w2c`），那是绕**世界原点**
    转 —— 头心不在原点时大角度会把头甩出画面（踩过）。

    Args:
        w2c: `[4,4]` world→camera（任意设备/类型）
        center: `[3]` 旋转中心（世界坐标），通常取当前帧网格的质心
        yaw_deg: 偏航（绕世界 y 轴）
        pitch_deg: 俯仰（绕世界 x 轴）

    Returns:
        `[4,4]` CPU float32
    """
    c = as_cpu(center).reshape(3)
    R = rotation_x(pitch_deg) @ rotation_y(yaw_deg)
    M = torch.eye(4, dtype=torch.float32)
    M[:3, :3] = R[:3, :3]
    M[:3, 3] = c - R[:3, :3] @ c
    return as_cpu(w2c) @ M


def roll_w2c(w2c, roll_deg: float) -> torch.Tensor:
    """绕相机自身光轴滚转（相当于歪头看）。

    在**相机空间**左乘旋转：只改变成像的旋转，不移动相机中心。
    """
    return rotation_z(roll_deg) @ as_cpu(w2c)


def project_point(K, w2c, point) -> tuple[float, float]:
    """把世界点投到像素坐标，用于自检（如"头心是否还在画面内"）。"""
    p = torch.cat([as_cpu(point).reshape(3), torch.ones(1)])
    cam = (as_cpu(w2c) @ p)[:3]
    if float(cam[2]) == 0.0:
        raise ValueError("投影点在相机平面上（z=0），无法投影")
    uv = as_cpu(K) @ (cam / cam[2])
    return float(uv[0]), float(uv[1])
