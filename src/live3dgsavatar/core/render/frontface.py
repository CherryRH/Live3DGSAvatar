"""前表面高斯门控。

RGBAvatar/3DGS 的高斯没有天然背面剔除；背面高斯可能在正面视角下
贡献颜色，到了大角度就变成“正面很好、侧面破洞/糊”。这里用高斯的
最短轴（硬表面阶段已对齐 FLAME 法线）和相机方向做点积，只保留朝向
相机的高斯不透明度。
"""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F

from ..types import Camera, GaussianSet


def _quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(dim=-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    return torch.stack([
        torch.stack([ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)], -1),
        torch.stack([2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)], -1),
        torch.stack([2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz], -1),
    ], dim=-2)


def frontface_gaussians(gaussians: GaussianSet, camera: Camera,
                        mesh=None, face_id=None) -> GaussianSet:
    """把背向相机的高斯 opacity 置 0（保留参数，不破坏梯度/存储）。

    硬表面高斯的旋转轴方向有 ± 二义性；若给出 mesh/face_id，直接用
    绑定面的几何法线判断朝向，避免把正面一起剔掉。
    """
    if gaussians.space != "world":
        raise ValueError("frontface_gaussians 需要世界空间高斯")
    if mesh is not None and face_id is not None:
        tri = mesh.verts[:, mesh.faces[face_id]]       # [B,N,3,3]
        fn = torch.cross(tri[:, :, 1] - tri[:, :, 0],
                         tri[:, :, 2] - tri[:, :, 0], dim=-1)
        normal = F.normalize(fn, dim=-1)
    else:
        rot = _quat_to_matrix(gaussians.rotation)      # [B,N,3,3]
        scale = gaussians.scaling
        onehot = F.one_hot(scale.argmin(dim=-1), 3).to(scale.dtype)
        normal = (rot * onehot[..., None, :]).sum(-1)  # [B,N,3]
    view = camera.position.to(normal.device, normal.dtype)[:, None, :] - gaussians.xyz
    front = (normal * view).sum(-1) > 0                 # [B,N]
    return replace(gaussians, opacity=gaussians.opacity * front[..., None])
