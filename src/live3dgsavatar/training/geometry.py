"""硬表面几何正则：让高斯贴近 FLAME 表面、法向对齐、法向尺度短。

这里只做**逐高斯**的几何约束，不依赖光栅化：
- 绑定面 + 重心坐标给出每个高斯的表面锚点；
- 面法线给出表面朝向；
- 四元数给出的最短轴朝向表面法线；
- 最短轴尺度被压到切向尺度以下。

这些量都能直接反传，适合硬表面阶段从 UV/FLAME 初始化开始训练。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..core.types import GaussianSet, Mesh


def face_normals(mesh: Mesh) -> torch.Tensor:
    """`[B,F,3]` 三角形面法线（未归一化）。"""
    tri = mesh.verts[:, mesh.faces]
    n = torch.cross(tri[:, :, 1] - tri[:, :, 0],
                    tri[:, :, 2] - tri[:, :, 0], dim=-1)
    return n


def _quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """WXYZ 单位四元数 → `[...,3,3]` 旋转矩阵。"""
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


def hard_surface_terms(
    gaussians: GaussianSet,
    mesh: Mesh,
    face_id: torch.Tensor,
    face_bary: torch.Tensor,
    *,
    max_scale: float = 0.0012,
) -> dict[str, torch.Tensor]:
    """返回硬表面几何项；每项均为可反传标量。

    Args:
        gaussians: `[B,N,·]` 世界空间高斯（`space="world"`）
        mesh: `[B,V,3]` 当前帧网格
        face_id: `[N]` 绑定的面下标
        face_bary: `[N,3]` 重心坐标
    """
    if gaussians.space != "world":
        raise ValueError("hard_surface_terms 需要世界空间高斯")
    tri = mesh.verts[:, mesh.faces[face_id]]                 # [B,N,3,3]
    bary = face_bary.to(device=tri.device, dtype=tri.dtype)  # [N,3]
    anchor = (tri * bary[None, :, :, None]).sum(dim=-2)      # [B,N,3]

    # 1) 中心到 FLAME 表面的距离
    point = (gaussians.xyz - anchor).pow(2).sum(-1).mean()

    # 2) 法向：最短轴 vs 面法线
    fn = face_normals(mesh)                                  # [B,F,3]
    fn = F.normalize(fn, dim=-1)[:, face_id]                 # [B,N,3]
    rot = _quat_to_matrix(gaussians.rotation)                # [B,N,3,3]
    scale = gaussians.scaling.clamp_min(1e-12)
    axis_idx = scale.argmin(dim=-1)                          # [B,N]
    onehot = F.one_hot(axis_idx, 3).to(scale.dtype)          # [B,N,3]
    axis = (rot * onehot[..., None, :]).sum(-1)              # [B,N,3]
    normal_axis = (scale * onehot).sum(-1)                   # [B,N]
    tangent_values = torch.where(
        onehot.bool(), torch.full_like(scale, -1.0), scale)
    tangent = torch.where(
        onehot.bool(), torch.full_like(scale, float("inf")), scale)
    tangent_min = tangent.min(dim=-1).values                  # [B,N]
    tangent_max = tangent_values.max(dim=-1).values           # [B,N]
    cos = (axis * fn).sum(-1).abs().clamp(max=1.0)
    normal = (1.0 - cos).mean()
    # 3) 法向尺度应短于切向：目标 normal/tangent <= ratio
    ratio = normal_axis / tangent_min.clamp_min(1e-12)
    scale_short = torch.relu(ratio - 0.5).pow(2).mean()
    # 4) 小圆片：切向不要超过上限，且两轴尽量相等
    size = torch.relu(tangent_max - float(max_scale)).pow(2).mean()
    tangent_for_log = torch.where(
        onehot.bool(), torch.ones_like(scale), scale)
    tangent_logs = torch.log(tangent_for_log.clamp_min(1e-12))
    # 用 tanh 归一化，避免大比值时 clamp 把梯度截成 0。
    log_ratio = (tangent_logs.max(dim=-1).values
                 - tangent_logs.min(dim=-1).values)
    iso = torch.tanh(0.5 * log_ratio).pow(2).mean()
    return {"surface_point": point, "surface_normal": normal,
            "surface_scale": scale_short, "surface_size": size,
            "surface_iso": iso}
