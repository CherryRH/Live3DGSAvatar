"""相机矩阵与内核之间的边界转换。

**本模块是「行主序 → 列主序」转置的唯一发生地**（见 docs/CONVENTIONS.md §2.2）。

参照实现把这个转置散落在每个调用点（`camera.get_w2v.transpose(0, 1)` 与
`camera.get_full_proj.transpose(0, 1)`），并被三份重复的渲染函数各复制一遍。
集中到此处后，任何矩阵布局问题只需检查一个函数。
"""

from __future__ import annotations

import torch


def to_kernel_matrix(m: torch.Tensor) -> torch.Tensor:
    """把行主序的 `[4, 4]` 相机矩阵转成内核所需的列主序连续张量。

    Args:
        m: `[4, 4]`（或 `[B, 4, 4]`，此时返回同形状），`p_clip = M @ p_world` 语义

    Returns:
        转置后的连续张量。CUDA 内核按列主序读取，故必须先转置，
        否则会出现「相机绕某轴镜像 / 渲染全黑」这类典型症状。
    """
    if m.shape[-2:] != (4, 4):
        raise ValueError(f"期望 [..., 4, 4]，实际 {tuple(m.shape)}")
    return m.transpose(-1, -2).contiguous()


def project_points(
    points: torch.Tensor,      # [..., 3]
    full_proj: torch.Tensor,   # [4, 4] 或 [B, 4, 4]
    width: int,
    height: int,
    in_viewport: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把世界点投到 NDC（或像素）。

    与参照实现 `Camera.project_points` 同语义：先右乘 `full_projᵀ`，
    再做透视除法，最后用 `0 < z < 1` 过滤。

    Returns:
        (points_proj [..., 3], filter_mask [...])
    """
    pts = torch.nn.functional.pad(points, [0, 1], value=1.0)
    proj = full_proj.to(pts.device, pts.dtype)
    proj_pts = pts @ proj.transpose(-1, -2)
    proj_pts = proj_pts[..., :3] / (proj_pts[..., 3:4] + 1e-7)
    mask = (proj_pts[..., 2] > 0.0) & (proj_pts[..., 2] < 1.0)
    if in_viewport:
        proj_pts = proj_pts.clone()
        proj_pts[..., 0] = (proj_pts[..., 0] * 0.5 + 0.5) * width
        proj_pts[..., 1] = (proj_pts[..., 1] * 0.5 + 0.5) * height
    return proj_pts, mask
