"""按三角面分配采样容量：把"一个 texel 一个高斯"换成"一个面若干高斯"。

这是重构的第一步，只做**采样点的分配**，不涉及训练循环：

    counts[f]  = 面 f 分到几个高斯（按 UV 面积，至少 1 个）
    (face_id, bary)  每个高斯的绑定（面内任意子位置，不再限于 texel 中心）

这样采样密度不再由 `tex_size` 决定，而是可以按 `density` 独立设定；同时每个
非退化三角面至少有一个采样点，消掉"30% 的三角面没有 texel"。

## 为什么用低差异序列

同一面上多个采样点若随机撒，会扎堆；用确定性的低差异序列（黄金比例 + 平方根
映射）能在任意 `n` 下都铺得比较均匀，且**跨运行可复现**——训练与评测的采样
位置不会因为随机种子而漂移。

## 与旧路径的关系

旧路径 `build_binding` 用 UV 光栅化得到 texel 中心，`valid_mask` 是 `[tex_size²]`。
新路径的 `valid_mask` 是全 True 的 `[N]`（每个分配到的点都有效），其余字段一致，
所以 `GaussianAvatar` 与渲染路径不需要改动。
"""

from __future__ import annotations

import numpy as np

__all__ = ["face_area_uv", "face_capacity", "allocate_bary", "capacity_binding",
           "capacity_summary"]

#: 黄金比例共轭，用于低差异序列的第二维
_GOLDEN = 0.6180339887498949


def face_area_uv(uvs, uv_faces) -> np.ndarray:
    """每个三角面的 UV 面积（`[F]`，归一化 UV 单位）。"""
    uv = np.asarray(uvs, dtype=np.float64)
    tri = uv[np.asarray(uv_faces, dtype=np.int64)]
    return 0.5 * np.abs(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]))


def face_capacity(uvs, uv_faces, tex_size: int, *, density: float = 2.0,
                  min_count: int = 1) -> np.ndarray:
    """每个面分到的高斯数（`[F]` int64）。

    Args:
        uvs / uv_faces: 模板 UV
        tex_size: 基准分辨率；`tex_size²` 个 texel 对应 `density=1`
        density: 采样密度倍数（相对"每 texel 一个点"）
        min_count: 面积大于 0 的面至少分几个（`1` 保证没有空面）

    Returns:
        每面的采样数；UV 面积退化为 0 的面返回 0
    """
    if density <= 0:
        raise ValueError(f"density 必须为正，实际 {density}")
    if min_count < 0:
        raise ValueError(f"min_count 不能为负，实际 {min_count}")
    area = face_area_uv(uvs, uv_faces)
    counts = np.ceil(area * (tex_size ** 2) * float(density)).astype(np.int64)
    counts = np.maximum(counts, int(min_count))
    counts[area <= 0] = 0
    return counts


def allocate_bary(counts, *, share_axis: str = "u") -> tuple[np.ndarray, np.ndarray]:
    """把 per-face 计数展开成 `(face_id, bary)`。

    Args:
        counts: `[F]` 每面采样数

    Returns:
        `face_id [N]` int64、`bary [N,3]` float32，`bary` 之和为 1 且非负
    """
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    face_id = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    bary = np.empty((total, 3), dtype=np.float32)
    cursor = 0
    for f, n in enumerate(counts):
        if n <= 0:
            continue
        i = np.arange(n, dtype=np.float64)
        # 第一维分层、第二维黄金比例：任意 n 下都铺得开
        u1 = (i + 0.5) / n
        u2 = np.mod(i * _GOLDEN, 1.0)
        s = np.sqrt(u1)
        bary[cursor:cursor + n, 0] = 1.0 - s
        bary[cursor:cursor + n, 1] = s * (1.0 - u2)
        bary[cursor:cursor + n, 2] = s * u2
        cursor += n
    return face_id, bary


def capacity_binding(uvs, uv_faces, tex_size: int, *, density: float = 2.0,
                     min_count: int = 1, device: str = "cpu"):
    """按容量分配直接构造 `Binding`（不经过 UV 光栅化）。"""
    import torch

    from .bind import Binding

    counts = face_capacity(uvs, uv_faces, tex_size, density=density,
                           min_count=min_count)
    face_id, bary = allocate_bary(counts)
    return Binding(
        face_id=torch.from_numpy(face_id).to(device),
        face_bary=torch.from_numpy(bary).to(device),
        valid_mask=torch.ones(len(face_id), dtype=torch.bool, device=device),
    )


def capacity_summary(uvs, uv_faces, tex_size: int,
                     densities=(1.0, 2.0, 3.0)) -> list[dict]:
    """不同密度下的容量概况（离线报告用）。"""
    area = face_area_uv(uvs, uv_faces)
    out = []
    for d in densities:
        counts = face_capacity(uvs, uv_faces, tex_size, density=d)
        out.append({
            "density": d,
            "total": int(counts.sum()),
            "per_face_median": int(np.median(counts)),
            "faces_with_one": float((counts == 1).mean()),
            "zero_faces": int((counts == 0).sum()),
            "area_uv": float(area.sum()),
        })
    return out
