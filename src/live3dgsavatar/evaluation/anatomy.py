"""高斯解剖：容量都用在哪了，以及还能重分配到哪。

这里是**纯统计**，不碰 GPU、不读文件：输入张量/数组，输出可直接进报告的
字典与掩码。GPU 相关的部分（离面距离、逐面颜色采样）留在脚本里。

为什么单独一层：这次诊断（不透明度/离面距离/尺度/覆盖率 + 消融）以后会反复用，
不应该每次重写；同时"死点重分配"也要用同一套 `dead_mask` /
`faces_without_gaussians`，两边口径必须一致。
"""

from __future__ import annotations

import numpy as np

from ..core.color import (DEAD_OPACITY, SH_C0, rgb_from_sh_dc,
                          sh_dc_from_rgb)

__all__ = [
    "DEAD_OPACITY",
    "SH_C0",
    "summarize",
    "histogram_counts",
    "dead_mask",
    "faces_without_gaussians",
    "sh_dc_from_rgb",
    "rgb_from_sh_dc",
    "allocate_targets",
]


def summarize(values, *, quantiles=(50, 90, 99)) -> dict:
    """`[N]` 的分布摘要：min / mean / max / 分位数。空数组返回 `{"n": 0}`。"""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return {"n": 0}
    out = {"n": int(v.size), "min": float(v.min()), "mean": float(v.mean()),
           "max": float(v.max())}
    for q in quantiles:
        out[f"p{int(q)}"] = float(np.percentile(v, float(q)))
    return out


def histogram_counts(values, edges) -> dict:
    """按给定边界统计直方图，返回 `{"edges": [...], "counts": [...]}`。"""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    e = np.asarray(edges, dtype=np.float64).reshape(-1)
    if e.size < 2:
        raise ValueError("edges 至少要有两个边界")
    if np.any(np.diff(e) <= 0):
        raise ValueError("edges 必须严格递增")
    counts, _ = np.histogram(v, bins=e)
    return {"edges": [float(x) for x in e], "counts": [int(x) for x in counts]}


def dead_mask(opacity, eps: float = DEAD_OPACITY) -> np.ndarray:
    """物理不透明度 `< eps` 的高斯（返回 bool `[N]`）。"""
    return np.asarray(opacity, dtype=np.float64).reshape(-1) < float(eps)


def faces_without_gaussians(face_id, n_faces: int) -> np.ndarray:
    """哪些三角面一个高斯都没绑（bool `[F]`）。

    绑定关系来自 `avatar.binding_face_id`（每个高斯绑一个面）；这些"空面"
    是容量可以优先重分配的位置。
    """
    ids = np.asarray(face_id).reshape(-1).astype(np.int64)
    n = int(n_faces)
    if n <= 0:
        raise ValueError(f"n_faces 必须为正，实际 {n_faces}")
    used = np.unique(ids[(ids >= 0) & (ids < n)])
    mask = np.ones(n, dtype=bool)
    mask[used] = False
    return mask


def allocate_targets(target_faces, n_points: int) -> np.ndarray:
    """把 `n_points` 个空位分给 `target_faces`（**一面一个**，保持简单）。

    点数多于面数时只分配前 `n_points` 个面；少于时截断。返回要绑定的面 id
    `[n]`。刻意不做"按面积加权/一面多个"：那属于容量分配策略，等原型验证
    有效后再作为独立策略加进来，不在第一版里混入未经验证的启发式。
    """
    faces = np.asarray(target_faces).reshape(-1).astype(np.int64)
    n = int(n_points)
    if n <= 0 or faces.size == 0:
        return np.zeros(0, dtype=np.int64)
    return faces[:n].copy()
