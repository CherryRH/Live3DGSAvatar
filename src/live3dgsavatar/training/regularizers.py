"""训练期的结构正则：让 UV 邻域上的参数保持连贯。

## 为什么需要

`xyz`（每个 texel 的切空间偏移）是自由参数，只受光度项约束。实测基线模型上：

- 偏移中位 19 mm，而相邻 texel 的偏移差（UV 拉普拉斯）中位 **16.7 mm**——
  也就是说邻域之间几乎互不相关，偏移场是一片高频噪声；
- 这种"每个 texel 各干各的"解在观测视角下能拟合，但换个角度就容易互相错位。

这里的正则只有一句话：**每个 texel 的参数应当接近它 UV 邻域的平均**。它不改表示、
不改格式，只是在损失里加一项。

## 量纲

用**相对**拉普拉斯：`mean|v_i − mean(v_nb)| / mean|v_i|`，无量纲，便于给权重。
基线模型上这个值约 0.86（噪声很大），平滑后应显著下降。
"""

from __future__ import annotations

import numpy as np

__all__ = ["uv_adjacency", "laplacian_penalty"]


def uv_adjacency(uv, *, k: int = 4, max_dist: float = 1.5 / 256.0):
    """按 UV 距离取每个 texel 的邻居。

    Args:
        uv: `[N, 2]` 每个高斯的 UV 坐标（由 `bary @ face_uv` 得到）
        k: 邻居数
        max_dist: 超过该 UV 距离视为"不在邻域"（避免跨 UV 岛误连）；
            默认 1.5 个 texel（tex_size=256 时约 0.006）

    Returns:
        `[N, k]` int64 邻居下标；找不到足够邻居时用自身补齐（自环不产生梯度）
    """
    from scipy.spatial import cKDTree

    uv = np.asarray(uv, dtype=np.float64)
    n = len(uv)
    tree = cKDTree(uv)
    dist, idx = tree.query(uv, k=min(k + 1, n))
    if idx.ndim == 1:
        idx = idx[:, None]
        dist = dist[:, None]
    out = np.repeat(np.arange(n)[:, None], k, axis=1)
    for i in range(n):
        cand, d = idx[i, 1:], dist[i, 1:]
        take = np.nonzero(d <= max_dist)[0][:k]
        if take.size == 0:
            continue                      # 孤立点：保留自环，惩罚为 0
        out[i, :take.size] = cand[take]
        # 邻居不足时用**已有邻居**补齐，不要用自身（否则会稀释邻域均值）
        out[i, take.size:] = cand[take[0]]
    return out


def laplacian_penalty(values, neighbors):
    """相对拉普拉斯：`mean|v_i − mean(v_nb)| / (mean|v_i| + eps)`。

    Args:
        values: `[N, C]` 参数（如 `xyz` 或 log 尺度）
        neighbors: `[N, k]` 邻居下标（`uv_adjacency` 的输出）

    Returns:
        标量张量；`values` 全零时返回 0，避免 0/0。
    """
    import torch

    v = values
    nb = neighbors.to(v.device)
    neigh_mean = v[nb].mean(dim=1)                    # [N, C]，自环不影响
    diff = (v - neigh_mean).abs().mean()
    scale = v.abs().mean().clamp(min=1e-8)
    return diff / scale
