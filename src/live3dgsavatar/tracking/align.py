"""RDF 坐标空间 → 数据集坐标空间的对齐（相似变换）。

## 为什么需要它

RDF 与数据集用的是**同一套相机**：

| | RDF | 数据集（`render_inputs.npz`） |
|---|---|---|
| 内参 | `fov_y = 30°`、512² → fx = 256/tan(15°) = **955.4** | `K[0,0] = 955.4` |
| 相机旋转 | `diag(1, -1, -1)` | `w2v` 的旋转 = `diag(1, -1, -1)` |
| 相机位置 | `(0, 0, 1)` | `w2v` 推出 `(0, 0, 1)` |

内参与相机位姿都相同，**但重建出的头部深度不同**（RDF 的 `camera_distance=1.0`
只是初值，深度由单目先验决定）。在相机内参固定时，沿相机中心整体缩放
**不改变投影** —— 所以两者的差异必然落在"绕相机中心的相似变换"上。

于是有两个层次的模型，按由简到繁依次验证：

1. `fit_scale_about_camera`：只有一个参数 `s`（`p = cam + s·(p - cam)`）。
   若这一层就能拟合到像素级，就没有必要引入更多自由度。
2. `fit_similarity`：完整的 `p = s·R·p + t`（Umeyama）。作为兜底，
   也用来检验"是否只是缩放"这个假设。

**先验越多越好**：先验少（1 个参数）时拟合不会被噪声带偏，
而且外推到未见数据时更稳。所以先用 1 参数模型，不够再放开。
"""

from __future__ import annotations

import torch


def fit_similarity(src: torch.Tensor, dst: torch.Tensor) -> dict[str, torch.Tensor]:
    """Umeyama 最小二乘相似变换：求 `s, R, t` 使 `dst ≈ s·R·src + t`。

    **不含反射**（`det(R) = +1`）：镜像会把左右脸对调，得到"看起来还行
    但耳朵左右颠倒"的结果，且误差指标不会明显变差。

    Args:
        src: `[N, 3]` 源点
        dst: `[N, 3]` 目标点（与 src 逐点对应）

    Returns:
        `{"scale": float, "rotation": [3,3], "translation": [3]}`
    """
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(
            f"需要两个同形状的 [N,3]，实际 {tuple(src.shape)} vs {tuple(dst.shape)}")
    if src.shape[0] < 3:
        raise ValueError(f"至少需要 3 个点，实际 {src.shape[0]}")

    a = src.to(torch.float64)
    b = dst.to(torch.float64)

    mu_a = a.mean(dim=0)
    mu_b = b.mean(dim=0)
    ac = a - mu_a
    bc = b - mu_b

    var_a = (ac ** 2).sum() / a.shape[0]
    if float(var_a) <= 0:
        raise ValueError("源点全部重合，无法拟合相似变换")

    cov = (bc.T @ ac) / a.shape[0]

    u, d, vt = torch.linalg.svd(cov)
    sign = torch.ones(3, dtype=torch.float64)
    if float(torch.det(u) * torch.det(vt)) < 0:
        sign[-1] = -1.0
    rot = u @ torch.diag(sign) @ vt
    scale = float((d * sign).sum() / var_a)
    trans = mu_b - scale * (rot @ mu_a)

    return {"scale": scale, "rotation": rot, "translation": trans}


def apply_similarity(points: torch.Tensor, params: dict) -> torch.Tensor:
    """`p' = s·R·p + t`（保持输入 dtype）。"""
    p = points.to(torch.float64)
    rot = params["rotation"]
    out = params["scale"] * (p @ rot.T) + params["translation"]
    return out.to(points.dtype)


def fit_scale_about_camera(
    src: torch.Tensor,
    dst: torch.Tensor,
    camera_center: torch.Tensor | None = None,
) -> float:
    """单参数模型：只允许"绕相机中心缩放"，求最优 `s`。

    `p' = c + s·(p - c)`，其中 `c` 是相机中心。
    这是相机内参与位姿都相同时，**唯一不改变投影**的自由度 ——
    若数据与这个模型吻合，说明 RDF 只是把深度估偏了，没有别的系统误差。

    Args:
        src: `[N, 3]` RDF 坐标
        dst: `[N, 3]` 数据集坐标
        camera_center: `[3]`；默认 `(0, 0, 1)`（两边的相机位置）

    Returns:
        标量 `s`
    """
    if camera_center is None:
        camera_center = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    c = camera_center.to(torch.float64)

    a = src.to(torch.float64) - c
    b = dst.to(torch.float64) - c
    denom = (a ** 2).sum()
    if float(denom) <= 0:
        raise ValueError("源点全部落在相机中心，无法求缩放")
    return float((a * b).sum() / denom)


def residuals(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """对齐后的残差统计（单位与坐标一致，通常是"米"量级）。"""
    d = (pred.to(torch.float64) - target.to(torch.float64)).norm(dim=-1)
    return {
        "max": float(d.max()),
        "mean": float(d.mean()),
        "p95": float(torch.quantile(d, 0.95)),
    }
