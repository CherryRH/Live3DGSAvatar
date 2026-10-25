"""图像指标：PSNR / 最大绝对误差 / 分组统计。

**只处理 `[0,255]` uint8 的 HWC 图像**，纯 numpy，不依赖 torch 与 CUDA ——
所以渲染测试、在线更新对比、GUI 实时对比可以共用同一套口径。

## 口径（全项目唯一）

- 像素范围 0–255，MSE 按通道平均
- 两张**完全相同**的图的 PSNR 记为 `99.0`（而不是 inf）：inf 会让
  "中位数/平均值"这类聚合失去意义，也会在 JSON 里变成 `Infinity`
- 掩码内 PSNR 只统计 `mask` 为真的像素；mask 为空返回 ``None``

`psnr_pair` 是热路径版本：GUI 每帧都要算，实测"原地相减 + einsum"比
`((a-b)**2).mean()` 快 4 倍以上（13 ms → 2.8 ms），且结果一致。
"""

from __future__ import annotations

import numpy as np

#: 完全相同时的 PSNR 上限（dB）
PSNR_CAP = 99.0


def _psnr_from_mse(mse: float) -> float:
    return PSNR_CAP if mse <= 0.0 else float(10.0 * np.log10((255.0 ** 2) / mse))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """整幅 PSNR（dB）。"""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    diff = a - b
    return _psnr_from_mse(float(np.einsum("ijk,ijk->", diff, diff)) / diff.size)


def psnr_masked(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None) -> float | None:
    """只在 `mask` 为真的像素上算 PSNR；mask 为空返回 ``None``。"""
    if mask is None or not bool(np.any(mask)):
        return None
    a = np.asarray(a, dtype=np.float32)[mask]
    b = np.asarray(b, dtype=np.float32)[mask]
    diff = a - b
    return _psnr_from_mse(float(np.einsum("ij,ij->", diff, diff)) / diff.size)


def psnr_pair(render: np.ndarray, target: np.ndarray,
              mask: np.ndarray | None = None):
    """一次算出 `(全图 PSNR, 掩码内 PSNR)`；尺寸不一致返回 `(None, None)`。

    GUI 每帧都调用，所以做了两件事：只分配一块临时数组（原地相减）、
    用 einsum 求平方和。
    """
    a = np.asarray(render)
    b = np.asarray(target)
    if a.shape != b.shape:
        return None, None

    diff = a.astype(np.float32)
    np.subtract(diff, b, out=diff)
    full = _psnr_from_mse(float(np.einsum("ijk,ijk->", diff, diff)) / diff.size)
    if mask is None or not bool(np.any(mask)):
        return full, None
    sub = diff[mask]
    return full, _psnr_from_mse(float(np.einsum("ij,ij->", sub, sub)) / sub.size)


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """最大绝对误差（0–255 尺度），用来抓"局部崩了但 PSNR 还凑合"。"""
    return float(np.abs(np.asarray(a, np.int16) - np.asarray(b, np.int16)).max())


class Accum:
    """累积一组标量，输出中位/最小/最大/均值。

    `add(None)` 会被忽略 —— 指标函数在"算不出来"时返回 ``None``，
    调用方不必逐个判空。
    """

    def __init__(self) -> None:
        self.values: list[float] = []

    def add(self, value: float | None) -> None:
        if value is not None and not np.isnan(value):
            self.values.append(float(value))

    def summary(self) -> dict | None:
        if not self.values:
            return None
        v = np.array(self.values)
        return {"median": float(np.median(v)), "min": float(v.min()),
                "max": float(v.max()), "mean": float(v.mean()),
                "n": int(v.size)}

    def __len__(self) -> int:
        return len(self.values)
