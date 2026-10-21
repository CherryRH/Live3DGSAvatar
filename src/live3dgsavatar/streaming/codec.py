"""帧编码：把 `[H,W,3] uint8` 压成 JPEG 字节。

## 为什么需要

原始 RGB 一帧 512×512×3 = **768 KiB**。同机回环无所谓（855 MiB/s），
但跨网络时 30 fps 就要 22.5 MiB/s（约 190 Mbps），超过多数家宽上行。
实测在本项目的渲染图（黑底人像）上，JPEG q=85 只有 **24 KiB**（32×），
编码 0.5 ms —— 带宽立刻不再是瓶颈。

## 为什么是 JPEG 而不是 PNG/WebP

- PNG 无损，但在这类图上仍有几百 KiB，且编码慢一个数量级
- WebP 压缩率更好，但编码明显更慢（PIL 里尤其），而我们要的是**每帧都编**
- JPEG 质量损失对"预览 + 审查"足够，且浏览器解码有硬件加速

## 依赖

只用 Pillow（项目已依赖，用于读写数据集图像）。没有它时 `jpeg_available()`
返回 False，调用方应回退到原始 RGB 而不是崩掉。
"""

from __future__ import annotations

import io

import numpy as np

#: 默认质量。85 是"肉眼看不出"的常用档，本项目实测 24 KiB/帧。
DEFAULT_QUALITY = 85


def jpeg_available() -> bool:
    """Pillow 是否可用。"""
    try:
        import PIL.Image  # noqa: F401
        return True
    except ImportError:                                        # pragma: no cover
        return False


def encode_jpeg(rgb: np.ndarray, quality: int = DEFAULT_QUALITY) -> bytes:
    """把 `[H,W,3] uint8` 编成 JPEG 字节。

    Args:
        rgb: 行优先 RGB，`uint8`（`core` 的渲染输出就是这个形状）
        quality: 1–100；越高质量越大。默认 85

    Raises:
        ValueError: 形状或 dtype 不对
        RuntimeError: 没装 Pillow
    """
    from PIL import Image

    arr = np.ascontiguousarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"rgb 形状应为 [H,W,3]，实际 {tuple(arr.shape)}")
    if arr.dtype != np.uint8:
        raise ValueError(f"rgb 应为 uint8，实际 {arr.dtype}")

    q = int(quality)
    if not (1 <= q <= 100):
        raise ValueError(f"quality 应在 1–100，实际 {q}")

    buf = io.BytesIO()
    # `optimize=False`：那是多趟 Huffman 优化，省 3–5% 但慢一倍，不值得
    Image.fromarray(arr, mode="RGB").save(
        buf, format="JPEG", quality=q, optimize=False)
    return buf.getvalue()
