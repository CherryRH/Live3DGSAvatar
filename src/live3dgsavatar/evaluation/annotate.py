"""输出图片/视频里的文字标签统一用这个字体。"""

from __future__ import annotations

from functools import lru_cache

#: 标签统一字号（原来用 PIL 默认字体，约 11 px，看不清）
LABEL_SIZE = 22


@lru_cache(maxsize=8)
def label_font(size: int = LABEL_SIZE):
    """尽量用 DejaVuSans；Pillow 新版可直接给默认字体指定字号。"""
    from PIL import ImageFont

    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()
