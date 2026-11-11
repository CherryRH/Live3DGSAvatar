"""颜色与物理量的共享约定：SH DC ↔ RGB、死点不透明度阈值。

放在 `core/` 是因为 `training/` 与 `evaluation/` 都要用，但两者之间不允许互相
依赖（见 `docs/ARCHITECTURE.md` §2.3）。这里只有纯 numpy 运算，不做 I/O。
"""

from __future__ import annotations

import numpy as np

__all__ = ["SH_C0", "DEAD_OPACITY", "sh_dc_from_rgb", "rgb_from_sh_dc"]

#: SH DC 的归一化常数（与 3DGS / 参照实现一致）
SH_C0 = 0.28209479177387814

#: 物理不透明度低于此值视为"死点"：对渲染的贡献可忽略。
#: biden/train1 实测 51.9% 的高斯落在这一档，删掉它们渲染逐像素不变。
DEAD_OPACITY = 0.01


def sh_dc_from_rgb(rgb) -> np.ndarray:
    """`[0,1]` RGB → PLY 里存的 SH DC 系数（与 `extract_texture` 互逆）。"""
    return (np.asarray(rgb, dtype=np.float64) - 0.5) / SH_C0


def rgb_from_sh_dc(f_dc) -> np.ndarray:
    """SH DC 系数 → `[0,1]` RGB。"""
    return np.asarray(f_dc, dtype=np.float64) * SH_C0 + 0.5
