"""驱动参数求解：外部追踪器（RDF）→ 本项目可渲染的驱动参数与网格。

分层约定：`tracking/` 可依赖 `core/types`，但**不得**被 `core/` 导入 ——
驱动来源（数据集 / 摄像头 / 桥接进程）是可替换的外部依赖。

模块：

- `param_map`：RDF 的 118 维 `x` → 129 维 `blend_weight`（纯函数）
- `align`    ：RDF 坐标空间 → 数据集坐标空间的相似变换（纯函数）
"""

from . import align, param_map

__all__ = ["align", "param_map"]
