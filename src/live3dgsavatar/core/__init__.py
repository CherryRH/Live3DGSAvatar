"""核心层：类型契约、变形算子、光栅化封装。

依赖规则（见 docs/ARCHITECTURE.md）：
- 本层**不得** import `data/`、`training/`、`app/`、`tracking/`；
- 本层**不得**做文件 I/O 与参数解析（`core/io/` 的 PLY 序列化除外）；
- CUDA 扩展只允许出现在 `core/render/` 与 `core/deform/` 的显式白名单文件。
"""

from . import deform, io, render  # noqa: F401
from .avatar import AvatarConfig, GaussianAvatar
from .types import (
    Camera,
    GaussianSet,
    Mesh,
    RenderOutput,
    Space,
)

__all__ = [
    "AvatarConfig",
    "Camera",
    "GaussianAvatar",
    "GaussianSet",
    "Mesh",
    "RenderOutput",
    "Space",
    "deform",
    "io",
    "render",
]
