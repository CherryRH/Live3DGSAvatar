"""核心层：类型契约、变形算子、光栅化封装。

依赖规则（见 docs/ARCHITECTURE.md §2.3）：
- 本层**不得** import `data/`、`training/`、`app/`、`tracking/`；
- 本层**不得**做文件 I/O 与参数解析；
- 只有 `core.render` 允许接触 CUDA 扩展。
"""

from . import deform, io, render  # noqa: F401
from .avatar import AvatarConfig, GaussianAvatar
from .types import (
    Camera,
    Frame,
    GaussianSet,
    Mesh,
    RenderOutput,
    Space,
)

__all__ = [
    "AvatarConfig",
    "Camera",
    "Frame",
    "GaussianAvatar",
    "GaussianSet",
    "Mesh",
    "RenderOutput",
    "Space",
    "deform",
    "io",
    "render",
]
