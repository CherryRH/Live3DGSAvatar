"""「几何 + 驱动参数 → 图像」的唯一组合路径。

**为什么要有这个函数**：同一条组合（`deform` → `rasterize`）以前在
`app/session.py`、`scripts/render_test.py`、`scripts/profile_deform.py`
里各写一遍。热更新也要用它，如果各处自己拼装，**任何一处漏掉
`scaling_modifier` 或背景色都会造成"同一模型两处结果不一致"**。

因此把顺序固定在这里，调用方只决定"用哪些帧、什么相机、要不要梯度"。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..types import Camera, GaussianSet, Mesh, RenderOutput

if TYPE_CHECKING:  # 仅类型；避免 core 内部循环导入
    from ..avatar import GaussianAvatar
    from .rasterizer import SimpleRasterizer


def render_frame(
    avatar: "GaussianAvatar",
    rasterizer: "SimpleRasterizer",
    mesh: Mesh,
    blend_weight: torch.Tensor,
    camera: Camera,
    background: torch.Tensor,
    *,
    target_image: torch.Tensor | None = None,
    build_graph: bool = False,
) -> RenderOutput:
    """单次前向：网格 + 驱动参数 → 高斯 → 图像。

    Args:
        avatar: 参数容器
        rasterizer: 光栅化器（推理用 `SimpleRasterizer`）
        mesh: `[B, V, 3]` 世界空间网格
        blend_weight: `[B, D]` 驱动参数
        camera: 与该批帧匹配的相机
        background: `[3]` 或 `[B, 3]` 背景色
        target_image: 训练态才需要（改造者用它算 `est_color`）
        build_graph: ``True`` 时**保留计算图**（热更新/训练需要反传）。
            **默认 False**：推理路径不该承担构图开销。

    Returns:
        `RenderOutput`；``build_graph=False`` 时其张量不带梯度。
    """
    ctx = torch.enable_grad() if build_graph else torch.no_grad()
    with ctx:
        gaussians: GaussianSet = avatar.deform(mesh, blend_weight)
        return rasterizer.render(gaussians, camera, background,
                                 target_image=target_image)
