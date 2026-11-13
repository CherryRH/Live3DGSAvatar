"""「几何 + 驱动参数 → 图像」的唯一组合路径。

`deform` → `rasterize` 只在 `render_frame` / `render_gaussians` 里组合。
GUI、评测、缺陷定位、训练、后训练都走这里，避免各处漏掉背景色、缩放或
无梯度上下文。需要复用同一份形变结果渲染多个相机时走 `render_gaussians`。
调用方只决定"用哪些帧、什么相机、要不要梯度"。
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

from ..types import Camera, GaussianSet, Mesh, RenderOutput

if TYPE_CHECKING:  # 仅类型；避免 core 内部循环导入
    from ..avatar import GaussianAvatar
    from .rasterizer import SimpleRasterizer


def _synchronize() -> None:
    """计时前同步 CUDA（无 GPU 时是 no-op）。"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def render_gaussians(
    gaussians: GaussianSet,
    rasterizer: "SimpleRasterizer",
    camera: Camera,
    background: torch.Tensor,
    *,
    target_image: torch.Tensor | None = None,
    build_graph: bool = False,
) -> RenderOutput:
    """只做光栅化：已经 `deform` 好的高斯 → 图像。

    需要复用同一份形变结果渲染多个相机（showcase、批量诊断）时用它；
    `render_frame` 是它的“deform + 本函数”组合。
    """
    ctx = torch.enable_grad() if build_graph else torch.no_grad()
    with ctx:
        return rasterizer.render(gaussians, camera, background,
                                 target_image=target_image)


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
    timings: dict[str, float] | None = None,
    sync: bool = False,
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
        timings: 若给出，写入 `deform` / `rasterize` 两个毫秒数。
        sync: 计时前是否 `torch.cuda.synchronize()`；仅 `timings` 给定时有意义。

    Returns:
        `RenderOutput`；``build_graph=False`` 时其张量不带梯度。
    """
    ctx = torch.enable_grad() if build_graph else torch.no_grad()
    with ctx:
        if sync:
            _synchronize()
        t0 = time.perf_counter()
        gaussians: GaussianSet = avatar.deform(mesh, blend_weight)
        if sync:
            _synchronize()
        if timings is not None:
            timings["deform"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        out = rasterizer.render(gaussians, camera, background,
                                target_image=target_image)
        if sync:
            _synchronize()
        if timings is not None:
            timings["rasterize"] = (time.perf_counter() - t0) * 1000.0
        return out
