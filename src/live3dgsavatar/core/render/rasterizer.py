"""光栅化封装。**整个代码库中唯一允许出现 CUDA 扩展调用的地方。**

对上层只暴露 `Rasterizer` 协议与两个实现：

- :class:`SimpleRasterizer` —— 推理：逐帧前向，不分配梯度缓冲
- :class:`BatchRasterizer`  —— 训练：`BatchGaussianRasterizer` + 预分配缓冲 + 多流

背景色语义（与参照实现对齐）见 docs/CONVENTIONS.md §5.1。
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from ..types import Camera, GaussianSet, RenderOutput
from .camera_utils import to_kernel_matrix


def _normalize_bg(bg_color: torch.Tensor, batch_size: int) -> torch.Tensor:
    """把背景色规整为 `[B, 3]`。

    接受 `[3]` / `[1, 3]` / `[B, 3]`；`[B,3,1,1]`（训练态随机背景）会被 reshape 成 `[B,3]`。
    参照实现内部做 `bg_color.reshape(-1, 3)` 后按需 repeat，此处保持同一语义。
    """
    bg = bg_color.reshape(-1, 3)
    if bg.shape[0] == 1 and batch_size != 1:
        bg = bg.repeat(batch_size, 1)
    if bg.shape[0] != batch_size:
        raise ValueError(
            f"背景色行数 {bg.shape[0]} 与 batch {batch_size} 不匹配")
    return bg.contiguous()


def _screenspace_points(xyz: torch.Tensor) -> torch.Tensor:
    """`means2D`：仅用于接住屏幕空间梯度；推理态关闭梯度。

    形状与 `xyz` 相同（单帧 `[N,3]`）。
    """
    if torch.is_grad_enabled() and xyz.requires_grad:
        pts = torch.zeros_like(xyz, requires_grad=True) + 0
        pts.retain_grad()
        return pts
    return torch.zeros_like(xyz)


class SimpleRasterizer:
    """推理用：逐帧构造 `GaussianRasterizer`。

    对应参照实现的 `diff_renderer.render_gs_batch`（单帧 `render_gs` 的批循环）。
    """

    def __init__(self, sh_degree: int = 0, scaling_modifier: float = 1.0) -> None:
        self.sh_degree = sh_degree
        self.scaling_modifier = scaling_modifier

    def render(
        self,
        gaussians: GaussianSet,
        camera: Camera,
        bg_color: torch.Tensor,
        target_image: Optional[torch.Tensor] = None,
    ) -> RenderOutput:
        from diff_gaussian_rasterization import (  # noqa: PLC0415
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        if gaussians.space != "world":
            raise ValueError(
                f"光栅化器只接受世界空间高斯，实际 space={gaussians.space!r}")

        b, n = gaussians.batch_size, gaussians.num_gaussians
        if camera.batch_size not in (1, b):
            raise ValueError(
                f"相机 batch {camera.batch_size} 与高斯 batch {b} 不匹配")

        bg = _normalize_bg(bg_color, b)
        means2d = _screenspace_points(gaussians.xyz)

        colors, alphas, est_colors, est_weights, radii = [], [], [], [], []
        for i in range(b):
            ci = 0 if camera.batch_size == 1 else i
            settings = GaussianRasterizationSettings(
                image_height=int(camera.height),
                image_width=int(camera.width),
                tanfovx=float(camera.tan_fov_x[ci]),
                tanfovy=float(camera.tan_fov_y[ci]),
                bg=bg[i],
                scale_modifier=self.scaling_modifier,
                viewmatrix=to_kernel_matrix(camera.w2c[ci]),
                projmatrix=to_kernel_matrix(camera.full_proj[ci]),
                sh_degree=self.sh_degree,
                campos=camera.position[ci].contiguous(),
                prefiltered=False,
                debug=False,
            )
            rasterizer = GaussianRasterizer(raster_settings=settings)
            color, alpha, est_color, est_weight, radius = rasterizer(
                means3D=gaussians.xyz[i],
                means2D=means2d[i],
                shs=gaussians.color[i],
                colors_precomp=None,
                opacities=gaussians.opacity[i],
                scales=gaussians.scaling[i],
                rotations=gaussians.rotation[i],
                cov3D_precomp=None,
                target_image=target_image[i] if target_image is not None else None,
            )
            colors.append(color)
            alphas.append(alpha)
            est_colors.append(est_color)
            est_weights.append(est_weight)
            radii.append(radius)

        return RenderOutput(
            color=torch.stack(colors),
            alpha=torch.stack(alphas),
            est_color=torch.stack(est_colors),
            est_weight=torch.stack(est_weights),
            radii=torch.stack(radii),
        )


class BatchRasterizer:
    """训练用：`BatchGaussianRasterizer` + 预分配张量 + 多 CUDA 流。

    对应参照实现的 `diff_renderer.batch_gaussian.BatchGaussianRenderer`。
    缓冲在构造时按 `max_batch_size` / `max_gaussian_size` 一次性分配，
    每步只做 zero_ + 调用，避免反复申请显存。

    Args:
        max_batch_size / max_gaussian_size: 预分配上限，超出即报错
        num_streams: 批并行使用的 CUDA 流数（参照实现固定为 3）
    """

    def __init__(
        self,
        camera: Camera,
        max_batch_size: int,
        max_gaussian_size: int,
        bg_color: torch.Tensor,
        sh_degree: int = 0,
        scaling_modifier: float = 1.0,
        num_streams: int = 3,
    ) -> None:
        from diff_gaussian_rasterization import (  # noqa: PLC0415
            BatchGaussianRasterizer,
            GaussianRasterizationSettings,
        )

        self.max_batch_size = max_batch_size
        self.max_gaussian_size = max_gaussian_size
        self.scaling_modifier = scaling_modifier
        self.sh_degree = sh_degree

        bg = _normalize_bg(bg_color, 1)
        ci = 0
        self._settings = GaussianRasterizationSettings(
            image_height=int(camera.height),
            image_width=int(camera.width),
            tanfovx=float(camera.tan_fov_x[ci]),
            tanfovy=float(camera.tan_fov_y[ci]),
            bg=bg[0],
            scale_modifier=scaling_modifier,
            viewmatrix=to_kernel_matrix(camera.w2c[ci]),
            projmatrix=to_kernel_matrix(camera.full_proj[ci]),
            sh_degree=sh_degree,
            campos=camera.position[ci].contiguous(),
            prefiltered=False,
            debug=False,
        )
        self._rasterizer = BatchGaussianRasterizer(
            max_gaussian_size=max_gaussian_size,
            max_batch_size=max_batch_size,
            raster_settings=self._settings,
        )
        if num_streams != 3:
            # 参照实现的流数硬编码为 3；此处显式提示而非静默忽略
            self._rasterizer.raster_tensors.stream_list = [
                torch.cuda.Stream() for _ in range(num_streams)
            ]

    def render(
        self,
        gaussians: GaussianSet,
        bg_color: torch.Tensor,
        target_image: Optional[torch.Tensor] = None,
    ) -> RenderOutput:
        b, n = gaussians.batch_size, gaussians.num_gaussians
        if b > self.max_batch_size:
            raise ValueError(f"batch {b} 超过预分配上限 {self.max_batch_size}")
        if n != self.max_gaussian_size:
            raise ValueError(
                f"高斯数 {n} 与预分配 {self.max_gaussian_size} 不一致；"
                "训练态要求每帧高斯数恒定（绑定后不增删）")

        bg = bg_color.reshape(-1, 3)
        if bg.shape[0] == 1:
            bg = bg.repeat(b, 1)
        # 训练态需要 [B,3] 的背景（随机背景色时逐样本不同）
        self._rasterizer.raster_settings.bg = bg.contiguous()

        g = GaussianSet(
            xyz=gaussians.xyz.contiguous(),
            rotation=gaussians.rotation.contiguous(),
            scaling=gaussians.scaling.contiguous(),
            opacity=gaussians.opacity.contiguous(),
            color=gaussians.color.contiguous(),
            space="world",
        )
        means2d = _screenspace_points(g.xyz)
        if target_image is not None:
            target_image = target_image.contiguous()

        color, alpha, est_color, est_weight, radii = self._rasterizer(
            means3D=g.xyz,
            means2D=means2d,
            opacities=g.opacity,
            shs=g.color,
            scales=g.scaling,
            rotations=g.rotation,
            target_image=target_image,
        )
        return RenderOutput(color=color, alpha=alpha, est_color=est_color,
                            est_weight=est_weight, radii=radii)
