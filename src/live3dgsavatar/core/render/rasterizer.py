"""光栅化封装。**整个代码库中唯一允许出现 CUDA 扩展调用的地方。**

对上层只暴露 :class:`SimpleRasterizer`：逐帧前向，不分配梯度缓冲。
训练/推理/热更新共用这一条路径，避免多套光栅化实现产生不可比的结果。

背景色语义与参照实现对齐。
"""

from __future__ import annotations

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


def _normalize_target_image(
        target_image: Optional[torch.Tensor], camera: Camera,
        batch_size: int) -> Optional[torch.Tensor]:
    """把训练态的 `target_image` 规整成内核要求的 **RGBA 4 通道**。

    `diff_gaussian_rasterization` 的改造版在 kernel 里固定按 RGBA 读取::

        target_image[3 * H * W + pix_id] > 0.5   // 第 4 通道 = 写回掩码

    C++ 绑定同样按 `batch_index * 4 * H * W` 偏移。只传 3 通道 RGB 时，
    内核会越过张量末尾读满一整张平面 —— 是否越界取决于显存分配位置，
    表现为随机的 `CUDA error: an illegal memory access`（曾让多轮训练/验证
    在起始步或存档时崩掉，且难以复现）。这里统一补齐第 4 通道，把问题
    收敛在唯一允许调用 CUDA 扩展的地方。
    """
    if target_image is None:
        return None
    if target_image.ndim != 4 or target_image.shape[1] not in (3, 4):
        raise ValueError(
            "target_image 形状应为 [B,C,H,W]，C 为 3（RGB）或 4（RGBA），"
            f"实际 {tuple(target_image.shape)}")
    if int(target_image.shape[0]) != int(batch_size):
        raise ValueError(
            f"target_image batch {target_image.shape[0]} 与高斯 batch "
            f"{batch_size} 不匹配")
    h, w = int(camera.height), int(camera.width)
    if (int(target_image.shape[2]), int(target_image.shape[3])) != (h, w):
        raise ValueError(
            f"target_image 分辨率 {tuple(target_image.shape[2:])} 与相机 "
            f"{(h, w)} 不匹配")
    if target_image.shape[1] == 4:
        return target_image
    # 缺省第 4 通道取 1：整幅图都允许写回 est_color/est_weight。
    # 需要"只在掩码内写回"的调用方应自己传 4 通道。
    alpha = torch.ones_like(target_image[:, :1])
    return torch.cat([target_image, alpha], dim=1)


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
        target_image = _normalize_target_image(target_image, camera, b)
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
