"""训练、验证、评测共用的**唯一**前向入口。

之前三者各写一遍：训练走 `deform → clamp → frontface → render_gaussians`，
验证直接 `render_frame` —— 早停选的模型和训练时的模型不是同一个。把"怎么
构造高斯、要不要裁尺度、要不要剔背面"收成一个 `ForwardOptions`，调用方只
声明意图。

用法::

    opts = ForwardOptions(build_graph=True, frontface=True,
                          clamp_max_scale=0.002)
    res = forward_avatar(avatar, rasterizer, mesh, weight, camera, bg,
                         options=opts, target_image=target)
    res.out.color / res.out.alpha / res.gaussians
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..core.avatar import GaussianAvatar
    from ..core.render import RenderOutput, SimpleRasterizer
    from ..core.types import Camera, Mesh


@dataclass(frozen=True)
class ForwardOptions:
    """前向的行为开关（训练与验证必须用同一份）。

    Attributes:
        build_graph: 是否保留计算图（训练 True，验证/评测 False）。
        frontface: 是否剔除背面高斯（几何初始化/排查时可用）。
        clamp_max_scale: 切向尺度硬上限（米）；``None`` 表示不裁。
    """

    build_graph: bool = False
    frontface: bool = False
    clamp_max_scale: float | None = None


@dataclass
class ForwardResult:
    """一次前向的结果：渲染输出 +（可选）中间高斯。

    `gaussians` 只在需要几何约束时非空——走快速路径
    （`render_frame`）时没有中间高斯可拿。
    """

    out: "RenderOutput"
    gaussians: object | None = None


def forward_avatar(avatar: "GaussianAvatar", rasterizer: "SimpleRasterizer",
                   mesh: "Mesh", weight, camera: "Camera", background,
                   *, options: ForwardOptions | None = None,
                   target_image=None) -> ForwardResult:
    """按 `options` 跑一次前向；训练/验证/评测都从这里走。"""
    from ..core.render import render_frame, render_gaussians
    from ..core.render.frontface import frontface_gaussians

    opts = options or ForwardOptions()
    if opts.frontface or opts.clamp_max_scale is not None:
        gs = avatar.deform(mesh, weight)
        if opts.clamp_max_scale is not None:
            gs.scaling = gs.scaling.clamp(max=float(opts.clamp_max_scale))
        if opts.frontface:
            gs = frontface_gaussians(gs, camera, mesh, avatar.binding_face_id)
        out = render_gaussians(gs, rasterizer, camera, background,
                               target_image=target_image,
                               build_graph=opts.build_graph)
        return ForwardResult(out, gs)
    out = render_frame(avatar, rasterizer, mesh, weight, camera, background,
                       target_image=target_image,
                       build_graph=opts.build_graph)
    return ForwardResult(out, None)
