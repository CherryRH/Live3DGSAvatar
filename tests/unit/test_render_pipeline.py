"""渲染管线的**纯 torch** 回归测试（不需要 CUDA 扩展）。

`render_frame` 是训练/评测/GUI 共用的那一条路径，但它默认不开前表面门控；
`frontface=True` 这条分支只在少数入口用到，历史上就因为"函数被删了、调用点
没改"而留过一颗哑弹。这里用假的 avatar / rasterizer 把两条分支都走一遍，
保证调用点与实际 API 对得上。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import live3dgsavatar  # noqa: E402, F401
from live3dgsavatar.core.render.pipeline import render_frame  # noqa: E402
from live3dgsavatar.core.types import (Camera, GaussianSet, Mesh,  # noqa: E402
                                       RenderOutput)


class _FakeAvatar:
    """只提供 `deform` 的最小 avatar；门控需要 binding_face_id。"""

    def __init__(self, n: int = 4) -> None:
        self.num_gaussians = n
        self.binding_face_id = torch.zeros(n, dtype=torch.long)

    def deform(self, mesh: Mesh, blend_weight=None) -> GaussianSet:
        n = self.num_gaussians
        b = mesh.batch_size
        return GaussianSet(
            xyz=torch.zeros(b, n, 3),
            rotation=torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(b, n, 1),
            scaling=torch.full((b, n, 3), 0.001),
            opacity=torch.full((b, n, 1), 0.5),
            color=torch.full((b, n, 1, 3), 0.5),
            space="world",
        )


class _FakeRasterizer:
    """记录收到的高斯数，返回固定尺寸的假渲染结果。"""

    def __init__(self) -> None:
        self.calls = 0

    def render(self, gaussians, camera, bg_color, target_image=None):
        self.calls += 1
        b = gaussians.batch_size
        return RenderOutput(color=torch.zeros(b, 3, camera.height, camera.width),
                            alpha=torch.ones(b, 1, camera.height, camera.width))


def _fixture():
    mesh = Mesh(
        verts=torch.tensor([[[0.0, 0.0, 0.0],
                             [0.1, 0.0, 0.0],
                             [0.0, 0.1, 0.0]]]),
        faces=torch.tensor([[0, 1, 2]]),
        uvs=torch.zeros(3, 2),
        uv_faces=torch.tensor([[0, 1, 2]]),
    )
    camera = Camera(K=torch.eye(3), w2c=torch.eye(4), width=8, height=8)
    return mesh, camera


def test_render_frame_without_frontface_renders() -> None:
    mesh, camera = _fixture()
    rasterizer = _FakeRasterizer()
    out = render_frame(_FakeAvatar(), rasterizer, mesh, None, camera,
                       torch.zeros(3))
    assert rasterizer.calls == 1
    assert out.color.shape == (1, 3, 8, 8)


def test_render_frame_frontface_branch_uses_existing_api() -> None:
    """`frontface=True` 必须能走到真实实现（曾因删函数留下断引用）。"""
    mesh, camera = _fixture()
    rasterizer = _FakeRasterizer()
    out = render_frame(_FakeAvatar(), rasterizer, mesh, None, camera,
                       torch.zeros(3), frontface=True)
    assert rasterizer.calls == 1
    assert out.alpha.shape == (1, 1, 8, 8)


def test_frontface_gaussians_only_zeroes_opacity() -> None:
    """门控只能把 opacity 置 0，不能改形状/数值（形状契约由调用方依赖）。"""
    from live3dgsavatar.core.render.frontface import frontface_gaussians

    mesh, camera = _fixture()
    avatar = _FakeAvatar()
    gs = avatar.deform(mesh)
    out = frontface_gaussians(gs, camera, mesh, avatar.binding_face_id)

    assert out.opacity.shape == gs.opacity.shape
    assert out.xyz.shape == gs.xyz.shape
    keep = (out.opacity == gs.opacity) | (out.opacity == 0)
    assert bool(keep.all()), "门控只允许把 opacity 清零，不应改写数值"
