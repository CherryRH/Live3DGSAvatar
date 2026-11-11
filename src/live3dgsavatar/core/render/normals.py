"""屏幕空间法线图：从当前帧的 FLAME 网格渲染出来，供几何条件扩散使用。

与 `faceid.face_id_map` 用同一套光栅化约定（nvdiffrast、y 翻转），
输出**相机空间**单位法线，编码成 `[0,1]` 的 RGB —— 即 ControlNet 期望的格式。

用途：给单视角扩散模型一个像素级对齐的几何条件（"这里应该有表面、朝向如何"），
这样它才能在模型没铺高斯的区域生成与姿态一致的纹理。条件来自 FLAME 网格，
而不是渲染出的高斯，所以孔洞不会让条件也一起消失。
"""

from __future__ import annotations

import torch

from .faceid import to_clip

__all__ = ["vertex_normals", "normal_map", "encode_normals"]


def vertex_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """`[1, V, 3]` 顶点 → `[1, V, 3]` 平滑法线（按面积加权，朝外）。"""
    faces = faces.to(torch.int64)
    v = verts[0]
    f = v[faces]                                   # [F, 3, 3]
    n = torch.cross(f[:, 1] - f[:, 0], f[:, 2] - f[:, 0], dim=-1)  # 面积加权
    out = torch.zeros_like(v)
    for k in range(3):
        out.index_add_(0, faces[:, k], n)
    norm = out.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return (out / norm).unsqueeze(0)


def encode_normals(n: torch.Tensor, convention: str = "opengl") -> torch.Tensor:
    """相机空间法线 → `[0,1]` 编码（`n*0.5+0.5`）。

    `opengl`（默认）会先把 `(x, y, z)` 转成 `(x, -y, -z)`：本项目的相机是
    +y 朝下、+z 朝前，而 ControlNet 训练用的法线是 +y 朝上、+z 朝向观察者。
    不转换的话法线图看起来"上下颠倒"，生成结果也会跟着错。
    """
    unit = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    if convention == "opengl":
        unit = unit * torch.tensor([1.0, -1.0, -1.0], device=n.device,
                                   dtype=n.dtype)
    return (unit * 0.5 + 0.5).clamp(0, 1)


def normal_map(verts: torch.Tensor, faces: torch.Tensor, camera,
               glctx, flip_y: bool = False,
               convention: str = "opengl") -> tuple[torch.Tensor, torch.Tensor]:
    """渲染相机空间法线图。

    Args:
        verts: `[1, V, 3]` 当前帧世界坐标顶点（FLAME 网格）
        faces: `[F, 3]`
        camera: batch 为 1 的 `Camera`
        glctx: nvdiffrast 上下文
        flip_y: 同 `faceid`；默认 `False`（实测直接送入即与渲染对齐）

    Args:
        convention: `"opengl"`（默认，给 ControlNet 用）或 `"camera"`（本项目
            相机约定：+x 右、+y 下、+z 朝前）。两者差一个 `(x, -y, -z)`：
            3DGS 的相机是 +y 朝下、+z 朝前，而 ControlNet 训练用的法线是
            +y 朝上、+z 朝向观察者。**编码前必须转换**，否则法线图看起来
            像"上下颠倒"（G 通道反了），生成结果也会跟着错。

    Returns:
        `(rgb, mask)`：`rgb` 为 `[H, W, 3]`、值域 `[0,1]` 的相机空间法线图
        （背景置 0）；`mask` 为 `[H, W]` bool，表示该像素落在网格上。
    """
    if convention not in ("opengl", "camera"):
        raise ValueError(f"convention 只能是 opengl/camera，实际 {convention!r}")
    import nvdiffrast.torch as dr

    pos = to_clip(verts, camera)
    if flip_y:
        pos = pos.clone()
        pos[..., 1] = -pos[..., 1]
    rast, _ = dr.rasterize(glctx, pos, faces.to(torch.int32),
                           resolution=(int(camera.height), int(camera.width)))

    # 世界法线 → 相机空间：R = w2c[:3, :3]
    world_n = vertex_normals(verts, faces)
    rot = camera.w2c[0, :3, :3].to(world_n.dtype)
    cam_n = world_n[0] @ rot.transpose(0, 1)
    interp, _ = dr.interpolate(cam_n.unsqueeze(0).contiguous(), rast,
                               faces.to(torch.int32))
    mask = rast[0, ..., 3] > 0
    rgb = encode_normals(interp[0], convention)
    rgb = rgb * mask.unsqueeze(-1).to(rgb.dtype)
    return rgb, mask
