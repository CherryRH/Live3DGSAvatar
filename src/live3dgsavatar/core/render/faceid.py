"""屏幕空间的人脸 id 图：把每个像素归到"哪个三角面"。

用途是**把图像空间的缺陷归因到几何/texel**：有了 id 图，模糊、孔洞、精修差分
这些逐像素的量就能累加到三角面（进而到 texel），于是"视角 × 表情"的组合爆炸
被消掉了 —— 缺陷是表示空间的一个量，视角只是访问它的途径。

用 nvdiffrast 在屏幕空间光栅化，与 UV 域的 `build_binding` 用的是同一个库、
同一套约定。

⚠️ **y 方向**：我们的投影矩阵是 3DGS 约定（NDC 的 +y 向下），nvdiffrast 的
`rasterize` 输出按图像行序（第 0 行是图像顶部）。实测表明**不需要翻转**：
直接送入就能与渲染结果对齐；翻转反而会上下镜像（真机上验证过：
法线图/掩码出现"脖子朝上、头顶朝下"）。

因此 `flip_y` 默认 `False`。若换相机约定或 nvdiffrast 版本，用
`detect_flip()` 用 alpha 轮廓自动判定，别靠推导。
"""

from __future__ import annotations

import torch

__all__ = ["to_clip", "rasterize_faces", "face_id_map", "detect_flip"]


def to_clip(verts: torch.Tensor, camera) -> torch.Tensor:
    """世界坐标顶点 → 裁剪空间 `[1, V, 4]`（与 `Camera.full_proj` 同语义）。

    Args:
        verts: `[1, V, 3]` 世界坐标
        camera: `core.types.Camera`（batch 为 1）

    Returns:
        `[1, V, 4]`，其中 `p_clip = full_proj @ [p, 1]`
    """
    if verts.ndim != 3 or verts.shape[-1] != 3:
        raise ValueError(f"verts 应为 [B, V, 3]，实际 {tuple(verts.shape)}")
    proj = camera.full_proj.to(verts.device, verts.dtype)
    if proj.shape[0] != verts.shape[0]:
        raise ValueError(
            f"相机批大小 {proj.shape[0]} 与顶点批大小 {verts.shape[0]} 不一致")
    homo = torch.nn.functional.pad(verts, [0, 1], value=1.0)
    return homo @ proj.transpose(-1, -2)


def rasterize_faces(clip_verts: torch.Tensor, faces: torch.Tensor,
                    resolution: tuple[int, int], glctx,
                    flip_y: bool = False) -> torch.Tensor:
    """光栅化面 id，返回 `[H, W]` int32，`-1` 表示背景。

    Args:
        clip_verts: `[1, V, 4]` 裁剪空间顶点（`to_clip` 的输出）
        faces: `[F, 3]` 三角面索引
        resolution: `(H, W)`
        glctx: nvdiffrast 上下文
        flip_y: 是否把 y 取反（见模块说明，默认要翻）
    """
    import nvdiffrast.torch as dr  # noqa: PLC0415

    pos = clip_verts
    if flip_y:
        pos = pos.clone()
        pos[..., 1] = -pos[..., 1]
    rast, _ = dr.rasterize(glctx, pos, faces.to(torch.int32),
                           resolution=(int(resolution[0]), int(resolution[1])))
    # nvdiffrast 的 id 从 1 开始，0 表示无覆盖 → 转成 -1 表示背景
    ids = rast[0, ..., 3].to(torch.int32) - 1
    return ids


def detect_flip(verts: torch.Tensor, faces: torch.Tensor, camera, glctx,
                alpha, threshold: float = 0.5) -> tuple[bool, dict]:
    """用 alpha 轮廓判定是否需要翻 y，返回 `(flip, {选项: IoU})`。

    光栅化方向不该靠推导（投影约定 + 库的行序很容易错），直接用渲染出来的
    alpha 轮廓对齐：哪个方向的重合度高就用哪个。一次运行判一次即可。
    """
    import numpy as np

    a = np.asarray(alpha) > threshold
    scores = {}
    for flip in (False, True):
        ids = face_id_map(verts, faces, camera, glctx, flip_y=flip)
        m = ids.detach().cpu().numpy() >= 0
        union = int((m | a).sum())
        scores[flip] = float((m & a).sum()) / union if union else 1.0
    best = max(scores, key=scores.get)
    return bool(best), scores


def face_id_map(verts: torch.Tensor, faces: torch.Tensor, camera,
                glctx, flip_y: bool = False) -> torch.Tensor:
    """世界顶点 + 相机 → `[H, W]` 面 id 图（`-1` 为背景）。"""
    clip = to_clip(verts, camera)
    return rasterize_faces(clip, faces, (camera.height, camera.width), glctx,
                           flip_y=flip_y)
