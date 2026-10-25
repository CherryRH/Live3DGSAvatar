"""在 CAP4D 生成的图上跑我们自己的链路（**让 RDF 拟合图，而不是自己算网格**）。

## 为什么不用 CAP4D 的 `flame/*.npz` + 自己的相机换算

试过一次，PSNR 只有 2.4 dB —— 头都不在同一块区域。原因是那条路要同时对齐
四件东西：CAP4D 的裁剪框 → 内参、它那套 FLAME2023 的 identity、姿态约定、
以及我们模型的世界空间。任何一处偏差都会整体错位，而错位时 PSNR 完全没有
诊断价值。

## 现在的做法

**让 RDF 去拟合这张图**。RDF 的相机（fov_y=30°、位于 (0,0,1)、
`rot=diag(1,-1,-1)`）与我们数据集的相机**完全一致**（`f=955.4` 就是
`0.5*512/tan(15°)`），所以：

    CAP4D 图 → RDF 拟合 → (x, 5023 顶点) → 补牙齿 + 129 维 → 用数据集相机渲染

中间不需要任何相机/尺度/identity 的换算，网格与相机天然自洽。
这也是实时链路本来的形态（RDF 出顶点，我们补牙、映射、渲染）。

## 分工

- `scripts/rdf_dump.py`（realdenseface 环境）：图 → `x` + `verts`
- 本模块（live3dgs 环境）：`verts` → 5083 顶点；`x` → 129 维驱动参数
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

#: RDF 的原生顶点数（补牙前的网格）
RDF_VERTS = 5023


def load_rdf_dump(path: Path) -> dict:
    """读 `scripts/rdf_dump.py` 产出的 npz。

    必须含 `x (T,118)`、`verts (T,5023,3)`、`names (T,)`；
    `image_*` / `camera_*` 字段仅用于核对，本模块不依赖它们
    （相机取自数据集，与 RDF 的虚拟相机一致）。
    """
    path = Path(path)
    with np.load(path, allow_pickle=True) as d:
        missing = {"x", "verts", "names"} - set(d.files)
        if missing:
            raise ValueError(f"{path} 缺少字段：{sorted(missing)}")
        out = {
            "x": np.asarray(d["x"], dtype=np.float32),
            "verts": np.asarray(d["verts"], dtype=np.float32),
            "names": [str(n) for n in d["names"]],
        }
        if "image_width" in d.files:
            out["width"] = int(np.asarray(d["image_width"]).reshape(-1)[0])
            out["height"] = int(np.asarray(d["image_height"]).reshape(-1)[0])
    if out["verts"].shape[1] != RDF_VERTS:
        raise ValueError(
            f"verts 应为 RDF 的 {RDF_VERTS} 顶点，实际 {out['verts'].shape}")
    if out["x"].shape[1] != 118:
        raise ValueError(f"x 应为 118 维，实际 {out['x'].shape}")
    return out


def to_frames(dump: dict, flame_dir: Path, device: str | torch.device = "cpu"):
    """把 RDF 的网格与参数拼成渲染要的 `frames`。

    两件已经验证过的转换：
    - **补牙齿**：RDF 只有 5023 顶点，模型是 5083（`tracking/mesh_build.py`）
    - **参数映射**：118 维 `x` → 129 维 `blend_weight`（`tracking/param_map.py`）
    """
    from ..tracking.mesh_build import TeethBuilder
    from ..tracking.param_map import to_blend_weight

    teeth = TeethBuilder(Path(flame_dir))
    frames: list[dict] = []
    for i in range(dump["verts"].shape[0]):
        base = torch.as_tensor(dump["verts"][i], dtype=torch.float32, device=device)
        mesh = teeth.build(base.to(torch.float64)).to(torch.float32)
        weight = to_blend_weight(
            torch.as_tensor(dump["x"][i], dtype=torch.float32, device=device))
        frames.append({"mesh": mesh, "blend_weight": weight.reshape(-1),
                       "name": dump["names"][i]})
    return frames


def images_dir(cap4d_dir: Path) -> Path:
    """CAP4D 生成图所在目录（兼容给到 `generated_images` 这一层）。"""
    cap4d_dir = Path(cap4d_dir)
    for cand in (cap4d_dir / "generated_images", cap4d_dir):
        if (cand / "images").is_dir():
            return cand / "images"
    raise FileNotFoundError(f"{cap4d_dir} 下找不到 generated_images/images")
