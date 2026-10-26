"""可渲染目录的读取：`images/` + `render_inputs.npz`。

数据集、增强集、CAP4D 生成图用**同一套目录约定**，npz 内容按键判别：

- `mesh` + `blend_weight`：已烘焙，可直接渲染
- `x` + `verts`：RDF 跟踪结果（5023 顶点、118 维），需补牙与映射

下面保留了最初针对 CAP4D 生成图的说明（为什么让 RDF 拟合，而不是自己算网格）。

---

在 CAP4D 生成的图上跑我们自己的链路（**让 RDF 拟合图，而不是自己算网格**）。

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

#: 可渲染目录里的帧文件（与数据集同名）
FRAMES_FILE = "render_inputs.npz"

#: 旧命名，仍可读但会提示改名
LEGACY_FRAMES_FILES = ("rdf.npz", "cap4d_rdf.npz")

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


def find_frames_file(directory: Path) -> Path:
    """在目录里找帧文件：优先 `render_inputs.npz`，其次旧命名。

    Raises:
        FileNotFoundError: 都不存在，报错里写明该目录应有的结构
    """
    directory = Path(directory)
    canonical = directory / FRAMES_FILE
    if canonical.is_file():
        return canonical
    for name in LEGACY_FRAMES_FILES:
        legacy = directory / name
        if legacy.is_file():
            import logging

            logging.getLogger(__name__).warning(
                "%s 使用旧文件名 %s；建议改名为 %s",
                directory, name, FRAMES_FILE)
            return legacy
    raise FileNotFoundError(
        f"{directory} 里没有 {FRAMES_FILE}。可渲染目录的约定是：\n"
        f"  {directory}/images/        图像\n"
        f"  {directory}/{FRAMES_FILE}   帧数据（mesh 或 RDF 的 x/verts）")


def frames_schema(path: Path) -> str:
    """判别 npz 内容：``"baked"``（mesh/blend_weight）或 ``"rdf"``（x/verts）。"""
    with np.load(Path(path), allow_pickle=True) as d:
        keys = set(d.files)
    if {"mesh", "blend_weight"} <= keys:
        return "baked"
    if {"x", "verts"} <= keys:
        return "rdf"
    raise ValueError(
        f"{path} 既不是烘焙格式（mesh/blend_weight）也不是 RDF 格式（x/verts）；"
        f"实际字段 {sorted(keys)}")


def load_frames_dir(directory: Path, flame_dir: Path,
                    device: str | torch.device = "cpu") -> list[dict]:
    """读一个可渲染目录，返回渲染要的 `frames`。"""
    path = find_frames_file(directory)
    if frames_schema(path) == "rdf":
        return to_frames(load_rdf_dump(path), Path(flame_dir), device=device)

    with np.load(path, allow_pickle=True) as d:
        mesh = np.asarray(d["mesh"], dtype=np.float32)
        weight = np.asarray(d["blend_weight"], dtype=np.float32)
        names = ([str(n) for n in d["names"]] if "names" in d.files
                 else [f"{i:05d}" for i in range(mesh.shape[0])])
    return [{"mesh": torch.as_tensor(mesh[i], device=device),
             "blend_weight": torch.as_tensor(weight[i], device=device),
             "name": names[i]} for i in range(mesh.shape[0])]


def images_dir(cap4d_dir: Path) -> Path:
    """帧目录里的 `images/`：给定目录本身，或它的 `generated_images/` 子目录。"""
    cap4d_dir = Path(cap4d_dir)
    for cand in (cap4d_dir, cap4d_dir / "generated_images"):
        if (cand / "images").is_dir():
            return cand / "images"
    raise FileNotFoundError(
        f"{cap4d_dir} 下找不到 images/（可直接给 generated_images 目录）")
