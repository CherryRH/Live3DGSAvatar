"""RDF 跟踪结果目录 → 渲染帧。

跟踪目录的约定与数据集相同（`docs/CONVENTIONS.md` §6）：

```
<目录>/images/ + render_inputs.npz
```

`render_inputs.npz` 有两种 schema，按内容判别：

- `mesh` + `blend_weight`：已烘焙，可直接渲染（数据集、增强集）
- `x` + `verts`：RDF 跟踪输出（5023 顶点、118 维），需补牙与映射为 129 维

后者是 `scripts/rdf_dump.py` 的产物：RDF 的虚拟相机与数据集的相机一致，
所以不必做任何相机/尺度换算，只需补牙齿与参数映射。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

#: 可渲染目录里的帧文件（与数据集同名）
FRAMES_FILE = "render_inputs.npz"

#: 旧命名，仍可读但会提示改名
LEGACY_FRAMES_FILES = ("rdf.npz",)

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
