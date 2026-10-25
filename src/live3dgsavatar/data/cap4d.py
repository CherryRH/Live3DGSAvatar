"""把 CAP4D 的生成结果读成"我们能渲染的一帧"。

CAP4D 的产物是：`generated_images/images/NNNNN.png`（512²，**白底**）
加 `generated_images/flame/NNNNN.npz`（每帧的 `shape/expr/rot/tra/eye_rot/`
`fx/fy/cx/cy/extr/crop_box`）。本模块把它转成我们渲染链路要的
`(mesh, blend_weight, K)`。

## 三处约定要对齐

**① 相机**：CAP4D 在投影前把顶点做 `diag(1,-1,-1)` 翻转，`extr=I`；
而我们的数据集本来就是 `w2v = diag(1,-1,-1)`、相机在原点。两者**恰好一致**，
所以这里固定 `R = diag(1,-1,-1)`、`T = 0`，只有 `K` 随帧变化。

**② 内参**：生成图是**裁剪后**的 512² 图（`crop_box` 是正方形，见
`get_square_bbox`），所以 `scale = target / side`、`cx' = (cx - x0) * scale`。

**③ 头姿**：CAP4D 的 `rot/tra` 是"世界空间里的头姿"。我们训练侧的约定是
**root=单位阵、把旋转放在 neck 关节**，并平移修正
`transl = R·j_neck − j_neck + t`（见 `data/checkpoint_loader.py`）。
本模块直接复用同一式子，这样得到的网格与训练时的网格在同一个空间。

## 表情

CAP4D 的 `expr` 是 65 维，其中**前 64 维是 FLAME 表情、第 65 维是眨眼**
（`flowface` 把基的最后一列换成了 `blink_blendshape`）。我们的
`blend_weight` 要 100 维表情 + 2 维眼睑，所以：前 64 维照搬，后 36 维补零，
眨眼值填进眼睑段。

## shape

CAP4D 存的是它那套 FLAME2023 的 150 维 identity，**不能**直接喂给我们的
FLAME 基。主体身份应当由 `fit_subject_shape()` 从**我们自己的数据集网格**
反解出来 —— 那里有已知的 expr/pose/trans，只有 300 维 shape 未知。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

#: CAP4D 的 expr 里真正属于 FLAME 的维数（第 65 维是眨眼）
CAP4D_FLAME_EXPR = 64

#: 生成图的边长
CAP4D_RESOLUTION = 512

#: 我们的 blend_weight 分段（与 `data/checkpoint_loader.py` 一致）
BLEND_DIM = 129


@dataclass
class Cap4dFrame:
    """一帧 CAP4D 结果（尚未转成网格）。"""

    index: int
    image: Path
    expr: np.ndarray        # [64]   FLAME 表情
    blink: float            # 第 65 维（眨眼），我们填进眼睑
    rot: np.ndarray         # [3]    轴角
    tra: np.ndarray         # [3]
    eye_rot: np.ndarray     # [3]
    K: np.ndarray           # [3,3]  裁剪后的内参


# ------------------------------------------------------------------ 读 --


def read_cap4d_dir(cap4d_dir: Path, *, limit: int = 0) -> list[Cap4dFrame]:
    """读 `<cap4d_dir>/generated_images/` 下的图与参数。

    也接受直接给到 `generated_images` 这一层（两者都试）。
    """
    cap4d_dir = Path(cap4d_dir)
    for candidate in (cap4d_dir / "generated_images", cap4d_dir):
        if (candidate / "images").is_dir() and (candidate / "flame").is_dir():
            root = candidate
            break
    else:
        raise FileNotFoundError(
            f"{cap4d_dir} 下找不到 generated_images/{{images,flame}}")

    images = sorted(p for p in (root / "images").iterdir()
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if limit > 0:
        images = images[:limit]

    frames: list[Cap4dFrame] = []
    for i, img in enumerate(images):
        npz = root / "flame" / f"{i:05d}.npz"
        if not npz.is_file():
            continue
        with np.load(npz, allow_pickle=True) as d:
            expr = np.asarray(d["expr"]).reshape(-1)[:CAP4D_FLAME_EXPR]
            # 第 65 维是眨眼；没有就按 0
            raw_expr = np.asarray(d["expr"]).reshape(-1)
            blink = float(raw_expr[CAP4D_FLAME_EXPR]) if raw_expr.size > CAP4D_FLAME_EXPR else 0.0
            K = crop_intrinsics(
                fx=float(np.asarray(d["fx"]).reshape(-1)[0]),
                fy=float(np.asarray(d["fy"]).reshape(-1)[0]),
                cx=float(np.asarray(d["cx"]).reshape(-1)[0]),
                cy=float(np.asarray(d["cy"]).reshape(-1)[0]),
                crop_box=np.asarray(d["crop_box"]).reshape(-1)[:4],
                target=CAP4D_RESOLUTION)
            frames.append(Cap4dFrame(
                index=i, image=img,
                expr=expr.astype(np.float64), blink=blink,
                rot=np.asarray(d["rot"]).reshape(-1)[:3].astype(np.float64),
                tra=np.asarray(d["tra"]).reshape(-1)[:3].astype(np.float64),
                eye_rot=np.asarray(d["eye_rot"]).reshape(-1)[:3].astype(np.float64),
                K=K))
    if not frames:
        raise ValueError(f"{root} 里没有可用的帧（检查 flame/*.npz 是否齐全）")
    return frames


def crop_intrinsics(*, fx: float, fy: float, cx: float, cy: float,
                    crop_box, target: int = CAP4D_RESOLUTION) -> np.ndarray:
    """把原图内参换算成"裁剪 + 缩放到 target²"之后的内参。

    `crop_box = [x0, y0, x1, y1]` 是**正方形**（`get_square_bbox`），
    所以两个轴共用同一个缩放系数。
    """
    x0, y0, x1, y1 = (float(v) for v in crop_box)
    side = x1 - x0
    if side <= 0:
        raise ValueError(f"crop_box 非法：{crop_box}")
    scale = float(target) / side
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = fx * scale
    K[1, 1] = fy * scale
    K[0, 2] = (cx - x0) * scale
    K[1, 2] = (cy - y0) * scale
    return K


# ------------------------------------------------------- CAP4D → RDF 参数 --


def build_rdf_x(frame: Cap4dFrame) -> np.ndarray:
    """把一帧 CAP4D 参数转成 RDF 的 118 维 `x`。

    布局（见 `tracking/param_map.py`）::

        [exp 100][global 3][neck 3][jaw 3][eye_l 3][eye_r 3][trans 3]

    CAP4D 的姿态是"根旋转 + 平移"（`verts = R·v + t`），所以把 `rot` 放进
    **global**、把 neck/jaw 置零；眼球两只共用一组 `eye_rot`。

    ⚠️ **不要**用我们自己的 FLAME2020 去算网格：实测它与本数据集网格的
    重建误差 4.4–5.4 mm（误差随表情幅度增长），对不上 —— 模型的网格是
    FLAME2023 口径（与 `tracking/mesh_build.py` 的说明一致）。所以网格必须
    由 RDF 的 FLAME2023 算，本函数只负责把参数搬过去。
    """
    x = np.zeros(118, dtype=np.float64)
    x[:CAP4D_FLAME_EXPR] = frame.expr[:CAP4D_FLAME_EXPR]
    x[100:103] = frame.rot                 # global
    x[106:109] = 0.0                       # jaw（CAP4D 的条件几何是闭颌）
    x[109:112] = frame.eye_rot             # eye_l
    x[112:115] = frame.eye_rot             # eye_r
    x[115:118] = frame.tra
    return x


# ------------------------------------------------------- 读转换后的中间文件 --


def load_converted(path: Path) -> dict:
    """读 `export_cap4d_frames.py` 产出的 npz。

    必须包含：`mesh (T,5023,3)`、`x (T,118)`、`K (T,3,3)`、`names (T,)`；
    可选 `images_dir`（字符串）。
    """
    path = Path(path)
    with np.load(path, allow_pickle=True) as d:
        need = {"mesh", "x", "K", "names"}
        missing = need - set(d.files)
        if missing:
            raise ValueError(f"{path} 缺少字段：{sorted(missing)}")
        out = {
            "mesh": np.asarray(d["mesh"], dtype=np.float32),
            "x": np.asarray(d["x"], dtype=np.float32),
            "K": np.asarray(d["K"], dtype=np.float32),
            "names": [str(n) for n in d["names"]],
        }
        if "images_dir" in d.files:
            out["images_dir"] = Path(str(d["images_dir"]))
    if out["mesh"].shape[1] != 5023:
        raise ValueError(f"mesh 应为 5023 顶点的基础网格，实际 {out['mesh'].shape}")
    return out


def to_frames(converted: dict, flame_dir: Path, device: str | torch.device = "cpu"):
    """把转换结果拼成我们渲染要的 `frames` 与逐帧 `K`。

    两件"我们已经验证过"的事：
    - **补牙齿**：RDF 只有 5023 顶点，模型训练时是 5083（多 60 个牙齿），
      用 `tracking/mesh_build.py` 的嘴唇锚定法补。
    - **参数映射**：RDF 的 118 维 → 模型的 129 维 `blend_weight`
      （`tracking/param_map.py`）。
    """
    from ..tracking.mesh_build import TeethBuilder
    from ..tracking.param_map import to_blend_weight

    teeth = TeethBuilder(Path(flame_dir))
    frames: list[dict] = []
    ks: list[torch.Tensor] = []
    for i in range(converted["mesh"].shape[0]):
        base = torch.as_tensor(converted["mesh"][i], dtype=torch.float32,
                               device=device)
        mesh = teeth.build(base.to(torch.float64)).to(torch.float32)
        weight = to_blend_weight(
            torch.as_tensor(converted["x"][i], dtype=torch.float32, device=device))
        name = converted["names"][i]
        frames.append({"mesh": mesh, "blend_weight": weight.reshape(-1),
                       "name": name})
        ks.append(torch.as_tensor(converted["K"][i], dtype=torch.float32,
                                  device=device))
    return frames, ks
