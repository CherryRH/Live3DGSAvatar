"""缺陷信号：把"哪里渲染坏了"做成可计算的量，并归因到面/texel。

## 为什么不按 (视角, 表情) 统计

视角 × 表情是笛卡尔积，数量爆炸且大部分重复。缺陷本身是**表示空间**的性质
（某个 texel 没被观测好、某个高斯飞在外面），视角与表情只是访问它的途径。
所以这里所有信号最终都落到 **per-face / per-texel** 数组上：

    图像空间量（孔洞、锐度、精修差分）──按面 id 聚合──▶ per-face ──binding──▶ per-texel

于是"覆盖角度"与"覆盖表情"变成两个独立的选择问题，详见 `views.py` 与后续的选择器。

## 信号一览

| 信号 | 来源 | 抓什么 |
|---|---|---|
| `coverage` | 面 id 图跨帧可见性 | 从没被看到过的区域 |
| `floater` | 高斯到绑定面的距离 | 飞点 / 离面漂移 |
| `alpha_hole` | 渲染 alpha vs 面 id 图 | 该有前景却是空的 |
| `sharpness` | 图像梯度 | 糊 / 涂抹 |
| `delta` | 精修前后差分 | 增强器认为需要改的地方 |

除 `coverage` 需要面 id 图（GPU）外，其余都能用 CPU 张量/数组计算，便于单测。

⚠️ **UV 掩码有两种形状**（见 `deform/bind.py`）：刚构建 `Binding` 时是 `[tex_size²]`，
从 `.ply` 载入时退化成 `[N]` 全 True（文件只存有效 texel，位置信息已丢失）。
所以铺回 UV 图必须用 `texel_index`（可由一次 UV 光栅化恢复），不能想当然地拿
`valid_binding_mask` 当 `[tex_size²]` 用。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

__all__ = ["DefectMap", "attribute_to_faces", "per_texel", "coverage_counts",
           "floater_offsets", "alpha_hole_ratio", "sharpness_map",
           "refine_delta"]


@dataclass
class DefectMap:
    """一次缺陷分析的结果（逐面 + 逐 texel + 一个总览）。

    `per_face` / `per_texel` 的键与信号名一致（`coverage`、`floater`、…），
    值是一维 numpy 数组。`score` 是便于排序/阈值化的标量（越大越差）。
    """

    per_face: dict[str, np.ndarray] = field(default_factory=dict)
    per_texel: dict[str, np.ndarray] = field(default_factory=dict)
    score: float = 0.0

    def top_faces(self, key: str = "coverage", k: int = 10) -> list[int]:
        """按某个信号取最差的 k 个面（默认覆盖最少的）。"""
        values = self.per_face.get(key)
        if values is None or values.size == 0:
            return []
        order = np.argsort(values)[:k]
        return [int(i) for i in order]

    def summary(self) -> dict:
        out: dict = {"score": round(float(self.score), 4)}
        for name, values in self.per_face.items():
            if values.size == 0:
                continue
            out[name] = {"mean": round(float(np.mean(values)), 4),
                         "min": round(float(np.min(values)), 4),
                         "max": round(float(np.max(values)), 4)}
        return out


def attribute_to_faces(values: np.ndarray, face_id: np.ndarray, n_faces: int,
                       reduce: str = "mean") -> np.ndarray:
    """把逐像素的量按面 id 聚合到 `[n_faces]`。

    Args:
        values: `[H, W]` 逐像素量（孔洞、锐度、差分…）
        face_id: `[H, W]` 面 id，`-1` 为背景
        n_faces: 面总数
        reduce: `"mean"` / `"max"` / `"sum"`；没被任何像素覆盖的面填 0
    """
    values = np.asarray(values, dtype=np.float64)
    face_id = np.asarray(face_id)
    if values.shape != face_id.shape:
        raise ValueError(f"形状不一致：values {values.shape} vs face_id {face_id.shape}")
    valid = face_id >= 0
    out = np.zeros(n_faces, dtype=np.float64)
    if not bool(valid.any()):
        return out
    ids = face_id[valid]
    vals = values[valid]
    if reduce == "sum":
        np.add.at(out, ids, vals)
    elif reduce == "max":
        np.maximum.at(out, ids, vals)
    elif reduce == "mean":
        np.add.at(out, ids, vals)
        np.maximum(out, 0, out=out)
        counts = np.bincount(ids, minlength=n_faces).astype(np.float64)
        nz = counts > 0
        out[nz] /= counts[nz]
    else:                                              # pragma: no cover
        raise ValueError(f"未知 reduce：{reduce}")
    return out


def per_texel(per_face: np.ndarray, texel_face_id: torch.Tensor | np.ndarray,
              tex_size: int, *, texel_index: torch.Tensor | np.ndarray | None = None,
              valid_mask: torch.Tensor | np.ndarray | None = None) -> np.ndarray:
    """把逐面数组铺回 UV 图：返回 `[tex_size²]`，无效 texel 为 `NaN`。

    `texel_face_id` 是 `Binding.face_id`（每个高斯绑定的面），顺序与高斯一致。

    铺回位置有两种给法（**必须给一种**）：

    - `texel_index`：`[N]`，每个高斯对应的展平 texel 下标。从 `.ply` 载入的模型
      必须用这个 —— 文件里只存有效 texel，UV 掩码退化成全 True，位置信息已丢失；
    - `valid_mask`：`[tex_size²]` 的 UV 掩码（**刚构建** `Binding` 时才有）。

    Raises:
        ValueError: 只给了 `[N]` 掩码（从 PLY 载入的情形）却没有 `texel_index`
    """
    face = np.asarray(texel_face_id.detach().cpu() if hasattr(texel_face_id, "detach")
                      else texel_face_id, dtype=np.int64).reshape(-1)
    values = np.asarray(per_face, dtype=np.float64)
    full = np.full(tex_size * tex_size, np.nan, dtype=np.float64)

    if texel_index is not None:
        idx = np.asarray(texel_index.detach().cpu()
                         if hasattr(texel_index, "detach") else texel_index,
                         dtype=np.int64).reshape(-1)
        if idx.size != face.size:
            raise ValueError(
                f"texel_index 长度 {idx.size} 与高斯数 {face.size} 不一致")
        full[idx] = values[face]
        return full

    if valid_mask is not None:
        mask = np.asarray(valid_mask.detach().cpu() if hasattr(valid_mask, "detach")
                          else valid_mask, dtype=bool).reshape(-1)
        if mask.size == tex_size * tex_size:
            full[mask] = values[face]
            return full
        raise ValueError(
            f"valid_mask 长度 {mask.size} 不是 tex_size²={tex_size * tex_size}："
            "从 .ply 载入的模型只保留有效 texel，掩码位置信息已丢失；"
            "请改传 texel_index（可由一次 UV 光栅化得到）")

    raise ValueError("per_texel 需要 texel_index 或 [tex_size²] 的 valid_mask")


def coverage_counts(face_id_maps: list[np.ndarray], n_faces: int) -> np.ndarray:
    """每个面在给定视角集合里**被看到多少次**。

    没有出现在任何视角里的面 = 从未被观测 → 缺陷候选；这也是"该渲染哪些视角"
    的依据（贪心覆盖的目标）。
    """
    counts = np.zeros(n_faces, dtype=np.int64)
    for ids in face_id_maps:
        valid = np.asarray(ids)[np.asarray(ids) >= 0]
        if valid.size:
            counts += np.bincount(valid, minlength=n_faces)[:n_faces]
    return counts


def floater_offsets(xyz: torch.Tensor, mesh_verts: torch.Tensor,
                    mesh_faces: torch.Tensor, face_id: torch.Tensor,
                    face_bary: torch.Tensor) -> torch.Tensor:
    """每个高斯到其绑定面表面的距离（`[N]`，与世界坐标同尺度）。

    距离明显大于中位数的高斯就是"飞点"：它们不属于皮肤表面，渲染时表现为
    悬空的小团块。只依赖张量，CPU 上即可验证。
    """
    if xyz.ndim != 2 or xyz.shape[-1] != 3:
        raise ValueError(f"xyz 应为 [N, 3]，实际 {tuple(xyz.shape)}")
    faces = mesh_faces.reshape(-1, 3).to(torch.long)
    idx = face_id.reshape(-1).to(torch.long)
    bary = face_bary.reshape(-1, 3).to(xyz.dtype)
    if idx.numel() != xyz.shape[0] or bary.shape[0] != xyz.shape[0]:
        raise ValueError("xyz / face_id / face_bary 的个数不一致")
    verts = mesh_verts.reshape(-1, 3).to(xyz.dtype)
    corners = verts[faces[idx]]                       # [N, 3, 3]
    surface = (corners * bary.unsqueeze(-1)).sum(dim=1)
    return (xyz - surface).norm(dim=-1)


def alpha_hole_ratio(alpha: np.ndarray, face_id: np.ndarray,
                     n_faces: int) -> np.ndarray:
    """逐面"该有前景却是空"的比例：`alpha < 0.5` 的面内像素占比。

    用面 id 图作为"这里应当有表面"的参照，渲染 alpha 作为实际覆盖，
    两者之差就是孔洞。
    """
    alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)
    face_id = np.asarray(face_id).reshape(-1)
    if alpha.shape != face_id.shape:
        raise ValueError(f"形状不一致：alpha {alpha.shape} vs face_id {face_id.shape}")
    front = face_id >= 0
    out = np.zeros(n_faces, dtype=np.float64)
    if not bool(front.any()):
        return out
    ids = face_id[front]
    holes = (alpha[front] < 0.5).astype(np.float64)
    np.add.at(out, ids, holes)
    counts = np.bincount(ids, minlength=n_faces).astype(np.float64)
    nz = counts > 0
    out[nz] /= counts[nz]
    return out


def sharpness_map(image: np.ndarray) -> np.ndarray:
    """逐像素梯度幅值（0–255 尺度）：越小越糊。"""
    rgb = np.asarray(image, dtype=np.float32)
    grey = rgb.mean(axis=-1)
    gy, gx = np.gradient(grey)
    return np.hypot(gx, gy) * 255.0


def refine_delta(render: np.ndarray, refined: np.ndarray) -> np.ndarray:
    """精修前后的逐像素差分（通道平均，0–255）：越大说明增强器改得越多。

    这是唯一"自带标签"的缺陷信号——增强器改动大的地方，就是原始渲染不可靠
    的地方。同时它天然是后续微调的**监督权重图**。
    """
    a = np.asarray(render, dtype=np.float32)
    b = np.asarray(refined, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"尺寸不一致：{a.shape} vs {b.shape}")
    return np.abs(a - b).mean(axis=-1)


def coverage_defect(coverage: np.ndarray, saturate: float = 50.0) -> np.ndarray:
    """覆盖率 → 缺陷权重（`0` 表示看过足够多次，`1` 表示从未看过）。

    不能用"按分位数归一化"：实测各高斯的观测次数从 0 到 18000 横跨四个数量级，
    按 P90 归一化会让"看过 850 次"的中位数也算作 86% 缺陷 —— 那等于把整张脸都
    标成坏的，选择器就退化成"谁看得最多先选谁"。

    这里改成**饱和**：看过 `saturate` 次即视为充分，权重线性降到 0。
    """
    cov = np.asarray(coverage, dtype=np.float64).reshape(-1)
    if saturate <= 0:
        raise ValueError(f"saturate 必须为正，实际 {saturate}")
    return np.clip(1.0 - cov / float(saturate), 0.0, 1.0)


def hole_rate(hole_hits: np.ndarray, visible: np.ndarray) -> np.ndarray:
    """孔洞"命中率"：某个面在**多少比例的候选里**出现孔洞。

    用最大值不行：1100 个候选里几乎每个面都会在某个极端视角下漏一次，
    取最大值等于把所有面都标成缺陷。命中率能区分"偶发"与"总是漏"。
    """
    hits = np.asarray(hole_hits, dtype=np.float64).reshape(-1)
    vis = np.asarray(visible, dtype=np.float64).reshape(-1)
    if hits.shape != vis.shape:
        raise ValueError(f"形状不一致：{hits.shape} vs {vis.shape}")
    out = np.zeros_like(hits)
    nz = vis > 0
    out[nz] = hits[nz] / vis[nz]
    return out


def build_defect_map(*, per_face: dict[str, np.ndarray],
                     texel_face_id: torch.Tensor | np.ndarray,
                     tex_size: int,
                     texel_index: torch.Tensor | np.ndarray | None = None,
                     valid_mask: torch.Tensor | np.ndarray | None = None,
                     score: float = 0.0) -> DefectMap:
    """把逐面信号打包成 `DefectMap`（同时铺回 UV 图）。

    总览分数默认取"覆盖率不足"的直观度量：`1 - mean(coverage)/max(coverage)`，
    调用方也可显式给 `score`。
    """
    per_face = {k: np.asarray(v, dtype=np.float64) for k, v in per_face.items()}
    if score == 0.0 and "coverage" in per_face:
        cov = per_face["coverage"]
        peak = float(cov.max()) if cov.size else 0.0
        score = float(1.0 - (cov.mean() / peak)) if peak > 0 else 0.0
    per_tex = {k: per_texel(v, texel_face_id, tex_size, texel_index=texel_index,
                            valid_mask=valid_mask)
               for k, v in per_face.items()}
    return DefectMap(per_face=per_face, per_texel=per_tex, score=score)
