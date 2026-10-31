"""表情候选池：从数据里取"这个人常做的表情"，并补上"没做过但需要覆盖"的方向。

## 为什么不能直接写 `dim_k = 愤怒`

FLAME / RDF 的表情是 **PCA 系数**，维度按方差排序但**没有语义标签**。实测本数据
100 维的标准差在 0.23–1.29 之间且非单调，说明各维尺度也不一致 —— 把某一维当成
"皱眉"或"愤怒"是没有依据的。

所以候选池由两部分构成：

1. **聚类中心**（`cluster`）：这个人实际做过的表情模式，覆盖其习惯动作；
2. **主轴极值**（`axis_extremes`）：沿方差最大的若干方向取 ±kσ，覆盖"没做过但
   参数空间里合理"的方向。它不声称语义，但在几何上是系统的。

若之后确实需要**有名字**的原型（沮丧 / 愤怒 / 皱眉…），由 `load_named` 从
配置里读入 —— 这些向量应当来自**带标签数据的均值**（例如按情感标签对帧取平均），
而不是手工编造某一维。`named_prototype_names` 只负责注册名字，不猜数值。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["ExpressionSpec", "expression_matrix", "load_expressions",
           "cluster", "cluster_frames", "axis_extremes", "load_named",
           "merge_specs"]

#: 表情段在 `blend_weight` 中的范围（前 100 维，见 `tracking/param_map.py`）
EXPR_SLICE = slice(0, 100)


@dataclass(frozen=True)
class ExpressionSpec:
    """一个表情候选：名字 + 参数向量（`[D]` float32）。"""

    name: str
    vector: np.ndarray

    def __post_init__(self) -> None:
        v = np.asarray(self.vector, dtype=np.float32).reshape(-1)
        object.__setattr__(self, "vector", v)

    @property
    def magnitude(self) -> float:
        return float(np.linalg.norm(self.vector))

    def to_dict(self) -> dict:
        return {"name": self.name, "magnitude": round(self.magnitude, 3)}


def expression_matrix(weights: np.ndarray) -> np.ndarray:
    """`blend_weight [N,129]` → 表情段 `[N,100] float32`。"""
    w = np.asarray(weights, dtype=np.float32)
    if w.ndim != 2 or w.shape[1] < EXPR_SLICE.stop:
        raise ValueError(f"blend_weight 应为 [N,>=100]，实际 {w.shape}")
    return np.ascontiguousarray(w[:, EXPR_SLICE])


def load_expressions(dataset_dir: Path) -> np.ndarray:
    """从标准数据集的 `render_inputs.npz` 读表情矩阵。"""
    path = Path(dataset_dir) / "render_inputs.npz"
    if not path.is_file():
        raise FileNotFoundError(f"找不到 {path}")
    with np.load(path, allow_pickle=True) as d:
        return expression_matrix(d["blend_weight"])


def _kmeans(X: np.ndarray, k: int, *, seed: int = 0, iters: int = 40
            ) -> np.ndarray:
    """朴素的 k-means++（numpy 实现，避免引入 sklearn 依赖）。"""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    # k-means++ 初始化
    centers = [X[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min(((X[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        total = d2.sum()
        prob = d2 / total if total > 0 else np.full(n, 1.0 / n)
        centers.append(X[rng.choice(n, p=prob)])
    C = np.array(centers, dtype=np.float64)
    for _ in range(iters):
        dist = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        label = dist.argmin(axis=1)
        moved = False
        for j in range(k):
            members = X[label == j]
            if members.size == 0:
                continue
            new = members.mean(axis=0)
            if not np.allclose(new, C[j]):
                C[j] = new
                moved = True
        if not moved:
            break
    return C.astype(np.float32)


def cluster(X: np.ndarray, k: int | None = None, *, min_k: int = 4,
            max_k: int = 16, seed: int = 0) -> list[ExpressionSpec]:
    """对表情做聚类，返回聚类中心作为候选。

    `k` 缺省时按样本量取 `sqrt(N/2)` 并夹到 `[min_k, max_k]` —— 目的是"覆盖习惯
    表情"，不是精确建模分布。簇数过少（例如数据里几乎只有一两种极端表情）时，
    用 `axis_extremes` 补方向。

    Args:
        X: `[N, D]` 表情矩阵
        k: 簇数；`None` 表示按样本量估算
        min_k / max_k: k 的夹取范围
        seed: 初始化种子（结果可复现）

    Returns:
        `ExpressionSpec` 列表，名字形如 `cluster00`
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError(f"表情矩阵应为 [N,D]，实际 {X.shape}")
    n = X.shape[0]
    if k is None:
        # 只在"没指定簇数"时按经验范围夹取；显式 k 由调用方负责语义
        k = int(round(np.sqrt(max(n, 1) / 2.0)))
        k = int(min(max(k, min_k), max_k))
    k = int(min(max(k, 1), n))          # 显式 k 只受样本数约束
    if k == 1:
        return [ExpressionSpec("cluster00", X.mean(axis=0))]
    centers = _kmeans(X.astype(np.float64), k, seed=seed)
    order = np.argsort(np.linalg.norm(centers, axis=1))
    return [ExpressionSpec(f"cluster{i:02d}", centers[j]) for i, j in enumerate(order)]


def cluster_frames(X: np.ndarray, k: int | None = None, *, min_k: int = 4,
                   max_k: int = 12, seed: int = 0) -> list[ExpressionSpec]:
    """聚类后取**真实帧**（每簇的 medoid）作为表情候选。

    与 `cluster` 的区别：这里返回的向量来自数据集里真实存在的一帧，因此
    姿态、表情、眼镜/发型都与该帧完全自洽 —— 直接拿那一帧当基帧渲染，
    不会出现"把一个外来表情硬套到另一帧的姿态上"的畸变。

    代价是每个簇只代表一个样本（不是均值），但候选池本来就只需要"覆盖"。

    Returns:
        `ExpressionSpec` 列表，名字形如 `frame02581`（便于回查原图）
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError(f"表情矩阵应为 [N,D]，实际 {X.shape}")
    n = X.shape[0]
    if k is None:
        k = int(min(max(int(round(np.sqrt(max(n, 1) / 2.0))), min_k), max_k))
    k = int(min(max(k, 1), n))
    centers = cluster(X, k=k, seed=seed)
    out: list[ExpressionSpec] = []
    for spec in centers:
        dist = np.linalg.norm(X - spec.vector[None], axis=1)
        idx = int(np.argmin(dist))
        out.append(ExpressionSpec(f"frame{idx:05d}", X[idx]))
    return out


def axis_extremes(X: np.ndarray, k: int = 6, alpha: float = 1.5, *,
                  seed: int = 0) -> list[ExpressionSpec]:
    """沿方差最大的 k 个方向取 `mean ± alpha·σ`，作为"没做过的方向"的兜底。

    这些方向是**统计方向而非语义方向**（见模块说明）：不保证对应"皱眉"之类，
    但能在参数空间里系统性地铺开，补充聚类覆盖不到的区间。

    Args:
        X: `[N, D]`
        k: 取几个主轴（每个轴出两个候选，共 2k）
        alpha: 偏离倍数；1.5 已足够超出常见范围而不至于离谱

    Returns:
        `ExpressionSpec` 列表，名字形如 `axis00+` / `axis00-`
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError(f"至少需要 2 帧表情，实际 {X.shape}")
    k = int(max(1, min(k, X.shape[1])))
    mean = X.mean(axis=0)
    centered = X - mean
    # 主轴：对中心化数据做 SVD，右奇异向量即主轴
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    sigma = centered.std(axis=0)
    out: list[ExpressionSpec] = []
    for i in range(min(k, vt.shape[0])):
        direction = vt[i]
        scale = float(np.linalg.norm(centered @ direction)) / np.sqrt(len(X))
        for sign, tag in ((1.0, "+"), (-1.0, "-")):
            vec = mean + sign * alpha * scale * direction
            out.append(ExpressionSpec(f"axis{i:02d}{tag}", vec.astype(np.float32)))
    return out


def load_named(mapping: dict[str, list[float]] | None) -> list[ExpressionSpec]:
    """把"有名有姓"的原型（沮丧 / 愤怒 / 皱眉…）转成候选。

    `mapping` 应由**带标签数据的统计**得到（例如某情感标签下所有帧的表情均值），
    不要手填某一维 —— 见模块说明。传 `None` 或空字典时返回空列表。
    """
    out: list[ExpressionSpec] = []
    for name, vector in (mapping or {}).items():
        out.append(ExpressionSpec(str(name), np.asarray(vector, dtype=np.float32)))
    return out


def merge_specs(*groups: list[ExpressionSpec], cos_threshold: float = 0.995
                ) -> list[ExpressionSpec]:
    """合并候选并去掉**几乎重复**的向量（余弦相似度高于阈值）。

    聚类中心与主轴极值可能撞在一起；去重能省下后面的精修预算。
    """
    kept: list[ExpressionSpec] = []
    for spec in [s for group in groups for s in group]:
        v = spec.vector
        nv = np.linalg.norm(v)
        duplicate = False
        for prev in kept:
            pv = prev.vector
            denom = nv * np.linalg.norm(pv)
            if denom == 0:
                continue
            if float(v @ pv / denom) > cos_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(spec)
    return kept
