"""候选减选：用缺陷质量做贪心集合覆盖。

## 问题

候选是"视角 × 表情"的组合。渲染它们很便宜（几百张，零点几秒），但**精修很贵**
（每张几百毫秒到几秒）。所以需要一个选择器：给定每个候选能"看到"哪些有缺陷的
texel，挑出十几张，覆盖尽量多的缺陷质量。

## 为什么是集合覆盖而不是网格扫描

缺陷在 texel 空间是**稀疏**的（大多数 texel 早就被真实数据看得很好），均匀撒点会
把预算花在完好区域。贪心覆盖对"稀疏目标 + 预算有限"是标准做法，且天然可解释：
每一步都报出"这一张新增覆盖了多少缺陷质量"。

## 接口

候选只需要提供 `covers`（它可见的 texel 下标数组）。**如何得到 `covers` 由调用方
决定**（用面 id 图求可见性、用高斯投影、或离线缓存的可见性表），本模块只做纯计算。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Candidate", "CoverageResult", "greedy_cover"]


@dataclass
class Candidate:
    """一个待选的 (视角, 表情) 组合。

    Attributes:
        label: 便于报告的名字，如 `yaw+30_pitch0_expr07`
        covers: 该组合**可见**的 texel 下标（`int` 数组，可重复可无序）
        payload: 调用方要用回的东西（`ViewSpec`、表情下标等），本模块不解释
    """

    label: str
    covers: np.ndarray
    payload: object | None = None

    def __post_init__(self) -> None:
        idx = np.asarray(self.covers, dtype=np.int64).reshape(-1)
        object.__setattr__(self, "covers", idx)


@dataclass
class CoverageResult:
    """选择结果与每一步的边际收益。"""

    chosen: list[Candidate] = field(default_factory=list)
    gains: list[float] = field(default_factory=list)
    total_mass: float = 0.0

    @property
    def covered_mass(self) -> float:
        return float(sum(self.gains))

    @property
    def covered_ratio(self) -> float:
        if self.total_mass <= 0:
            return 0.0
        return min(1.0, self.covered_mass / self.total_mass)

    def to_dict(self) -> dict:
        return {"chosen": [c.label for c in self.chosen],
                "gains": [round(float(g), 2) for g in self.gains],
                "covered_mass": round(self.covered_mass, 2),
                "total_mass": round(self.total_mass, 2),
                "covered_ratio": round(self.covered_ratio, 4)}


def greedy_cover(candidates: list[Candidate], mass: np.ndarray, *,
                 budget: int = 16, min_gain_ratio: float = 0.01) -> CoverageResult:
    """贪心选候选，最大化覆盖的缺陷质量。

    Args:
        candidates: 候选列表
        mass: `[n_texel]` 每个 texel 的缺陷质量（非负；越大越该修）
        budget: 最多选几个
        min_gain_ratio: 单次新增覆盖低于 `总质量的该比例` 就提前停止 ——
            避免为了一点点缺陷把预算花完

    Returns:
        `CoverageResult`；`gains` 是每一步**新增**的覆盖质量，可直接用于报告

    Notes:
        复杂度 `O(budget × Σ|covers|)`。重复的 texel 下标在评分时会去重，
        所以调用方不必保证 `covers` 唯一。
    """
    mass = np.asarray(mass, dtype=np.float64).reshape(-1)
    if candidates is None or len(candidates) == 0 or mass.size == 0:
        return CoverageResult(total_mass=0.0)
    if budget <= 0:
        raise ValueError(f"budget 必须为正，实际 {budget}")

    nz = mass > 0
    total = float(mass[nz].sum())
    if total <= 0:
        return CoverageResult(total_mass=0.0)

    covered = np.zeros(mass.size, dtype=bool)
    remaining = list(range(len(candidates)))
    result = CoverageResult(total_mass=total)

    while remaining and len(result.chosen) < budget:
        best_i = -1
        best_gain = 0.0
        for i in remaining:
            idx = candidates[i].covers
            if idx.size == 0:
                continue
            fresh = np.unique(idx[(idx >= 0) & (idx < mass.size)])
            if fresh.size == 0:
                continue
            gain = float(mass[fresh[~covered[fresh]]].sum())
            if gain > best_gain:
                best_gain, best_i = gain, i
        if best_i < 0 or best_gain <= 0:
            break
        if best_gain < min_gain_ratio * total:
            break

        result.chosen.append(candidates[best_i])
        result.gains.append(best_gain)
        idx = candidates[best_i].covers
        covered[idx[(idx >= 0) & (idx < mass.size)]] = True
        remaining.remove(best_i)

    return result
