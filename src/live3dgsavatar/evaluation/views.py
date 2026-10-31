"""候选视角网格：把"看哪些角度"变成可枚举、可选的东西。

评估与精修共用这一份定义。两条经验：

- **角度不需要很密**：头半径约 0.09、相机距离 0.6 时，10° 约对应 512² 上的
  65 px 位移，相邻视角重叠很大，局部缺陷会在连续多个视角里出现；
- **侧脸要加密**：轮廓最容易崩坏的区域在 `|yaw|` 较大处，那里降到 5° 一档。

网格本身只是**候选池**（几百张，渲染只要零点几秒）；真正贵的是精修，
由 `protocol.select_for_refine` 按缺陷质量贪心挑出十几张。
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ViewSpec", "candidate_views", "grid_size"]


@dataclass(frozen=True)
class ViewSpec:
    """一个待渲染的视角（偏航 / 俯仰 / 滚转，单位度）。"""

    yaw: float
    pitch: float
    roll: float = 0.0

    @property
    def label(self) -> str:
        """稳定的文件名标签，例如 `yaw+30_pitch-10_roll+15`。"""
        return f"yaw{self.yaw:+.0f}_pitch{self.pitch:+.0f}_roll{self.roll:+.0f}"

    def to_dict(self) -> dict:
        return {"yaw": self.yaw, "pitch": self.pitch, "roll": self.roll,
                "label": self.label}


def _axis_values(limit: float, step: float, dense_from: float,
                 dense_step: float) -> list[float]:
    """生成一维角度序列：中间按 `step`，超过 `dense_from` 后按 `dense_step`。

    结果对 0 对称、去重、升序。
    """
    values: set[float] = set()
    n = int(round(limit / step))
    for i in range(-n, n + 1):
        v = round(i * step, 6)
        if abs(v) < dense_from - 1e-9:
            values.add(v)
    if dense_from <= limit:
        m = int(round((limit - dense_from) / dense_step))
        for j in range(m + 1):
            for sign in (1.0, -1.0):
                v = round(sign * (dense_from + j * dense_step), 6)
                if v <= limit + 1e-9:
                    values.add(v)
    return sorted(values)


def candidate_views(yaw_limit: float = 40.0, pitch_limit: float = 20.0,
                    step: float = 10.0, *, dense_from: float = 30.0,
                    dense_step: float = 5.0, rolls: tuple[float, ...] = (0.0,)
                    ) -> list[ViewSpec]:
    """枚举候选视角。

    Args:
        yaw_limit / pitch_limit: 偏航 / 俯仰范围（度）。默认 ±40° / ±20°。
        step: 基础间距（度）。
        dense_from: 从多大的 `|yaw|` 开始加密。
        dense_step: 加密后的间距。
        rolls: 需要哪些滚转角（默认只取 0；评估时可给 `(-15, 0, 15)`）。

    Returns:
        按 `(roll, pitch, yaw)` 排序的视角列表；默认参数下 55 个。
    """
    if yaw_limit <= 0 or pitch_limit < 0 or step <= 0:
        raise ValueError(f"网格参数非法：yaw={yaw_limit}, pitch={pitch_limit}, step={step}")
    yaws = _axis_values(yaw_limit, step, dense_from, dense_step)
    pitches = _axis_values(pitch_limit, step, dense_from, dense_step)
    out = [ViewSpec(y, p, r) for r in rolls for p in pitches for y in yaws]
    return out


def grid_size(*args, **kwargs) -> int:
    """候选视角数量（不必真的构造列表）。"""
    return len(candidate_views(*args, **kwargs))
