"""训练调度：学习率曲线、形变基暖启动、早停判据。

三件事集中在这里，都是纯函数或纯状态，便于单测（不需要 GPU）：

- `expon_lr`：对数空间线性插值的学习率曲线，可选起步延迟
- `blend_active`：前 N 步只用模板网格拟合静态外观，之后才启用形变基
- `EarlyStopping`：按验证指标早停
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def expon_lr(iteration: int, lr_init: float, lr_final: float,
             max_steps: int, delay_steps: int = 0,
             delay_mult: float = 1.0) -> float:
    """3DGS 的指数学习率曲线（**按上游语义移植**）。

        log_lerp   = exp(lerp(log(lr_init), log(lr_final), t))，t = clip(step / max_steps)
        delay_rate = delay_mult + (1 - delay_mult) * sin(π/2 · clip(step / delay_steps))
        lr         = delay_rate * log_lerp

    ⚠️ `delay_mult` **只有配 `delay_steps > 0` 才起作用**；`delay_steps = 0`
    时 `delay_rate ≡ 1`，设了 `delay_mult` 也没有效果。本项目的默认计划
    不启用起步延迟（两个值分别是 0 与 1.0），需要时再显式打开。

    Args:
        iteration: 当前步（从 0 开始）
        lr_init / lr_final: 初始/终止学习率（须为正）
        max_steps: 衰减到 `lr_final` 的步数
        delay_steps: 起步延迟的步数窗口；0 表示不延迟
        delay_mult: 延迟窗口内的起始倍率
    """
    if max_steps <= 0:
        raise ValueError(f"max_steps 必须为正，实际 {max_steps}")
    if lr_init <= 0 or lr_final <= 0:
        raise ValueError(f"lr 必须为正：lr_init={lr_init}, lr_final={lr_final}")
    if delay_steps < 0:
        raise ValueError(f"delay_steps 不能为负，实际 {delay_steps}")

    step = max(int(iteration), 0)
    if int(delay_steps) > 0:
        frac = min(max(step / float(delay_steps), 0.0), 1.0)
        delay_rate = delay_mult + (1.0 - delay_mult) * math.sin(0.5 * math.pi * frac)
    else:
        delay_rate = 1.0

    t = min(max(step / float(max_steps), 0.0), 1.0)
    log_lerp = math.exp(math.log(lr_init) * (1.0 - t) + math.log(lr_final) * t)
    return delay_rate * log_lerp


def blend_active(iteration: int, start: int) -> bool:
    """第 `iteration` 步是否启用 `blend_weight`（形变基）。

    暖启动：先只用模板网格拟合静态外观，再打开形变基。这样形变基不会在
    颜色还很差时被拉偏（形状与颜色同时从零开始学，容易互相带偏）。
    """
    return int(iteration) >= int(start)


@dataclass
class EarlyStopping:
    """按验证指标的早停（指标越大越好，例如 PSNR）。

    Attributes:
        patience: 连续多少次验证没有提升就停
        min_delta: 认为"有提升"的最小幅度
    """

    patience: int = 10
    min_delta: float = 1e-3
    best: float | None = None
    bad: int = 0

    def update(self, value: float) -> bool:
        """喂一个新指标；返回是否应当停止。"""
        if self.best is None or value > self.best + self.min_delta:
            self.best = value
            self.bad = 0
            return False
        self.bad += 1
        return self.bad >= self.patience

    @property
    def improved(self) -> bool:
        """上一次 `update` 是否刷新了最好值（`bad == 0` 且已有 best）。"""
        return self.best is not None and self.bad == 0
