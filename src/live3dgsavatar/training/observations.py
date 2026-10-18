"""热更新的输入契约：一帧"被观察到的"画面 + 它对应的几何与驱动参数。

**为什么要有这一层**：热更新只应关心"给定这些观测，如何改进模型"，
而不应关心观测是**从数据集读的**还是**从摄像头来的**。所以这里只定义
纯数据契约，`training/` 因此**不依赖 `data/` 或 `app/`**（见分层规则）。

```
数据集  ─┐
         ├─► FrameObservation ─► OnlineUpdater ─► 更新后的模型
摄像头  ─┘                        （只依赖 core/）
```
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..core.types import Camera, Mesh


@dataclass
class FrameObservation:
    """单帧观测。

    Attributes:
        mesh: 世界空间网格顶点 `[1, V, 3]`（与训练时同一拓扑）
        blend_weight: 驱动参数 `[1, D]`（与训练时同一参数空间）
        target: 目标图像 `[1, 3, H, W]`，值域 `[0, 1]`
        mask: 可选的前景掩码 `[1, 1, H, W]`，`True/1` 处参与损失。
            背景预测通常不准，把背景排除能明显改善拟合。
        name: 可选标识（帧名/时间戳），只用于日志
    """

    mesh: Mesh
    blend_weight: torch.Tensor
    target: torch.Tensor
    mask: torch.Tensor | None = None
    name: str = ""

    def __post_init__(self) -> None:
        if self.blend_weight.ndim != 2:
            raise ValueError(
                f"blend_weight 形状应为 [B, D]，实际 {tuple(self.blend_weight.shape)}")
        # 批大小必须与 target 一致：错位会让"这一帧的参数驱动了另一帧的画面"，
        # 而症状只是"拟合效果差"，极难定位 —— 因此在契约层拦下。
        if self.blend_weight.shape[0] != self.target.shape[0]:
            raise ValueError(
                f"blend_weight 批大小 {self.blend_weight.shape[0]} 与 "
                f"target 批大小 {self.target.shape[0]} 不一致")
        if self.target.ndim != 4 or self.target.shape[1] != 3:
            raise ValueError(
                f"target 形状应为 [B, 3, H, W]，实际 {tuple(self.target.shape)}")
        if self.mask is not None and self.mask.shape != (
                self.target.shape[0], 1, self.target.shape[2], self.target.shape[3]):
            raise ValueError(
                f"mask 形状应为 {self.target.shape[0], 1, *self.target.shape[2:]}，"
                f"实际 {tuple(self.mask.shape)}")

    @property
    def batch_size(self) -> int:
        return int(self.target.shape[0])

    @property
    def height(self) -> int:
        return int(self.target.shape[2])

    @property
    def width(self) -> int:
        return int(self.target.shape[3])

    def to(self, device: torch.device | str) -> "FrameObservation":
        """把全部张量搬到同一设备（`Mesh`/`Camera` 由调用方自行处理）。"""
        return FrameObservation(
            mesh=self.mesh,
            blend_weight=self.blend_weight.to(device),
            target=self.target.to(device),
            mask=None if self.mask is None else self.mask.to(device),
            name=self.name,
        )


@dataclass
class BatchObservation:
    """若干帧的集合（热更新一次通常用一小段视频）。

    **刻意不做 sampler**：基础能力只需要"给我一批帧"，采样策略
    （如 local-global 双池）是后续可插拔的独立组件。
    """

    frames: list[FrameObservation]
    camera: Camera

    def __len__(self) -> int:
        return len(self.frames)

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("BatchObservation 至少需要一帧")
        h, w = self.frames[0].height, self.frames[0].width
        for f in self.frames:
            if (f.height, f.width) != (h, w):
                raise ValueError(
                    f"同一批观测的分辨率必须一致：{f.name or '?'} 为 "
                    f"{f.width}x{f.height}，期望 {w}x{h}")
        if self.camera.height != h or self.camera.width != w:
            raise ValueError(
                f"相机分辨率 {self.camera.width}x{self.camera.height} 与观测 {w}x{h} 不一致")
