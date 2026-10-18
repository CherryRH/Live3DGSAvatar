"""热更新的损失。

**只依赖 `core/` 与 torch**，不读文件、不建模型 —— 这样损失可以脱离
渲染器单独测试（给两张图就能算），也便于以后替换成别的判据。

默认判据是 `PSNR` 的同源量（MSE），因为用户的验收标准就是 PSNR：
**优化目标与验收判据一致**，避免"优化涨了但 PSNR 没涨"。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..core.types import RenderOutput


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """掩码均方误差。

    Args:
        prediction: `[B, 3, H, W]`，值域 `[0, 1]`（已 clamp 或本就如此）
        target: 同形状
        mask: `[B, 1, H, W]`，`True/非零` 处参与统计。``None`` 表示全图

    Returns:
        标量张量（可反向传播）
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction {tuple(prediction.shape)} 与 target {tuple(target.shape)} 形状不一致")
    diff2 = (prediction - target) ** 2
    if mask is None:
        return diff2.mean()
    m = mask.to(diff2.dtype)
    # 按"有效像素数 × 通道数"归一，避免前景大小不同导致损失量级漂移
    denom = m.sum() * diff2.shape[1]
    if float(denom) <= 0:
        raise ValueError("mask 全为 0，没有可优化的像素")
    return (diff2 * m).sum() / denom


def masked_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-10,
) -> float:
    """**验收判据**：掩码 PSNR（dB）。不可微，只用于评估与日志。"""
    with torch.no_grad():
        mse = float(masked_mse(prediction.detach(), target.detach(), mask))
    return -10.0 * torch.log10(torch.tensor(max(mse, eps))).item()


@dataclass
class PhotometricLoss:
    """光度损失：掩码 MSE + 可选的额外项。

    为什么**不用** `fused-ssim`：它是训练用的 CUDA 扩展（本项目默认不依赖），
    而热更新的优化目标与验收判据都是 PSNR/MSE，**同源**即可；
    引入 SSIM 只会让"优化目标"与"验收判据"错位。

    Attributes:
        mask_weight: 掩码外像素的权重（0 表示完全忽略背景）。
            背景通常预测不准，忽略它能显著改善前景拟合。
    """

    mask_weight: float = 0.0

    def __call__(
        self,
        output: RenderOutput,
        observation,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """返回 `(标量损失, 供日志的指标字典)`。"""
        color = output.color
        target = observation.target
        mask = observation.mask

        if mask is None or self.mask_weight >= 1.0:
            loss = masked_mse(color, target, mask)
        else:
            # 掩码内为主，掩码外按 mask_weight 加权（默认 0 = 完全忽略）
            inside = masked_mse(color, target, mask)
            outside_m = (~mask.to(torch.bool)).to(color.dtype)
            if float(outside_m.sum()) <= 0:
                loss = inside
            else:
                outside = masked_mse(color, target, outside_m)
                loss = inside + self.mask_weight * outside

        mse = float(masked_mse(color.detach(), target.detach(), mask))
        metrics = {
            "loss": float(loss.detach()),
            "psnr": -10.0 * torch.log10(torch.tensor(max(mse, 1e-10))).item(),
            "psnr_full": masked_psnr(color, target, None),
        }
        return loss, metrics
