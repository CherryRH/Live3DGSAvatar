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


def weighted_l1(prediction: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor | None = None, mask_weight: float = 0.0,
                sample_weight: torch.Tensor | None = None) -> torch.Tensor:
    """带**空间权重**与**逐样本权重**的 L1（唯一实现）。

    - 空间：mask 内权重 1，mask 外 `mask_weight`（0 = 不看背景）
    - 样本：`sample_weight [B]`，给增强数据打折（None = 全 1）

    归一化分母同时含空间权重与样本权重，所以"权重整体缩放"不改变损失。
    """
    diff = (prediction - target).abs()
    if mask is None:
        per_sample = diff.flatten(1).mean(dim=1)
    else:
        w = mask.to(diff.dtype) + float(mask_weight) * (~mask).to(diff.dtype)
        num = (diff * w).flatten(1).sum(dim=1)
        den = (w.flatten(1).sum(dim=1) * diff.shape[1]).clamp(min=1.0)
        per_sample = num / den
    if sample_weight is None:
        return per_sample.mean()
    sw = sample_weight.to(per_sample.dtype).reshape(-1)
    if sw.numel() != per_sample.numel():
        raise ValueError(
            f"sample_weight 长度 {sw.numel()} 与 batch {per_sample.numel()} 不一致")
    if float(sw.sum()) <= 0:
        return per_sample.new_zeros(())
    return (per_sample * sw).sum() / sw.sum()


def masked_l1(prediction: torch.Tensor, target: torch.Tensor,
              mask: torch.Tensor | None = None) -> torch.Tensor:
    """掩码内的平均绝对误差（= `weighted_l1` 的默认参数情形）。"""
    return weighted_l1(prediction, target, mask)


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
        l1: 优化目标用 L1 而不是 L2。
            L1 对少量离群像素更稳健（高光、镜面反射、跟踪抖动的边缘），
            但收敛更慢。**只影响优化目标**；验收用的 PSNR 始终由 MSE 计算 ——
            否则"优化目标"与"验收判据"会错位。
    """

    mask_weight: float = 0.0
    l1: bool = False

    def __call__(
        self,
        output: RenderOutput,
        observation,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """返回 `(标量损失, 供日志的指标字典)`。"""
        color = output.color
        target = observation.target
        mask = observation.mask

        def core(pred, tgt, m):
            return masked_l1(pred, tgt, m) if self.l1 else masked_mse(pred, tgt, m)

        if mask is None or self.mask_weight >= 1.0:
            loss = core(color, target, mask)
        else:
            # 掩码内为主，掩码外按 mask_weight 加权（默认 0 = 完全忽略）
            inside = core(color, target, mask)
            outside_m = (~mask.to(torch.bool)).to(color.dtype)
            if float(outside_m.sum()) <= 0:
                loss = inside
            else:
                outside = core(color, target, outside_m)
                loss = inside + self.mask_weight * outside

        mse = float(masked_mse(color.detach(), target.detach(), mask))
        metrics = {
            "loss": float(loss.detach()),
            "psnr": -10.0 * torch.log10(torch.tensor(max(mse, 1e-10))).item(),
            "psnr_full": masked_psnr(color, target, None),
        }
        return loss, metrics
