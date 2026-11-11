"""热更新的损失。

**只依赖 `core/` 与 torch**，不读文件、不建模型 —— 这样损失可以脱离
渲染器单独测试（给两张图就能算），也便于以后替换成别的判据。

默认判据是 `PSNR` 的同源量（MSE），因为用户的验收标准就是 PSNR：
**优化目标与验收判据一致**，避免"优化涨了但 PSNR 没涨"。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

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
                sample_weight: torch.Tensor | None = None,
                weight_map: torch.Tensor | None = None) -> torch.Tensor:
    """带**空间权重**与**逐样本权重**的 L1（唯一实现）。

    - 空间：mask 内权重 1，mask 外 `mask_weight`（0 = 不看背景）
    - 样本：`sample_weight [B]`，给增强数据打折（None = 全 1）
    - 像素：`weight_map [B,1,H,W]`，与空间权重相乘（用于"只在缺陷区域监督"）

    归一化分母同时含空间权重与样本权重，所以"权重整体缩放"不改变损失。
    """
    diff = (prediction - target).abs()
    if mask is None:
        if weight_map is None:
            per_sample = diff.flatten(1).mean(dim=1)
        else:
            w = weight_map.to(diff.dtype).expand_as(diff)
            den = w.flatten(1).sum(dim=1).clamp(min=1e-6)
            per_sample = (diff * w).flatten(1).sum(dim=1) / den
    else:
        w = mask.to(diff.dtype) + float(mask_weight) * (~mask).to(diff.dtype)
        if weight_map is not None:
            w = w * weight_map.to(diff.dtype)
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


def local_ssim_loss(prediction: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor | None = None, *,
                    window_size: int = 11,
                    sigma: float = 1.5) -> torch.Tensor:
    """`1 − SSIM` 的纯 PyTorch 实现（11×11 高斯窗，逐通道）。

    为什么不用 vendored 的 `fused_ssim`：它在本机这套构建上 backward 会
    触发 `CUDA illegal memory access`（2026-11-11 实测；单独跑算子不复现，
    与光栅化 backward 同进程时必崩）。GAF 的 L_img 需要 SSIM 项，所以这里
    用 conv2d 实现，慢一些但可靠，且不依赖 CUDA 扩展。

    Args:
        prediction / target: `[B, 3, H, W]`，值域 `[0, 1]`
        mask: `[B, 1, H, W]`；只在这些像素上平均（`None` 表示全图）
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction {tuple(prediction.shape)} 与 target "
            f"{tuple(target.shape)} 形状不一致")
    b, c, h, w = prediction.shape
    window_size = min(int(window_size), h, w)
    if window_size % 2 == 0:
        window_size = max(3, window_size - 1)
    device, dtype = prediction.device, prediction.dtype
    coords = (torch.arange(window_size, device=device, dtype=dtype)
              - (window_size - 1) / 2.0)
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    win = (g[:, None] @ g[None, :]).expand(c, 1, window_size, window_size)
    win = win.contiguous()
    pad = window_size // 2

    mu_p = F.conv2d(prediction, win, padding=pad, groups=c)
    mu_t = F.conv2d(target, win, padding=pad, groups=c)
    mu_pp, mu_tt, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t
    var_p = F.conv2d(prediction * prediction, win, padding=pad, groups=c) - mu_pp
    var_t = F.conv2d(target * target, win, padding=pad, groups=c) - mu_tt
    cov = F.conv2d(prediction * target, win, padding=pad, groups=c) - mu_pt
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = (((2.0 * mu_pt + c1) * (2.0 * cov + c2))
                / ((mu_pp + mu_tt + c1) * (var_p + var_t + c2)))
    if mask is None:
        return 1.0 - ssim_map.mean()
    m = mask.to(dtype)
    if m.shape[1] == 1 and c > 1:
        m = m.expand(-1, c, -1, -1)
    return 1.0 - (ssim_map * m).sum() / m.sum().clamp_min(1.0)


def alpha_hole_loss(alpha: torch.Tensor, observed: torch.Tensor, *,
                    threshold: float = 0.5) -> torch.Tensor:
    """观测表面上的孔洞软惩罚：`mean(relu(threshold − alpha))`。

    `observed` 是"FLAME 表面 ∩ 真实前景"：只罚**数据支持过**的表面，
    脖子/胸口（表面但数据里是背景）不在内，符合"不补缺失解剖"。
    """
    if alpha.shape != observed.shape:
        raise ValueError(f"alpha {tuple(alpha.shape)} 与 observed "
                         f"{tuple(observed.shape)} 形状不一致")
    m = observed.to(dtype=alpha.dtype)
    gap = torch.relu(float(threshold) - alpha)
    return (gap * m).sum() / m.sum().clamp_min(1.0)


def surface_spill_loss(alpha: torch.Tensor, surface: torch.Tensor, *,
                       margin: int = 12, alpha_min: float = 0.05,
                       alpha_max: float = 0.6) -> torch.Tensor:
    """新视角：FLAME 表面（膨胀 `margin` 像素）之外的**半透明**溢出惩罚。

    只罚 `alpha ∈ (alpha_min, alpha_max)` 的带内贡献：实色头发/背景噪声不罚，
    目标是薄雾/虚影（半透明、离头较远的那层）。带内峰值归一化到 1。
    """
    if alpha.shape != surface.shape:
        raise ValueError(f"alpha {tuple(alpha.shape)} 与 surface "
                         f"{tuple(surface.shape)} 形状不一致")
    lo, hi = float(alpha_min), float(alpha_max)
    if hi <= lo:
        raise ValueError(f"alpha_max 必须大于 alpha_min，实际 {lo}, {hi}")
    s = surface.to(dtype=alpha.dtype)
    k = int(margin) * 2 + 1
    if k > 1:
        s = F.max_pool2d(s, kernel_size=k, stride=1, padding=k // 2)
    outside = (1.0 - s).clamp_min(0.0)
    band = torch.relu(alpha - lo) * torch.relu(hi - alpha)
    scale = max(1e-6, ((hi - lo) / 2.0) ** 2)
    return ((band / scale) * outside).sum() / outside.sum().clamp_min(1.0)


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
