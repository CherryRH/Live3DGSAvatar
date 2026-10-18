"""闭式颜色校正：只拟合一个 `3×3` 矩阵（+可选偏置），**不需要反向传播**。

## 为什么先做这个

实测（`crh_1`，3 帧一致）：仅用 9 个参数、闭式最小二乘、零迭代，
就能把"模型颜色 vs 目标图片"的误差降低约 **1 dB**，且**跨帧可迁移**。
说明模型与当前拍摄条件之间存在**系统性颜色偏差**（白平衡/曝光/色调），
这是任何"重拟合"都该优先消除的部分。

## 它修什么、不修什么

| 能修 | 不能修 |
|---|---|
| 全局色调/白平衡偏移 | 逐区域的局部误差 |
| 曝光差异 | 几何/姿态误差 |
| 轻微的通道串扰 | 未观测区域（头顶等） |

## 使用位置

它作用于**渲染输出**，而不是模型参数 —— 因此**不修改 `model.ply`**，
可与梯度微调叠加（先校正颜色，再做精调）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .losses import masked_psnr


@dataclass
class ColorCorrection:
    """渲染输出的仿射颜色校正：`rgb' = rgb @ M + b`。

    Attributes:
        matrix: `[3, 3]` 线性项（初始为单位阵）
        bias: `[3]` 偏置项（初始为 0）
    """

    matrix: torch.Tensor
    bias: torch.Tensor

    @classmethod
    def identity(cls, device="cpu", dtype=torch.float32) -> "ColorCorrection":
        return cls(torch.eye(3, device=device, dtype=dtype),
                   torch.zeros(3, device=device, dtype=dtype))

    def __call__(self, rgb: torch.Tensor) -> torch.Tensor:
        """应用校正，返回同形状（clamp 到 [0,1]）。

        支持两种输入：
        - 图像 `[B, 3, H, W]`（正常渲染输出）
        - 点集 `[M, 3]`（离线分析用）
        """
        m = self.matrix.to(rgb.dtype)
        b = self.bias.to(rgb.dtype)
        if rgb.ndim == 2:                      # 点集
            return (rgb @ m.T + b).clamp(0.0, 1.0)
        if rgb.ndim != 4:
            raise ValueError(f"期望 [B,3,H,W] 或 [M,3]，实际 {tuple(rgb.shape)}")
        x = rgb.permute(0, 2, 3, 1)            # [B,H,W,3]
        y = x @ m.T + b
        return y.permute(0, 3, 1, 2).clamp(0.0, 1.0)

    def to(self, device) -> "ColorCorrection":
        return ColorCorrection(self.matrix.to(device), self.bias.to(device))


def fit_color_correction(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    with_bias: bool = True,
) -> tuple[ColorCorrection, dict[str, float]]:
    """最小二乘拟合颜色校正（闭式，单步求解）。

    支持两种输入形状：

    - 图像 `[B, 3, H, W]`（正常用法）
    - 点集 `[M, 3]`（如"逐高斯颜色 vs 图像同位置颜色"，用于离线分析）

    Args:
        prediction: 当前渲染 `[B, 3, H, W]` 或点集 `[M, 3]`
        target: 同形状
        mask: `[B, 1, H, W]`，只在这些像素上拟合
        with_bias: 是否拟合偏置（`False` 则只有 3×3，9 个参数）

    Returns:
        `(ColorCorrection, 指标)`。指标含拟合前后的 PSNR，便于判断是否值得应用。
    """
    p = prediction.detach().float()
    t = target.detach().float()
    if p.shape != t.shape:
        raise ValueError(f"形状不一致：{tuple(p.shape)} vs {tuple(t.shape)}")

    # 展平成点集 [M, 3]；点集输入直接使用
    if p.ndim == 2:
        ph, th = p, t
        if mask is not None:
            raise ValueError("点集输入不支持 mask（请先在调用方筛点）")
    elif p.ndim == 4:
        ph = p.permute(0, 2, 3, 1).reshape(-1, 3)
        th = t.permute(0, 2, 3, 1).reshape(-1, 3)
        if mask is not None:
            # 掩码是 [B,1,H,W] → 每个像素一个布尔值；按**行**选择，
            # 不能用 expand 成 [M,3] 再索引（会把结果压成一维，lstsq 直接报错）
            sel = mask.to(torch.bool).reshape(-1)          # [M]
            if sel.numel() != ph.shape[0]:
                raise ValueError(
                    f"mask 像素数 {sel.numel()} 与图像像素数 {ph.shape[0]} 不一致")
            ph, th = ph[sel], th[sel]
    else:
        raise ValueError(f"prediction 应为 [B,3,H,W] 或 [M,3]，实际 {tuple(p.shape)}")
    if ph.shape[0] < 8:
        raise ValueError(f"有效像素太少（{ph.shape[0]}），无法拟合颜色校正")

    n_in = 3 + (1 if with_bias else 0)
    A = torch.cat([ph, torch.ones_like(ph[:, :1])], dim=1) if with_bias else ph
    # 最小二乘：A @ W ≈ th，W 为 [n_in, 3]
    W = torch.linalg.lstsq(A, th).solution

    matrix = W[:3].T.contiguous()                    # [3,3]，使 rgb' = rgb @ M^T
    bias = W[3].contiguous() if with_bias else torch.zeros(3, dtype=W.dtype)
    corr = ColorCorrection(matrix, bias)

    before = masked_psnr(prediction, target, mask)
    after = masked_psnr(corr(prediction), target, mask)
    n_params = int(matrix.numel() + (3 if with_bias else 0))
    metrics = {"psnr_before": before, "psnr_after": after,
               "psnr_delta": after - before,
               "n_pixels": int(ph.shape[0]), "n_params": n_params}
    return corr, metrics
