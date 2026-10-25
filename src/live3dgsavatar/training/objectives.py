"""目标函数：把若干损失项按权重组合，并返回**每项的值**。

设计要点：

- 各项权重是 `Objective` 的字段（配置），不是代码里的分支
- 返回值带分项，日志与报告直接用它，不必重算
- `mask` 一律可选：背景不可信时把背景排除（在线更新、白底合成图都靠它）
- 分阶段启用（例如 LPIPS 到某一步再开）由 `lpips_from` 表达

已实现的项：

| 项 | 作用 | 备注 |
|---|---|---|
| `l1` | 主损失 | 对离群像素稳健 |
| `ssim` | 结构相似度 | 用 `fused_ssim`（submodule 里有），比手写实现快 |
| `lpips` | 感知损失 | 默认关；开启时按需加载 VGG 权重 |
| `alpha` | 不透明度对齐 | 合成数据/白底图有用 |
| `sparsity` | 驱动基稀疏 | 让 K 个基各管一块，避免互相抵消 |
| `orth` | 基正交 | 减少基之间冗余 |
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .losses import weighted_l1


def _as_mask(mask: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor | None:
    """把 mask 规整到 `[B,1,H,W]` 的 bool；`None` 原样返回。"""
    if mask is None:
        return None
    m = mask
    if m.ndim == 2:
        m = m[None, None]
    elif m.ndim == 3:
        m = m[:, None] if m.shape[0] == ref.shape[0] else m[None]
    if m.shape[0] == 1 and ref.shape[0] > 1:
        m = m.expand(ref.shape[0], -1, -1, -1)
    if m.dtype != torch.bool:
        m = m > 0.5
    return m


def _ssim(pred: torch.Tensor, target: torch.Tensor,
          mask: torch.Tensor | None) -> torch.Tensor:
    """`1 - SSIM`。有 mask 时只在 mask 内统计（把外面填成 target 以减少干扰）。"""
    try:
        from fused_ssim import fused_ssim
    except ImportError:                                  # pragma: no cover
        fused_ssim = None

    p, t = pred, target
    if mask is not None:
        # 用 target 填 mask 外：SSIM 是局部窗口量，填 0 会在边界造成假结构
        p = torch.where(mask, p, t)
        t = torch.where(mask, t, t)
    if fused_ssim is not None and p.is_cuda:
        return 1.0 - fused_ssim(p, t, train=False)
    # CPU / 无 fused_ssim：退回均值+方差的简化 SSIM（仅用于自检）
    # `unbiased=False`：与下面的 cov（按均值算）同一口径，否则"完全相同"也会差一点
    mu_p, mu_t = p.mean(), t.mean()
    var_p, var_t = p.var(unbiased=False), t.var(unbiased=False)
    cov = ((p - mu_p) * (t - mu_t)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * mu_p * mu_t + c1) * (2 * cov + c2)) / (
        (mu_p ** 2 + mu_t ** 2 + c1) * (var_p + var_t + c2))
    return 1.0 - score


@dataclass
class Objective:
    """加权损失。

    Attributes:
        l1 / ssim / lpips / alpha: 各项权重；0 表示不计算（省时间）
        mask_weight: mask 外像素的权重（0 = 完全不看背景）
        lpips_from: 第几步开始启用 LPIPS（前几步颜色还没成形，感知损失意义不大）
    """

    l1: float = 1.0
    ssim: float = 0.0
    lpips: float = 0.0
    alpha: float = 0.0
    mask_weight: float = 0.0
    lpips_from: int = 20_000
    _lpips_model: object | None = field(default=None, repr=False)

    # ------------------------------------------------------------ 主入口 --

    def __call__(self, *, pred: torch.Tensor, target: torch.Tensor,
                 mask: torch.Tensor | None = None,
                 sample_weight: torch.Tensor | None = None,
                 alpha: torch.Tensor | None = None,
                 target_alpha: torch.Tensor | None = None,
                 iteration: int = 0,
                 extras: dict[str, torch.Tensor] | None = None
                 ) -> tuple[torch.Tensor, dict[str, float]]:
        """算总损失。

        Args:
            pred: `[B,3,H,W]` 渲染结果（已叠背景）
            target: `[B,3,H,W]` 目标图
            mask: `[B,1,H,W]` 前景掩码；`None` 表示全图
            alpha: `[B,1,H,W]` 渲染的不透明度
            target_alpha: `[B,1,H,W]` 目标掩码（作为不透明度监督）
            iteration: 当前步（用于 LPIPS 的启用时机）
            extras: 额外的正则项，如 `{"sparsity": ..., "orth": ...}`

        Returns:
            `(total, parts)`，`parts` 是各项的**标量值**（用于日志）
        """
        if target.device != pred.device:
            raise RuntimeError(
                f"pred 在 {pred.device}，target 在 {target.device}；"
                "数据源与训练设备不一致 —— 应由 Trainer 统一搬到训练设备")
        m = _as_mask(mask, pred)
        parts: dict[str, float] = {}
        total = pred.new_zeros(())

        if self.l1 > 0:
            v = weighted_l1(pred, target, m, self.mask_weight, sample_weight)
            total = total + self.l1 * v
            parts["l1"] = float(v.detach())

        if self.ssim > 0:
            v = _ssim(pred, target, m)
            total = total + self.ssim * v
            parts["ssim"] = float(v.detach())

        if self.lpips > 0 and iteration >= self.lpips_from:
            v = self._lpips(pred, target, m)
            total = total + self.lpips * v
            parts["lpips"] = float(v.detach())

        if self.alpha > 0 and alpha is not None and target_alpha is not None:
            ta = target_alpha.to(alpha.dtype)
            v = weighted_l1(alpha, ta, m, self.mask_weight, sample_weight)
            total = total + self.alpha * v
            parts["alpha"] = float(v.detach())

        for name, value in (extras or {}).items():
            weight = getattr(self, name, 0.0)
            if weight > 0:
                total = total + weight * value
                parts[name] = float(value.detach())

        parts["total"] = float(total.detach())
        return total, parts

    # ------------------------------------------------------------ 内部 --

    def _lpips(self, pred: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor | None) -> torch.Tensor:
        import lpips as lpips_lib

        if self._lpips_model is None:
            model = lpips_lib.LPIPS(net="vgg").to(pred.device).eval()
            for p in model.parameters():
                p.requires_grad_(False)
            self._lpips_model = model
        p, t = pred, target
        if mask is not None:
            # LPIPS 没有 mask 参数；用 target 填外面，避免背景干扰
            p = torch.where(mask, p, t)
        return self._lpips_model(p * 2 - 1, t * 2 - 1).mean()

    @classmethod
    def rgba_avatar_defaults(cls) -> "Objective":
        """完整训练默认：只有 L1，但**背景也参与**（`mask_weight=1`）。

        为什么背景必须参与：只在前景 mask 内算损失时，落在人像之外的
        高斯完全拿不到梯度 —— 它们的位置/颜色会在训练中漂移，推理时表现
        为"头周围出现大片发散的高斯球"。把背景一起监督（渲染背景色已知），
        这些高斯会被压成透明或被拉回目标颜色。

        代价是背景预测不完全准时会带来一点噪声；用 `mask_weight < 1`
        可以折中（例如 0.2）。
        """
        return cls(l1=1.0, ssim=0.0, lpips=0.0, alpha=0.0, mask_weight=1.0)

    @classmethod
    def appearance_only(cls, *, l1: bool = True, mask_weight: float = 0.0
                        ) -> "Objective":
        """在线微调用：只有像素 L1（+可选 mask）。"""
        return cls(l1=1.0 if l1 else 0.0, mask_weight=mask_weight)
