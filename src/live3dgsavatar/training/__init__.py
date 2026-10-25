"""训练与在线更新层。

**分层**：本层向下依赖 `core/`（渲染与类型），**不得**被 `core/` 依赖；
也**不依赖** `data/`、`app/` —— 观测以 `FrameObservation` 传入，
因此数据集与摄像头两条来源共用同一份更新逻辑。

现有能力：

| 模块 | 作用 | 需要 GPU |
|---|---|---|
| `color_fit` | 闭式颜色校正（9/12 参数，零迭代） | ❌ |
| `losses` | 掩码 MSE / PSNR（优化目标与验收判据同源） | ❌ |
| `observations` | 观测契约（帧 + 相机） | ❌ |
| `updater` | 梯度微调（颜色 / 不透明度），带时间预算 | ✅ |
| `plans` | 参数计划：哪些组参与优化、各自学习率曲线 | ❌ |
| `schedule` | 学习率曲线、blend 暖启动、早停判据 | ❌ |
| `objectives` | 加权损失（L1/SSIM/LPIPS/alpha），返回分项 | ❌ |
| `trainer` | 通用训练循环 + 钩子（验证/存档/日志/时间预算）| 取决于注入的 `render_fn` |

`trainer` 通过**注入 `render_fn`** 与渲染解耦：训练逻辑（调度、钩子、
早停）可以脱离 CUDA 扩展单独测试。`updater` 将来会收敛成
`trainer + ParameterPlan.appearance_only()` 的薄壳（见 docs/TRAINING.md）。

**边界**：热更新只改外观，不碰几何与驱动参数 —— 原因见 `updater` 模块文档。
"""

from .color_fit import ColorCorrection, fit_color_correction
from .losses import (PhotometricLoss, masked_l1, masked_mse,
                     masked_psnr, weighted_l1)
from .observations import BatchObservation, FrameObservation
from .updater import OnlineUpdater, UpdateConfig, UpdateReport, UpdateState

__all__ = [
    "BatchObservation",
    "ColorCorrection",
    "FrameObservation",
    "OnlineUpdater",
    "PhotometricLoss",
    "UpdateConfig",
    "UpdateReport",
    "UpdateState",
    "fit_color_correction",
    "masked_l1",
    "masked_mse",
    "masked_psnr",
    "weighted_l1",
]
