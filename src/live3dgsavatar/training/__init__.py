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

**边界**：热更新只改外观，不碰几何与驱动参数 —— 原因见 `updater` 模块文档。
"""

from .color_fit import ColorCorrection, fit_color_correction
from .losses import PhotometricLoss, masked_mse, masked_psnr
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
    "masked_mse",
    "masked_psnr",
]
