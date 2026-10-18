"""RDF 输出参数 → 本项目 129 维驱动参数。

## 两边的布局

RDF（`OnlineGNOptimizer.dim_x = num_expressions + 18`）：:

    x = [exp 100][pose 15][trans 3]
    pose = [global 3][neck 3][jaw 3][eye_l 3][eye_r 3]     # 轴角

本项目（模型训练时用的 `blend_weight`）：:

    [  0:100] 表情
    [100:106] 头姿 rot6d
    [106:112] 下颌 rot6d
    [112:124] 眼球 rot6d（两组）
    [124:126] 眼睑
    [126:129] 平移

## 三个必须处理的差异

1. **轴角 → rot6d**：rot6d 取旋转矩阵的**前两行**（与 `rotation_6d_to_matrix` 互逆）。
2. **头姿是复合的**：RDF 把全局旋转拆成 `global` 与 `neck` 两级，
   其 `_extract_head_rot` 返回 `R_global @ R_neck`。本项目的头姿段按
   **复合结果** 填入（可用 `compose_head=False` 改回只看 global，供标定时对比）。
3. **眼睑**：RDF 不输出眼睑，填 0（数据集的眼睑段也是全 0，见验证记录）。

## 平移为什么不能直接用

RDF 的 `trans` 在**它自己的相机空间**里（虚拟相机在 (0,0,1)、fov_y=30°），
而模型的 `blend_weight` 来自训练管线，处在数据集的世界空间。

一个 3D 点在相机内参固定时，沿相机中心整体缩放**不改变投影** ——
所以两者的差异至少包含一个"绕相机中心的缩放"，这正是需要在真实数据上
标定的量（见 `align.py`）。**本模块不猜这个变换**：调用方可以把标定后的
平移传进来（`translation=`），否则原样透传。
"""

from __future__ import annotations

import torch

#: RDF 的表情维数（`--num_expressions 100`）。
NUM_EXPRESSIONS = 100

#: 姿态段的分段（含结束下标）。顺序即 RDF 的轴角排列。
POSE_SLICES: tuple[tuple[str, int, int], ...] = (
    ("global", 0, 3),
    ("neck", 3, 6),
    ("jaw", 6, 9),
    ("eye_l", 9, 12),
    ("eye_r", 12, 15),
)

#: `x` 的总维数。
DIM_X = NUM_EXPRESSIONS + 15 + 3

#: `blend_weight` 的总维数。
DIM_BLEND_WEIGHT = 129

#: `blend_weight` 各段的起止下标（便于诊断与断言）。
BW_SLICES: dict[str, tuple[int, int]] = {
    "expression": (0, 100),
    "head": (100, 106),
    "jaw": (106, 112),
    "eyes": (112, 124),
    "eyelids": (124, 126),
    "translation": (126, 129),
}


# ------------------------------------------------------------ 旋转工具 --


def rotvec_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    """轴角 `[..., 3]` → 旋转矩阵 `[..., 3, 3]`（Rodrigues）。

    与 `data/flame_model.rodrigues` 同式，但这里独立实现：
    `tracking/` 不应依赖 `data/`（数据层有自己的 FLAME 资产约定）。
    """
    theta = rotvec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    k = rotvec / theta
    zero = torch.zeros_like(theta[..., 0])
    K = torch.stack([
        torch.stack([zero, -k[..., 2], k[..., 1]], dim=-1),
        torch.stack([k[..., 2], zero, -k[..., 0]], dim=-1),
        torch.stack([-k[..., 1], k[..., 0], zero], dim=-1),
    ], dim=-2)
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device).expand_as(K)
    sin = torch.sin(theta)[..., None]
    cos = torch.cos(theta)[..., None]
    return eye + sin * K + (1.0 - cos) * (K @ K)


def matrix_to_rot6d(mat: torch.Tensor) -> torch.Tensor:
    """`[..., 3, 3]` → `[..., 6]`，取**前两行**。

    与 `rotation_6d_to_matrix` 互逆：那边把 6 个数按 `(b1, b2, b3)` 堆成行。
    """
    return mat[..., :2, :].reshape(*mat.shape[:-2], 6)


# --------------------------------------------------------------- 分段 --


def split_pose(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """把 `x` 的姿态段拆成 5 个轴角 `[B, 3]`。"""
    if x.shape[-1] != DIM_X:
        raise ValueError(f"x 的维数应为 {DIM_X}，实际 {x.shape[-1]}")
    pose = x[..., NUM_EXPRESSIONS:NUM_EXPRESSIONS + 15]
    return {name: pose[..., a:b] for name, a, b in POSE_SLICES}


def head_matrix(x: torch.Tensor, *, compose: bool = True) -> torch.Tensor:
    """头姿旋转矩阵 `[B, 3, 3]`。

    Args:
        compose: ``True``（默认）返回 `R_global @ R_neck`（与 RDF 的
            `_extract_head_rot` 一致）；``False`` 只返回 `R_global`。
            标定时两种假设都要试 —— 训练管线用的是哪种只能由数据回答。
    """
    parts = split_pose(x)
    rot = rotvec_to_matrix(parts["global"])
    if compose:
        rot = rot @ rotvec_to_matrix(parts["neck"])
    return rot


# ------------------------------------------------------------ 主转换 --


def to_blend_weight(
    x: torch.Tensor,
    *,
    translation: torch.Tensor | None = None,
    compose_head: bool = True,
) -> torch.Tensor:
    """RDF 的 `x` → 129 维驱动参数。

    Args:
        x: `[B, 118]`（或 `[118]`，会补 batch 维）
        translation: `[B, 3]`，**已标定**的平移。``None`` 时原样透传 RDF 的
            `trans` —— 只在"仅关心旋转/表情"的诊断里这么做，
            真正渲染时应当传入标定结果。
        compose_head: 头姿是否用 `R_global @ R_neck`

    Returns:
        `[B, 129]` float32
    """
    single = x.ndim == 1
    if single:
        x = x[None]

    if x.shape[-1] != DIM_X:
        raise ValueError(f"x 的维数应为 {DIM_X}，实际 {x.shape[-1]}")

    parts = split_pose(x)
    exp = x[..., :NUM_EXPRESSIONS]

    head6 = matrix_to_rot6d(head_matrix(x, compose=compose_head))
    jaw6 = matrix_to_rot6d(rotvec_to_matrix(parts["jaw"]))
    eye6 = torch.cat([
        matrix_to_rot6d(rotvec_to_matrix(parts["eye_l"])),
        matrix_to_rot6d(rotvec_to_matrix(parts["eye_r"])),
    ], dim=-1)
    # RDF 不估计眼睑。数据集的眼睑段也是全 0，因此填 0 与训练分布一致。
    eyelids = torch.zeros(*x.shape[:-1], 2, dtype=x.dtype, device=x.device)

    trans = x[..., NUM_EXPRESSIONS + 15:] if translation is None else translation

    out = torch.cat([exp, head6, jaw6, eye6, eyelids, trans], dim=-1)
    if out.shape[-1] != DIM_BLEND_WEIGHT:
        raise AssertionError(
            f"组装后的维数 {out.shape[-1]} != {DIM_BLEND_WEIGHT}")
    return out.to(torch.float32) if single is False else out[0].to(torch.float32)


# ------------------------------------------------------------- 诊断 --


def segment_errors(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """逐段比较两组 129 维参数，返回每段的 `max|Δ|`。

    用于标定时判断"哪一段已经对了、哪一段还没对" —— 比只看整体误差有用得多：
    表情段一致说明 RDF 的输出口径与训练一致；只有平移差说明只需标定变换。
    """
    if pred.shape[-1] != target.shape[-1]:
        raise ValueError(
            f"维数不一致：{pred.shape[-1]} vs {target.shape[-1]}")
    out = {}
    for name, (a, b) in BW_SLICES.items():
        out[name] = float(
            (pred[..., a:b].to(torch.float64) - target[..., a:b].to(torch.float64)
             ).abs().max())
    return out
