"""混合：驱动参数 → 切空间高斯。

对应参照实现的 `GaussianModel.get_batch_attributes`：

    weight = project_weight(blend_weight)          # [B, D] → [B, K]
    _xyz, _rot, _color = linear_blending(weight, _xyz, _rotation, _feature_dc,
                                         _xyz_b, _rotation_b, _feature_b)
    opacity = sigmoid(_opacity)                    # ⚠️ 不参与混合
    scaling = exp(_scaling)                        # ⚠️ 不参与混合
    rotation = normalize(_rot)
    sh = _color                                   # [B, N, 1, 3]

`linear_blending` 的 CUDA 实现（`cuda_utils/linear_blending.cu`）::

    out[b, n] = base[n] + Σ_l weight[b, l] · basis[l, n]

即**基是 `[K, N, ·]`**，与批次无关；`weight` 为 `[B, K]`。

⚠️ 参照实现在注释里特别提醒：`blend_weight` 若被多 unsqueeze 一次会产生
「奇怪的渲染结果」。本实现用显式形状检查把这类错误变成异常，而不是静默出错。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..types import GaussianSet


def linear_blending(
    weights: torch.Tensor,     # [B, K]
    base: torch.Tensor,        # [N, ...]
    basis: torch.Tensor,       # [K, N, ...]
) -> torch.Tensor:
    """`base + Σ_k weights[:, k] · basis[k]`，返回 `[B, N, ...]`。

    纯 PyTorch 等价实现。参照实现走 CUDA 算子以省显存/提速；
    本层保留可读版本，CUDA 版在 `BlendField` 中可选启用（见 `use_cuda_kernel`）。
    """
    if weights.ndim != 2:
        raise ValueError(f"weights 形状应为 [B, K]，实际 {tuple(weights.shape)}")
    if basis.ndim != base.ndim + 1:
        raise ValueError(
            f"basis 形状应为 [K, N, ...]（比 base 多一维），"
            f"实际 base={tuple(base.shape)}, basis={tuple(basis.shape)}")
    if basis.shape[0] != weights.shape[1]:
        raise ValueError(
            f"K 不一致：weights 的 K={weights.shape[1]}，basis 的 K={basis.shape[0]}")
    if basis.shape[1:] != base.shape:
        raise ValueError(
            f"基与基态形状不一致：base={tuple(base.shape)}, basis[1:]={tuple(basis.shape[1:])}")

    # `base + Σ_k w_k · basis_k`，把 K 维做成一次矩阵收缩。
    #
    # ⚠️ **不要写成 `(w[:, :, None, ...] * basis[None]).sum(dim=1)`**：
    #    那会先物化 `[B, K, N, ·]` 中间张量（K=20、N=60353 时 13.8 MiB），
    #    实测比下面的写法慢 **3.2×**（1.88 ms → 0.59 ms／属性，K=20、N=60353）。
    #    这是纯 PyTorch 侧的写法问题，与是否使用 CUDA 内核无关。
    #
    # 数值说明：收缩顺序与"逐 k 累加"不同，float32 下差异约 3.8e-06
    #（远低于本层 1e-5 的判据）。逐位一致并非该层的契约 —— 参照实现自身
    # 在 CUDA 版与 Python 回退版之间也有同类求和顺序差异。
    b = weights.shape[0]
    tail = base.shape[1:]
    flat_base = base.reshape(base.shape[0], -1)                 # [N, P]
    flat_basis = basis.reshape(basis.shape[0], basis.shape[1], -1)   # [K, N, P]
    out = torch.tensordot(weights, flat_basis, dims=([1], [0]))  # [B, N, P]
    return out.reshape(b, base.shape[0], *tail) + base.unsqueeze(0)


class GaussianBlendField:
    """`BlendField` 的实现：约简高斯 blendshape。

    Args:
        base_xyz / base_rotation / base_color: 基态，`[N, 3]` / `[N, 4]` / `[N, 1, 3]`
        basis_xyz / basis_rotation / basis_color: 基，`[K, N, ·]`
        base_opacity / base_scaling: logit/log 空间，`[N, 1]` / `[N, 3]`
        weight_module: 可选的 `D → K` 映射（MLP 或线性层）；
            为 ``None`` 时要求传入的 `blend_weight` 已经是 K 维
        num_basis_in: 当 `weight_module` 非 None 时，取 `blend_weight` 的前若干维
        use_cuda_kernel: 用 CUDA 版 `linear_blending` 替代纯 PyTorch 版。
            **默认 False**，理由见下。

    Note:
        默认不走 CUDA 内核是**有意为之**。实测 `diff_gaussian_rasterization.linear_blending`
        在 CUDA 不可用时（例如 CPU 张量、无可用设备）会**静默返回全零**而不报错，
        参照实现默认走这条路。纯 PyTorch 版在本项目实测下与 CUDA 版数值等价
        （见 tests/unit/test_deform.py 与 tests/equivalence/），且形状错误会显式报错，
        因此作为默认。
        训练态若需省显存，在确认设备可用后显式传 `use_cuda_kernel=True`。
    """

    def __init__(
        self,
        base_xyz: torch.Tensor,
        base_rotation: torch.Tensor,
        base_color: torch.Tensor,
        base_opacity: torch.Tensor,
        base_scaling: torch.Tensor,
        basis_xyz: torch.Tensor,
        basis_rotation: torch.Tensor,
        basis_color: torch.Tensor,
        weight_module: torch.nn.Module | None = None,
        num_basis_in: int | None = None,
        use_cuda_kernel: bool = False,
    ) -> None:
        self.base_xyz = base_xyz
        self.base_rotation = base_rotation
        self.base_color = base_color
        self.base_opacity = base_opacity
        self.base_scaling = base_scaling
        self.basis_xyz = basis_xyz
        self.basis_rotation = basis_rotation
        self.basis_color = basis_color
        self.weight_module = weight_module
        self.num_basis_in = num_basis_in
        self.use_cuda_kernel = use_cuda_kernel

    @property
    def num_basis_blend(self) -> int:
        return self.basis_xyz.shape[0]

    # ------------------------------------------------------------- 权重投影 --
    def project_weight(self, blend_weight: torch.Tensor) -> torch.Tensor:
        """`[B, D]` → `[B, K]`。

        与参照实现 `GaussianModel.project_weight` 一致：只取前 `num_basis_in` 维，
        再经 `weight_module` 约简。**未配置 `weight_module` 时原样返回**，
        以支持「直接用基权重」的消融配置。
        """
        if self.weight_module is None:
            return blend_weight
        w = blend_weight
        if self.num_basis_in is not None:
            w = w[:, : self.num_basis_in]
        return self.weight_module(w)

    # --------------------------------------------------------------- 主流程 --
    def __call__(self, blend_weight: torch.Tensor | None, batch_size: int = 1) -> GaussianSet:
        """`blend_weight` `[B, D]` → 切空间 `GaussianSet`。

        Args:
            blend_weight: `[B, D]`；``None`` 表示**只用基态、不做任何混合**
                （对应参照实现在 `blend_start_iter` 之前的行为）。
                此时仍需 `batch_size` 来确定输出的批次维。
        """
        if blend_weight is None:
            b = batch_size
            xyz = self.base_xyz.unsqueeze(0).expand(b, -1, 3)
            rot = self.base_rotation.unsqueeze(0).expand(b, -1, 4)
            color = self.base_color.unsqueeze(0).expand(b, -1, 1, 3)
        else:
            if blend_weight.ndim != 2:
                raise ValueError(
                    f"blend_weight 形状应为 [B, D]，实际 {tuple(blend_weight.shape)}"
                    "（不要预先 unsqueeze，本方法自行处理批次维）")
            weights = self.project_weight(blend_weight)
            b = weights.shape[0]
            if self.use_cuda_kernel:
                xyz, rot, color = self._linear_blending_cuda(weights)
            else:
                xyz = linear_blending(weights, self.base_xyz, self.basis_xyz)
                rot = linear_blending(weights, self.base_rotation, self.basis_rotation)
                color = linear_blending(weights, self.base_color, self.basis_color)

        n = xyz.shape[1]

        opacity = torch.sigmoid(self.base_opacity).unsqueeze(0).expand(b, n, 1)
        scaling = torch.exp(self.base_scaling).unsqueeze(0).expand(b, n, 3)

        return GaussianSet(
            xyz=xyz,
            rotation=F.normalize(rot, dim=-1),
            scaling=scaling,
            opacity=opacity,
            color=color,
            space="tangent",
        )

    def _linear_blending_cuda(self, weights: torch.Tensor):
        """走 CUDA 版 `linear_blending`（`[K,N,·]` 的基布局）。

        ⚠️ 该算子在 CUDA 不可用时会**静默返回全零**而非报错，因此在此显式拦截。
        """
        if not self.base_xyz.is_cuda:
            raise RuntimeError(
                "use_cuda_kernel=True 要求输入在 CUDA 设备上："
                "该内核在 CPU 张量上不报错、直接返回全零（静默失败），故拒绝执行")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "use_cuda_kernel=True 但 torch.cuda.is_available() 为 False；"
                "改用纯 PyTorch 路径（use_cuda_kernel=False）")

        from diff_gaussian_rasterization import linear_blending as cuda_lb  # noqa: PLC0415

        return cuda_lb(
            weights.contiguous(),
            self.base_xyz.contiguous(), self.base_rotation.contiguous(),
            self.base_color.contiguous(),
            self.basis_xyz.contiguous(), self.basis_rotation.contiguous(),
            self.basis_color.contiguous(),
        )
