"""高斯 avatar 的参数容器。

职责边界（见 docs/ARCHITECTURE.md §3.7）：
- **只持有参数与绑定**，不持有优化器、不做渲染、不做 I/O 之外的副作用；
- 序列化只做「参数 ↔ PLY」的映射，模型定义（层数/维度）由 `AvatarConfig` 显式给出。

参数形状（N = 高斯数，K = 约简基数量）::

    xyz      [N, 3]        opacity [N, 1]（logit）    scaling [N, 3]（log）
    rotation [N, 4]        color   [N, 1, 3]（SH DC）
    xyz_b      [K, N, 3]      rotation_b [K, N, 4]      color_b [K, N, 1, 3]

⚠️ 只对 **xyz / rotation / color** 做 blendshape 混合；
**opacity 与 scaling 不参与**（参照实现亦然，是约简高斯 blendshape 的关键设计）。
"""

from __future__ import annotations

from pathlib import Path as Path_type

import math
from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import nn

from .deform.bind import Binding, MeshBinder
from .deform.blend import GaussianBlendField
from .types import GaussianSet, Mesh

# SH 0 阶归一化常数；rgb2sh0/shh0_2rgb 与之互逆
SH_C0 = 0.28209479177387814


@dataclass
class AvatarConfig:
    """模型结构定义。**必须完整描述网络**，否则 PLY 中的 `weight_module` 无法还原。

    Attributes:
        tex_size: UV 图边长，决定高斯数量（有效 texel 数）
        num_basis_in: 驱动参数维度 D（离线 FLAME 为 129，NeRSemble 为 100）
        num_basis_blend: 约简后的基数量 K
        mlp_hidden: 权重投影 MLP 的隐藏层宽度；空元组表示单层线性映射
        use_weight_proj: 是否启用权重投影；False 时直接用前 K 维驱动参数
        init_opacity / init_scaling: 初始化值（激活前空间由 `inverse_*` 处理）
    """

    tex_size: int = 256
    num_basis_in: int = 129
    num_basis_blend: int = 20
    mlp_hidden: tuple[int, ...] = (128, 128)
    use_weight_proj: bool = True
    init_opacity: float = 0.5
    init_scaling: float = 0.0008
    sh_degree: int = 0

    def __post_init__(self) -> None:
        if self.tex_size <= 0:
            raise ValueError(f"tex_size 必须为正，实际 {self.tex_size}")
        if self.num_basis_blend <= 0:
            raise ValueError(f"num_basis_blend 必须为正，实际 {self.num_basis_blend}")
        if self.use_weight_proj and self.num_basis_in <= 0:
            raise ValueError("启用 use_weight_proj 时 num_basis_in 必须为正")

    @classmethod
    def from_yaml(cls, path, *, section: str = "model") -> "AvatarConfig":
        """从（RGBAvatar 格式的）模型 `config.yaml` 读结构。

        兼容点：
        - 键名直接沿用 RGBAvatar（`tex_size` / `num_basis_in` / `num_basis_blend`
          / `init_opacity` / `init_scaling` / `use_weight_proj`）
        - `use_blend` / `use_mlp_proj` 这两个旧开关：`use_blend=False` 等价于
          不做权重投影（直接用前 K 维），`use_mlp_proj=False` 等价于单层线性映射
        - 缺的键用本类默认值，不报错 —— 老 config 可能没有 `mlp_hidden`

        Args:
            path: yaml 路径
            section: 取哪一节（默认 `model`）
        """
        import yaml

        raw = yaml.safe_load(Path_type(path).read_text(encoding="utf-8")) or {}
        sec = raw.get(section) or {}
        if not isinstance(sec, dict):
            raise ValueError(f"{path} 的 {section!r} 节不是映射")

        tex_size = int(sec.get("tex_size", cls.tex_size))
        num_basis_in = int(sec.get("num_basis_in", cls.num_basis_in))
        num_basis_blend = int(sec.get("num_basis_blend", cls.num_basis_blend))
        use_weight_proj = bool(sec.get(
            "use_weight_proj", sec.get("use_blend", cls.use_weight_proj)))
        use_mlp_proj = bool(sec.get("use_mlp_proj", True))
        hidden = sec.get("mlp_hidden")
        mlp_hidden = tuple(int(v) for v in hidden) if hidden is not None else (
            cls.mlp_hidden if use_mlp_proj else ())
        return cls(
            tex_size=tex_size,
            num_basis_in=num_basis_in,
            num_basis_blend=num_basis_blend,
            mlp_hidden=mlp_hidden,
            use_weight_proj=use_weight_proj,
            init_opacity=float(sec.get("init_opacity", cls.init_opacity)),
            init_scaling=float(sec.get("init_scaling", cls.init_scaling)),
            sh_degree=int(sec.get("sh_degree", cls.sh_degree)),
        )

    @property
    def num_basis(self) -> int:
        """实际参与线性混合的基数量。"""
        return self.num_basis_blend if self.use_weight_proj else self.num_basis_in

    def build_weight_module(self) -> nn.Module:
        """按配置构造 `D → K` 的权重投影模块。"""
        if not self.use_weight_proj:
            return nn.Identity()
        layers: list[nn.Module] = []
        prev = self.num_basis_in
        for hidden in self.mlp_hidden:
            layers += [nn.Linear(prev, hidden), nn.ReLU()]
            prev = hidden
        layers.append(nn.Linear(prev, self.num_basis_blend))
        return nn.Sequential(*layers)


def inverse_sigmoid(x: float) -> float:
    return math.log(x / (1.0 - x))


def rgb2sh0(rgb: torch.Tensor) -> torch.Tensor:
    """`[0,1]` 颜色 → SH DC 系数。"""
    return (rgb - 0.5) / SH_C0


def sh0_to_rgb(sh: torch.Tensor) -> torch.Tensor:
    """SH DC 系数 → `[0,1]` 颜色（`rgb2sh0` 的逆）。"""
    return sh * SH_C0 + 0.5


class GaussianAvatar(nn.Module):
    """约简高斯 blendshape avatar 的参数容器。

    参数以 `nn.Parameter` 持有，可直接交给优化器；未启用训练时也可只做前向。
    """

    def __init__(self, config: AvatarConfig, binding: Binding) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("binding_face_id", binding.face_id.to(torch.long))
        self.register_buffer("binding_face_bary", binding.face_bary.to(torch.float32))
        self.register_buffer("valid_binding_mask", binding.valid_mask.to(torch.bool))

        n = int(self.binding_face_id.shape[0])
        k = config.num_basis
        self.num_gaussians = n
        self.num_basis = k

        # 基态：位置为 0（真实位置由绑定面决定），颜色为 0（由 fast_forward 或训练填充）
        self.xyz = nn.Parameter(torch.zeros(n, 3))
        self.opacity = nn.Parameter(
            torch.full((n, 1), inverse_sigmoid(config.init_opacity)))
        self.scaling = nn.Parameter(
            torch.full((n, 3), math.log(config.init_scaling)))
        self.rotation = nn.Parameter(
            torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(n, 1))
        self.color = nn.Parameter(torch.zeros(n, 1, 3))

        # 基：初始化为 0，残差由训练学习
        self.xyz_b = nn.Parameter(torch.zeros(k, n, 3))
        self.rotation_b = nn.Parameter(torch.zeros(k, n, 4))
        self.color_b = nn.Parameter(torch.zeros(k, n, 1, 3))

        self.weight_module = config.build_weight_module()

    # ------------------------------------------------------------ 属性分组 --
    def parameter_groups(self) -> dict[str, nn.Parameter]:
        """按语义分组，供优化器分别设置学习率。"""
        return {
            "xyz": self.xyz,
            "opacity": self.opacity,
            "scaling": self.scaling,
            "rotation": self.rotation,
            "f_dc": self.color,
            "xyz_b": self.xyz_b,
            "rotation_b": self.rotation_b,
            "f_dc_b": self.color_b,
            # 权重投影 MLP：训练时要单独给学习率（RGBAvatar 的 weight_module_lr），
            # 而它不在上面那些"基"里 —— 之前漏了这一组，训练脚本只能自己再拼。
            **({"weight_module": self.weight_module} if
               self.config.use_weight_proj else {}),
        }

    def to_binding(self) -> Binding:
        return Binding(
            face_id=self.binding_face_id,
            face_bary=self.binding_face_bary,
            valid_mask=self.valid_binding_mask,
        )

    def build_blend_field(self, use_cuda_kernel: bool = False) -> GaussianBlendField:
        """构造 `BlendField`：驱动参数 → 切空间高斯。"""
        wm = self.weight_module if self.config.use_weight_proj else None
        return GaussianBlendField(
            base_xyz=self.xyz,
            base_rotation=self.rotation,
            base_color=self.color,
            base_opacity=self.opacity,
            base_scaling=self.scaling,
            basis_xyz=self.xyz_b,
            basis_rotation=self.rotation_b,
            basis_color=self.color_b,
            weight_module=wm,
            num_basis_in=self.config.num_basis_in,
            use_cuda_kernel=use_cuda_kernel,
        )

    def build_binder(self) -> MeshBinder:
        """返回绑定器。

        **同一实例会被复用**：`MeshBinder` 内部缓存了「去重后的绑定面 + 反查索引」
        （纯模板依赖）。若每次调用都新建实例，该缓存每帧都会失效，
        `torch.unique` 就会每帧重跑一次（CPU 上约 2 ms）—— 那比不做优化还慢。
        `deform()` 每帧都会取一次 binder，因此这里必须缓存。
        """
        cached = getattr(self, "_binder_cache", None)
        if cached is None:
            cached = MeshBinder(self.to_binding())
            self._binder_cache = cached                 # type: ignore[attr-defined]
        return cached

    # ---------------------------------------------------------------- 前向 --
    def deform(
        self,
        mesh: Mesh,
        blend_weight: torch.Tensor | None = None,
        use_cuda_kernel: bool = False,
    ) -> GaussianSet:
        """网格 + 驱动参数 → 世界空间高斯。

        Args:
            mesh: `[B, V, 3]`（须与 `blend_weight` 的 batch 一致，且拓扑与模板相同）
            blend_weight: `[B, D]`；``None`` 表示只用基态（不做混合）
        """
        field = self.build_blend_field(use_cuda_kernel=use_cuda_kernel)
        # blend_weight=None → 只用基态（对应参照 blend_start_iter 之前的行为）
        tangent = field(blend_weight, batch_size=mesh.batch_size)
        tangent = tangent.to(mesh.verts.device)
        return self.build_binder().bind(tangent, mesh)

    # ------------------------------------------------------------ 纹理视图 --
    def extract_texture(self) -> torch.Tensor:
        """把高斯颜色铺回 UV 图，返回 `[tex_size, tex_size, 3]`。

        对应参照实现的 `BindingModel.extract_texture`：`_feature_dc` 存的是 SH 系数，
        先转成 `[0,1]` 颜色再写入有效 texel。

        ⚠️ **仅当绑定来自 UV 光栅化时可用**（此时 `valid_binding_mask` 是
        `[tex_size²]`，每个高斯对应一个已知 texel）。从 `.ply` 加载时文件里
        只存了有效 texel（`N` 个高斯），**原始 texel 位置信息不在文件里**，
        因此无法重建 UV 图 —— 此时显式报错，而不是画出一张错位的纹理图。
        """
        size = self.config.tex_size
        n_texel = size * size
        mask = self.valid_binding_mask
        if int(mask.numel()) != n_texel:
            raise RuntimeError(
                f"无法重建 UV 纹理图：valid_binding_mask 有 {int(mask.numel())} 项，"
                f"应为 tex_size²={n_texel}。\n"
                "从 .ply 加载时只保留了有效 texel，原始 UV 位置未存盘。"
                "若确需纹理图，请在构建绑定的那一侧（UV 光栅化时）调用本方法，"
                "或把 texel 索引一并写进 .ply。")
        flat = torch.zeros(n_texel, 3, dtype=self.color.dtype, device=self.color.device)
        flat[mask] = sh0_to_rgb(self.color.squeeze(1))
        return flat.reshape(size, size, 3)

    # -------------------------------------------------------------- 设备 --
    def to(self, *args, **kwargs):  # type: ignore[override]
        module = super().to(*args, **kwargs)
        self.binding_face_id = self.binding_face_id.to(
            module.binding_face_bary.device)
        return module
