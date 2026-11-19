"""高斯 avatar 的参数容器。

职责边界：
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


def _opt_float(value) -> float | None:
    """yaml 里的可选浮点：缺失/None 保持 None。"""
    return None if value is None else float(value)


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
        init_scale_tangent / init_scale_normal: 盘状初始化。给了就按
            (切向, 切向, 法向) 逐轴设尺度，高斯一开始就是贴在绑定面上的小圆片；
            缺省（None）沿用 `init_scaling` 的各向同性球。UV 网格的 texel 间距
            决定了切向下限：圆片比 texel 还小就会在网格上留下条纹/空洞，
            所以盘状初始化应该给到 texel 间距量级（256² 头上约 2.2 mm）。
    """

    tex_size: int = 256
    num_basis_in: int = 129
    num_basis_blend: int = 20
    mlp_hidden: tuple[int, ...] = (128, 128)
    use_weight_proj: bool = True
    init_opacity: float = 0.5
    init_scaling: float = 0.0008
    init_scale_tangent: float | None = None
    init_scale_normal: float | None = None
    sh_degree: int = 0

    def __post_init__(self) -> None:
        if self.tex_size <= 0:
            raise ValueError(f"tex_size 必须为正，实际 {self.tex_size}")
        if self.num_basis_blend <= 0:
            raise ValueError(f"num_basis_blend 必须为正，实际 {self.num_basis_blend}")
        if self.use_weight_proj and self.num_basis_in <= 0:
            raise ValueError("启用 use_weight_proj 时 num_basis_in 必须为正")
        for name in ("init_scale_tangent", "init_scale_normal"):
            value = getattr(self, name)
            if value is not None and float(value) <= 0:
                raise ValueError(f"{name} 必须为正，实际 {value}")

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
            init_scale_tangent=_opt_float(sec.get("init_scale_tangent")),
            init_scale_normal=_opt_float(sec.get("init_scale_normal")),
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
        if (config.init_scale_tangent is None
                and config.init_scale_normal is None):
            self.scaling = nn.Parameter(
                torch.full((n, 3), math.log(config.init_scaling)))
        else:
            # 盘状初始化：切空间 (x, y) 是面内两轴、z 是法向。单位旋转下
            # 局部 z 轴正好落在绑定面法线上，所以只压 z 就能得到贴面小圆片。
            st = float(config.init_scale_tangent if config.init_scale_tangent
                       is not None else config.init_scaling)
            sn = float(config.init_scale_normal if config.init_scale_normal
                       is not None else config.init_scaling)
            scale = torch.tensor([st, st, sn]).log()
            self.scaling = nn.Parameter(scale.repeat(n, 1))
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
            # 它不属于上面的基，但优化器必须能单独配置它。
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

    def split_unified(self, indices: torch.Tensor, *,
                      scale_factor: float = 0.5,
                      opacity_factor: float = 0.5,
                      jitter: float = 0.8,
                      pattern: str = "random",
                      phase: int = 0,
                      mode: str = "copy") -> int:
        """统一集合分裂：含 blendshape 基。

        Args:
            indices: 要分裂的高斯（按残差挑出来的"欠重建"位置）。
            scale_factor / opacity_factor: 副本的尺度/不透明度倍率。
            jitter: 位移倍数（单位：1σ）。**必须 > 0**：原地复制会得到
                两个完全重合的高斯，既不增加覆盖，又让后续 split 反复选中
                同一批位置。
            pattern: `random` 随机方向；`quad` 切平面 4 方向轮转。
            phase: quad 方向的轮转相位（每轮 45°）。
            mode:
                ``copy``      父高斯不动，追加一个"更小更淡"的副本。总量增加，
                              画面偏软（历史默认）。
                ``subdivide`` 父高斯替换成**两个**子高斯：两者都缩到
                              `scale_factor`、**透明度不变**，位置沿 ±方向
                              分开。radiance 总量近似不变，但等效分辨率翻倍 ——
                              用来补精细纹理。
        """
        idx = indices.to(self.xyz.device).long()
        if idx.numel() == 0:
            return 0
        total = int(self.num_gaussians + idx.numel())
        if total > int(self.config.tex_size) ** 2:
            raise RuntimeError(
                f"分裂后 {total} 超过 tex_size={self.config.tex_size} 容量")
        def cat(a, b):
            return torch.cat([a, b], dim=0)
        # 副本位置 = 父位置 + 方向 × 父尺度 × jitter（切空间内，
        # 与 `xyz` 同一坐标系，绑定变换会把它一起带到世界空间）。
        #
        # pattern:
        #   random  随机方向（探索，但同一 texel 多次分裂方向不固定）
        #   quad    切平面内 4 个正交方向轮转，phase 每轮转 45°
        #           —— 相当于把一个 texel 细分成 4 个子格，适合补高频纹理
        base_xyz = self.xyz.data[idx]
        base_scaling = self.scaling.data[idx]
        base_opacity = self.opacity.data[idx]
        if float(jitter) > 0:
            sigma = torch.exp(base_scaling) * float(jitter)
            if pattern == "quad":
                k = torch.arange(idx.numel(), device=base_xyz.device)
                ang = (k % 4).to(base_xyz.dtype) * (math.pi / 2)
                ang = ang + float(phase) * (math.pi / 4)
                zero = torch.zeros_like(ang)
                direction = torch.stack([torch.cos(ang), torch.sin(ang), zero],
                                        dim=-1)
                offset = direction * sigma
            else:
                offset = torch.randn_like(base_xyz) * sigma
        else:
            offset = torch.zeros_like(base_xyz)

        if mode == "subdivide":
            # 父 -> 两个子：都缩到 scale_factor，透明度不变，位置沿 ± 分开
            child_xyz = base_xyz - offset
            parent_xyz = base_xyz + offset
            child_scaling = base_scaling + math.log(float(scale_factor))
            child_opacity = base_opacity
            new_xyz = child_xyz
            new_scaling = child_scaling
            new_opacity = child_opacity
            # 原地改父（先复制，避免之后 concat 时别名）
            xyz_all = self.xyz.data.clone()
            scaling_all = self.scaling.data.clone()
            xyz_all[idx] = parent_xyz
            scaling_all[idx] = child_scaling
            opacity_all = self.opacity.data
        else:
            new_xyz = base_xyz + offset
            new_scaling = base_scaling + math.log(float(scale_factor))
            new_opacity = base_opacity + math.log(float(opacity_factor))
            xyz_all = self.xyz.data
            scaling_all = self.scaling.data
            opacity_all = self.opacity.data
        self.binding_face_id = cat(self.binding_face_id, self.binding_face_id[idx])
        self.binding_face_bary = cat(self.binding_face_bary, self.binding_face_bary[idx])
        self.valid_binding_mask = cat(self.valid_binding_mask,
                                      self.valid_binding_mask[idx])
        self.xyz = nn.Parameter(cat(xyz_all, new_xyz))
        self.opacity = nn.Parameter(cat(opacity_all, new_opacity))
        self.scaling = nn.Parameter(cat(scaling_all, new_scaling))
        self.rotation = nn.Parameter(cat(self.rotation.data, self.rotation.data[idx]))
        self.color = nn.Parameter(cat(self.color.data, self.color.data[idx]))
        self.xyz_b = nn.Parameter(torch.cat(
            [self.xyz_b.data, self.xyz_b.data[:, idx]], dim=1))
        self.rotation_b = nn.Parameter(torch.cat(
            [self.rotation_b.data, self.rotation_b.data[:, idx]], dim=1))
        self.color_b = nn.Parameter(torch.cat(
            [self.color_b.data, self.color_b.data[:, idx]], dim=1))
        self.num_gaussians = total
        self._binder_cache = None
        return int(idx.numel())

    def prune_unified(self, keep: torch.Tensor) -> int:
        """按 bool mask 剪枝。"""
        keep = keep.to(self.xyz.device).bool()
        removed = int((~keep).sum())
        if removed == 0:
            return 0
        if int(keep.sum()) == 0:
            raise RuntimeError("剪枝后模型为空")
        self.binding_face_id = self.binding_face_id[keep].clone()
        self.binding_face_bary = self.binding_face_bary[keep].clone()
        self.valid_binding_mask = self.valid_binding_mask[keep].clone()
        for name in ("xyz", "opacity", "scaling", "rotation", "color"):
            setattr(self, name, nn.Parameter(getattr(self, name).data[keep].clone()))
        self.xyz_b = nn.Parameter(self.xyz_b.data[:, keep].clone())
        self.rotation_b = nn.Parameter(self.rotation_b.data[:, keep].clone())
        self.color_b = nn.Parameter(self.color_b.data[:, keep].clone())
        self.num_gaussians = int(keep.sum())
        self._binder_cache = None
        return removed

    def residual_indices(self, face_score: torch.Tensor, count: int, *,
                         min_opacity: float = 0.01,
                         max_offset: float = 0.08) -> torch.Tensor:
        """按绑定面残差为统一集合选择分裂候选。"""
        score = face_score.to(self.binding_face_id.device)[self.binding_face_id]
        with torch.no_grad():
            op = torch.sigmoid(self.opacity).squeeze(-1)
            off = self.xyz.norm(dim=-1)
            ok = (op >= min_opacity) & (off <= max_offset)
            score = torch.where(ok, score, torch.zeros_like(score))
            k = min(int(count), int(ok.sum()))
            if k <= 0:
                return torch.empty(0, dtype=torch.long, device=self.xyz.device)
        return torch.topk(score, k).indices

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
