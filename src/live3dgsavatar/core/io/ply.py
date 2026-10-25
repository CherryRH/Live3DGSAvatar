"""GaussianAvatar ↔ PLY 序列化。

**互操作契约**：写出的 PLY 必须能被参照实现（RGBAvatar）的 `load_ply` 读取，
反之亦然。这是 `tests/equivalence` 能用同一份 `model.ply` 驱动新旧两套实现的前提。

顶点属性顺序（与参照实现一致）::

    x y z  nx ny nz  opacity
    scale_0..2  rot_0..3  f_dc_0..2
    f_rest_0..44                      # 恒为 0（sh_degree=0 占位，保留以兼容 3DGS 工具链）
    xyz_b_0..3K-1
    rot_b_0..4K-1
    f_dc_b_0..3K-1
    weight_module                     # 展平后的 MLP 参数，前 P 行有值、其余补零
    [face_id]                         # i4，可选
    [face_bary_0..2]                  # f4，可选

相对参照实现修掉的两个缺陷（见 docs/MIGRATION.md）：

- **K4** `save_ply` 引用了不存在的 `self.binding_face_id`，任何模型都保存失败；
- **K3** `load_ply` 从配置读 `num_basis_blend` 而非文件列数，配置与文件不一致时静默出错。
  本实现在 PLY 头写一行 `comment gaussian_config ...` 自描述，并在加载时与文件列数交叉校验。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from ..avatar import AvatarConfig, GaussianAvatar
from ..deform import Binding

# f_rest 占位列数：sh_degree=0 时无高阶 SH，参照实现固定写 45 个零
F_REST_COLS = 45
CONFIG_COMMENT_PREFIX = "gaussian_config "


def _flatten_module(module: torch.nn.Module) -> np.ndarray:
    parts = [p.detach().reshape(-1).cpu().numpy() for p in module.state_dict().values()]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def _load_flattened_module(flat: torch.Tensor, module: torch.nn.Module) -> None:
    state = module.state_dict()
    offset = 0
    for key, param in state.items():
        numel = param.numel()
        chunk = flat[offset:offset + numel].reshape(param.shape)
        if chunk.numel() != numel:
            raise ValueError(
                f"weight_module 参数量不足：还原 {key} 需要 {numel} 个，"
                f"仅剩 {max(0, flat.numel() - offset)} 个。"
                "通常是 AvatarConfig.mlp_hidden 与实际模型不符")
        state[key] = chunk.to(param.dtype)
        offset += numel
    module.load_state_dict(state)


def save_ply(avatar: GaussianAvatar, path: str | Path) -> None:
    """把 avatar 写成 PLY。绑定信息存在时一并写入。"""
    path = Path(path)
    n = avatar.num_gaussians
    k = avatar.num_basis
    cfg = avatar.config

    xyz = avatar.xyz.detach().cpu().numpy()
    normal = np.zeros_like(xyz)
    opacity = avatar.opacity.detach().cpu().numpy()
    scaling = avatar.scaling.detach().cpu().numpy()
    rotation = avatar.rotation.detach().cpu().numpy()
    f_dc = avatar.color.detach().cpu().transpose(1, 2).reshape(n, -1).numpy()
    f_rest = np.zeros([n, F_REST_COLS], dtype=xyz.dtype)

    xyz_b = avatar.xyz_b.detach().cpu().transpose(0, 1).reshape(n, -1).numpy()
    rot_b = avatar.rotation_b.detach().cpu().transpose(0, 1).reshape(n, -1).numpy()
    f_dc_b = avatar.color_b.detach().cpu().transpose(0, 1).reshape(n, -1).numpy()

    # weight_module 被压进单列：前 P 行有值、其余补零（参照实现的约定）。
    # 因此要求 P <= N。参照架构（D→128→128→K）在 D=129/K=20 时为 35732 个参数，
    # 而 tex_size=256 时 N≈60000，余量充足；只有把 tex_size 缩到很小才会触发。
    flat_module = _flatten_module(avatar.weight_module).reshape(-1, 1)
    if flat_module.shape[0] > n:
        raise ValueError(
            f"weight_module 参数量 {flat_module.shape[0]} 超过高斯数 {n}，"
            "无法写入单列（参照实现的 PLY 约定）。"
            f"请把 tex_size 增大到至少 {int(math.ceil(math.sqrt(flat_module.shape[0])))}，"
            "或缩小 AvatarConfig.mlp_hidden")
    module_col = np.zeros([n, 1], dtype=np.float32)
    module_col[:flat_module.shape[0], 0] = flat_module[:, 0]

    names = ["x", "y", "z", "nx", "ny", "nz", "opacity"]
    names += [f"scale_{i}" for i in range(scaling.shape[1])]
    names += [f"rot_{i}" for i in range(rotation.shape[1])]
    names += [f"f_dc_{i}" for i in range(f_dc.shape[1])]
    names += [f"f_rest_{i}" for i in range(f_rest.shape[1])]
    names += [f"xyz_b_{i}" for i in range(xyz_b.shape[1])]
    names += [f"rot_b_{i}" for i in range(rot_b.shape[1])]
    names += [f"f_dc_b_{i}" for i in range(f_dc_b.shape[1])]
    names += ["weight_module"]

    has_binding = avatar.binding_face_id.numel() == n and avatar.binding_face_bary.numel() > 0
    face_id = bary = None
    if has_binding:
        face_id = avatar.binding_face_id.detach().cpu().numpy().astype(np.int32)[..., None]
        bary = avatar.binding_face_bary.detach().cpu().numpy().astype(np.float32)

    # 属性表：先全部按 float32，再把绑定列替换成正确的整型 dtype
    dtype: list[tuple[str, str]] = [(name, "f4") for name in names]
    if has_binding:
        dtype += [("face_id", "i4"),
                  ("face_bary_0", "f4"), ("face_bary_1", "f4"), ("face_bary_2", "f4")]

    attrs = np.concatenate(
        [xyz, normal, opacity, scaling, rotation, f_dc, f_rest, xyz_b, rot_b, f_dc_b, module_col],
        axis=1)
    if has_binding:
        attrs = np.concatenate([attrs, face_id, bary], axis=1)

    elements = np.empty(n, dtype=dtype)
    elements[:] = list(map(tuple, attrs))
    el = PlyElement.describe(elements, "vertex")

    config = {
        "tex_size": cfg.tex_size,
        "num_basis_in": cfg.num_basis_in,
        "num_basis_blend": cfg.num_basis_blend,
        "mlp_hidden": list(cfg.mlp_hidden),
        "use_weight_proj": cfg.use_weight_proj,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([el], text=False, comments=[CONFIG_COMMENT_PREFIX + json.dumps(config)]).write(str(path))


def _read_config_comment(plydata: PlyData) -> dict | None:
    for comment in getattr(plydata, "comments", []) or []:
        text = comment.decode("utf-8", "ignore") if isinstance(comment, bytes) else str(comment)
        if text.startswith(CONFIG_COMMENT_PREFIX):
            return json.loads(text[len(CONFIG_COMMENT_PREFIX):])
    return None


def read_embedded_config(path: str | Path) -> "AvatarConfig | None":
    """读 PLY 头里内嵌的 `gaussian_config`；没有则返回 ``None``。

    参照实现写出的 PLY 没有这条注释，调用方需要自己回退到配置默认值 ——
    先用这个函数**探测**，比"调 `load_ply(config=None)` 再捕异常"清楚得多。
    """
    plydata = PlyData.read(str(path))
    meta = _read_config_comment(plydata)
    if meta is None:
        return None
    return AvatarConfig(
        tex_size=int(meta["tex_size"]),
        num_basis_in=int(meta["num_basis_in"]),
        num_basis_blend=int(meta["num_basis_blend"]),
        mlp_hidden=tuple(meta.get("mlp_hidden", ())),
        use_weight_proj=bool(meta.get("use_weight_proj", True)),
    )


def read_ply_metadata(path: str | Path) -> dict:
    """只读 PLY 头，返回配置与各基的列数。用于加载前的交叉校验。"""
    plydata = PlyData.read(str(path))
    props = [p.name for p in plydata.elements[0].properties]
    n = len(plydata.elements[0])

    def count(prefix: str) -> int:
        return sum(1 for p in props if p.startswith(prefix))

    return {
        "num_gaussians": n,
        "num_basis_blend_from_file": count("xyz_b_") // 3,
        "has_binding": "face_id" in props,
        "config": _read_config_comment(plydata),
    }


def load_ply(
    path: str | Path,
    config: AvatarConfig | None = None,
    device: torch.device | str = "cuda",
    train: bool = False,
) -> GaussianAvatar:
    """从 PLY 载入 avatar。

    Args:
        config: 模型结构。为 ``None`` 时尝试从 PLY 内的 `comment gaussian_config` 还原；
            若文件没有该注释（例如参照实现写出的文件）则必须显式提供。
        device: 目标设备
        train: 是否把参数包成 `requires_grad` 的 `Parameter`（默认 False，推理用）

    Raises:
        ValueError: 配置与文件列数不一致（这是参照实现会静默出错的地方）。
    """
    path = Path(path)
    plydata = PlyData.read(str(path))
    el = plydata.elements[0]
    props = [p.name for p in el.properties]

    xyz = np.stack([np.asarray(el[c], dtype=np.float32) for c in ("x", "y", "z")], axis=1)
    n = xyz.shape[0]

    opacity = np.asarray(el["opacity"], dtype=np.float32)[:, None]
    scaling_names = sorted((p for p in props if p.startswith("scale_")),
                           key=lambda s: int(s.split("_")[-1]))
    scaling = np.stack([np.asarray(el[c], dtype=np.float32) for c in scaling_names], axis=1)
    rotation = np.stack([np.asarray(el[f"rot_{i}"], dtype=np.float32) for i in range(4)], axis=1)
    f_dc = np.stack([np.asarray(el[f"f_dc_{i}"], dtype=np.float32) for i in range(3)],
                    axis=1).reshape(n, 3, 1)

    k_from_file = sum(1 for p in props if p.startswith("xyz_b_")) // 3
    if k_from_file <= 0:
        raise ValueError(f"{path} 缺少 xyz_b_* 基参数列，无法载入 blendshape 基")

    # ---- 配置解析与一致性校验（修掉参照实现的 K3 缺陷）----
    meta = _read_config_comment(plydata)
    if config is None:
        if meta is None:
            raise ValueError(
                f"{path} 未内嵌 gaussian_config 注释（可能是参照实现写出的文件），"
                "必须显式传入 AvatarConfig")
        config = AvatarConfig(
            tex_size=meta["tex_size"], num_basis_in=meta["num_basis_in"],
            num_basis_blend=meta["num_basis_blend"],
            mlp_hidden=tuple(meta.get("mlp_hidden", ())),
            use_weight_proj=meta.get("use_weight_proj", True))
    if config.num_basis != k_from_file:
        raise ValueError(
            f"num_basis 不一致：配置给出 {config.num_basis}，文件有 {k_from_file} 列。"
            "（参照实现在此会静默出错；请修正 AvatarConfig 或改用正确的文件）")

    # tex_size 不参与存储（它只决定「一个 UV texel = 一个高斯」的容量上界），
    # 因此**无法**从文件内容反推。但若配置的 tex_size 连现有高斯数都装不下，
    # 说明配置写错了 —— 这类错误必须显式报出，否则会静默带进后续保存/重建。
    capacity = config.tex_size * config.tex_size
    if n > capacity:
        raise ValueError(
            f"tex_size 与文件不一致：tex_size={config.tex_size} 最多容纳 "
            f"{capacity} 个高斯，而文件有 {n} 个。"
            "请修正 configs/render.yaml 的 model.network.tex_size")

    xyz_b = np.stack([np.asarray(el[f"xyz_b_{i}"], dtype=np.float32)
                      for i in range(k_from_file * 3)], axis=1).reshape(n, k_from_file, 3)
    rot_b = np.stack([np.asarray(el[f"rot_b_{i}"], dtype=np.float32)
                      for i in range(k_from_file * 4)], axis=1).reshape(n, k_from_file, 4)
    f_dc_b = np.stack([np.asarray(el[f"f_dc_b_{i}"], dtype=np.float32)
                       for i in range(k_from_file * 3)], axis=1).reshape(n, k_from_file, 1, 3)

    flat_module = torch.from_numpy(np.asarray(el["weight_module"], dtype=np.float32))

    if "face_id" not in props:
        raise ValueError(
            f"{path} 缺少绑定信息（face_id / face_bary_*）。"
            "请确认保存时 avatar 已带有 Binding")
    face_id = torch.from_numpy(np.asarray(el["face_id"], dtype=np.int64))
    bary_names = sorted((p for p in props if p.startswith("face_bary_")),
                        key=lambda s: int(s.split("_")[-1]))
    if len(bary_names) != 3:
        raise ValueError(f"期望 3 个 face_bary_* 列，实际 {len(bary_names)}")
    face_bary = torch.from_numpy(
        np.stack([np.asarray(el[c], dtype=np.float32) for c in bary_names], axis=1))

    binding = Binding(face_id=face_id, face_bary=face_bary,
                      valid_mask=torch.ones(n, dtype=torch.bool))
    avatar = GaussianAvatar(config, binding)
    flat = np.asarray(el["f_rest_0"], dtype=np.float32) if "f_rest_0" in props else None
    if flat is not None and float(np.abs(flat).max()) > 0:
        # 占位列非零说明文件里其实有高阶 SH；本项目不支持，显式拒绝而非静默忽略
        raise ValueError(
            f"{path} 的 f_rest_* 非零，说明含高阶 SH；本项目 sh_degree=0，不支持")

    with torch.no_grad():
        avatar.xyz.copy_(torch.from_numpy(xyz))
        avatar.opacity.copy_(torch.from_numpy(opacity))
        avatar.scaling.copy_(torch.from_numpy(scaling))
        avatar.rotation.copy_(torch.from_numpy(rotation))
        avatar.color.copy_(torch.from_numpy(f_dc).transpose(1, 2).contiguous())
        avatar.xyz_b.copy_(torch.from_numpy(xyz_b).transpose(0, 1).contiguous())
        avatar.rotation_b.copy_(torch.from_numpy(rot_b).transpose(0, 1).contiguous())
        avatar.color_b.copy_(torch.from_numpy(f_dc_b).transpose(0, 1).contiguous())
    _load_flattened_module(flat_module, avatar.weight_module)

    avatar = avatar.to(device)
    if train:
        avatar.requires_grad_(True)
    return avatar
