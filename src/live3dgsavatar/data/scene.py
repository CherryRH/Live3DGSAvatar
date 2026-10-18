"""从参照仓库取出模板几何与相机参数，适配成本项目的 `Mesh` / `Camera`。

**用途**：数据集（INSTA 格式）里的网格顶点是 tracker 已经算好的世界空间顶点，
本项目需要的只是「拓扑 + UV + 顶点 + 相机」。这里把参照仓库的加载逻辑薄封装一次，
避免在多个脚本里重复。

**层次**：这是**数据层**（`data/`）。`core/` 不允许依赖参照仓库，
也不允许 import 本模块 —— 它只接受显式传入的 `Mesh` / `Camera` / 张量。

**为什么经由参照仓库**：INSTA 数据集的读取（3DMM 参数、相机、mesh 顶点）
目前复用参照实现的 `FLAMEDataset` 与 FLAME 模板，避免重复实现。
将来若自研数据层，只需替换本模块，上层（脚本 / GUI 后端）不用改。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from live3dgsavatar.config import (Config, load_config, repo_root,
                                   resolve_data_dir)

from . import flame_topology



@dataclass
class Scene:
    """渲染一帧所需的全部输入。

    **设备契约**：所有张量（`frames` 里的、拓扑、相机）都在 `load_scene(device=...)`
    指定的同一设备上。调用方直接把它们组成 `Mesh`/`Camera` 即可，
    **不需要也不应该**再逐个 `.to(device)` —— 漏掉任何一个都会触发
    「indices should be either on cpu or on the same device」这类错误。
    """

    frames: list                      # list[dict]，每个含 mesh / blend_weight
    faces: torch.Tensor               # [F,3] int32
    uvs: torch.Tensor                 # [Vuv,2]
    uv_faces: torch.Tensor            # [F,3] int32
    K: torch.Tensor                   # [3,3] float32
    R: torch.Tensor                   # [3,3]
    T: torch.Tensor                   # [3]
    width: int
    height: int
    #: 数据集**总共**多少帧（可能大于已载入的 `len(frames)`）
    dataset_frames: int = 0

    @property
    def num_frames(self) -> int:
        """已载入的帧数。"""
        return len(self.frames)

    @property
    def total_frames(self) -> int:
        """数据集总帧数；未载入的帧无法访问。"""
        return self.dataset_frames or len(self.frames)

    def frames_tensor(self, idx: list[int], device: str | None = None):
        """按索引取网格顶点，堆成 `[B, V, 3]`。

        `device=None` 表示**跟随 Scene 自身**（帧数据已在目标设备上）。
        显式传 `device` 仍支持，用于 `render_test` 这类需要强制指定的场景。
        """
        out = torch.stack([self.frames[i]["mesh"] for i in idx])
        return out if device is None else out.to(device)

    def weights_tensor(self, idx: list[int], device: str | None = None):
        """按索引取驱动参数，堆成 `[B, D]`（`device=None` 跟随 Scene）。"""
        out = torch.stack([self.frames[i]["blend_weight"] for i in idx])
        return out if device is None else out.to(device)


def load_scene(
    config: Config | None = None,
    frames: int | None = None,
    subject: str | None = None,
    device: str | None = None,
) -> Scene:
    """加载模板几何、相机与若干帧的网格顶点。

    所有路径与切分来自配置（`configs/system.yaml`），**本模块不含任何硬编码路径**。

    Args:
        config: 配置对象；``None`` 时用 `load_config()` 读仓库根的 configs/
        frames: 取前多少帧；``None`` 用 `render.frames`；``-1`` 表示全部
        subject: 覆盖 `subject`（人物名，同时是模型的一级目录名）
        device: 目标设备；``None`` 用 `runtime.device`（``cpu`` 便于无 GPU 自检）
    """
    cfg = config or load_config()
    if subject is not None:
        cfg.set("subject", subject)
    if device is None:
        device = str(cfg.get("runtime.device", "cuda"))

    data_dir = resolve_data_dir(cfg)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"数据集目录不存在：{data_dir}\n"
            f"（由 paths.data_dir 显式给出，或 "
            f"paths.data_root={cfg.get('paths.data_root')} + subject={cfg.subject} 推出）")

    manifest = data_dir / MANIFEST_NAME
    if manifest.exists():
        scene = _load_from_manifest(cfg, data_dir, manifest, frames, device)
    elif (data_dir / "checkpoint").is_dir():
        scene = _load_from_checkpoint(cfg, data_dir, frames, device)
    else:
        raise FileNotFoundError(
            f"数据集格式无法识别：{data_dir}\n"
            f"需要 {MANIFEST_NAME}（自包含清单），或 checkpoint/（每帧 3DMM 参数）。\n"
            "见 docs/CONFIG.md「数据集格式」。")
    _warn_if_images_missing(data_dir, cfg)
    return scene


# ------------------------------------------------------- checkpoint 格式 --


def _load_from_checkpoint(
    cfg: Config,
    data_dir: Path,
    frames: int | None,
    device: str,
) -> Scene:
    """参照格式：`<subject>/{images,checkpoint}/`，每帧一个 `.frame`。

    需要跑 FLAME 前向把 3DMM 参数变成网格顶点 —— 由本项目自己的
    `flame_model.FlameModel` 完成（不依赖任何外部仓库）。
    """
    from .checkpoint_loader import CheckpointDataset

    topo = _flame_topology(cfg, device)
    ds = CheckpointDataset(data_dir)
    total = len(ds)
    n = total if frames is None or frames < 0 else min(int(frames), total)

    flame = _flame_model(cfg)
    out = []
    for i in range(n):
        mesh, weight = ds.mesh_and_weight(i, flame)
        out.append({"mesh": mesh.to(device), "blend_weight": weight.to(device)})

    if not out:
        raise ValueError(f"{data_dir.name} 里没有可用帧（frames={frames}）")

    _check_manifest_contract(cfg, out, topo)
    return Scene(
        frames=out,
        faces=topo["faces"],
        uvs=topo["uvs"],
        uv_faces=topo["uv_faces"],
        K=ds.camera_K.to(device),
        R=ds.camera_R.to(device),
        T=ds.camera_T.to(device),
        width=int(ds.img_size[0]),
        height=int(ds.img_size[1]),
        dataset_frames=total,
    )


#: FLAME 前向模型（进程内缓存：加载 50 MB pickle 不便宜）
_FLAME_MODEL: dict | None = None


def _flame_model(cfg: Config):
    """取（并缓存）`FlameModel`。只在 checkpoint 格式下才需要。"""
    global _FLAME_MODEL
    if _FLAME_MODEL is None or _FLAME_MODEL.get("dir") != _flame_dir(cfg):
        from .flame_model import FlameModel  # noqa: PLC0415

        _FLAME_MODEL = {"dir": _flame_dir(cfg),
                        "model": FlameModel(_flame_dir(cfg))}
    return _FLAME_MODEL["model"]


# --------------------------------------------------------- manifest 格式 --

#: 自包含清单的文件名。存在它时走 manifest 分支，不再需要 FLAME 文件。
MANIFEST_NAME = "render_inputs.npz"

#: FLAME 资产目录（本项目自带，不再依赖参照仓库）
_FLAME_DIR_REL = "data/FLAME2020"


def _flame_dir(cfg: Config) -> Path:
    """FLAME 资产目录：`<repo>/data/FLAME2020`（可用 `paths.flame_dir` 覆盖）。"""
    explicit = cfg.get("paths.flame_dir")
    if explicit:
        p = Path(str(explicit))
        return p if p.is_absolute() else (repo_root() / p)
    return repo_root() / _FLAME_DIR_REL


def _cache_dir(cfg: Config) -> Path:
    """拓扑缓存目录：`<output_dir>/cache`（已 gitignore）。"""
    out = cfg.get("paths.output_dir") or (repo_root() / "output")
    return Path(str(out)) / "cache"


def _flame_topology(cfg: Config, device: str) -> dict:
    """取 FLAME 拓扑（`faces`/`uvs`/`uv_faces`），放到 `device`。

    缓存以 CPU 为准、每次调用再搬设备（否则"先 cuda 后 cpu"会拿到错设备）。
    """
    return flame_topology.load(_flame_dir(cfg), _cache_dir(cfg), device)


def _load_from_manifest(
    cfg: Config,
    data_dir: Path,
    manifest: Path,
    frames: int | None,
    device: str,
) -> Scene:
    """manifest 格式：`render_inputs.npz` 已算好逐帧 `mesh` / `blend_weight`。

    字段约定（见数据集 README）::

        mesh          [N, V, 3]   世界空间网格顶点
        blend_weight  [N, D]      驱动参数
        K [3,3], w2v [4,4], width, height
        names         [N]         帧名，与 images/ 对应

    好处：**不需要 3DMM 参数、不需要 FLAME 前向、不需要 chumpy**。
    拓扑（面表/UV）仍从 FLAME 资产取，但只取一次、与帧无关。
    """
    import numpy as np  # noqa: PLC0415

    topo = _flame_topology(cfg, device)

    with np.load(manifest, allow_pickle=False) as d:
        missing = [k for k in ("mesh", "blend_weight", "K", "w2v", "width", "height")
                   if k not in d.files]
        if missing:
            raise ValueError(
                f"{manifest.name} 缺少字段 {missing}；"
                f"实际含 {(list(d.files))}")

        meshes = d["mesh"]
        weights = d["blend_weight"]
        total = int(meshes.shape[0])
        if weights.shape[0] != total:
            raise ValueError(
                f"mesh 帧数 {total} 与 blend_weight 帧数 {weights.shape[0]} 不一致")

        n = total if frames is None or frames < 0 else min(int(frames), total)
        out = [
            {"mesh": torch.from_numpy(meshes[i].astype(np.float32, copy=True)).to(device),
             "blend_weight": torch.from_numpy(weights[i].astype(np.float32, copy=True)).to(device)}
            for i in range(n)
        ]

        K = torch.from_numpy(np.asarray(d["K"], dtype=np.float32).copy()).to(device)
        w2v = np.asarray(d["w2v"], dtype=np.float32)
        R = torch.from_numpy(w2v[:3, :3].copy()).to(device)
        T = torch.from_numpy(w2v[:3, 3].copy()).to(device)
        width = int(d["width"])
        height = int(d["height"])

    if not out:
        raise ValueError(f"{manifest.name} 里没有可用的帧（frames={frames}）")

    f0 = out[0]

    _check_manifest_contract(cfg, out, topo)

    return Scene(
        frames=out,
        faces=topo["faces"],
        uvs=topo["uvs"],
        uv_faces=topo["uv_faces"],
        K=K,
        R=R,
        T=T,
        width=width,
        height=height,
        dataset_frames=total,
    )


def _check_manifest_contract(cfg: Config, frames: list[dict], topo: dict) -> None:
    """加载后的契约自检（两种格式共用）。

    这些错误如果留到下游才暴露，报错会离根因很远：
    维度错在 MLP 的 `Linear` 里才炸；顶点数错会让 `face_id` 指向错误的面，
    表现为"模型错乱"，极难与其它问题区分。
    """
    f0 = frames[0]
    if f0["mesh"].dtype != torch.float32 or f0["blend_weight"].dtype != torch.float32:
        raise ValueError("mesh / blend_weight 应为 float32")

    d_expect = int(cfg.get("model.network.num_basis_in", 129))
    if int(f0["blend_weight"].numel()) != d_expect:
        raise ValueError(
            f"blend_weight 维度 {f0['blend_weight'].numel()} 与配置 "
            f"num_basis_in={d_expect} 不一致")

    v_actual = int(f0["mesh"].shape[0])
    v_index_max = int(topo["faces"].max())
    if v_index_max >= v_actual:
        raise ValueError(
            f"网格顶点数 {v_actual} 不足：FLAME 面表引用了顶点 {v_index_max}；"
            "通常是用了不同版本的 FLAME 资产，或数据集与模型不是同一套网格")


def _warn_if_images_missing(data_dir: Path, cfg: Config) -> None:
    """数据集原图用于 PSNR 的 GT；缺失时只在需要 GT 的场景下才会失败。

    这里只做**轻量提示**，不抛错 —— GUI 只渲染、不需要 GT。
    """
    sub = str(cfg.get("paths.image_subdir", "images"))
    img_dir = data_dir / sub
    if not img_dir.is_dir():
        import logging  # noqa: PLC0415

        logging.getLogger(__name__).info(
            "数据集原图目录不存在（%s）——只影响 PSNR 对比，不影响渲染", img_dir)
