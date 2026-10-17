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

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from live3dgsavatar.config import Config, load_config, resolve_data_dir

# 仓库根：src/live3dgsavatar/data/scene.py → 上溯 4 层
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

# ⚠️ 必须在**导入 FLAME 之前**打补丁（`FLAME.__init__` 里 pickle.load → import chumpy）。
#    `live3dgsavatar/__init__.py` 导入即施加 numpy 兼容补丁。
import live3dgsavatar  # noqa: E402, F401  （勿删：导入即生效）



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
    ref_root = cfg.paths.reference_root
    if ref_root is None:
        raise ValueError(
            "缺少 paths.reference_root（参照仓库根目录）。"
            "请在 configs/system.yaml 设置或设 LIVE3DGS_REFERENCE_ROOT。")
    ref_root = Path(ref_root)
    split = str(cfg.runtime.split)
    if frames is None:
        frames = int(cfg.get("render.frames", 1))
    if not data_dir.exists():
        raise FileNotFoundError(
            f"数据集目录不存在：{data_dir}\n"
            f"（由 paths.data_dir 显式给出，或 "
            f"paths.data_root={cfg.paths.data_root} + subject={cfg.subject} 推出；"
            "请检查 configs/system.yaml 或设 LIVE3DGS_DATA_ROOT）")

    # 两种数据格式：优先 manifest（自包含），否则走参照的 checkpoint 格式
    manifest = data_dir / MANIFEST_NAME
    if manifest.exists():
        scene = _load_from_manifest(cfg, data_dir, manifest, frames, ref_root, device)
    else:
        scene = _load_from_checkpoint(cfg, data_dir, frames, ref_root, split, device)
    _warn_if_images_missing(data_dir, cfg)
    return scene


def _load_from_checkpoint(
    cfg: Config,
    data_dir: Path,
    frames: int | None,
    ref_root: Path,
    split: str,
    device: str,
) -> Scene:
    """旧格式：参照实现的 `<subject>/{images,checkpoint}/`。

    需要把 3DMM 参数经 FLAME 前向算成网格顶点，因此**依赖参照仓库与 chumpy**。
    """
    # 参照侧的加载管线（只读参照仓库）；同属 data/ 层，包内相对导入
    from .reference import reference_workspace  # noqa: PLC0415

    with reference_workspace(ref_root):
        from dataset import FLAMEDataset  # noqa: PLC0415
        from submodules.flame import FLAME, FlameConfig  # noqa: PLC0415

        # ⚠️ 必须用 FlameConfig 的**默认值**（dtype=float64）。
        #    FLAMEDataset 把姿态/形状参数存为 float64，而 FLAME 的 v_template / shapedirs
        #    跟随 cfg.dtype；把 dtype 改成 float32 会让
        #    `template_vertices(float32) + blend_shapes(betas(float64), ...)`
        #    抛 `expected scalar type Double but found Float`。
        #    网格顶点在 FLAMEDataset 内部已降到 float32（`mesh_verts`），无需在此转换。
        flame = FLAME(FlameConfig()).to(device)
        ds = FLAMEDataset(
            flame, str(data_dir), split=split,
            pin_memory=bool(cfg.get("dataset.pin_memory", False)),
            use_shape_weight=bool(cfg.get("dataset.use_shape_weight", True)),
            use_pose_weight=bool(cfg.get("dataset.use_pose_weight", True)))

        # 数据集总帧数 vs 已载入帧数：GUI 只需要载入一部分，
        # 但控件上的"数据集帧"滑块要反映**总数**。
        total = ds.__len__()
        n = total if frames < 0 else min(frames, total)
        out = [
            {"mesh": ds.mesh_verts[i].to(device),
             "blend_weight": ds.blend_weight[i].to(device)}
            for i in range(n)
        ]

        # 契约自检：下游 `core/` 只接受 float32（CUDA 内核限制）
        for f in out[:1]:
            assert f["mesh"].dtype == torch.float32, (
                f"mesh 顶点应为 float32（core/ 与 CUDA 内核要求），实际 {f['mesh'].dtype}")
            assert f["blend_weight"].dtype == torch.float32, (
                f"blend_weight 应为 float32，实际 {f['blend_weight'].dtype}")
        return Scene(
            frames=out,
            faces=flame.faces.detach().clone().to(torch.int32),
            uvs=flame.uvs.to(torch.float32).detach().clone(),
            uv_faces=flame.uv_faces.to(torch.int32).detach().clone(),
            K=torch.from_numpy(ds.camera_intri).to(torch.float32),
            R=torch.from_numpy(ds.camera_extri[:3, :3]).to(torch.float32),
            T=torch.from_numpy(ds.camera_extri[:3, 3]).to(torch.float32),
            width=int(ds.image_width),
            height=int(ds.image_height),
            dataset_frames=total,
        )


# --------------------------------------------------------- manifest 格式 --

#: 自包含清单的文件名。存在它时走 manifest 分支，不再需要 FLAME 文件。
MANIFEST_NAME = "render_inputs.npz"

#: FLAME 拓扑缓冲（`faces`/`uvs`/`uv_faces`）。
#: 网格顶点由清单直接给出，但**渲染仍需拓扑**（`face_id` 是面索引）。
#: 该拓扑只与 FLAME 资产有关、与数据集无关，故进程内缓存一份共用。
_FLAME_TOPOLOGY: dict | None = None


def _flame_topology(ref_root: Path, device: str) -> dict:
    """取 FLAME（含牙齿扩展）的 `faces` / `uvs` / `uv_faces`，并放到 `device`。

    为什么要 FLAME 资产：`model.ply` 里的 `face_id` 是**面索引**，
    而面表只存在于 FLAME 模板中。但**不需要任何逐帧 3DMM 参数**，
    也不需要数据集里的 `checkpoint/`——所以比旧格式轻得多。

    ⚠️ 缓存**以 CPU 为准**（设备无关），每次调用再 `.to(device)`。
    否则先加载 cuda 再加载 cpu 时会拿到错误设备上的张量。
    """
    global _FLAME_TOPOLOGY
    if _FLAME_TOPOLOGY is None:
        from .reference import reference_workspace  # noqa: PLC0415

        with reference_workspace(ref_root):
            from submodules.flame import FLAME, FlameConfig  # noqa: PLC0415

            # ⚠️ 用默认 FlameConfig：`add_mouth=True` 会追加 60 个牙齿顶点，
            #    面表随之变为 10032 行，正好覆盖 `model.ply` 的 `face_id` 上界 10031。
            flame = FLAME(FlameConfig())
            _FLAME_TOPOLOGY = {
                "faces": flame.faces.detach().clone().to("cpu", torch.int32),
                "uvs": flame.uvs.detach().clone().to("cpu", torch.float32),
                "uv_faces": flame.uv_faces.detach().clone().to("cpu", torch.int32),
            }
    return {k: v.to(device) for k, v in _FLAME_TOPOLOGY.items()}


def _load_from_manifest(
    cfg: Config,
    data_dir: Path,
    manifest: Path,
    frames: int | None,
    ref_root: Path,
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

    topo = _flame_topology(ref_root, device)

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

    # 契约自检：core/ 只接受 float32
    if f0["mesh"].dtype != torch.float32 or f0["blend_weight"].dtype != torch.float32:
        raise ValueError("mesh / blend_weight 应为 float32")

    # 维度契约最先查：错维会在 MLP 的 Linear 里才爆，离根因太远
    d_expect = int(cfg.get("model.network.num_basis_in", 129))
    if int(f0["blend_weight"].numel()) != d_expect:
        raise ValueError(
            f"blend_weight 维度 {f0['blend_weight'].numel()} 与配置 "
            f"num_basis_in={d_expect} 不一致")

    # 顶点数必须与拓扑一致：`face_id` 是**面索引**，顶点对不上会让绑定错位。
    # 面表里的最大**顶点**索引必须落在 mesh 的顶点范围内。
    v_actual = int(f0["mesh"].shape[0])
    v_index_max = int(topo["faces"].max())
    if v_index_max >= v_actual:
        raise ValueError(
            f"网格顶点数 {v_actual} 不足：FLAME 面表引用了顶点 {v_index_max}；"
            "通常是用了不同版本的 FLAME 资产，或数据集与模型不是同一套网格")

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
