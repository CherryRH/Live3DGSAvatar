"""FLAME 网格拓扑（面表 + UV），**只读资产、不依赖任何外部仓库**。

## 为什么只需要拓扑

渲染时每个高斯的位置由下式决定::

    xyz_world = R(face_id) @ xyz_tangent + Σᵢ baryᵢ · vᵢ
    #              ↑ 面索引                        ↑ 网格顶点（观测已给）

- `face_id`是**面索引** → 需要面表
- 网格顶点由观测（`render_inputs.npz`）或 FLAME 前向给出 → 本模块不管
- UV 只在**构建绑定**（训练侧）需要

所以这里只提供 `faces` / `uvs` / `uv_faces`，**不加载任何 3DMM 参数**。

## 牙齿扩展

RGBAvatar 在 FLAME 之后追加了 60 个牙齿顶点与 56 个面，使面表从 9976 变为
**10032** 行 —— 这正好是 `model.ply` 里 `face_id` 上界 10031 的来源。
**必须复现**，否则绑定会错位。牙齿几何由嘴唇顶点确定性推导，
不依赖任何学习参数（见 `docs/CONVENTIONS.md`）。

## 缓存

拓扑只取决于 FLAME 资产、与数据集无关，首次调用落盘到
`output/cache/flame_topology.npz`，后续直接读取（省去加载 50 MB pickle）。
文件里带**资产指纹**，FLAME 资产一变（2020↔2023、重新下载）自动失效重建。
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import torch

# ⚠️ 必须在 pickle.load 之前施加补丁：FLAME 资产与 `.frame` 里含
#    chumpy 对象，而 chumpy 在新版 numpy / Python 下需要别名补齐。
from .. import compat  # noqa: F401, E402  （导入即生效）

_log = logging.getLogger(__name__)

#: 追加的牙齿顶点数（上/下各 15 个根 + 15 个边缘）
NUM_TEETH_VERTS = 60

#: 牙齿三角面（相对牙齿顶点编号；形态由 RGBAvatar 的 `add_teeth` 固定）
_F_TEETH_UPPER = torch.tensor([
    [0, 31, 30], [0, 1, 31], [1, 32, 31], [1, 2, 32], [2, 33, 32],
    [2, 3, 33], [3, 34, 33], [3, 4, 34], [4, 35, 34], [4, 5, 35],
    [5, 36, 35], [5, 6, 36], [6, 37, 36], [6, 7, 37], [7, 8, 37],
    [8, 38, 37], [8, 9, 38], [9, 39, 38], [9, 10, 39], [10, 40, 39],
    [10, 11, 40], [11, 41, 40], [11, 12, 41], [12, 42, 41], [12, 13, 42],
    [13, 43, 42], [13, 14, 43], [14, 44, 43],
], dtype=torch.int64)

_F_TEETH_LOWER = torch.tensor([
    [45, 46, 15], [46, 16, 15], [46, 47, 16], [47, 17, 16], [47, 48, 17],
    [48, 18, 17], [48, 49, 18], [49, 19, 18], [49, 50, 19], [50, 20, 19],
    [50, 51, 20], [51, 21, 20], [51, 52, 21], [52, 22, 21], [52, 23, 22],
    [52, 53, 23], [53, 24, 23], [53, 54, 24], [54, 25, 24], [54, 55, 25],
    [55, 26, 25], [55, 56, 26], [56, 27, 26], [56, 57, 27], [57, 28, 27],
    [57, 58, 28], [58, 29, 28], [58, 59, 29],
], dtype=torch.int64)


def asset_paths(flame_dir: Path) -> tuple[Path, Path]:
    """FLAME 资产路径：`(generic_model.pkl, flame_uv.npz)`。"""
    d = Path(flame_dir)
    return d / "generic_model.pkl", d / "flame_uv.npz"


def fingerprint(flame_dir: Path) -> str:
    """资产指纹（大小 + mtime），用于判断缓存是否仍然有效。

    拓扑只取决于资产；资产一换而缓存未失效，`face_id` 就会指向错误的面，
    表现为"模型错乱"且极难与其它问题区分。
    """
    parts = []
    for p in asset_paths(flame_dir):
        try:
            st = p.stat()
            parts.append(f"{p.name}:{st.st_size}:{int(st.st_mtime)}")
        except OSError:
            parts.append(f"{p.name}:missing")
    parts.append(f"teeth:{NUM_TEETH_VERTS}")
    return "|".join(parts)


def build(flame_dir: Path) -> dict[str, torch.Tensor]:
    """从 FLAME 资产构建拓扑（含牙齿扩展）。**只读 pickle，无需 FLAME 代码。**"""
    geom_path, uv_path = asset_paths(flame_dir)
    if not geom_path.exists():
        raise FileNotFoundError(
            f"缺少 FLAME 模型：{geom_path}\n"
            "请从 https://flame.is.tue.mpg.de/ 下载 FLAME 2020 并解压到 data/FLAME2020/")
    if not uv_path.exists():
        raise FileNotFoundError(
            f"缺少 FLAME UV 资产：{uv_path}（随本项目提供，应已在仓库中）")

    with open(geom_path, "rb") as fh:
        data = pickle.load(fh, encoding="latin1")
    faces = torch.from_numpy(np.asarray(data["f"], dtype=np.int64)).to(torch.int32)
    n_verts = int(np.asarray(data["v_template"]).shape[0])

    uv = np.load(uv_path)
    uvs = torch.from_numpy(np.asarray(uv["vt"], dtype=np.float32))
    uv_faces = torch.from_numpy(np.asarray(uv["ft"], dtype=np.int64)).to(torch.int32)
    n_uv = int(uvs.shape[0])

    return {
        "faces": torch.cat([faces, _F_TEETH_UPPER.to(torch.int32) + n_verts,
                            _F_TEETH_LOWER.to(torch.int32) + n_verts], dim=0),
        "uvs": torch.cat([uvs, _teeth_uvs(n_uv)], dim=0),
        "uv_faces": torch.cat([uv_faces,
                               _F_TEETH_UPPER.to(torch.int32) + n_uv,
                               _F_TEETH_LOWER.to(torch.int32) + n_uv], dim=0),
    }


def _teeth_uvs(offset: int) -> torch.Tensor:
    """牙齿顶点的 UV（与 RGBAvatar 的 `add_teeth` 同一布局）。

    15 × 4 的网格，贴在 UV 图右上角一块未被占用的区域
    （`u ∈ [0.25,0.75]`、`v ∈ [0.955,0.995]`）。
    """
    # ⚠️ 用 float64 生成再降精度：与参照实现的逐步舍入一致，
    #    否则 float32 逐步会带来 ~6e-08 的差异（虽然极小，但没有理由不一致）。
    u = torch.linspace(0.75, 0.25, 15, dtype=torch.float64)
    v = torch.linspace(0.995, 0.955, 3, dtype=torch.float64)
    v = v[[0, 2, 1, 1]]
    grid = torch.meshgrid(u, v, indexing="ij")
    uv = torch.stack(grid, dim=-1).permute(1, 0, 2).reshape(NUM_TEETH_VERTS, 2)
    return uv.to(torch.float32)


def cache_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / "flame_topology.npz"


def read_cache(cache_dir: Path, flame_dir: Path) -> dict[str, torch.Tensor] | None:
    """读缓存；指纹不匹配返回 ``None``。"""
    path = cache_path(cache_dir)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as d:
            if str(d["fingerprint"]) != fingerprint(flame_dir):
                _log.info("拓扑缓存指纹不匹配（FLAME 资产已变），重新生成")
                return None
            return {
                "faces": torch.from_numpy(np.asarray(d["faces"], dtype=np.int32)),
                "uvs": torch.from_numpy(np.asarray(d["uvs"], dtype=np.float32)),
                "uv_faces": torch.from_numpy(np.asarray(d["uv_faces"], dtype=np.int32)),
            }
    except (OSError, KeyError, ValueError) as e:
        _log.info("拓扑缓存不可用（%s），回退到资产", e)
        return None


def write_cache(cache_dir: Path, flame_dir: Path, topo: dict[str, torch.Tensor]) -> None:
    path = cache_path(cache_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, fingerprint=np.array(fingerprint(flame_dir)),
                 **{k: v.numpy() for k, v in topo.items()})
    except OSError as e:
        _log.info("写拓扑缓存失败（%s），忽略", e)


def clear_cache(cache_dir: Path) -> bool:
    path = cache_path(cache_dir)
    if path.exists():
        path.unlink()
        return True
    return False


_IN_MEMORY: dict[str, torch.Tensor] | None = None


def load(flame_dir: Path, cache_dir: Path, device: str) -> dict[str, torch.Tensor]:
    """取拓扑（优先内存 → 磁盘缓存 → 资产），统一放到 `device`。

    设备无关：缓存始终以 CPU 张量保存，每次调用再 `.to(device)`。
    """
    global _IN_MEMORY
    if _IN_MEMORY is None:
        _IN_MEMORY = read_cache(cache_dir, flame_dir)
        if _IN_MEMORY is None:
            _IN_MEMORY = build(flame_dir)
            write_cache(cache_dir, flame_dir, _IN_MEMORY)
    return {k: v.to(device) for k, v in _IN_MEMORY.items()}


def reset_memory_cache() -> None:
    """清空进程内缓存（测试用）。"""
    global _IN_MEMORY
    _IN_MEMORY = None
