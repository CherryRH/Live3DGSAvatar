"""数据集发现：列出 `paths.data_root` 下可用来**驱动**的数据集。

## 为什么单独一个模块

界面上要能"用任意已有数据集驱动当前模型"，就得先知道有哪些数据集、各有多少帧、
能不能提供原始画面。这些信息只依赖目录结构和一个 npz 的 shape，
**不需要 torch、不需要 GPU** —— 放在 `data/` 里单独实现，
后端接口（`/api/datasets`）就能在毫秒级返回，也不会因为缺 CUDA 而失败。

## 两种格式

| 格式 | 判据 | 能否驱动 | 能否喂给 RDF |
|---|---|---|---|
| `manifest` | 有 `render_inputs.npz` | 能（帧内已含网格与参数） | 需 `input/` |
| `checkpoint` | 有 `checkpoint/*.frame` | 能（需 FLAME 前向） | 需 `input/` |

`duda` 属于后一种，且没有 `input/`，所以只能在"数据集模式"下驱动。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import Config, repo_root

#: 认为是"可用数据集"的标志文件/目录
_MANIFEST = "render_inputs.npz"
_CHECKPOINT_DIR = "checkpoint"


@dataclass
class DatasetInfo:
    """一个可用于驱动的数据集。"""

    name: str
    path: Path
    frames: int
    format: str          # manifest | checkpoint | unknown
    has_images: bool     # images/：抠图目标（训练/评价用）
    has_input: bool      # input/：原始画面（RDF 模式取帧用）

    @property
    def can_drive(self) -> bool:
        return self.frames > 0 and self.format in ("manifest", "checkpoint")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "frames": int(self.frames),
            "format": self.format,
            "has_images": bool(self.has_images),
            "has_input": bool(self.has_input),
            "can_drive": bool(self.can_drive),
        }


def data_root(cfg: Config) -> Path:
    """数据集根目录（`paths.data_root`，相对路径按仓库根解析）。"""
    raw = cfg.get("paths.data_root")
    if raw is None:
        raise ValueError("缺少 paths.data_root")
    p = Path(str(raw)).expanduser()
    return p if p.is_absolute() else (repo_root() / p)


def _manifest_frames(path: Path) -> int:
    """从 npz 的 `mesh` 形状读帧数（mmap，不把数据读进内存）。"""
    import numpy as np

    try:
        with np.load(path, mmap_mode="r") as z:
            for key in ("mesh", "blend_weight"):
                if key in z.files:
                    return int(z[key].shape[0])
    except (OSError, ValueError):
        return 0
    return 0


def _checkpoint_frames(path: Path) -> int:
    try:
        return sum(1 for _ in path.glob("*.frame"))
    except OSError:
        return 0


def inspect(cfg: Config, name: str) -> DatasetInfo | None:
    """检查单个数据集目录；不存在或不可用时返回 ``None``。"""
    path = data_root(cfg) / name
    if not path.is_dir():
        return None

    manifest = path / _MANIFEST
    ckpt_dir = path / _CHECKPOINT_DIR
    if manifest.exists():
        fmt, frames = "manifest", _manifest_frames(manifest)
    elif ckpt_dir.is_dir():
        fmt, frames = "checkpoint", _checkpoint_frames(ckpt_dir)
    else:
        fmt, frames = "unknown", 0

    return DatasetInfo(
        name=name,
        path=path,
        frames=frames,
        format=fmt,
        has_images=(path / "images").is_dir(),
        has_input=(path / "input").is_dir(),
    )


def list_datasets(cfg: Config) -> list[DatasetInfo]:
    """列出所有可用数据集（按名字排序）。

    不可用的目录（缺 npz 与 checkpoint）会被跳过 —— 界面上不该出现
    "选了却驱动不了"的选项。
    """
    root = data_root(cfg)
    if not root.is_dir():
        return []
    out: list[DatasetInfo] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if not child.is_dir() or child.name.startswith("."):
            continue
        info = inspect(cfg, child.name)
        if info is not None and info.can_drive:
            out.append(info)
    return out


def resolve_dataset_dir(cfg: Config, name: str) -> Path:
    """按名字取数据集目录，并确认它确实可用。

    Raises:
        FileNotFoundError: 目录不存在
        ValueError: 目录存在但没有可驱动的数据
    """
    info = inspect(cfg, name)
    if info is None:
        root = data_root(cfg)
        raise FileNotFoundError(f"数据集不存在：{root / name}")
    if not info.can_drive:
        raise ValueError(
            f"数据集不可驱动：{info.path}\n"
            f"需要 {_MANIFEST} 或 {_CHECKPOINT_DIR}/*.frame")
    return info.path
