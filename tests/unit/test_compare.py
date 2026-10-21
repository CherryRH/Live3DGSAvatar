"""对比视图：目标帧（`images/`）与 PSNR（**不需要 GPU**）。

守三件事：

1. **分辨率对齐**：渲染可能被 `scale` 缩小，目标图必须缩到同一尺寸 ——
   否则并排显示错位、PSNR 也算错
2. **掩码口径**：PSNR 按 alpha>127 取前景，与离线评估一致
3. **缺失时不崩**：没有 `images/` 时返回 `None`，界面只是不显示对比
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from live3dgsavatar.app.session import Session, _psnr_pair
from live3dgsavatar.config import load_config


def _cfg(**over):
    cfg = load_config(files=("system.yaml", "render.yaml", "train.yaml"), env=False)
    cfg.set("runtime.device", "cpu")
    for k, v in over.items():
        cfg.set(k, v)
    return cfg


def _session_with(tmp_path: Path, frames: int = 3) -> Session:
    """造一个带 `images/` 的最小数据集，并接到 session 上。"""
    from PIL import Image

    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True)
    for i in range(frames):
        rgba = np.zeros((8, 8, 4), np.uint8)
        rgba[..., :3] = (40 * (i + 1)) % 256     # 保持 uint8 范围
        rgba[2:6, 2:6, 3] = 255          # 中间 4×4 是前景
        Image.fromarray(rgba).save(img_dir / f"{i:05d}.png")

    session = Session(_cfg())
    session._active_dataset_dir = tmp_path
    session.scene = type("S", (), {"num_frames": frames, "total_frames": frames,
                                   "width": 8, "height": 8})()
    from live3dgsavatar.app.protocol import FrameTimeline

    session.timeline = FrameTimeline(frames)
    return session


# ------------------------------------------------------------- PSNR --


def test_psnr_pair_identical_is_capped() -> None:
    """完全相同不应返回 inf（JSON 传不了），给 99 作为上限。"""
    a = np.full((4, 4, 3), 100, np.uint8)
    mask = np.ones((4, 4), bool)
    full, masked = _psnr_pair(a, a, mask)
    assert full == 99.0 and masked == 99.0


def test_psnr_pair_known_value() -> None:
    """整体差 10 级（0–255）的 PSNR ≈ 20·log10(255/10) ≈ 28.13 dB。"""
    a = np.zeros((4, 4, 3), np.uint8)
    b = np.full((4, 4, 3), 10, np.uint8)
    full, masked = _psnr_pair(a, b, np.ones((4, 4), bool))
    # 0–255 下差 10 级 → 20·log10(255/10) ≈ 28.13 dB
    assert abs(full - 28.13) < 0.05, full
    assert abs(masked - 28.13) < 0.05, masked


def test_psnr_pair_masks_background() -> None:
    """背景差得再多，掩码内 PSNR 也不受影响。"""
    a = np.zeros((4, 4, 3), np.uint8)
    b = np.zeros((4, 4, 3), np.uint8)
    b[0, :] = 255                        # 只在第 0 行（掩码外）差异
    mask = np.zeros((4, 4), bool)
    mask[2:, :] = True

    full, masked = _psnr_pair(a, b, mask)
    assert full < 15.0, f"全图应受影响：{full}"
    assert masked == 99.0, f"掩码内应完全相同：{masked}"


def test_psnr_pair_handles_size_mismatch() -> None:
    full, masked = _psnr_pair(np.zeros((2, 2, 3), np.uint8),
                              np.zeros((4, 4, 3), np.uint8), None)
    assert full is None and masked is None


# --------------------------------------------------------- 目标帧 --


def test_compare_target_reads_current_frame(tmp_path) -> None:
    session = _session_with(tmp_path)
    rgb = torch.zeros(8, 8, 3, dtype=torch.uint8)

    target, psnr = session._compare_target(rgb)
    assert target is not None
    assert tuple(target.shape) == (8, 8, 3)
    # 第 0 帧的 RGB 是 40
    assert int(target[0, 0, 0]) == 40
    assert psnr is not None and psnr > 0

    session.timeline.set_index(2)
    target2, _ = session._compare_target(rgb)
    assert int(target2[0, 0, 0]) == 120, "应跟随帧号取对应帧"


def test_compare_target_resizes_to_render_size(tmp_path) -> None:
    """渲染被缩小（scale<1）时，目标图必须缩到同一尺寸。"""
    session = _session_with(tmp_path)
    small = torch.zeros(4, 4, 3, dtype=torch.uint8)
    target, _ = session._compare_target(small)
    assert tuple(target.shape) == (4, 4, 3)


def test_compare_target_missing_file_returns_none(tmp_path) -> None:
    """数据集没有该帧的 images/ 时不该崩，只是没有对比。"""
    session = _session_with(tmp_path)
    session.timeline.set_index(2)
    (tmp_path / "images" / "00002.png").unlink()
    target, psnr = session._compare_target(torch.zeros(8, 8, 3, dtype=torch.uint8))
    assert target is None and psnr is None


def test_compare_target_cached_by_frame(tmp_path) -> None:
    """同一帧重复渲染时不该反复读盘（渲染可能 45 fps 刷同一帧）。"""
    session = _session_with(tmp_path)
    rgb = torch.zeros(8, 8, 3, dtype=torch.uint8)
    session._compare_target(rgb)
    cached = session._target_cache
    assert cached is not None

    session._compare_target(rgb)
    # 命中缓存 ⇒ 复用**同一块像素数组**（没有重新解码）
    assert session._target_cache[1] is cached[1], "同一帧应命中缓存"

    session.timeline.set_index(1)
    session._compare_target(rgb)
    assert session._target_cache[1] is not cached[1], "换帧后应重新读"


def test_target_pool_is_bounded_and_reuses_arrays(tmp_path) -> None:
    """目标帧池要复用同一块像素数组，并且有上限（不能无限吃内存）。

    为什么需要池：顺序播放时每帧都要新图，而解一张 512² PNG 要 7–10 ms，
    比渲染本身还慢。池 + 后台预取把这段成本移出了渲染线程。
    """
    session = _session_with(tmp_path, frames=20)
    rgb = torch.zeros(8, 8, 3, dtype=torch.uint8)

    first = session._compare_target(rgb)[0]
    second = session._compare_target(rgb)[0]
    # `torch.from_numpy` 每次都新建张量对象，所以比的是**内存**是否共享
    assert np.shares_memory(first.numpy(), second.numpy()), \
        "同一帧应复用同一块数组（没有重新解码）"

    for i in range(20):
        session.timeline.set_index(i)
        session._compare_target(rgb)
    # 上限 8：池子里只留最近几帧，播放时不至于把整个数据集读进内存
    assert len(session._target_store) <= 8, len(session._target_store)


def test_prefetch_is_deduplicated(tmp_path) -> None:
    """同一帧被反复排队时只应有一个在飞任务（避免线程爆炸）。"""
    session = _session_with(tmp_path, frames=10)
    for _ in range(5):
        session._prefetch_target(3, 8, 8)
    # 在飞集合最多一个 key（重复提交会被拦下）
    assert len(session._target_inflight) <= 1, session._target_inflight
