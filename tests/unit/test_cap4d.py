"""在 CAP4D 生成图上跑我们链路的数据侧（`data/cap4d.py`）。

选型结论（写在这里免得以后又走回头路）：**不用 CAP4D 的 flame 参数自己算网格**，
而是让 RDF 拟合生成图。试过前者，PSNR 只有 2.4 dB —— 头都不在同一区域，
因为要同时对四件事（裁剪→内参、它的 identity、姿态约定、我们的世界空间）。
RDF 的虚拟相机与本数据集相机一致，所以这条路不需要任何换算。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import live3dgsavatar  # noqa: E402, F401

from live3dgsavatar.data import cap4d as cap  # noqa: E402

FLAME_DIR = REPO_ROOT / "data" / "FLAME2020"


def _dump(tmp_path: Path, n: int = 2, verts: int = 5023) -> Path:
    p = tmp_path / "cap4d_rdf.npz"
    np.savez(p,
             x=np.zeros((n, 118), np.float32),
             verts=np.zeros((n, verts, 3), np.float32),
             names=np.array([f"{i:05d}.png" for i in range(n)]),
             image_width=np.int32(512), image_height=np.int32(512))
    return p


# ------------------------------------------------------------------ 读 --


def test_load_rdf_dump_reads_fields(tmp_path) -> None:
    out = cap.load_rdf_dump(_dump(tmp_path))
    assert out["x"].shape == (2, 118)
    assert out["verts"].shape == (2, 5023, 3)
    assert out["names"] == ["00000.png", "00001.png"]
    assert out["width"] == 512 and out["height"] == 512


def test_load_rdf_dump_rejects_wrong_vertex_count(tmp_path) -> None:
    """很可能是把 CAP4D 自己的 5023 之外的东西混进来了，要立刻报。"""
    with pytest.raises(ValueError, match="5023"):
        cap.load_rdf_dump(_dump(tmp_path, verts=5083))


def test_load_rdf_dump_rejects_wrong_x_dim(tmp_path) -> None:
    p = tmp_path / "bad.npz"
    np.savez(p, x=np.zeros((1, 129), np.float32),
             verts=np.zeros((1, 5023, 3), np.float32),
             names=np.array(["a.png"]))
    with pytest.raises(ValueError, match="118"):
        cap.load_rdf_dump(p)


def test_load_rdf_dump_reports_missing_fields(tmp_path) -> None:
    p = tmp_path / "bad.npz"
    np.savez(p, x=np.zeros((1, 118), np.float32))
    with pytest.raises(ValueError, match="缺少字段"):
        cap.load_rdf_dump(p)


def test_images_dir_accepts_both_levels(tmp_path) -> None:
    gen = tmp_path / "generated_images" / "images"
    gen.mkdir(parents=True)
    assert cap.images_dir(tmp_path) == gen
    assert cap.images_dir(tmp_path / "generated_images") == gen


def test_images_dir_reports_missing(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="generated_images"):
        cap.images_dir(tmp_path)


# ------------------------------------------------------------ 组装 frames --


def test_to_frames_adds_teeth_and_maps_weights(tmp_path) -> None:
    """5023 → 5083（补 60 颗牙），118 → 129 维驱动参数。"""
    if not (FLAME_DIR / "generic_model.pkl").is_file():
        pytest.skip("没有 FLAME 资产")

    dump = cap.load_rdf_dump(_dump(tmp_path, n=1))
    frames = cap.to_frames(dump, FLAME_DIR)

    assert len(frames) == 1
    assert frames[0]["mesh"].shape == (5083, 3)
    assert frames[0]["blend_weight"].shape == (129,)
    assert torch.isfinite(frames[0]["mesh"]).all()


def test_to_frames_keeps_frame_order(tmp_path) -> None:
    if not (FLAME_DIR / "generic_model.pkl").is_file():
        pytest.skip("没有 FLAME 资产")
    dump = cap.load_rdf_dump(_dump(tmp_path, n=3))
    frames = cap.to_frames(dump, FLAME_DIR)
    assert [f["name"] for f in frames] == ["00000.png", "00001.png", "00002.png"]
