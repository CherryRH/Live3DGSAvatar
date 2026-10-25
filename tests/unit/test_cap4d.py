"""CAP4D 结果 → 我们能渲染的一帧（`data/cap4d.py`）。

真正在 GPU 上渲染要等用户跑，但**转换的数学**必须在这里锁住：
内参换算、rot6d、网格与驱动参数的一致性、以及主体 shape 能否被反解回来。
这些错了不会崩，只会让渲染出的对比图悄悄偏离生成图。
"""

from __future__ import annotations

import json
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


def _flame():
    from live3dgsavatar.data.flame_model import FlameModel

    if not (FLAME_DIR / "generic_model.pkl").is_file():
        pytest.skip("没有 FLAME2020 资产")
    return FlameModel(FLAME_DIR, torch.float64)


def _frame(expr=None, rot=None, tra=None, eye=None, K=None) -> cap.Cap4dFrame:
    return cap.Cap4dFrame(
        index=0, image=Path("00000.png"),
        expr=np.zeros(64) if expr is None else expr,
        blink=0.0,
        rot=np.zeros(3) if rot is None else rot,
        tra=np.zeros(3) if tra is None else tra,
        eye_rot=np.zeros(3) if eye is None else eye,
        K=np.eye(3) if K is None else K)


# ------------------------------------------------------------------ 内参 --


def test_crop_intrinsics_matches_manual_scaling() -> None:
    """裁剪 + 缩放到 512：两个轴共用同一个系数（crop_box 是正方形）。"""
    K = cap.crop_intrinsics(fx=1000.0, fy=1000.0, cx=500.0, cy=400.0,
                            crop_box=[100.0, 60.0, 300.0, 260.0], target=512)
    # side = 200，scale = 2.56
    assert K[0, 0] == pytest.approx(1000.0 * 2.56)
    assert K[1, 1] == pytest.approx(1000.0 * 2.56)
    assert K[0, 2] == pytest.approx((500.0 - 100.0) * 2.56)
    assert K[1, 2] == pytest.approx((400.0 - 60.0) * 2.56)
    assert K[2, 2] == 1.0 and K[2, 0] == 0.0


def test_crop_intrinsics_rejects_degenerate_box() -> None:
    with pytest.raises(ValueError, match="crop_box"):
        cap.crop_intrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0,
                            crop_box=[10.0, 10.0, 10.0, 20.0])


# --------------------------------------------------------- CAP4D → RDF 参数 --


def test_build_rdf_x_layout() -> None:
    """CAP4D 的 rot/tra 要落进 RDF 的 global/trans 段，neck/jaw 置零。"""
    expr = np.arange(64, dtype=np.float64) / 64.0
    frame = _frame(expr=expr, rot=[0.1, 0.2, 0.3], tra=[0, 0, -1.0],
                   eye=[0.01, 0.02, 0.03])
    x = cap.build_rdf_x(frame)

    assert x.shape == (118,)
    assert np.allclose(x[:64], expr), "前 64 维表情应原样搬运"
    assert np.allclose(x[64:100], 0.0), "后 36 维表情补零"
    assert np.allclose(x[100:103], [0.1, 0.2, 0.3]), "rot → global"
    assert np.allclose(x[103:106], 0.0), "neck 置零"
    assert np.allclose(x[106:109], 0.0), "jaw 置零（CAP4D 条件几何是闭颌）"
    assert np.allclose(x[109:112], [0.01, 0.02, 0.03]), "eye_l"
    assert np.allclose(x[112:115], [0.01, 0.02, 0.03]), "eye_r"
    assert np.allclose(x[115:118], [0, 0, -1.0]), "tra → trans"


# --------------------------------------------------- 读转换后的中间文件 --


def test_load_converted_validates_fields(tmp_path) -> None:
    p = tmp_path / "conv.npz"
    np.savez(p, mesh=np.zeros((2, 5023, 3), np.float32),
             x=np.zeros((2, 118), np.float32),
             K=np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)),
             names=np.array(["a.png", "b.png"]))
    out = cap.load_converted(p)
    assert out["mesh"].shape == (2, 5023, 3)
    assert out["names"] == ["a.png", "b.png"]


def test_load_converted_rejects_wrong_topology(tmp_path) -> None:
    p = tmp_path / "conv.npz"
    np.savez(p, mesh=np.zeros((2, 5083, 3), np.float32),
             x=np.zeros((2, 118), np.float32),
             K=np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)),
             names=np.array(["a.png", "b.png"]))
    with pytest.raises(ValueError, match="5023"):
        cap.load_converted(p)


def test_load_converted_reports_missing_fields(tmp_path) -> None:
    p = tmp_path / "conv.npz"
    np.savez(p, mesh=np.zeros((1, 5023, 3), np.float32))
    with pytest.raises(ValueError, match="缺少字段"):
        cap.load_converted(p)


def test_to_frames_adds_teeth_and_maps_weights(tmp_path) -> None:
    """5083 顶点（补 60 颗牙）+ 129 维驱动参数 + 逐帧内参。"""
    flame_dir = REPO_ROOT / "data" / "FLAME2020"
    if not (flame_dir / "generic_model.pkl").is_file():
        pytest.skip("没有 FLAME 资产")

    # 用 FLAME 自身的一帧当"RDF 网格"（形状对就行，这里只验流程）
    from live3dgsavatar.data.flame_model import FlameModel

    flame = FlameModel(flame_dir, torch.float64)
    base = flame.forward(
        torch.zeros(1, 300, dtype=torch.float64),
        torch.zeros(1, 100, dtype=torch.float64),
        torch.eye(3, dtype=torch.float64).reshape(1, 3, 3),
        torch.eye(3, dtype=torch.float64).reshape(1, 3, 3),
        torch.eye(3, dtype=torch.float64).reshape(1, 1, 3, 3).expand(1, 2, 3, 3),
        torch.zeros(1, 2, dtype=torch.float64))[0][:5023]

    converted = {"mesh": base[None].numpy().astype(np.float32),
                 "x": np.zeros((1, 118), np.float32),
                 "K": np.eye(3, dtype=np.float32)[None],
                 "names": ["00000.png"]}
    frames, ks = cap.to_frames(converted, flame_dir)

    assert len(frames) == 1
    assert frames[0]["mesh"].shape == (5083, 3)
    assert frames[0]["blend_weight"].shape == (129,)
    assert ks[0].shape == (3, 3)


# ------------------------------------------------------------- 读目录 --


def _write_cap4d_dir(root: Path, n: int = 2) -> Path:
    from PIL import Image

    gen = root / "generated_images"
    (gen / "images").mkdir(parents=True)
    (gen / "flame").mkdir(parents=True)
    for i in range(n):
        Image.fromarray(np.full((512, 512, 3), 250, np.uint8)).save(
            gen / "images" / f"{i:05d}.png")
        np.savez(gen / "flame" / f"{i:05d}.npz",
                 expr=np.zeros(65, np.float32),
                 rot=np.zeros(3, np.float32),
                 tra=np.array([0, 0, -1], np.float32),
                 eye_rot=np.zeros(3, np.float32),
                 fx=np.array([[900.0]], np.float32),
                 fy=np.array([[900.0]], np.float32),
                 cx=np.array([[256.0]], np.float32),
                 cy=np.array([[256.0]], np.float32),
                 extr=np.eye(4, dtype=np.float32)[None],
                 crop_box=np.array([6.0, 6.0, 506.0, 506.0], np.float32))
    return gen


def test_read_cap4d_dir_parses_frames(tmp_path) -> None:
    gen = _write_cap4d_dir(tmp_path, n=2)
    frames = cap.read_cap4d_dir(tmp_path)

    assert len(frames) == 2
    assert frames[0].expr.shape == (64,)
    # side = 500 → scale = 512/500 = 1.024
    assert frames[0].K[0, 0] == pytest.approx(900.0 * 1.024)
    assert frames[0].K[0, 2] == pytest.approx((256.0 - 6.0) * 1.024)
    # 也接受直接给到 generated_images 这一层
    assert len(cap.read_cap4d_dir(gen)) == 2


def test_read_cap4d_dir_reports_missing_layout(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="generated_images"):
        cap.read_cap4d_dir(tmp_path)


def test_read_cap4d_dir_skips_frames_without_params(tmp_path) -> None:
    gen = _write_cap4d_dir(tmp_path, n=2)
    (gen / "flame" / "00001.npz").unlink()
    frames = cap.read_cap4d_dir(tmp_path)
    assert len(frames) == 1
