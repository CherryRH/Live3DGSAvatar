"""CAP4D 导出脚本里**可离线验证**的部分：几何、参考帧选择、文件结构。

真跑 RDF 需要 `realdenseface` 环境（cpython-311 的 `flame_solver`）与 GPU，
这里测不到。但导出正确与否的核心是**几何**：Kabsch 求解、轴角转换、
投影约定。这些是纯 numpy，必须在这里锁住 —— 否则到了服务器上只能靠肉眼
判断"生成的图看起来对不对"。

另外锁住**文件结构**：CAP4D 的 `ReferenceDataset` 只认特定键与目录布局，
写错一个名字就要等到它的 `__getitem__` 才炸。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from script_loader import load_script

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "export_cap4d.py"


def _load():
    return load_script(SCRIPT.name)


# ------------------------------------------------------------------ 几何 --


def test_kabsch_recovers_rigid_transform() -> None:
    """给定刚体变换生成的目标点，必须精确还原 (R, t)。"""
    bench = _load()
    rng = np.random.default_rng(0)
    src = rng.normal(size=(200, 3)) * 0.1

    rv = np.array([0.12, -0.34, 0.21])
    R_true = bench._rotvec_to_mat(rv)
    t_true = np.array([0.02, -0.03, 1.5])

    dst = src @ R_true.T + t_true
    R, t = bench.kabsch(src, dst)

    assert np.allclose(R, R_true, atol=1e-9)
    assert np.allclose(t, t_true, atol=1e-9)


def test_kabsch_never_returns_a_reflection() -> None:
    """镜像数据会让朴素最小二乘得到 det=-1 的"旋转"，那会让渲染左右翻转。"""
    bench = _load()
    rng = np.random.default_rng(1)
    src = rng.normal(size=(100, 3))
    mirrored = src * np.array([-1.0, 1.0, 1.0])       # 反射

    R, _t = bench.kabsch(src, mirrored)
    assert abs(np.linalg.det(R) - 1.0) < 1e-9, np.linalg.det(R)


def test_rotvec_roundtrip() -> None:
    """轴角 → 旋转矩阵 → 轴角，必须回到原值（含接近 180° 的分支）。"""
    bench = _load()
    for rv in ([0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.1, -0.2, 0.3],
               [0.0, 0.0, np.pi - 1e-7], [np.pi - 1e-9, 0.0, 0.0]):
        R = bench._rotvec_to_mat(rv)
        back = bench.rotmat_to_rotvec(R)
        R2 = bench._rotvec_to_mat(back)
        assert np.allclose(R, R2, atol=1e-8), (rv, back)


def test_project_matches_pinhole_formula() -> None:
    bench = _load()
    pts = np.array([[0.0, 0.0, 2.0], [0.5, -0.25, 1.0], [-0.1, 0.2, -2.0]])
    uv = bench.project(pts, fx=500.0, fy=600.0, cx=256.0, cy=255.5)
    assert np.allclose(uv[0], [256.0, 255.5])
    assert np.allclose(uv[1], [0.5 / 1.0 * 500.0 + 256.0,
                               -0.25 / 1.0 * 600.0 + 255.5])
    # z 为负时不能除出反号（脚本里会先警告）
    assert np.isfinite(uv[2]).all()


# ------------------------------------------------------- 参考帧选择 --


def test_select_references_spreads_yaw() -> None:
    """4 张参考要覆盖尽量宽的转角，而不是挤在中间。"""
    bench = _load()
    ys = list(np.linspace(-40, 40, 81))
    picks = bench.select_references(ys, 4)
    assert len(picks) == 4
    assert picks[0] == 0 and picks[-1] == 80, picks
    picked = [ys[i] for i in picks]
    assert picked[0] < -30 and picked[-1] > 30, picked


def test_select_references_handles_few_frames() -> None:
    bench = _load()
    assert bench.select_references([5.0], 4) == [0]
    assert bench.select_references([], 4) == []
    assert len(bench.select_references([1.0, 2.0], 4)) == 2


# ------------------------------------------------------------ 文件结构 --


def test_build_fit_npz_has_every_key_cap4d_reads() -> None:
    """CAP4D 的 `ReferenceDataset` 只会读这些键，名字/形状不能错。"""
    bench = _load()
    n, v = 5, 20
    data = bench.build_fit_npz(
        identity=np.zeros(300, np.float32),
        xs=np.zeros((n, 118), np.float32),
        idx=np.arange(n),
        rots=[np.zeros(3) for _ in range(n)],
        tras=[np.zeros(3) for _ in range(n)],
        eye_rot=np.zeros((n, 3), np.float32),
        jaw=np.zeros((n, 3), np.float32),
        verts=np.zeros((n, v, 3), np.float32),
        residuals=np.zeros(n, np.float32),
        reproj=np.zeros(n, np.float32),
        names=[f"{i:05d}.png" for i in range(n)],
        fx=500.0, fy=500.0, cx=256.0, cy=256.0,
        width=512, height=512, fps=30.0)

    # 键名与形状必须对上 CAP4D 的读取逻辑。
    # ⚠️ shape 必须是 **150** 维：它的 skinner 有
    #    `assert shape_params.shape[1] == self.n_shape_params`（n_shape_params=150），
    #    而 RDF 的 identity 是 300 维 —— 曾经因此直接崩在 DataLoader worker 里。
    assert data["shape"].shape == (150,)
    assert data["shape_full"].shape == (300,)
    assert np.allclose(data["shape"], data["shape_full"][:150])
    assert data["expr"].shape == (n, bench.CAP4D_EXPR_DIM)
    for key in ("rot", "tra", "eye_rot"):
        assert data[key].shape == (n, 3), key
    for key in ("fx", "fy", "cx", "cy"):
        assert data[key].shape == (1, 1), key
    assert data["extr"].shape == (1, 4, 4)
    assert np.allclose(data["extr"][0], np.eye(4)), "extr 应为单位阵"
    assert data["resolutions"].shape == (1, 2)
    assert data["camera_order"].tolist() == ["cam0"]
    # 所有数组都必须是 numpy 能直接 savez 的类型
    for key, value in data.items():
        assert isinstance(value, np.ndarray), key


def test_write_images_produces_cap4d_layout(tmp_path) -> None:
    """CAP4D 按 `images/<cam>/<idx>.png` 取图，且索引与 fit.npz 行号一致。"""
    from PIL import Image

    bench = _load()
    src = tmp_path / "src"
    src.mkdir()
    for i in range(3):
        rgba = np.zeros((8, 8, 4), np.uint8)
        rgba[..., :3] = 30 * i
        rgba[2:6, 2:6, 3] = 255
        Image.fromarray(rgba).save(src / f"{i:05d}.png")
    names = [f"{i:05d}.png" for i in range(3)]

    out = tmp_path / "out"
    written = bench.write_images(src, names, np.array([2, 0]), out)
    assert written == 2

    # 重新编号成 0..1，且顺序跟随 idx
    assert (out / "images" / "cam0" / "00000.png").is_file()
    assert (out / "images" / "cam0" / "00001.png").is_file()
    assert (out / "bg" / "cam0" / "00000.png").is_file()
    assert not (out / "images" / "cam0" / "00002.png").exists()

    alpha = np.asarray(Image.open(out / "bg" / "cam0" / "00000.png"))
    assert alpha.shape == (8, 8), alpha.shape
    assert alpha[3, 3] == 255 and alpha[0, 0] == 0, "前景/背景权重没写对"


def test_find_images_prefers_requested_subdir(tmp_path) -> None:
    bench = _load()
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "00000.png").write_bytes(b"x")
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "00000.png").write_bytes(b"y")

    d, files = bench.find_images(tmp_path, "images")
    assert d == tmp_path / "images" and len(files) == 1

    # 指定的子目录不存在时回退到 input/
    d, files = bench.find_images(tmp_path, "nope")
    assert d == tmp_path / "input" and len(files) == 1


def test_find_images_reports_clearly_when_empty(tmp_path) -> None:
    bench = _load()
    with pytest.raises(FileNotFoundError, match="找不到图像"):
        bench.find_images(tmp_path, "images")


def test_reference_json_matches_written_frames(tmp_path) -> None:
    """`reference_images.json` 里的下标必须真的对应已写出的图。"""
    from PIL import Image

    bench = _load()
    src = tmp_path / "src"
    src.mkdir()
    for i in range(4):
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(src / f"{i:05d}.png")

    out = tmp_path / "out"
    bench.write_images(src, [f"{i:05d}.png" for i in range(4)],
                       np.arange(4), out)
    refs = [["cam0", 1], ["cam0", 3]]
    (out / "reference_images.json").write_text(json.dumps(refs), encoding="utf-8")

    loaded = json.loads((out / "reference_images.json").read_text())
    for cam, t in loaded:
        assert (out / "images" / cam / f"{t:05d}.png").is_file()


def test_expression_last_slot_is_blink_and_must_be_zero() -> None:
    """CAP4D 的第 65 维表情是**眨眼**，不是 FLAME 的第 65 个分量。

    `flowface/flame/flame.py` 在加载基时做
    `expr_eigenvecs[:, :, -1] = blink_blendshape`，所以把 RDF 的第 65 个
    表情系数写进去，语义就错成了"眨眼"。
    """
    bench = _load()
    xs = np.ones((2, 118), np.float32)
    xs[:, :100] = np.arange(100, dtype=np.float32)      # 每维给不同值
    expr = bench._cap4d_expr(xs)

    assert expr.shape == (2, 65)
    assert np.allclose(expr[:, :64], xs[:, :64]), "前 64 维应原样来自 RDF"
    assert np.all(expr[:, 64] == 0.0), "第 65 维（眨眼）必须是 0"


def test_build_fit_npz_shape_is_truncated_to_cap4d_dim() -> None:
    """`shape` 必须是 150 维、`expr` 末位眨眼为 0（CAP4D 的硬性口径）。"""
    bench = _load()
    n = 3
    data = bench.build_fit_npz(
        identity=np.linspace(0, 1, 300).astype(np.float32),
        xs=np.ones((n, 118), np.float32),
        idx=np.arange(n),
        rots=[np.zeros(3)] * n, tras=[np.zeros(3)] * n,
        eye_rot=np.zeros((n, 3), np.float32), jaw=np.zeros((n, 3), np.float32),
        verts=np.zeros((n, 5, 3), np.float32), residuals=np.zeros(n, np.float32),
        reproj=np.zeros(n, np.float32), names=[f"{i:05d}.png" for i in range(n)],
        fx=1.0, fy=1.0, cx=1.0, cy=1.0, width=8, height=8, fps=30.0)
    assert data["shape"].shape == (150,)
    assert data["expr"].shape == (n, 65)
    assert np.all(data["expr"][:, 64] == 0.0)


def test_build_gen_data_matches_cap4d_format() -> None:
    """我们自己的表情池必须与 CAP4D 的 `gen_data.npz` 同构（末位眨眼=0）。"""
    bench = _load()
    xs = np.ones((4, 118), np.float32)
    xs[:, :64] = 0.5
    xs[:, 100 + 9:100 + 12] = 0.2        # 左眼
    xs[:, 100 + 12:100 + 15] = 0.4       # 右眼

    g = bench.build_gen_data(xs, expr_factor=0.8)
    assert set(g) == {"expr", "eye_rot"}
    assert g["expr"].shape == (4, 65)
    assert g["eye_rot"].shape == (4, 3)
    assert np.allclose(g["expr"][:, :64], 0.4), "应乘上 expr_factor"
    assert np.all(g["expr"][:, 64] == 0.0), "眨眼列必须为 0"
    assert np.allclose(g["eye_rot"], 0.3), "两眼取平均"


def test_select_references_prefers_low_expression_in_each_yaw_bin() -> None:
    """同样覆盖 4 个 yaw 区间时，优先选表情更小的帧（参考图决定身份先验）。"""
    bench = _load()
    ys = [-40, -38, -10, -8, 10, 12, 38, 40]
    expr = [5.0, 1.0, 4.0, 0.5, 3.0, 0.6, 6.0, 1.2]

    picks = bench.select_references(ys, 4, expr)
    assert len(picks) == 4
    # 每个区间内都应该是表情更小的那个
    for i in picks:
        assert expr[i] < 3.0, (i, expr[i])

    # 不传 expr 时退化为纯 yaw 分位：只要求覆盖两端
    plain = bench.select_references(ys, 4)
    assert len(plain) == 4
    assert plain[0] == 0 and plain[-1] == len(ys) - 1, plain
