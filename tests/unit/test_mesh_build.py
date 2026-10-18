"""牙齿顶点补全（`tracking/mesh_build.py`）。

RDF 只输出 5023 个顶点，而模型需要 5083 个 —— 差的 60 个是牙齿。
它们绑着 1408 个高斯（2.33%），缺了会让 `face_id` 指向不存在的顶点。

这里守三件事：

1. **刚体、无反射**：牙齿大小固定，允许缩放会得到"张嘴时牙齿变大"的假解
2. **基础顶点原样透传**：补牙齿不能动已经摆好的 5023 个点
3. **与真值的误差有上界**：已知约 7 mm（模板/identity 差异所致，见文档），
   写成断言是为了在实现变更时立刻发现退化
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from live3dgsavatar.config import load_config
from live3dgsavatar.tracking import mesh_build as MB


def _flame_dir() -> Path:
    """FLAME 资产目录：配置里没写就用仓库根的 `data/FLAME2020/`。"""
    from live3dgsavatar.config import repo_root

    explicit = load_config().get("paths.flame_dir")
    return Path(str(explicit)) if explicit else repo_root() / "data" / "FLAME2020"


def _builder() -> MB.TeethBuilder:
    p = _flame_dir()
    if not p.is_dir():
        pytest.skip(f"缺少 FLAME 资产：{p}")
    return MB.TeethBuilder(p)


# ------------------------------------------------------------- Kabsch --


def test_rigid_transform_recovers_known_rotation() -> None:
    torch.manual_seed(0)
    src = torch.randn(30, 3, dtype=torch.float64)
    # 绕 z 轴 30°
    a = np.deg2rad(30.0)
    rot_true = torch.tensor([[np.cos(a), -np.sin(a), 0.0],
                             [np.sin(a), np.cos(a), 0.0],
                             [0.0, 0.0, 1.0]], dtype=torch.float64)
    t_true = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)
    dst = src @ rot_true.T + t_true

    rot, t = MB.kabsch(src, dst)
    assert torch.allclose(rot, rot_true, atol=1e-9)
    assert torch.allclose(t, t_true, atol=1e-9)
    assert abs(float(torch.det(rot)) - 1.0) < 1e-9


def test_rigid_transform_refuses_reflection() -> None:
    """镜像输入也必须给 `det = +1` —— 否则牙齿会左右对调。"""
    torch.manual_seed(1)
    src = torch.randn(20, 3, dtype=torch.float64)
    mirrored = src * torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)
    rot, _ = MB.kabsch(src, mirrored)
    assert abs(float(torch.det(rot)) - 1.0) < 1e-9


def test_rigid_transform_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match=r"\[N,3\]"):
        MB.kabsch(torch.randn(5, 2), torch.randn(5, 2))


# -------------------------------------------------------------- 补全 --


def test_teeth_are_the_last_sixty_vertices() -> None:
    tb = _builder()
    assert tuple(tb.teeth_canonical.shape) == (MB.NUM_TEETH, 3)
    assert MB.NUM_FULL_VERTS == MB.NUM_BASE_VERTS + MB.NUM_TEETH


def test_build_preserves_base_vertices_verbatim() -> None:
    """补牙齿不能改动已经摆好的基础顶点 —— 它们是 RDF 的直接输出。"""
    tb = _builder()
    torch.manual_seed(2)
    base = torch.randn(3, MB.NUM_BASE_VERTS, 3, dtype=torch.float32) * 0.1

    out = tb.build(base)
    assert tuple(out.shape) == (3, MB.NUM_FULL_VERTS, 3)
    assert torch.equal(out[:, :MB.NUM_BASE_VERTS], base), "基础顶点被改动了"


def test_build_accepts_single_frame() -> None:
    tb = _builder()
    base = torch.randn(MB.NUM_BASE_VERTS, 3, dtype=torch.float32)
    assert tuple(tb.build(base).shape) == (MB.NUM_FULL_VERTS, 3)


def test_build_rejects_wrong_vertex_count() -> None:
    tb = _builder()
    with pytest.raises(ValueError, match="基础顶点"):
        tb.build(torch.randn(2, 1000, 3))


def test_teeth_follow_the_lips() -> None:
    """平移整张脸，牙齿必须跟着走同样的量（而不是留在原处）。"""
    tb = _builder()
    base = torch.zeros(1, MB.NUM_BASE_VERTS, 3, dtype=torch.float32)
    # 用 canonical 顶点构造一个"未摆姿态"的网格，再整体平移
    from live3dgsavatar.data.flame_model import FlameModel

    model = FlameModel(_flame_dir(), dtype=torch.float64)
    canon = model.v_template[:MB.NUM_BASE_VERTS].to(torch.float32)
    shift = torch.tensor([0.05, -0.02, 0.11], dtype=torch.float32)

    a = tb.build(canon[None].expand(1, -1, -1).clone())
    b = tb.build((canon + shift)[None].expand(1, -1, -1).clone())
    moved = (b[0, MB.NUM_BASE_VERTS:] - a[0, MB.NUM_BASE_VERTS:])
    assert torch.allclose(moved, shift.expand(MB.NUM_TEETH, 3), atol=1e-4), (
        "牙齿没有跟随嘴唇平移")


def test_teeth_error_against_ground_truth_is_within_known_bound() -> None:
    """与真值比较（需要先跑过 `scripts/rdf_dump.py`）。

    已知误差约 7 mm，来源是"我们的模板与 RDF 的 identity 不同" ——
    这是 shape 差异，刚体/相似变换都吸收不了（两种都试过，均未改善）。
    上界写得宽松，只用于捕捉**实现退化**，不用于追求精度。
    """
    cfg = load_config()
    rdf_path = Path(str(cfg.paths.output_dir)) / "rdf" / "crh_2.npz"
    data = Path(str(cfg.paths.data_root)) / "crh_2" / "render_inputs.npz"
    if not rdf_path.exists() or not data.exists():
        pytest.skip("缺少 RDF 导出或数据集真值（先跑 scripts/rdf_dump.py）")

    tb = _builder()
    rdf = np.load(rdf_path)
    gt = np.load(data)
    n = min(50, int(rdf["verts"].shape[0]), int(gt["mesh"].shape[0]))

    base = torch.from_numpy(np.asarray(rdf["verts"], np.float64)[:n])
    gt_teeth = torch.from_numpy(np.asarray(gt["mesh"], np.float64)[:n, MB.NUM_BASE_VERTS:])

    out = tb.build(base)
    err = (out[:, MB.NUM_BASE_VERTS:] - gt_teeth).norm(dim=-1)
    mean_mm = float(err.mean()) * 1000.0
    assert mean_mm < 12.0, f"牙齿平均误差 {mean_mm:.1f} mm，超过已知上界 12 mm"
