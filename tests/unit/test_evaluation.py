"""评估层：候选视角网格 + 有真值/无真值度量。

这一层的定位是"被脚本、训练钩子、GUI 共用"，所以它的**纯函数部分必须能离线测**：
视角网格、度量聚合、报告序列化都不依赖 CUDA。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import live3dgsavatar  # noqa: E402, F401

from live3dgsavatar.core.camera import (orbit_w2c, project_point,  # noqa: E402
                                        rotation_x, rotation_y, roll_w2c)
from live3dgsavatar.evaluation import (EvaluationReport, ViewSpec,  # noqa: E402
                                       axis_extremes, candidate_views, cluster,
                                       grid_size, summarize_no_reference,
                                       summarize_quality)


# ------------------------------------------------------------------ 视角网格 --


def test_default_grid_covers_40_by_20_with_densified_yaw() -> None:
    views = candidate_views()
    yaws = sorted({v.yaw for v in views})
    pitches = sorted({v.pitch for v in views})
    assert yaws[0] == -40.0 and yaws[-1] == 40.0
    assert pitches == [-20.0, -10.0, 0.0, 10.0, 20.0]
    # |yaw| >= 30 处加密到 5°
    assert 35.0 in yaws and -35.0 in yaws
    assert 15.0 not in yaws
    assert 0.0 in yaws and len(views) == 55


def test_grid_has_no_duplicates_and_is_sorted() -> None:
    views = candidate_views()
    assert len(set(views)) == len(views)
    keys = [(v.roll, v.pitch, v.yaw) for v in views]
    assert keys == sorted(keys)


def test_rolls_multiply_view_count() -> None:
    assert grid_size(rolls=(-15.0, 0.0, 15.0)) == 3 * grid_size()
    labels = {v.label for v in candidate_views(rolls=(-15.0, 0.0, 15.0))}
    assert "yaw+40_pitch+20_roll+15" in labels
    assert "yaw+40_pitch+20_roll-15" in labels


def test_view_spec_rejects_bad_grid() -> None:
    with pytest.raises(ValueError, match="网格参数非法"):
        candidate_views(yaw_limit=0)
    with pytest.raises(ValueError, match="网格参数非法"):
        candidate_views(step=0)


def test_view_spec_serialises() -> None:
    d = ViewSpec(30.0, -10.0, 5.0).to_dict()
    assert d == {"yaw": 30.0, "pitch": -10.0, "roll": 5.0,
                 "label": "yaw+30_pitch-10_roll+5"}


# -------------------------------------------------------------------- 相机 --


def test_orbit_keeps_point_fixed_on_screen() -> None:
    """绕某点旋转时，该点的投影必须不动（早前绕世界原点转，人飞出画面）。"""
    K = np.array([[955.4, 0, 255.5], [0, 955.4, 255.5], [0, 0, 1]], np.float32)
    w2v = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 1],
                    [0, 0, 0, 1]], np.float32)
    center = np.array([-0.01, -0.01, 0.39], np.float32)

    ref = None
    for yaw, pitch in ((0.0, 0.0), (-40.0, 0.0), (40.0, 0.0), (0.0, 20.0)):
        view = orbit_w2c(w2v, center, yaw, pitch)
        u, v = project_point(K, view, center)
        if ref is None:
            ref = (u, v)
        assert abs(u - ref[0]) < 0.5 and abs(v - ref[1]) < 0.5


def test_orbit_zero_is_identity_and_roll_keeps_center() -> None:
    import torch

    w2c = torch.tensor(np.eye(4), dtype=torch.float32)
    c = torch.tensor([0.0, 0.0, 0.4])
    assert torch.allclose(orbit_w2c(w2c, c, 0.0, 0.0), w2c)
    # 滚转在相机空间进行：相机中心不动
    view = roll_w2c(w2c, 15.0)
    assert float((view[:3, 3]).abs().max()) < 1e-6


def test_rotations_are_orthonormal() -> None:
    import torch

    for fn in (rotation_x, rotation_y):
        R = fn(37.0)[:3, :3]
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-6)
        assert abs(float(torch.det(R)) - 1.0) < 1e-6


# ---------------------------------------------------------------- 有真值度量 --


def test_summarize_quality_matches_manual_psnr() -> None:
    pred = np.zeros((8, 8, 3), np.uint8)
    gt_rgb = np.full((8, 8, 3), 128, np.uint8)
    mask = np.zeros((8, 8), bool)
    mask[2:6, 2:6] = True

    out = summarize_quality({0: pred}, {0: (gt_rgb, mask)})

    expect_full = 10 * np.log10(255 ** 2 / 128 ** 2)
    assert out["psnr_full"]["median"] == pytest.approx(expect_full, rel=1e-4)
    assert out["psnr_masked"]["n"] == 1
    assert out["max_abs_diff"]["max"] == 128


def test_summarize_quality_skips_missing_and_mismatched() -> None:
    img = np.zeros((8, 8, 3), np.uint8)
    other = np.zeros((4, 4, 3), np.uint8)
    out = summarize_quality({0: img, 1: img, 2: img},
                            {0: (img, np.ones((8, 8), bool)),
                             1: (other, np.ones((4, 4), bool))})
    assert out["psnr_full"]["n"] == 1


def test_summarize_quality_empty_when_no_overlap() -> None:
    img = np.zeros((8, 8, 3), np.uint8)
    assert summarize_quality({3: img}, {0: (img, np.ones((8, 8), bool))}) == {}


# -------------------------------------------------------------- 无真值度量 --


def test_no_reference_stats_capture_blur_and_foreground() -> None:
    """糊图锐度低、全黑图前景占比 0 —— 这两个量是新视角评估的主要抓手。"""
    checker = np.zeros((16, 16, 3), np.uint8)
    checker[::2, ::2] = 255
    blurred = np.full((16, 16, 3), 128, np.uint8)
    dark = np.zeros((16, 16, 3), np.uint8)

    sharp = summarize_no_reference({0: checker})
    soft = summarize_no_reference({0: blurred})
    assert sharp["sharpness"]["median"] > soft["sharpness"]["median"]
    assert summarize_no_reference({0: dark}) == {} or \
        summarize_no_reference({0: dark})["foreground"]["median"] == 0.0


def test_no_reference_uses_mask_when_given() -> None:
    img = np.full((8, 8, 3), 200, np.uint8)
    mask = np.zeros((8, 8), bool)
    mask[:4] = True
    out = summarize_no_reference({0: img}, masks={0: mask})
    assert out["foreground"]["median"] == pytest.approx(0.5)


# -------------------------------------------------------------------- 报告 --


def test_report_to_dict_is_json_serialisable() -> None:
    import json

    r = EvaluationReport(subject="s", work_name="w", dataset="d",
                         num_frames=3, dataset_frames=10, width=8, height=8,
                         quality={"psnr_full": {"median": 1.0}},
                         extra={"model_path": "m.ply"})
    text = json.dumps(r.to_dict())
    assert '"dataset": "d"' in text and r.has_reference is True

    empty = EvaluationReport(subject="s", work_name="w", dataset="d",
                             num_frames=0, dataset_frames=0, width=8, height=8)
    assert empty.has_reference is False
    assert "quality" not in empty.to_dict()


# ------------------------------------------------------------ 缺陷信号 --


def test_attribute_to_faces_mean_max_sum() -> None:
    """逐像素量按面 id 聚合：背景（-1）必须被忽略。"""
    from live3dgsavatar.evaluation import attribute_to_faces

    fid = np.array([[0, 0], [1, -1]])
    vals = np.array([[1.0, 3.0], [10.0, 99.0]])
    assert list(attribute_to_faces(vals, fid, 2, "mean")) == [2.0, 10.0]
    assert list(attribute_to_faces(vals, fid, 2, "max")) == [3.0, 10.0]
    assert list(attribute_to_faces(vals, fid, 2, "sum")) == [4.0, 10.0]
    # 没被覆盖的面保持 0（而不是 NaN）
    assert list(attribute_to_faces(vals, fid, 3, "mean"))[2] == 0.0


def test_attribute_to_faces_rejects_bad_input() -> None:
    from live3dgsavatar.evaluation import attribute_to_faces

    with pytest.raises(ValueError, match="形状不一致"):
        attribute_to_faces(np.zeros((2, 2)), np.zeros((3, 3), int), 1)
    with pytest.raises(ValueError, match="未知 reduce"):
        attribute_to_faces(np.zeros((2, 2)), np.zeros((2, 2), int), 1, "median")


def test_coverage_counts_accumulates_over_views() -> None:
    from live3dgsavatar.evaluation import coverage_counts

    a = np.array([[0, 0], [1, -1]])
    b = np.array([[0, -1], [-1, -1]])
    assert list(coverage_counts([a, b], 2)) == [3, 1]
    assert list(coverage_counts([], 2)) == [0, 0]


def test_alpha_hole_ratio_uses_face_id_as_reference() -> None:
    from live3dgsavatar.evaluation import alpha_hole_ratio

    alpha = np.array([[1.0, 0.0], [0.5, 1.0]])
    fid = np.array([[0, 0], [1, -1]])
    assert list(alpha_hole_ratio(alpha, fid, 2)) == pytest.approx([0.5, 0.0])


def test_floater_offsets_measure_distance_to_surface() -> None:
    import torch

    from live3dgsavatar.evaluation import floater_offsets

    verts = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = torch.tensor([[0, 1, 2]])
    xyz = torch.tensor([[0.25, 0.25, 0.0], [0.25, 0.25, 0.1]])
    fid = torch.tensor([0, 0])
    bary = torch.tensor([[0.5, 0.25, 0.25]] * 2)
    out = floater_offsets(xyz, verts, faces, fid, bary)
    assert pytest.approx(float(out[0]), abs=1e-6) == 0.0
    assert pytest.approx(float(out[1]), rel=1e-5) == 0.1


def test_sharpness_and_refine_delta() -> None:
    from live3dgsavatar.evaluation import refine_delta, sharpness_map

    flat = np.full((8, 8, 3), 128, np.uint8)
    edged = flat.copy()
    edged[4:] = 255
    assert sharpness_map(edged).mean() > sharpness_map(flat).mean()

    delta = refine_delta(flat, edged)
    assert delta.shape == (8, 8)
    assert delta[:4].max() == pytest.approx(0.0)
    assert delta[4:].mean() > 0


def test_per_texel_maps_back_through_binding() -> None:
    """per-face 值铺回 UV：无效 texel 为 NaN，有效 texel 取所属面的值。"""
    import torch

    from live3dgsavatar.evaluation import per_texel

    texel_face = torch.tensor([0, 1])
    valid = torch.tensor([True, True, False, False])
    out = per_texel(np.array([5.0, 7.0]), texel_face, tex_size=2, valid_mask=valid)
    assert out.shape == (4,)
    assert out[0] == 5.0 and out[1] == 7.0
    assert np.isnan(out[2]) and np.isnan(out[3])


def test_build_defect_map_summary_and_score() -> None:
    import torch

    from live3dgsavatar.evaluation import build_defect_map

    per_face = {"coverage": np.array([0.0, 4.0]),
                "floater": np.array([0.0, 0.02])}
    dm = build_defect_map(per_face=per_face,
                          texel_face_id=torch.tensor([0, 1]),
                          tex_size=2,
                          valid_mask=torch.tensor([True, True, False, False]))
    # 覆盖率不足 → 分数 > 0（面 0 从未被看到）
    assert dm.score > 0
    assert 0 in dm.top_faces("coverage", k=1)
    s = dm.summary()
    assert s["coverage"]["min"] == 0.0 and s["floater"]["max"] == 0.02


def test_face_id_rasterization_needs_gpu() -> None:
    """面 id 光栅化依赖 nvdiffrast + CUDA；无 GPU 时明确跳过。"""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("没有 CUDA")
    from live3dgsavatar.core.render.faceid import rasterize_faces  # noqa: F401


# ------------------------------------------------------------ 表情候选池 --


def test_expression_matrix_slices_blend_weight() -> None:
    from live3dgsavatar.evaluation import expression_matrix

    w = np.zeros((3, 129), np.float32)
    w[:, :100] = np.arange(100, dtype=np.float32)
    X = expression_matrix(w)
    assert X.shape == (3, 100) and X[0, 5] == 5.0
    with pytest.raises(ValueError, match="blend_weight"):
        expression_matrix(np.zeros((3, 50), np.float32))


def test_load_expressions_from_bundled_dataset() -> None:
    from live3dgsavatar.evaluation import load_expressions

    root = REPO_ROOT / "datasets" / "crh_1"
    if not (root / "render_inputs.npz").is_file():
        pytest.skip("缺少仓库内数据集")
    X = load_expressions(root)
    assert X.ndim == 2 and X.shape[1] == 100 and X.shape[0] > 100


def test_cluster_recovers_separated_groups() -> None:
    rng = np.random.default_rng(0)
    centers = np.array([[0.0] * 100, [5.0] * 100, [-5.0] * 100])
    X = np.vstack([c + rng.normal(0, 0.05, (40, 100)) for c in centers])
    specs = cluster(X, k=3)
    assert len(specs) == 3
    got = np.array([s.vector for s in specs])
    # 每个恢复出的中心都应贴近某个真实中心
    for c in centers:
        assert np.min(np.linalg.norm(got - c, axis=1)) < 0.5


def test_cluster_is_deterministic_and_clamps_k() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 100)).astype(np.float32)
    a = [s.vector for s in cluster(X, k=5, seed=3)]
    b = [s.vector for s in cluster(X, k=5, seed=3)]
    assert all(np.allclose(x, y) for x, y in zip(a, b))
    # 请求 999 个簇时不能超过样本数
    assert len(cluster(X, k=999)) <= len(X)


def test_axis_extremes_lie_along_dominant_direction() -> None:
    """主轴极值：只在方差大的方向上有偏移，且幅度约为 alpha·σ。"""
    rng = np.random.default_rng(2)
    X = rng.normal(0, 0.01, (60, 100))
    X[:, 7] = rng.normal(0, 2.0, 60)          # 只有第 7 维方差大
    specs = axis_extremes(X, k=1, alpha=1.5)
    assert len(specs) == 2
    spread = [s.vector[7] for s in specs]
    assert max(spread) > 1.0 and min(spread) < -1.0
    # 其它维基本不动
    mean_other = float(np.abs(specs[0].vector[50]).max())
    assert mean_other < 0.1


def test_merge_specs_drops_near_duplicates() -> None:
    from live3dgsavatar.evaluation import ExpressionSpec, merge_specs

    a = ExpressionSpec("a", np.ones(100, np.float32))
    b = ExpressionSpec("b", np.ones(100, np.float32) * 1.001)
    c = ExpressionSpec("c", -np.ones(100, np.float32))
    kept = merge_specs([a], [b, c])
    assert [s.name for s in kept] == ["a", "c"]


def test_load_named_passes_through_and_empty_by_default() -> None:
    from live3dgsavatar.evaluation import load_named

    assert load_named(None) == []
    specs = load_named({"angry": [0.0] * 100, "sad": [1.0] + [0.0] * 99})
    assert [s.name for s in specs] == ["angry", "sad"]
    assert specs[1].magnitude == pytest.approx(1.0)


# ------------------------------------------------------------ 贪心覆盖 --


def test_greedy_cover_picks_highest_mass_first() -> None:
    from live3dgsavatar.evaluation import Candidate, greedy_cover

    mass = np.zeros(10)
    mass[7] = 100.0                            # 只有 texel 7 有缺陷
    a = Candidate("a", [0, 1, 2])
    b = Candidate("b", [7])
    res = greedy_cover([a, b], mass, budget=2)
    assert [c.label for c in res.chosen] == ["b"]
    assert res.covered_ratio == 1.0


def test_greedy_cover_reports_marginal_gains() -> None:
    from live3dgsavatar.evaluation import Candidate, greedy_cover

    mass = np.array([1.0, 1.0, 5.0, 0.0])
    a = Candidate("a", [0, 1])
    b = Candidate("b", [1, 2])
    res = greedy_cover([a, b], mass, budget=2)
    assert [c.label for c in res.chosen] == ["b", "a"]
    assert res.gains[0] == pytest.approx(6.0)     # b 覆盖 texel1+2 = 1+5
    assert res.gains[1] == pytest.approx(1.0)     # a 只新增 texel0
    assert res.covered_mass == pytest.approx(7.0)


def test_greedy_cover_respects_budget_and_stops_early() -> None:
    from live3dgsavatar.evaluation import Candidate, greedy_cover

    mass = np.full(5, 1.0)
    cands = [Candidate(f"c{i}", [i]) for i in range(5)]
    assert len(greedy_cover(cands, mass, budget=2).chosen) == 2
    # 新增低于总质量的 50% 就停：第一张 20% < 50% → 一张都不选
    assert greedy_cover(cands, mass, budget=5, min_gain_ratio=0.5).chosen == []


def test_greedy_cover_handles_degenerate_inputs() -> None:
    from live3dgsavatar.evaluation import Candidate, greedy_cover

    assert greedy_cover([], np.zeros(3)).chosen == []
    assert greedy_cover([Candidate("a", [0])], np.zeros(3)).chosen == []
    with pytest.raises(ValueError, match="budget"):
        greedy_cover([Candidate("a", [0])], np.ones(3), budget=0)


def test_per_texel_accepts_explicit_indices() -> None:
    """从 .ply 载入的模型没有 UV 掩码位置信息，必须能用显式 texel 下标铺回。"""
    import torch

    from live3dgsavatar.evaluation import per_texel

    out = per_texel(np.array([2.0, 5.0]), torch.tensor([0, 1]), tex_size=2,
                    texel_index=np.array([0, 3]))
    assert out[0] == 2.0 and out[3] == 5.0
    assert np.isnan(out[1]) and np.isnan(out[2])


def test_per_texel_rejects_ply_style_mask_without_indices() -> None:
    """`[N]` 全 True 的掩码不能当成 `[tex_size²]` 用 —— 报错要说人话。"""
    import torch

    from live3dgsavatar.evaluation import per_texel

    with pytest.raises(ValueError, match="texel_index"):
        per_texel(np.array([1.0, 2.0]), torch.tensor([0, 1]), tex_size=2,
                  valid_mask=torch.ones(2, dtype=torch.bool))


def test_per_texel_requires_some_position_info() -> None:
    import torch

    from live3dgsavatar.evaluation import per_texel

    with pytest.raises(ValueError, match="需要 texel_index"):
        per_texel(np.array([1.0]), torch.tensor([0]), tex_size=2)


# -------------------------------------------------- 缺陷质量的两个归一化 --


def test_coverage_defect_saturates() -> None:
    """覆盖率缺陷必须饱和：看过 50 次以上就算充分，不能被分位数拉平。

    早前按 P90 归一化，导致"看过 850 次"的中位数也被判成 86% 缺陷，
    缺陷图几乎覆盖整张脸，选择器退化成"谁看得最多先选谁"。
    """
    from live3dgsavatar.evaluation import coverage_defect

    out = coverage_defect(np.array([0, 25, 50, 500, 5000]), saturate=50)
    assert out[0] == 1.0 and out[2] == 0.0 and out[3] == 0.0
    assert out[1] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="saturate"):
        coverage_defect(np.zeros(3), saturate=0)


def test_hole_rate_is_a_ratio_not_a_max() -> None:
    """孔洞用命中率：1100 个候选里几乎每个面都会偶发漏一次，取最大值等于全标坏。"""
    from live3dgsavatar.evaluation import hole_rate

    out = hole_rate(np.array([0, 1, 3, 10]), np.array([10, 10, 10, 10]))
    assert list(np.round(out, 2)) == [0.0, 0.1, 0.3, 1.0]
    # 没被看到的面不参与（保持 0，而不是 NaN 或误判）
    assert hole_rate(np.array([3, 0]), np.array([0, 0]))[1] == 0.0
    with pytest.raises(ValueError, match="形状不一致"):
        hole_rate(np.zeros(2), np.zeros(3))


def test_cluster_frames_returns_real_rows() -> None:
    """表情候选取真实帧（medoid）：向量必须与 X 的某一行完全一致。"""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 100)).astype(np.float32)
    from live3dgsavatar.evaluation import cluster_frames

    specs = cluster_frames(X, k=4)
    assert len(specs) == 4
    for spec in specs:
        idx = int(spec.name[5:])
        assert 0 <= idx < len(X)
        assert np.allclose(spec.vector, X[idx]), "候选应当就是某一帧的表情"
