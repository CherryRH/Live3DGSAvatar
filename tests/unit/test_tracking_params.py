"""RDF 参数转换与坐标对齐（**纯 CPU，不需要 GPU / RDF**）。

这里守的是三类"看起来对、实际错"的问题：

1. **布局错位**：129 维的分段一旦错位，渲染不会报错，只是模型变丑 ——
   而且很难与"模型本身不好"区分。
2. **旋转口径错**：头姿到底是 `R_global` 还是 `R_global @ R_neck`，
   两种写法都能跑出图，只有和真值比才知道哪个对。
3. **相似变换含反射**：Umeyama 若允许反射，会得到左右脸互换的"镜像解"，
   残差反而更小，但人看着别扭。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from live3dgsavatar.tracking import align as A
from live3dgsavatar.tracking import param_map as P


def _rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Gram-Schmidt 还原（仅测试用，与 core 里的实现同式）。"""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


# ------------------------------------------------------------- 布局 --


def test_blend_weight_dimensions_and_segments() -> None:
    x = torch.randn(3, P.DIM_X, dtype=torch.float32) * 0.1
    w = P.to_blend_weight(x)

    assert tuple(w.shape) == (3, P.DIM_BLEND_WEIGHT)
    assert w.dtype == torch.float32


def test_expression_is_passed_through_verbatim() -> None:
    """表情段必须原样透传：RDF 与训练侧用的是同一套 100 维表情基。"""
    x = torch.randn(2, P.DIM_X)
    w = P.to_blend_weight(x)
    assert torch.allclose(w[:, :100], x[:, :100], atol=1e-6)


def test_eyelids_are_zero() -> None:
    """RDF 不估计眼睑；数据集的眼睑段也是全 0。"""
    w = P.to_blend_weight(torch.randn(2, P.DIM_X))
    assert torch.equal(w[:, 124:126], torch.zeros(2, 2))


def test_translation_defaults_to_passthrough() -> None:
    x = torch.randn(2, P.DIM_X)
    w = P.to_blend_weight(x)
    assert torch.allclose(w[:, 126:129], x[:, 115:118], atol=1e-6)


def test_translation_override_is_used() -> None:
    """标定后的平移应当替换掉 RDF 的原始 trans。"""
    x = torch.randn(2, P.DIM_X)
    override = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    w = P.to_blend_weight(x, translation=override)
    assert torch.allclose(w[:, 126:129], override)


def test_single_vector_input_is_accepted() -> None:
    w = P.to_blend_weight(torch.randn(P.DIM_X))
    assert tuple(w.shape) == (P.DIM_BLEND_WEIGHT,)


def test_wrong_input_dim_is_rejected() -> None:
    with pytest.raises(ValueError, match="维数"):
        P.to_blend_weight(torch.randn(3, 117))


# ------------------------------------------------------------- 旋转 --


def test_rot6d_rows_are_orthonormal() -> None:
    """blend_weight 里的 rot6d 必须是**正交矩阵的前两行**。

    否则 `rotation_6d_to_matrix` 里的 Gram-Schmidt 会"修正"它 ——
    也就是说模型看到的旋转和你以为的不是同一个。
    """
    x = torch.randn(8, P.DIM_X) * 0.3
    w = P.to_blend_weight(x)

    for a, b, name in ((100, 106, "head"), (106, 112, "jaw")):
        m = _rot6d_to_matrix(w[:, a:b])
        eye = torch.eye(3).expand(m.shape[0], 3, 3)
        assert torch.allclose(m @ m.transpose(-1, -2), eye, atol=1e-5), name
        # 前两行本身已正交，Gram-Schmidt 不应改变它们
        assert torch.allclose(m[:, :2, :], w[:, a:b].reshape(-1, 2, 3), atol=1e-5), name

    eyes = w[:, 112:124].reshape(-1, 2, 2, 3)
    for i in range(2):
        m = _rot6d_to_matrix(eyes[:, i].reshape(-1, 6))
        assert torch.allclose(m @ m.transpose(-1, -2),
                              torch.eye(3).expand(m.shape[0], 3, 3), atol=1e-5)


def test_head_rotation_composes_global_and_neck() -> None:
    """默认头姿 = `R_global @ R_neck`（与 RDF 的 `_extract_head_rot` 一致）。"""
    x = torch.zeros(1, P.DIM_X)
    x[0, 100:103] = torch.tensor([0.4, -0.2, 0.1])   # global
    x[0, 103:106] = torch.tensor([0.1, 0.3, -0.2])   # neck

    parts = P.split_pose(x)
    r_global = P.rotvec_to_matrix(parts["global"])
    r_neck = P.rotvec_to_matrix(parts["neck"])

    composed = P.head_matrix(x, compose=True)
    assert torch.allclose(composed, r_global @ r_neck, atol=1e-6)

    only_global = P.head_matrix(x, compose=False)
    assert torch.allclose(only_global, r_global, atol=1e-6)
    assert not torch.allclose(composed, only_global, atol=1e-3), (
        "两级旋转的复合与只用 global 必须不同，否则这个开关没有意义")


def test_rotvec_zero_is_identity() -> None:
    m = P.rotvec_to_matrix(torch.zeros(2, 3))
    assert torch.allclose(m, torch.eye(3).expand(2, 3, 3), atol=1e-7)


# ------------------------------------------------------------- 诊断 --


def test_segment_errors_localizes_the_problem() -> None:
    """诊断要能指出"只有平移不对"，而不是笼统地报一个总误差。"""
    x = torch.randn(2, P.DIM_X)
    good = P.to_blend_weight(x)
    bad = good.clone()
    bad[:, 126:129] += 0.5

    err = P.segment_errors(bad, good)
    assert err["translation"] > 0.4
    for name in ("expression", "head", "jaw", "eyes", "eyelids"):
        assert err[name] < 1e-6, f"{name} 不应有误差"


# ------------------------------------------------------------- 对齐 --


def _random_rotation(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(4, generator=g)
    q = q / q.norm()
    w, x, y, z = q
    return torch.tensor([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=torch.float64)


def test_fit_similarity_recovers_known_transform() -> None:
    torch.manual_seed(0)
    src = torch.randn(50, 3, dtype=torch.float64)
    rot = _random_rotation()
    s, t = 0.6, torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)
    dst = s * (src @ rot.T) + t

    got = A.fit_similarity(src, dst)
    assert abs(got["scale"] - s) < 1e-9
    assert torch.allclose(got["rotation"], rot, atol=1e-9)
    assert torch.allclose(got["translation"], t, atol=1e-9)
    assert torch.allclose(A.apply_similarity(src, got), dst, atol=1e-9)


def test_fit_similarity_never_reflects() -> None:
    """镜像输入也必须给 `det(R) = +1` 的解。

    允许反射时残差会更小，但会把左右脸对调 —— 误差指标看不出问题，
    人一眼就看出不对。
    """
    torch.manual_seed(1)
    src = torch.randn(20, 3, dtype=torch.float64)
    mirrored = src * torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)

    got = A.fit_similarity(src, mirrored)
    assert abs(float(torch.det(got["rotation"])) - 1.0) < 1e-9
    # 不能完美拟合 —— 这正是"拒绝反射"的代价，也是应有的行为
    assert float(A.residuals(A.apply_similarity(src, got), mirrored)["max"]) > 1e-3


def test_fit_similarity_rejects_degenerate_input() -> None:
    same = torch.zeros(5, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match="重合"):
        A.fit_similarity(same, torch.randn(5, 3, dtype=torch.float64))

    with pytest.raises(ValueError, match=r"\[N,3\]"):
        A.fit_similarity(torch.randn(5, 2), torch.randn(5, 2))


def test_scale_about_camera_recovers_known_scale() -> None:
    """单参数模型：沿相机中心缩放，恢复已知 s。"""
    cam = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    src = torch.randn(40, 3, dtype=torch.float64) * 0.1 + torch.tensor([0.0, 0.0, 0.2])
    s = 0.55
    dst = cam + s * (src - cam)

    got = A.fit_scale_about_camera(src, dst, cam)
    assert abs(got - s) < 1e-9

    # 应用后应为零残差
    applied = cam + got * (src - cam)
    assert A.residuals(applied, dst)["max"] < 1e-9


def test_scale_about_camera_defaults_to_dataset_camera() -> None:
    """默认相机中心是 (0,0,1) —— 由数据集的 `w2v` 推出。

    不写死期望值（容易手算错），而是构造一对**确实**绕 (0,0,1) 缩放的点，
    再验证默认参数能还原它。
    """
    cam = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    src = torch.tensor([[0.0, 0.0, 0.5], [0.1, 0.0, 0.3]], dtype=torch.float64)
    s_true = 0.4
    dst = cam + s_true * (src - cam)

    got = A.fit_scale_about_camera(src, dst)     # 不传 camera_center
    assert abs(got - s_true) < 1e-9


def test_residuals_reports_distribution() -> None:
    pred = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                        dtype=torch.float64)
    target = torch.zeros(3, 3, dtype=torch.float64)
    r = A.residuals(pred, target)
    assert abs(r["max"] - 1.0) < 1e-9
    assert 0 < r["p95"] <= r["max"]
