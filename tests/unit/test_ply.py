"""PLY 序列化测试（**不需要 GPU**）。

验证：往返一致（参数逐元素相同）、元数据正确、
以及"配置与文件不一致时报错"这类契约（而不是静默读错）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


ATOL = 0.0  # PLY 是 float32 原样存取，应逐位一致


def _tiny_avatar(n: int = 64, k: int = 3, tex_size: int = 8):
    """构造一个极小 avatar（只用有效高斯，无需 UV 光栅化）。"""
    from live3dgsavatar.core.avatar import AvatarConfig, GaussianAvatar
    from live3dgsavatar.core.deform.bind import Binding

    binding = Binding(
        face_id=torch.zeros(n, dtype=torch.long),
        face_bary=torch.tensor([[1.0, 0.0, 0.0]] * n),
        valid_mask=torch.ones(n, dtype=torch.bool),
    )
    # MLP 要够小：`weight_module` 参数存成 PLY 的一列，参数量不能超过高斯数
    return GaussianAvatar(
        AvatarConfig(tex_size=tex_size, num_basis_in=8, num_basis_blend=k,
                     mlp_hidden=()), binding)


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().cpu().to(torch.float64) - b.detach().cpu().to(torch.float64)).abs().max())


def test_ply_roundtrip_is_exact(tmp_path) -> None:
    from live3dgsavatar.core.io import load_ply, save_ply

    avatar = _tiny_avatar()
    path = tmp_path / "avatar.ply"
    save_ply(avatar, path)
    assert path.exists() and path.stat().st_size > 0

    # 不传 config：应从 PLY 内嵌的 gaussian_config 还原
    loaded = load_ply(path, config=None, device="cpu")

    assert loaded.num_gaussians == avatar.num_gaussians
    assert loaded.num_basis == avatar.num_basis
    assert loaded.config.mlp_hidden == avatar.config.mlp_hidden
    assert loaded.config.num_basis_in == avatar.config.num_basis_in

    for name in ("xyz", "opacity", "scaling", "rotation", "color",
                 "xyz_b", "rotation_b", "color_b"):
        d = _max_diff(getattr(avatar, name), getattr(loaded, name))
        assert d <= ATOL, f"{name} 往返不一致：max|Δ| = {d:.3e}"

    assert torch.equal(loaded.binding_face_id, avatar.binding_face_id)
    d = _max_diff(loaded.binding_face_bary, avatar.binding_face_bary)
    assert d <= ATOL, f"face_bary 往返不一致：{d:.3e}"

    # weight_module 参数也应还原
    ref_flat = torch.cat([p.detach().reshape(-1) for p in avatar.weight_module.state_dict().values()])
    got_flat = torch.cat([p.detach().reshape(-1) for p in loaded.weight_module.state_dict().values()])
    d = _max_diff(ref_flat, got_flat)
    assert d <= ATOL, f"weight_module 往返不一致：max|Δ| = {d:.3e}"


def test_read_ply_metadata_reports_config(tmp_path) -> None:
    from live3dgsavatar.core.io import read_ply_metadata, save_ply

    avatar = _tiny_avatar(n=512, k=5)
    path = tmp_path / "a.ply"
    save_ply(avatar, path)

    meta = read_ply_metadata(path)
    assert meta["num_gaussians"] == 512
    assert meta["num_basis_blend_from_file"] == 5
    assert meta["has_binding"] is True
    assert meta["config"]["num_basis_blend"] == 5
    assert meta["config"]["mlp_hidden"] == []


def test_load_rejects_basis_count_mismatch(tmp_path) -> None:
    """K3：配置与文件列数不一致必须报错（参照实现会静默出错）。"""
    from live3dgsavatar.core.avatar import AvatarConfig
    from live3dgsavatar.core.io import load_ply, save_ply

    save_ply(_tiny_avatar(k=3), tmp_path / "a.ply")
    bad = AvatarConfig(tex_size=32, num_basis_in=7, num_basis_blend=9, mlp_hidden=())
    try:
        load_ply(tmp_path / "a.ply", config=bad, device="cpu")
    except ValueError as e:
        assert "num_basis" in str(e), f"报错信息应说明 num_basis 不一致，实际：{e}"
    else:
        raise AssertionError("基数量不一致应报错")


def test_save_rejects_weight_module_larger_than_gaussians(tmp_path) -> None:
    """weight_module 被压进单列，因此参数量必须 ≤ 高斯数；超出要报错而非写出坏文件。"""
    from live3dgsavatar.core.avatar import AvatarConfig, GaussianAvatar
    from live3dgsavatar.core.deform import Binding
    from live3dgsavatar.core.io import save_ply

    n, tex = 64, 8            # 64 个高斯
    binding = Binding(torch.zeros(n, dtype=torch.long),
                      torch.tensor([[1.0, 0.0, 0.0]]).repeat(n, 1),
                      torch.ones(tex * tex, dtype=torch.bool))
    cfg = AvatarConfig(tex_size=tex, num_basis_in=129, num_basis_blend=20,
                       mlp_hidden=(128, 128), use_weight_proj=True)
    avatar = GaussianAvatar(cfg, binding)   # 参数量 35732 >> 64

    try:
        save_ply(avatar, tmp_path / "bad.ply")
    except ValueError as e:
        assert "tex_size" in str(e), f"报错应给出 tex_size 建议，实际：{e}"
    else:
        raise AssertionError("参数量超过高斯数时应报错")


def test_load_rejects_tex_size_too_small_for_gaussians(tmp_path) -> None:
    """**回归测试**：`tex_size` 装不下文件里的高斯数时必须报错。

    `tex_size` 不参与存储（它只决定 UV texel 的容量上界 `tex_size²`），
    因此**无法**从文件反推；但配错时必须显式报出，
    否则会把错误配置静默带进后续的保存/重建。
    实测：`tex_size=128`（上界 16384）加载 60353 个高斯**不报错**。
    """
    from live3dgsavatar.core.avatar import AvatarConfig
    from live3dgsavatar.core.io import load_ply, save_ply

    path = tmp_path / "m.ply"
    avatar = _tiny_avatar(n=64, k=3, tex_size=8)          # 上界 8² = 64，刚好
    save_ply(avatar, path)

    # 正确配置可加载（config 从文件的自描述注释还原）
    loaded = load_ply(path, device="cpu")
    assert loaded.num_gaussians == 64

    # tex_size 太小 → 明确报错
    bad = AvatarConfig(tex_size=4, num_basis_in=7, num_basis_blend=3,
                       mlp_hidden=(), use_weight_proj=True)
    try:
        load_ply(path, config=bad, device="cpu")
    except ValueError as e:
        assert "tex_size" in str(e), f"报错信息应提到 tex_size，实际：{e}"
    else:
        raise AssertionError("tex_size=4 装不下 64 个高斯，应报错")


def test_read_embedded_config_roundtrip(tmp_path) -> None:
    """我们自己写出的 PLY 带 `gaussian_config`，读回来要能还原结构。"""
    import torch

    from live3dgsavatar.core.avatar import AvatarConfig, GaussianAvatar
    from live3dgsavatar.core.deform import Binding
    from live3dgsavatar.core.io.ply import read_embedded_config, save_ply

    # 高斯数要够放 MLP 参数（PLY 约定：weight_module 压进单列，行数 ≥ 参数量）
    n = 64
    cfg = AvatarConfig(tex_size=8, num_basis_in=4, num_basis_blend=3,
                       mlp_hidden=(4,), use_weight_proj=True)
    binding = Binding(face_id=torch.zeros(n, dtype=torch.long),
                      face_bary=torch.tensor([[0.5, 0.5, 0.0]]).repeat(n, 1),
                      valid_mask=torch.ones(n, dtype=torch.bool))
    avatar = GaussianAvatar(cfg, binding)
    path = tmp_path / "m.ply"
    save_ply(avatar, path)

    back = read_embedded_config(path)
    assert back is not None
    assert back.tex_size == cfg.tex_size
    assert back.num_basis_in == cfg.num_basis_in
    assert back.num_basis_blend == cfg.num_basis_blend
    assert tuple(back.mlp_hidden) == tuple(cfg.mlp_hidden)
    assert back.use_weight_proj is True


def test_read_embedded_config_returns_none_without_comment(tmp_path) -> None:
    """参照实现写出的文件没有这条注释，必须返回 None 让调用方回退。"""
    from live3dgsavatar.core.io.ply import read_embedded_config

    real = REPO_ROOT / "models" / "crh_1" / "test" / "model.ply"
    if real.is_file():
        assert read_embedded_config(real) is None
    else:
        pytest.skip("仓库里没有参照实现写出的模型")


# ------------------------------------------------------- 统一集合分裂/剪枝 --


def test_split_unified_offsets_copies_so_they_are_not_coincident() -> None:
    """分裂副本必须**偏离**父高斯：原地复制等于没加密。

    旧实现原样复制 xyz，副本与父高斯完全重合；配合"优化器指向旧参数"
    的 bug，加密阶段看起来在加高斯，实际一个都没训练（见 ISSUES）。
    """
    torch.manual_seed(0)
    av = _tiny_avatar(n=16, k=2, tex_size=8)
    before = int(av.num_gaussians)
    idx = torch.arange(8)

    added = av.split_unified(idx, jitter=0.8)

    assert added == 8
    assert int(av.num_gaussians) == before + 8
    parent = av.xyz.data[:before][idx]
    copy = av.xyz.data[before:]
    assert not torch.allclose(parent, copy), "副本位置不应与父高斯完全一致"
    # 偏移量应受父尺度约束：不超过父尺度（tensor 为 log 尺度）的若干倍
    sigma = torch.exp(av.scaling.data[:before][idx])
    offset = (copy - parent).abs()
    assert float((offset > 4.0 * sigma).float().mean()) == 0.0


def test_split_unified_keeps_basis_rows_consistent() -> None:
    av = _tiny_avatar(n=8, k=2, tex_size=8)
    idx = torch.tensor([1, 4])
    av.split_unified(idx, jitter=0.5)
    n = int(av.num_gaussians)
    assert av.xyz_b.shape == (av.num_basis, n, 3)
    assert av.rotation_b.shape == (av.num_basis, n, 4)
    assert av.color_b.shape == (av.num_basis, n, 1, 3)
    assert torch.equal(av.xyz_b[:, 8:], av.xyz_b[:, idx])


def test_prune_unified_drops_masked_rows() -> None:
    av = _tiny_avatar(n=8, k=2, tex_size=8)
    keep = torch.ones(8, dtype=torch.bool)
    keep[[2, 5]] = False
    removed = av.prune_unified(keep)
    assert removed == 2
    assert int(av.num_gaussians) == 6


# ------------------------------------------------------------ 盘状初始化 --


def test_disc_init_sets_anisotropic_scaling() -> None:
    """给了切向/法向尺度就按 (t, t, n) 逐轴初始化，否则保持各向同性。"""
    from live3dgsavatar.core.avatar import AvatarConfig, GaussianAvatar
    from live3dgsavatar.core.deform.bind import Binding

    binding = Binding(
        face_id=torch.zeros(4, dtype=torch.long),
        face_bary=torch.tensor([[1.0, 0.0, 0.0]] * 4),
        valid_mask=torch.ones(4, dtype=torch.bool),
    )

    disc_cfg = AvatarConfig(tex_size=8, num_basis_in=8, num_basis_blend=2,
                            mlp_hidden=(), init_scale_tangent=0.0022,
                            init_scale_normal=0.0004)
    av = GaussianAvatar(disc_cfg, binding)
    scale_mm = (av.scaling.detach().exp() * 1000).numpy()
    assert np.allclose(scale_mm[:, :2], 2.2, atol=1e-3)
    assert np.allclose(scale_mm[:, 2], 0.4, atol=1e-3)

    iso_cfg = AvatarConfig(tex_size=8, num_basis_in=8, num_basis_blend=2,
                           mlp_hidden=(), init_scaling=0.0008)
    av_iso = GaussianAvatar(iso_cfg, binding)
    assert np.allclose((av_iso.scaling.detach().exp() * 1000).numpy(), 0.8,
                       atol=1e-4)


def test_disc_init_keys_roundtrip_through_yaml(tmp_path) -> None:
    from live3dgsavatar.core.avatar import AvatarConfig

    path = tmp_path / "config.yaml"
    path.write_text(
        "model:\n"
        "  tex_size: 8\n"
        "  num_basis_in: 8\n"
        "  num_basis_blend: 2\n"
        "  mlp_hidden: []\n"
        "  init_scale_tangent: 0.0022\n"
        "  init_scale_normal: 0.0004\n",
        encoding="utf-8")

    cfg = AvatarConfig.from_yaml(path)
    assert cfg.init_scale_tangent == pytest.approx(0.0022)
    assert cfg.init_scale_normal == pytest.approx(0.0004)

    # 缺省（老 config）保持 None → 各向同性
    legacy = tmp_path / "legacy.yaml"
    legacy.write_text("model:\n  tex_size: 8\n  num_basis_blend: 2\n",
                      encoding="utf-8")
    old = AvatarConfig.from_yaml(legacy)
    assert old.init_scale_tangent is None
    assert old.init_scale_normal is None


def test_split_unified_quad_pattern_is_deterministic_and_offsets() -> None:
    """quad 方向 = 切平面 4 个正交方向轮转，用于把一个 texel 细分成 4 格。"""
    torch.manual_seed(0)
    av = _tiny_avatar(n=8, k=2, tex_size=8)
    before = int(av.num_gaussians)
    idx = torch.arange(4)

    av.split_unified(idx, jitter=1.0, pattern="quad", phase=0)
    parent = av.xyz.data[:before][idx]
    copy = av.xyz.data[before:]
    assert not torch.allclose(parent, copy)
    # 切平面内 (x, y) 有位移，法向 (z) 不动
    assert float((copy[..., 2] - parent[..., 2]).abs().max()) == pytest.approx(0.0, abs=1e-7)
    # 4 个副本方向两两不同
    d = copy - parent
    assert len({tuple(v.round(decimals=6).tolist()) for v in d}) == 4
