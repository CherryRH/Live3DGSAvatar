"""GUI 的模型结构解析顺序（不需要 GPU）。

顺序必须是：PLY 内嵌 gaussian_config → 模型目录 config.yaml →
configs/render.yaml 的 model.network。`render_test` 早已按这个顺序，
GUI 若继续只读 render.yaml，就会在换结构（tex_size/K）的模型上“评测能开、
GUI 开不了”。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from live3dgsavatar.app.session import resolve_avatar_config
from live3dgsavatar.config import Config
from live3dgsavatar.core.avatar import AvatarConfig
from live3dgsavatar.core.io.ply import CONFIG_COMMENT_PREFIX


def _write_ply(path: Path, config: dict | None = None) -> None:
    element = PlyElement.describe(np.zeros(1, dtype=[("x", "f4")]), "vertex")
    comments = ([CONFIG_COMMENT_PREFIX + json.dumps(config)]
                if config is not None else [])
    PlyData([element], text=False, comments=comments).write(str(path))


def _sidecar(path: Path, *, tex_size: int, num_basis_in: int,
             num_basis_blend: int) -> Path:
    path.write_text(
        "model:\n"
        f"  tex_size: {tex_size}\n"
        f"  num_basis_in: {num_basis_in}\n"
        f"  num_basis_blend: {num_basis_blend}\n"
        "  mlp_hidden: [64, 64]\n"
        "  use_weight_proj: true\n",
        encoding="utf-8")
    return path


def _fallback_cfg(tmp_path: Path) -> Config:
    return Config({
        "subject": "s", "work_name": "w",
        "paths": {"models_dir": str(tmp_path / "models")},
        "model": {"network": {
            "tex_size": 64, "num_basis_in": 32, "num_basis_blend": 8,
            "mlp_hidden": [32, 32], "use_weight_proj": False,
        }},
    })


def test_embedded_config_wins_over_sidecar(tmp_path: Path) -> None:
    ply = tmp_path / "model.ply"
    _write_ply(ply, {"tex_size": 256, "num_basis_in": 129,
                     "num_basis_blend": 20, "mlp_hidden": [128, 128],
                     "use_weight_proj": True})
    sidecar = _sidecar(tmp_path / "config.yaml", tex_size=128,
                       num_basis_in=64, num_basis_blend=16)

    cfg = resolve_avatar_config(_fallback_cfg(tmp_path), ply, model_cfg=sidecar)
    assert cfg.tex_size == 256 and cfg.num_basis_in == 129
    assert cfg.num_basis_blend == 20 and cfg.mlp_hidden == (128, 128)


def test_sidecar_config_wins_over_render_yaml(tmp_path: Path) -> None:
    ply = tmp_path / "model.ply"
    _write_ply(ply)
    sidecar = _sidecar(tmp_path / "config.yaml", tex_size=128,
                       num_basis_in=64, num_basis_blend=16)

    cfg = resolve_avatar_config(_fallback_cfg(tmp_path), ply, model_cfg=sidecar)
    assert cfg.tex_size == 128 and cfg.num_basis_in == 64
    assert cfg.num_basis_blend == 16 and cfg.mlp_hidden == (64, 64)


def test_render_yaml_is_last_resort(tmp_path: Path) -> None:
    ply = tmp_path / "model.ply"
    _write_ply(ply)
    cfg = resolve_avatar_config(_fallback_cfg(tmp_path), ply)
    assert cfg.tex_size == 64 and cfg.num_basis_in == 32
    assert cfg.num_basis_blend == 8 and cfg.use_weight_proj is False


def test_avatar_config_from_sidecar_matches_template() -> None:
    """仓库模板本身必须可被 `AvatarConfig` 读出（否则兜底路径也是坏的）。"""
    root = Path(__file__).resolve().parents[2]
    cfg = AvatarConfig.from_yaml(root / "configs" / "model.yaml")
    assert cfg.tex_size == 256 and cfg.num_basis_in == 129
    assert cfg.num_basis_blend == 20
