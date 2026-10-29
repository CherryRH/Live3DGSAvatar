"""`scripts/train.py` 里可离线验证的部分：结构配置的解析与回退。

真实训练要 GPU（CUDA 扩展 + nvdiffrast），这里只测"配置从哪来"。
这条回退链是有来由的：训练**新模型**时 `<models_dir>/<subject>/<work>/`
还不存在，若直接走 `resolve_model_config` 会报"还没有选择模型" ——
那句错误会让人以为数据或模型选错了，其实是"还没训过"。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from script_loader import load_script

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "train.py"


def _load():
    return load_script(SCRIPT.name)


def _write(path: Path, text: str = "model:\n  tex_size: 128\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ------------------------------------------------------------------ 回退链 --


def test_explicit_config_wins(tmp_path) -> None:
    m = _load()
    explicit = _write(tmp_path / "custom.yaml")
    models = tmp_path / "models"
    _write(models / "s" / "w" / "config.yaml")
    assert m.resolve_structure_config(models, "s", "w", explicit=explicit) == explicit


def test_uses_matching_work_config_when_present(tmp_path) -> None:
    m = _load()
    models = tmp_path / "models"
    direct = _write(models / "s" / "w" / "config.yaml")
    _write(models / "s" / "other" / "config.yaml")
    assert m.resolve_structure_config(models, "s", "w") == direct


def test_falls_back_to_sibling_work(tmp_path) -> None:
    """训练新版本时没有 `<work>/config.yaml`，用同对象下任一版本的结构。"""
    m = _load()
    models = tmp_path / "models"
    sibling = _write(models / "s" / "test" / "config.yaml")
    assert m.resolve_structure_config(models, "s", "smoke") == sibling


def test_falls_back_to_repo_default(tmp_path) -> None:
    m = _load()
    models = tmp_path / "models"
    models.mkdir()
    default = _write(tmp_path / "configs" / "model.yaml")
    assert m.resolve_structure_config(models, "s", "w",
                                      repo_default=default) == default


def test_reports_clearly_when_nothing_found(tmp_path) -> None:
    """全都没有时要说明"训练新模型需要结构配置"，而不是"没选模型"。"""
    m = _load()
    models = tmp_path / "models"
    models.mkdir()
    with pytest.raises(FileNotFoundError, match="model-config"):
        m.resolve_structure_config(models, "s", "w")


def test_explicit_missing_file_is_reported(tmp_path) -> None:
    m = _load()
    with pytest.raises(FileNotFoundError, match="不存在"):
        m.resolve_structure_config(tmp_path, "s", "w",
                                   explicit=tmp_path / "nope.yaml")


# --------------------------------------------------------------- 仓库模板 --


def test_repo_template_is_valid() -> None:
    """`configs/model.yaml` 必须能被 `AvatarConfig` 读出来。"""
    from live3dgsavatar.core.avatar import AvatarConfig

    path = REPO_ROOT / "configs" / "model.yaml"
    if not path.is_file():
        pytest.skip("没有仓库模板")
    cfg = AvatarConfig.from_yaml(path)
    assert cfg.tex_size > 0 and cfg.num_basis_in > 0 and cfg.num_basis_blend > 0


# --------------------------------------------------------------- 模型落地 --


def test_promote_copies_model_structure_and_report(tmp_path) -> None:
    """把训练产物装进 `models/<对象>/<版本>/`，渲染侧才认得出来。"""
    m = _load()
    src = tmp_path / "out"
    src.mkdir()
    (src / "model.ply").write_bytes(b"ply-data")
    (src / "config.yaml").write_text("model:\n  tex_size: 256\n", encoding="utf-8")
    (src / "report.json").write_text("{\"best_psnr\": 29.0}", encoding="utf-8")

    target = tmp_path / "models" / "s" / "w"
    out = m.promote_model(source_ply=src / "model.ply",
                          structure_config=src / "config.yaml",
                          report=src / "report.json", target=target)

    assert out == target
    assert (target / "model.ply").read_bytes() == b"ply-data"
    assert (target / "config.yaml").is_file()
    assert (target / "report.json").is_file()


def test_promote_refuses_to_overwrite_without_force(tmp_path) -> None:
    """覆盖已有模型是破坏性操作，必须显式 --force。"""
    m = _load()
    src = tmp_path / "model.ply"
    src.write_bytes(b"x")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model: {}\n", encoding="utf-8")
    target = tmp_path / "models" / "s" / "w"
    m.promote_model(source_ply=src, structure_config=cfg, report=None, target=target)

    with pytest.raises(FileExistsError, match="force"):
        m.promote_model(source_ply=src, structure_config=cfg, report=None,
                        target=target)
    # 加了 force 就可以
    m.promote_model(source_ply=src, structure_config=cfg, report=None,
                    target=target, force=True)


def test_promote_reports_missing_source(tmp_path) -> None:
    m = _load()
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model: {}\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="找不到"):
        m.promote_model(source_ply=tmp_path / "nope.ply",
                        structure_config=cfg, report=None,
                        target=tmp_path / "models" / "s" / "w")


def test_promote_without_report_is_allowed(tmp_path) -> None:
    m = _load()
    src = tmp_path / "model.ply"
    src.write_bytes(b"x")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model: {}\n", encoding="utf-8")
    target = tmp_path / "models" / "s" / "w"
    m.promote_model(source_ply=src, structure_config=cfg, report=None, target=target)
    assert not (target / "report.json").exists()


# ------------------------------------------------------ 数据集构建脚本 --


def test_build_dataset_new_dataset_from_baked_source(tmp_path) -> None:
    """已烘焙的来源 → 数据集（images/ + render_inputs.npz）。

    这是 `build_dataset.sh` 的后半段：前端只负责拿到跟踪结果，整理交给这里。
    """
    import numpy as np

    m = load_script("build_dataset.py")
    src = tmp_path / "src"
    (src / "images").mkdir(parents=True)
    n, v = 3, 6
    np.savez(src / "render_inputs.npz",
             mesh=np.zeros((n, v, 3), np.float32),
             blend_weight=np.zeros((n, 129), np.float32),
             K=np.eye(3), w2v=np.eye(4),
             width=np.int64(8), height=np.int64(8),
             names=np.array([f"{i:05d}" for i in range(n)]))
    from PIL import Image

    for i in range(n):
        Image.fromarray(np.zeros((8, 8, 4), np.uint8)).save(
            src / "images" / f"{i:05d}.png")

    out = tmp_path / "datasets" / "ds1"
    m.build(source=src, out_dir=out, flame_dir=tmp_path,
            base_dataset=None, limit=0, force=False)

    assert (out / "images").is_dir()
    with np.load(out / "render_inputs.npz", allow_pickle=True) as d:
        for key in ("mesh", "blend_weight", "K", "w2v", "width", "height", "names"):
            assert key in d.files, key
        assert int(d["mesh"].shape[0]) == n


def test_build_dataset_refuses_to_overwrite(tmp_path) -> None:
    m = load_script("build_dataset.py")
    out = tmp_path / "ds"
    out.mkdir()
    (out / "render_inputs.npz").write_bytes(b"x")
    with pytest.raises(FileExistsError, match="force"):
        m.build(source=tmp_path, out_dir=out, flame_dir=tmp_path,
                base_dataset=None, force=False)


def test_build_dataset_requires_images_for_every_frame(tmp_path) -> None:
    """图像与跟踪结果必须一一对应：缺一张就报错，不能悄悄跳过。"""
    import numpy as np

    m = load_script("build_dataset.py")
    src = tmp_path / "src"
    (src / "images").mkdir(parents=True)
    np.savez(src / "render_inputs.npz",
             mesh=np.zeros((2, 6, 3), np.float32),
             blend_weight=np.zeros((2, 129), np.float32),
             K=np.eye(3), w2v=np.eye(4),
             width=np.int64(8), height=np.int64(8),
             names=np.array(["0", "1"]))
    with pytest.raises(FileNotFoundError, match="缺少第 0 帧"):
        m.build(source=src, out_dir=tmp_path / "out", flame_dir=tmp_path,
                base_dataset=None, force=False)


def test_build_dataset_renames_images_to_five_digits(tmp_path) -> None:
    """构建出的数据集图像一律 `00000.png …`，与 npz 行序一致。"""
    import numpy as np

    m = load_script("build_dataset.py")
    src = tmp_path / "src"
    (src / "images").mkdir(parents=True)
    n = 2
    np.savez(src / "render_inputs.npz",
             mesh=np.zeros((n, 6, 3), np.float32),
             blend_weight=np.zeros((n, 129), np.float32),
             K=np.eye(3), w2v=np.eye(4),
             width=np.int64(8), height=np.int64(8),
             names=np.array(["frame_1", "frame_2"]))
    from PIL import Image

    for name in ("frame_1", "frame_2"):
        Image.fromarray(np.zeros((8, 8, 4), np.uint8)).save(
            src / "images" / f"{name}.png")

    out = tmp_path / "ds"
    m.build(source=src, out_dir=out, flame_dir=tmp_path, base_dataset=None)
    assert sorted(p.name for p in (out / "images").iterdir()) == [
        "00000.png", "00001.png"]
    with np.load(out / "render_inputs.npz", allow_pickle=True) as d:
        assert list(d["names"]) == ["00000", "00001"]


# ------------------------------------------------- SyncFix 输入的准备工作 --


def test_rotation_y_is_proper_rotation() -> None:
    """新视角靠绕 y 轴转相机实现，必须是正交、行列式为 1 的旋转。"""
    import torch

    m = load_script("make_syncfix_inputs.py")
    R = m.rotation_y(30.0)
    assert R.shape == (4, 4)
    assert torch.allclose(R[:3, :3] @ R[:3, :3].T, torch.eye(3), atol=1e-6)
    assert abs(float(torch.det(R[:3, :3])) - 1.0) < 1e-6
    assert torch.allclose(m.rotation_y(0.0), torch.eye(4))
    # 不改变相机中心到原点的距离（只是绕头转）
    assert torch.allclose(R[:3, 3], torch.zeros(3))


def test_to_uint8_range_and_shape() -> None:
    import torch

    m = load_script("make_syncfix_inputs.py")
    x = torch.zeros(1, 3, 4, 5)
    x[0, 0] = 1.0
    img = m.to_uint8(x)
    assert img.shape == (4, 5, 3) and img.dtype.name == "uint8"
    assert img[..., 0].max() == 255 and img[..., 1].max() == 0
