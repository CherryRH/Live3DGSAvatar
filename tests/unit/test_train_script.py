"""`scripts/train.py` 里可离线验证的部分：结构配置的解析与回退。

真实训练要 GPU（CUDA 扩展 + nvdiffrast），这里只测"配置从哪来"。
这条回退链是有来由的：训练**新模型**时 `<models_dir>/<subject>/<work>/`
还不存在，若直接走 `resolve_model_config` 会报"还没有选择模型" ——
那句错误会让人以为数据或模型选错了，其实是"还没训过"。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "train.py"


def _load():
    if "_train_script" in sys.modules:
        return sys.modules["_train_script"]
    spec = importlib.util.spec_from_file_location("_train_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_train_script"] = mod
    spec.loader.exec_module(mod)
    return mod


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


def test_promote_with_custom_filename_keeps_both_models(tmp_path) -> None:
    """增强模型用 `model_aug.ply` 落盘，与常规 `model.ply` 并存便于对比。"""
    m = _load()
    src = tmp_path / "model.ply"
    src.write_bytes(b"plain")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model: {}\n", encoding="utf-8")
    target = tmp_path / "models" / "s" / "w"

    m.promote_model(source_ply=src, structure_config=cfg, report=None,
                    target=target)
    aug = tmp_path / "model_aug.ply"
    aug.write_bytes(b"aug")
    m.promote_model(source_ply=aug, structure_config=cfg, report=None,
                    target=target, filename="model_aug.ply")

    assert (target / "model.ply").read_bytes() == b"plain"
    assert (target / "model_aug.ply").read_bytes() == b"aug"


def test_promote_is_idempotent_when_config_already_there(tmp_path) -> None:
    """结构配置就在目标目录里时（续训场景），不能再拷一次。

    真实故障：`resolve_structure_config` 回退选中了 `models/<s>/<w>/config.yaml`
    正是 promote 的目标，`shutil.copyfile` 抛 SameFileError，训练明明跑完了
    却在最后一步失败。
    """
    m = _load()
    target = tmp_path / "models" / "s" / "w"
    target.mkdir(parents=True)
    cfg = target / "config.yaml"
    cfg.write_text("model: {}\n", encoding="utf-8")
    ply = tmp_path / "model.ply"
    ply.write_bytes(b"x")

    out = m.promote_model(source_ply=ply, structure_config=cfg, report=None,
                          target=target)

    assert out == target
    assert (target / "model.ply").read_bytes() == b"x"
    assert (target / "config.yaml").read_text(encoding="utf-8") == "model: {}\n"


def test_promote_skips_when_source_and_dest_are_same(tmp_path) -> None:
    """源 PLY 已经就在目标位置时，也不应报错。"""
    m = _load()
    target = tmp_path / "models" / "s" / "w"
    target.mkdir(parents=True)
    ply = target / "model.ply"
    ply.write_bytes(b"x")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model: {}\n", encoding="utf-8")

    m.promote_model(source_ply=ply, structure_config=cfg, report=None,
                    target=target, force=True)
    assert ply.read_bytes() == b"x"
