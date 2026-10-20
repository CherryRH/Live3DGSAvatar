"""数据集发现：列出可用来**驱动**的数据集（**不需要 GPU**）。

界面要能"用任意已有数据集驱动当前模型"，就得先知道有哪些、各有多少帧、
能不能提供原始画面。这些只依赖目录结构和一个 npz 的 shape —— 这里守三件事：

1. **帧数读得对**：manifest 读 npz 的 shape（mmap，不加载数据）
2. **不可用的目录被跳过**：界面上不该出现"选了却驱动不了"的选项
3. **错误信息可执行**：缺 `input/` 要说清"RDF 模式需要原始画面"
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from live3dgsavatar.config import load_config, resolve_data_dir
from live3dgsavatar.data import datasets as D

_DATA = Path.home() / "Datasets" / "INSTA"


def _cfg_with_root(tmp_path: Path):
    cfg = load_config()
    cfg.set("paths.data_root", str(tmp_path))
    return cfg


def _make_manifest(root: Path, name: str, frames: int,
                   *, with_input: bool = True) -> Path:
    """造一个 manifest 格式的数据集目录（不真的写满帧数据）。"""
    d = root / name
    d.mkdir(parents=True)
    np.savez(d / "render_inputs.npz",
             mesh=np.zeros((frames, 4, 3), np.float32),
             blend_weight=np.zeros((frames, 129), np.float32))
    if with_input:
        (d / "input").mkdir()
    (d / "images").mkdir()
    return d


# ------------------------------------------------------------ 基本解析 --


def test_inspect_reads_manifest_frame_count(tmp_path) -> None:
    _make_manifest(tmp_path, "crh_1", 7)
    info = D.inspect(_cfg_with_root(tmp_path), "crh_1")

    assert info is not None
    assert info.frames == 7
    assert info.format == "manifest"
    assert info.has_input is True
    assert info.has_images is True
    assert info.can_drive is True


def test_inspect_reads_checkpoint_frame_count(tmp_path) -> None:
    d = tmp_path / "old"
    (d / "checkpoint").mkdir(parents=True)
    for i in range(3):
        (d / "checkpoint" / f"{i:05d}.frame").write_bytes(b"x")

    info = D.inspect(_cfg_with_root(tmp_path), "old")
    assert info is not None
    assert info.frames == 3
    assert info.format == "checkpoint"
    assert info.can_drive is True


def test_inspect_returns_none_for_missing_dataset(tmp_path) -> None:
    assert D.inspect(_cfg_with_root(tmp_path), "nope") is None


def test_list_datasets_skips_unusable_dirs(tmp_path) -> None:
    """没有 npz 也没有 checkpoint 的目录不能出现在列表里。"""
    _make_manifest(tmp_path, "good", 5)
    (tmp_path / "empty").mkdir()
    (tmp_path / "notes.txt").write_text("not a dataset")

    names = [i.name for i in D.list_datasets(_cfg_with_root(tmp_path))]
    assert names == ["good"], f"实际 {names}"


def test_list_datasets_is_sorted_and_serializable(tmp_path) -> None:
    for name in ("b", "a"):
        _make_manifest(tmp_path, name, 2)

    infos = D.list_datasets(_cfg_with_root(tmp_path))
    assert [i.name for i in infos] == ["a", "b"]
    payload = infos[0].to_dict()
    assert payload["name"] == "a" and payload["frames"] == 2
    assert payload["can_drive"] is True


# ------------------------------------------------------------ 错误路径 --


def test_resolve_dataset_dir_reports_missing(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="不存在"):
        D.resolve_dataset_dir(_cfg_with_root(tmp_path), "ghost")


def test_resolve_dataset_dir_reports_unusable(tmp_path) -> None:
    """目录存在但没有数据 → 错误信息要指出缺什么。"""
    (tmp_path / "hollow").mkdir()
    with pytest.raises(ValueError, match="render_inputs.npz|checkpoint"):
        D.resolve_dataset_dir(_cfg_with_root(tmp_path), "hollow")


# ------------------------------------------------------------- 真实数据 --


def test_real_datasets_are_discoverable() -> None:
    """在有数据集的机器上，至少要能列出本项目的两个对象。"""
    cfg = load_config()
    # 默认 `data_root` 是仓库内的 `datasets/`（可能为空）。若这里不可用，
    # 回退到常见的采集目录，让这条测试在有数据的机器上仍然生效。
    if not D.list_datasets(cfg) and _DATA.is_dir():
        cfg.set("paths.data_root", str(_DATA))
    infos = {i.name: i for i in D.list_datasets(cfg)}
    if not infos:
        pytest.skip("data_root 下没有数据集可用")

    for name in ("crh_1", "crh_2"):
        if name not in infos:
            continue
        info = infos[name]
        assert info.frames > 0
        assert info.format == "manifest"
        assert info.has_input is True, "crh 系列应有 input/ 供 RDF 取帧"

    # duda 是 checkpoint 格式且没有 input/ —— 只能"数据集模式"驱动
    if "duda" in infos:
        assert infos["duda"].has_input is False
        assert infos["duda"].format == "checkpoint"


def test_resolve_data_dir_matches_named_dataset(tmp_path) -> None:
    """`resolve_data_dir`（跟随 subject）应与按名字解析一致。"""
    _make_manifest(tmp_path, "alice", 3)
    cfg = _cfg_with_root(tmp_path)
    cfg.set("subject", "alice")
    assert resolve_data_dir(cfg) == D.resolve_dataset_dir(cfg, "alice")

# --------------------------------------------------------- 后端接口 --


def test_datasets_endpoint_lists_datasets_and_current() -> None:
    """`GET /api/datasets` 要给出列表、当前选择与"哪个是模型自带"。"""
    import asyncio

    from fastapi import FastAPI
    from fastapi.routing import APIRoute

    from live3dgsavatar.app.server import _install_routes
    from live3dgsavatar.app.session import Session

    cfg = load_config(files=("system.yaml", "render.yaml", "train.yaml"), env=False)
    app = FastAPI()
    session = Session(cfg)
    _install_routes(app, session)

    endpoint = None
    for route in app.routes:
        if isinstance(route, APIRoute) and route.path == "/api/datasets":
            endpoint = route.endpoint
    assert endpoint is not None, "缺少 /api/datasets 路由"

    payload = asyncio.run(endpoint())
    assert "datasets" in payload and "model_subject" in payload
    assert isinstance(payload["datasets"], list)
    for item in payload["datasets"]:
        assert {"name", "frames", "format", "has_input", "is_model"} <= set(item)
    assert payload["current"] == "", "默认跟随模型，应为空串"


# --------------------------------------------------------- 模型发现 --


def test_list_models_finds_ply_dirs_only(tmp_path) -> None:
    """只列含 `model.ply` 的目录；缺 ply 的不出现（选了也加载不了）。"""
    from live3dgsavatar.data import models as M

    good = tmp_path / "alice" / "run1"
    good.mkdir(parents=True)
    (good / "model.ply").write_bytes(b"ply")
    (good / "config.yaml").write_text("model: {}")

    no_ply = tmp_path / "alice" / "run2"
    no_ply.mkdir(parents=True)

    cfg = load_config(files=("system.yaml", "render.yaml", "train.yaml"), env=False)
    cfg.set("paths.models_dir", str(tmp_path))

    infos = M.list_models(cfg)
    assert [i.work_name for i in infos] == ["run1"]
    assert infos[0].subject == "alice"
    assert infos[0].has_config is True
    assert infos[0].to_dict()["label"] == "alice / run1"


def test_list_models_real_repo() -> None:
    """本仓库里至少能发现一个模型（用于面板下拉）。"""
    from live3dgsavatar.data import models as M

    cfg = load_config(files=("system.yaml", "render.yaml", "train.yaml"), env=False)
    infos = M.list_models(cfg)
    if not infos:
        pytest.skip("models/ 下没有模型")
    for i in infos:
        assert i.path.is_file() and i.path.name == "model.ply"


def test_models_endpoint_lists_models_and_current() -> None:
    import asyncio

    from fastapi import FastAPI
    from fastapi.routing import APIRoute

    from live3dgsavatar.app.server import _install_routes
    from live3dgsavatar.app.session import Session

    cfg = load_config(files=("system.yaml", "render.yaml", "train.yaml"), env=False)
    app = FastAPI()
    _install_routes(app, Session(cfg))

    endpoint = None
    for route in app.routes:
        if isinstance(route, APIRoute) and route.path == "/api/models":
            endpoint = route.endpoint
    assert endpoint is not None, "缺少 /api/models 路由"

    payload = asyncio.run(endpoint())
    assert "models" in payload and "current" in payload
    assert {"subject", "work_name"} <= set(payload["current"])
