"""设置 API 的行为（`GET/POST /api/settings`）。

**不需要 GPU、不需要 httpx**：直接取出路由函数调用，用真实的 `Session`
（未加载模型也算就绪）与真实的 `Config`，只把持久化目标重定向到临时文件。

这里守的是三件在真实部署里会出事的东西：

1. **只应用不持久化** → 重启后设置全丢，用户以为坏了
2. **只持久化不应用** → 用户点了保存却看不到变化
3. **部分应用** → 一批里有一项非法，其余已经改了，留下"改了一半"的状态
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRoute

from live3dgsavatar.app import settings as S
from live3dgsavatar.app.server import _install_routes
from live3dgsavatar.app.session import Session
from live3dgsavatar.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------ 装配 --


#: 测试固定加载的三份配置。
#: **刻意不含 `local.yaml`** —— 那是用户自己的设置（可能存在），
#: 读了会让断言随本机状态变化，测试就不再可重复。
BASE_FILES = ("system.yaml", "render.yaml", "train.yaml")


def _config():
    return load_config(files=BASE_FILES, env=False)


def _endpoints() -> tuple[dict, Session]:
    """装配真实路由与真实 Session，返回 `{(path, method): endpoint}`。"""
    app = FastAPI()
    session = Session(_config())
    _install_routes(app, session)
    table = {}
    for route in app.routes:
        if isinstance(route, APIRoute):
            for method in route.methods:
                table[(route.path, method)] = route.endpoint
    return table, session


def _get(table) -> dict:
    return asyncio.run(table[("/api/settings", "GET")]())


def _post(table, payload: dict) -> dict:
    return json.loads(asyncio.run(
        table[("/api/settings", "POST")](payload)).body)


@contextlib.contextmanager
def _sandboxed(tmp_path: Path):
    """把持久化目标重定向到临时文件，避免污染仓库里的 `configs/local.yaml`。

    用显式上下文管理器而不是夹具：本项目的零依赖测试运行器不支持
    "夹具依赖其它夹具"，手写反而更清楚，也不依赖 pytest 特性。
    """
    target = Path(tmp_path) / "local.yaml"
    original = S.overrides_path
    S.overrides_path = lambda: target
    try:
        yield target
    finally:
        S.overrides_path = original


@contextlib.contextmanager
def _failing_save():
    """模拟"只读文件系统"：存盘抛 OSError。"""
    original = S.save_overrides

    def boom(*_a, **_k):
        raise OSError("只读文件系统")

    S.save_overrides = boom
    try:
        yield
    finally:
        S.save_overrides = original


# ------------------------------------------------------------------ GET --


def test_settings_get_exposes_every_field(tmp_path) -> None:
    with _sandboxed(tmp_path):
        body = _get(_endpoints()[0])

        assert set(body) >= {"groups", "fields", "overrides_path"}
        assert [g["key"] for g in body["groups"]] == ["render", "train", "system"]
        # **全量展示**：锁定项也在里面，否则用户以为参数不存在
        assert len(body["fields"]) == len(S.FIELDS)
        assert any(not f["editable"] for f in body["fields"])
        assert any(f["editable"] for f in body["fields"])


def test_locked_fields_carry_a_reason(tmp_path) -> None:
    """锁定的项必须带原因，前端要显示"为什么不能改"。"""
    with _sandboxed(tmp_path):
        for f in _get(_endpoints()[0])["fields"]:
            if not f["editable"]:
                assert f["reason"], f"{f['key']} 锁定但没有原因"


# ----------------------------------------------------------------- POST --


def test_post_applies_render_params_immediately(tmp_path) -> None:
    """渲染参数必须**立刻**写进运行参数，下一帧就生效。"""
    with _sandboxed(tmp_path):
        table, session = _endpoints()

        body = _post(table, {"patch": {"render.target_fps": 24,
                                       "render.background": [1, 1, 1]}})
        assert body["ok"] is True
        assert set(body["applied"]) == {"render.target_fps", "render.background"}

        assert session.params.target_fps == 24.0
        assert session.params.background == [1.0, 1.0, 1.0]
        assert session.cfg.get("render.target_fps") == 24.0


def test_post_persists_only_the_patch(tmp_path) -> None:
    """持久化只写被改的键 —— 不要复制整份配置。"""
    with _sandboxed(tmp_path) as target:
        table, _ = _endpoints()
        _post(table, {"patch": {"train.online_update.color_l1": True}})

        assert S.load_overrides(target) == {"train.online_update.color_l1": True}


def test_two_posts_accumulate(tmp_path) -> None:
    """连续保存两次不能互相覆盖。"""
    with _sandboxed(tmp_path) as target:
        table, _ = _endpoints()
        _post(table, {"patch": {"render.target_fps": 30}})
        _post(table, {"patch": {"train.online_update.steps": 60}})

        saved = S.load_overrides(target)
        assert saved["render.target_fps"] == 30.0
        assert saved["train.online_update.steps"] == 60


def test_post_returns_authoritative_render_params(tmp_path) -> None:
    """响应里要带渲染参数的**权威值**，前端据此同步镜像控件。

    为什么不塞进 `status`：status 每 4 Hz 推一次，而这些值很少变；
    塞进去是持续的带宽浪费，也会让"状态"与"配置"两种语义混在一起。
    """
    with _sandboxed(tmp_path):
        body = _post(_endpoints()[0], {"patch": {"render.scale": 0.5}})
        assert body["render"]["scale"] == 0.5
        assert body["render"]["background"] == [0.0, 0.0, 0.0]
        # status 仍然只管运行状态，不含 render
        assert "render" not in body["status"]


def test_post_returns_updated_snapshot(tmp_path) -> None:
    """响应里带回最新快照，前端据此刷新控件（不必再发一次 GET）。"""
    with _sandboxed(tmp_path):
        body = _post(_endpoints()[0], {"patch": {"render.target_fps": 15}})
        values = {f["key"]: f["value"] for f in body["fields"]}
        assert values["render.target_fps"] == 15.0


def test_post_rejects_whole_batch_if_one_item_invalid(tmp_path) -> None:
    """**整批拒绝**：不能出现"前两项改了、第三项失败"的中间状态。"""
    with _sandboxed(tmp_path) as target:
        table, session = _endpoints()
        before = session.cfg.get("render.target_fps")

        with pytest.raises(HTTPException) as exc:
            asyncio.run(table[("/api/settings", "POST")]({
                "patch": {"render.target_fps": 30,
                          "model.network.tex_size": 1024}}))
        assert exc.value.status_code == 400
        # 模型结构是锁定项 → 连合法的 target_fps 也不该被应用
        assert session.cfg.get("render.target_fps") == before
        assert not target.exists(), "校验失败不应写盘"


@pytest.mark.parametrize("payload", [
    {},
    {"patch": {}},
    {"patch": "not-a-dict"},
    {"nopatch": {"render.target_fps": 30}},
])
def test_post_rejects_empty_or_malformed(tmp_path, payload) -> None:
    with _sandboxed(tmp_path):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(_endpoints()[0][("/api/settings", "POST")](payload))
        assert exc.value.status_code == 400


def test_post_rejects_out_of_range(tmp_path) -> None:
    with _sandboxed(tmp_path) as target:
        with pytest.raises(HTTPException) as exc:
            asyncio.run(_endpoints()[0][("/api/settings", "POST")](
                {"patch": {"render.target_fps": -1}}))
        assert exc.value.status_code == 400
        assert not target.exists()


def test_save_failure_still_applies_with_warning(tmp_path) -> None:
    """存盘失败（只读文件系统）不应回滚已生效的修改，但要明确警告。"""
    with _sandboxed(tmp_path), _failing_save():
        table, session = _endpoints()
        body = _post(table, {"patch": {"render.target_fps": 20}})

        assert body["ok"] is True
        assert body["saved_to"] is None
        assert body["warning"] and "保存失败" in body["warning"]
        assert session.params.target_fps == 20.0, "已生效的修改不应被回滚"


# -------------------------------------------------------- 重启后仍然生效 --


def test_saved_overrides_take_effect_on_reload(tmp_path) -> None:
    """保存过的设置必须在下次 `load_config` 时被读到（持久化的意义）。"""
    with _sandboxed(tmp_path) as target:
        _post(_endpoints()[0], {"patch": {"render.target_fps": 42,
                                         "train.online_update.enabled": False}})

        # 用临时目录模拟 configs/：复制真实 yaml + 写入刚保存的覆盖
        cfg_dir = Path(tmp_path) / "configs"
        shutil.copytree(REPO_ROOT / "configs", cfg_dir)
        shutil.copy(target, cfg_dir / "local.yaml")

        cfg = load_config(config_dir=cfg_dir, env=False)
        assert cfg.get("render.target_fps") == 42.0
        assert cfg.get("train.online_update.enabled") is False
