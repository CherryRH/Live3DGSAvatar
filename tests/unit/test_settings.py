"""设置面板的注册表与持久化（**不需要 GPU / FastAPI**）。

这里守的是三类**会静默失效**的错误：

1. **漂移**：`FIELDS` 漏了某个配置项 → 面板上看不到它，
   用户以为"没有这个参数"；或多了个不存在的键 → 保存时写入无效配置。
2. **假放开**：某个键标成 `editable=True`，但运行中改了不生效 →
   用户以为调好了，实际没变。
3. **越权写盘**：持久化写到带注释的 `system.yaml`，把文档变成机器产物。
"""

from __future__ import annotations

import pytest

from live3dgsavatar.app import settings as S
from live3dgsavatar.config import _FALLBACK, load_config


# ------------------------------------------------------------ 注册表完整性 --


def test_fields_cover_every_config_key_exactly() -> None:
    """`FIELDS` 必须与内置兜底键**完全一致**（不多不少）。

    多了：保存时会写进一个 `Config` 不认识/schema 没校验的键。
    少了：面板上看不到它，用户不知道有这个参数。
    """
    registered = set(S.BY_KEY)
    known = set(_FALLBACK)
    assert registered == known, (
        f"注册表与配置键不一致\n"
        f"  缺（面板看不到）：{sorted(known - registered)}\n"
        f"  多（配置里没有）：{sorted(registered - known)}")


def test_locked_fields_must_explain_why() -> None:
    """锁定字段必须给出 reason —— 否则用户只看到灰掉的框，不知道为什么。"""
    with pytest.raises(ValueError, match="reason"):
        S.Field(key="x", label="X", group="render", kind="int", editable=False)

    for f in S.FIELDS:
        if not f.editable:
            assert f.reason, f"{f.key} 锁定了但没有 reason"


def test_editable_keys_are_settable_on_config() -> None:
    """每个可编辑键都必须能在 `Config` 上真正读写（防拼错路径）。"""
    cfg = load_config()
    for key in S.EDITABLE_KEYS:
        value = cfg.get(key)
        assert value is not None, f"可编辑键 {key} 在配置里取不到值"
        cfg.set(key, value)
        assert cfg.get(key) == value, f"可编辑键 {key} 写入后取回不一致"


# ------------------------------------------------------------------ 校验 --


def test_validate_rejects_unknown_and_locked_keys() -> None:
    with pytest.raises(ValueError, match="未知"):
        S.validate_patch({"render.nonexistent": 1})

    # 锁定的键：错误信息里要带上原因，而不是只说"不允许"
    f = S.BY_KEY["model.network.tex_size"]
    with pytest.raises(ValueError) as exc:
        S.validate_patch({f.key: 512})
    assert f.label in str(exc.value)
    assert "锁定" in str(exc.value)


@pytest.mark.parametrize("key,value", [
    ("render.target_fps", -1),                 # 小于下界
    ("render.target_fps", 10_000),             # 大于上界
    ("render.scaling_modifier", 0.0),       # 下界外
    ("train.online_update.steps", 0),       # 下界外
    ("train.online_update.color_l1", "maybe"),   # 布尔值非法
    ("render.background", [0.0, 0.0]),      # 分量不足
    ("render.background", [0.0, 0.0, 2.0]),  # 分量越界
])
def test_validate_rejects_out_of_range(key: str, value) -> None:
    with pytest.raises(ValueError):
        S.validate_patch({key: value})


def test_validate_coerces_types() -> None:
    """前端传的都是 JSON，`"true"` 与 `3` 这类要能正确转换。"""
    out = S.validate_patch({
        "train.online_update.enabled": "false",
        "train.online_update.steps": 50,
        "render.target_fps": 30,          # 整数字面量 → float 字段
        "render.background": [1, 0, 0],
    })
    assert out["train.online_update.enabled"] is False
    assert out["train.online_update.steps"] == 50
    assert isinstance(out["render.target_fps"], float)
    assert out["render.background"] == [1.0, 0.0, 0.0]


def test_validate_rejects_fractional_int() -> None:
    with pytest.raises(ValueError, match="整数"):
        S.validate_patch({"train.online_update.steps": 3.5})


# ---------------------------------------------------------------- 快照 --


def test_snapshot_is_sorted_by_group_then_order() -> None:
    """前端直接按返回顺序渲染，所以排序必须在这里就正确。"""
    snap = S.snapshot(load_config())
    group_rank = {g["key"]: g["order"] for g in S.groups_payload()}

    ranks = [(group_rank[f["group"]], f["order"]) for f in snap]
    assert ranks == sorted(ranks), "快照没有按「分组 → 组内 order」排序"

    # 至少三个组都出现，且每组内部 order 递增
    seen = {}
    for f in snap:
        seen.setdefault(f["group"], []).append(f["order"])
    assert set(seen) == {"render", "train", "system"}
    for group, orders in seen.items():
        assert orders == sorted(orders), f"{group} 组内顺序不对"


def test_snapshot_exposes_metadata_needed_by_ui() -> None:
    """前端只靠这些字段渲染控件；缺一个就会显示成错误的控件。"""
    required = {"key", "label", "group", "kind", "editable", "reason",
                "help", "unit", "value"}
    for f in S.snapshot(load_config()):
        assert required <= set(f), f"{f['key']} 缺少字段：{required - set(f)}"
        if f["kind"] == "enum":
            assert f["choices"], f"{f['key']} 是 enum 但没有候选值"
        if f["kind"] in ("int", "float"):
            assert f["minimum"] is not None or f["editable"] is False


def test_snapshot_values_are_json_serializable() -> None:
    import json

    json.dumps(S.snapshot(load_config()))     # 不抛异常即可


# -------------------------------------------------------------- 持久化 --


def test_overrides_path_is_local_yaml() -> None:
    """持久化必须写 local.yaml：它在合并链末尾且已 gitignore。

    不能写 system.yaml / render.yaml / train.yaml —— 那些带大量注释，
    `yaml.safe_dump` 会把注释全丢掉。
    """
    p = S.overrides_path()
    assert p.name == "local.yaml"
    assert p.parent.name == "configs"


def test_save_overrides_roundtrip(tmp_path) -> None:
    target = tmp_path / "local.yaml"
    S.save_overrides({"render.target_fps": 30.0,
                      "train.online_update.enabled": False}, path=target)
    assert target.exists()
    assert S.load_overrides(target) == {
        "render.target_fps": 30.0,
        "train.online_update.enabled": False,
    }


def test_save_overrides_merges_instead_of_clobbering(tmp_path) -> None:
    """第二次保存不能把第一次写的内容抹掉（这是最容易踩的持久化 bug）。"""
    target = tmp_path / "local.yaml"
    S.save_overrides({"render.target_fps": 30.0}, path=target)
    S.save_overrides({"train.online_update.steps": 60}, path=target)

    got = S.load_overrides(target)
    assert got["render.target_fps"] == 30.0, "第一次保存的值被覆盖丢失"
    assert got["train.online_update.steps"] == 60


def test_overrides_are_applied_by_load_config(tmp_path) -> None:
    """写进 local.yaml 的值必须真的被 `load_config` 读到。

    用临时 configs/ 目录（复制真实 yaml + 写入覆盖），**不碰仓库的
    configs/local.yaml** —— 那里可能有用户自己的设置。
    """
    import shutil

    from live3dgsavatar.config import repo_root

    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    # ⚠️ **只复制三份基础配置**，不复制 local.yaml ——
    #    那是用户自己的设置，复制过来会让断言随本机状态变化。
    for name in ("system.yaml", "render.yaml", "train.yaml"):
        shutil.copy(repo_root() / "configs" / name, cfg_dir / name)
    # 覆盖值必须胜过 render.yaml 里的默认 60
    S.save_overrides({"render.target_fps": 24.0}, path=cfg_dir / "local.yaml")

    cfg = load_config(config_dir=cfg_dir, env=False)
    assert cfg.get("render.target_fps") == 24.0, "local.yaml 没有覆盖默认值"
    assert cfg.get("render.scaling_modifier") == 1.0, "未覆盖的项应保持默认"


def test_missing_overrides_returns_empty(tmp_path) -> None:
    assert S.load_overrides(tmp_path / "nope.yaml") == {}


def test_corrupt_overrides_does_not_crash(tmp_path) -> None:
    """坏掉的 local.yaml 不应让整个服务起不来。"""
    target = tmp_path / "local.yaml"
    target.write_text("this: [is: not, valid", encoding="utf-8")
    assert S.load_overrides(target) == {}
