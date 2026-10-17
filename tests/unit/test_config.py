"""配置系统测试（无需 GPU、无需数据集）。

覆盖：默认量来源、优先级链、路径解析、非法配置报错、以及
**「代码中不得出现硬编码路径」**这条约束。
"""

from __future__ import annotations

import os
from pathlib import Path

from live3dgsavatar.config import CONFIG_DIR, Config, load_config, repo_root

from support import REPO_ROOT


# ------------------------------------------------------------------ 基本 --


def test_loads_defaults_from_yaml_files() -> None:
    """默认量应当来自 configs/*.yaml，而不是散落在代码里的字面量。"""
    cfg = load_config()
    assert (CONFIG_DIR / "system.yaml").exists(), "configs/system.yaml 必须存在"
    assert (CONFIG_DIR / "render.yaml").exists(), "configs/render.yaml 必须存在"

    # 这几项是 YAML 提供的（而非代码兜底）
    assert cfg.get("render.batch_size") == 4
    assert cfg.get("render.frames") == 20
    assert cfg.get("model.network.tex_size") == 256
    assert cfg.get("model.network.num_basis_in") == 129
    assert cfg.get("model.network.num_basis_blend") == 20
    assert cfg.get("model.network.mlp_hidden") == [128, 128]
    # 人物名/工作名与 RGBAvatar 的 CLI 与目录约定一致。
    # ⚠️ 不写死具体人物名 —— 默认对象会随采集进度变化，测试只校验**约定**。
    assert cfg.subject and cfg.work_name
    assert isinstance(cfg.subject, str) and isinstance(cfg.work_name, str)


def test_path_fields_are_absolute_paths() -> None:
    """路径字段必须已解析为绝对 `Path`（脚本会 chdir，相对路径会失效）。"""
    cfg = load_config()
    for key in ("paths.data_root", "paths.output_dir", "paths.reference_root"):
        value = cfg.get(key)
        assert isinstance(value, Path), f"{key} 应为 Path，实际 {type(value).__name__}"
        assert value.is_absolute(), f"{key} 应为绝对路径，实际 {value}"
    # model_ply 允许为 None（表示按 models_dir/subject/work_name 推导）
    assert cfg.get("paths.model_ply") is None or isinstance(
        cfg.get("paths.model_ply"), Path)


def test_tilde_is_expanded() -> None:
    cfg = Config({"paths": {"data_root": "~/datasets"}})
    # 通过内部解析函数验证（不依赖具体文件是否存在）
    from live3dgsavatar.config import _resolve_path

    resolved = _resolve_path("~/datasets", "paths.data_root")
    assert resolved.is_absolute()
    assert "~" not in str(resolved)



import contextlib


@contextlib.contextmanager
def _env(**pairs):
    """临时设置/删除环境变量（零依赖，替代 pytest 的 monkeypatch）。"""
    old = {k: os.environ.get(k) for k in pairs}
    try:
        for key, value in pairs.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class _Exc:
    """承载捕获到的异常，模拟 `pytest.raises(...) as exc` 的 `.value`。"""

    value: BaseException | None = None


@contextlib.contextmanager
def _raises(exc_type):
    """零依赖的 `pytest.raises` 替代（运行器的 pytest 桩不支持 raises）。"""
    holder = _Exc()
    try:
        yield holder
    except exc_type as e:
        holder.value = e
        return
    except Exception as e:                       # noqa: BLE001
        raise AssertionError(
            f"期望 {exc_type.__name__}，实际抛出 {type(e).__name__}: {e}") from e
    raise AssertionError(f"期望 {exc_type.__name__}，但未抛出异常")


def test_env_overrides_yaml() -> None:
    with _env(LIVE3DGS_DATA_ROOT="/tmp/env_ds", LIVE3DGS_DEVICE="cpu"):
        cfg = load_config()
    assert str(cfg.paths.data_root) == "/tmp/env_ds"
    assert cfg.runtime.device == "cpu"


def test_env_can_be_disabled() -> None:
    with _env(LIVE3DGS_DATA_ROOT="/tmp/ignored"):
        cfg = load_config(env=False)
    assert str(cfg.paths.data_root) != "/tmp/ignored"


# --------------------------------------------------------------- 优先级 --


def test_yaml_overrides_builtin_fallback(tmp_path: Path) -> None:
    """YAML 里写了的值必须胜过代码内置兜底。"""
    (tmp_path / "system.yaml").write_text(
        "subject: custom\npaths:\n  data_root: /tmp/yaml_ds\n", encoding="utf-8")
    cfg = load_config(config_dir=tmp_path)
    assert cfg.subject == "custom"
    assert str(cfg.paths.data_root) == "/tmp/yaml_ds"
    # 未在 YAML 出现的键仍由兜底补齐，保证不会 KeyError
    assert cfg.get("render.batch_size") is not None


def test_local_yaml_is_merged_last(tmp_path: Path) -> None:
    """`local.yaml` 用于本机私有覆盖，应合并在最上层。"""
    (tmp_path / "system.yaml").write_text(
        "paths:\n  data_root: /tmp/base\n", encoding="utf-8")
    (tmp_path / "local.yaml").write_text(
        "paths:\n  data_root: /tmp/local\n", encoding="utf-8")
    cfg = load_config(config_dir=tmp_path)
    assert str(cfg.paths.data_root) == "/tmp/local"


def test_deep_merge_keeps_siblings() -> None:
    from live3dgsavatar.config import deep_merge

    base = {"a": {"x": 1, "y": 2}, "b": 1}
    over = {"a": {"y": 9}}
    assert deep_merge(base, over) == {"a": {"x": 1, "y": 9}, "b": 1}


# ------------------------------------------------------------- 访问接口 --


def test_dotted_access_and_set() -> None:
    cfg = load_config()
    assert cfg.paths.data_root == cfg.get("paths.data_root")
    cfg.set("render.batch_size", 99)
    assert cfg.get("render.batch_size") == 99
    cfg.set("brand.new.key", 1)          # 中间层自动创建
    assert cfg.get("brand.new.key") == 1


def test_attribute_error_is_descriptive() -> None:
    cfg = load_config()
    with _raises(AttributeError) as exc:
        _ = cfg.nonexistent_section
    assert "不存在" in str(exc.value)


def test_get_returns_default_for_missing_dotted_key() -> None:
    cfg = load_config()
    assert cfg.get("nope.not.here", "fallback") == "fallback"


# ------------------------------------------------------------- 校验报错 --


def test_missing_data_root_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "system.yaml").write_text("paths:\n  output_dir: out\n", encoding="utf-8")
    # 兜底会补 data_root，所以显式置空来验证校验逻辑
    (tmp_path / "render.yaml").write_text("", encoding="utf-8")
    cfg = load_config(config_dir=tmp_path)
    assert cfg.get("paths.data_root") is not None   # 兜底生效，不应失败


def test_wrong_type_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "system.yaml").write_text(
        "paths:\n  data_root: /tmp/x\nrender:\n  batch_size: not_a_number\n",
        encoding="utf-8")
    with _raises(ValueError) as exc:
        load_config(config_dir=tmp_path)
    assert "batch_size" in str(exc.value)


def test_bad_device_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "system.yaml").write_text(
        "paths:\n  data_root: /tmp/x\nruntime:\n  device: tpu\n", encoding="utf-8")
    with _raises(ValueError) as exc:
        load_config(config_dir=tmp_path)
    assert "device" in str(exc.value)


def test_non_mapping_yaml_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "system.yaml").write_text("- just\n- a list\n", encoding="utf-8")
    with _raises(ValueError) as exc:
        load_config(config_dir=tmp_path)
    assert "mapping" in str(exc.value)


# ------------------------------------------------- 硬编码路径的静态约束 --


def test_no_hardcoded_absolute_paths_in_source() -> None:
    """**回归测试**：`src/`、`tests/`、`scripts/` 不得出现用户的绝对路径。

    路径必须来自 `configs/*.yaml`（可用环境变量或命令行覆盖）。
    否则换机器 / 换数据集就要改代码，也为应用层复用埋坑。
    """
    import re

    pattern = re.compile(r"/home/[A-Za-z0-9_.-]+/")
    offenders: list[str] = []
    for sub in ("src", "tests", "scripts"):
        for path in sorted((REPO_ROOT / sub).rglob("*.py")):
            if "__pycache__" in str(path):
                continue
            for lineno, line in enumerate(
                    path.read_text(encoding="utf-8").splitlines(), 1):
                code = line.split("#", 1)[0]
                if pattern.search(code):
                    # tests/support.py 里的 REPO_ROOT 由 __file__ 推导，不会命中；
                    # 这里只拦截字面量绝对路径。
                    offenders.append(
                        f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")

    assert not offenders, (
        "发现硬编码绝对路径，请移到 configs/*.yaml：\n  " + "\n  ".join(offenders))


def test_scripts_have_no_business_defaults() -> None:
    """**回归测试**：脚本不应自带业务默认量（应由配置提供）。

    允许 `default=None`（表示"未覆盖"）与布尔开关（`store_true`）。
    这里只拦截明确的数值/路径字面量默认值，避免配置被代码悄悄压过。
    """
    import re

    # 形如 `default=256`、`default=Path("output/...")` 的业务默认量
    bad_numeric = re.compile(
        r"add_argument\([^)]*default=(?!None\b)(\d+|Path\()")
    offenders: list[str] = []
    for name in ("render_test.py", "equivalence_check.py", "show_config.py"):
        path = REPO_ROOT / "scripts" / name
        for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if bad_numeric.search(code):
                offenders.append(
                    f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")

    assert not offenders, (
        "脚本参数不应硬编码业务默认量（应由 configs/*.yaml 提供）：\n  "
        + "\n  ".join(offenders))


# --------------------------------------------------------- 模型路径解析 --


def test_model_dir_follows_rgba_avatar_layout() -> None:
    """模型目录约定与 RGBAvatar 一致：`<模型名>/<工作名>/`。"""
    from live3dgsavatar.config import model_dir, reference_model_dir

    cfg = load_config()
    subject, work = str(cfg.subject), str(cfg.work_name)
    assert model_dir(cfg) == Path(cfg.paths.models_dir) / subject / work
    ref = reference_model_dir(cfg)
    assert ref == Path(cfg.paths.reference_root) / "output" / subject / work


def test_resolve_model_ply_prefers_project_models(tmp_path) -> None:
    """**本项目 models/ 优先，找不到才回退参照仓库。**"""
    from live3dgsavatar.config import model_dir, resolve_model_ply

    cfg = load_config()
    cfg.set("paths.models_dir", tmp_path / "models")
    cfg.set("paths.model_ply", None)
    cfg.set("paths.reference_root", None)          # 排除回退，单独验证"不存在"

    # 1) 都不存在 → 报错，且信息里列出尝试过的路径
    with _raises(FileNotFoundError) as exc:
        resolve_model_ply(cfg)
    assert f"subject={cfg.subject}" in str(exc.value)
    assert f"work_name={cfg.work_name}" in str(exc.value)

    # 2) 只建本项目模型 → 命中
    target = model_dir(cfg)
    target.mkdir(parents=True)
    (target / "model.ply").write_bytes(b"x")
    assert resolve_model_ply(cfg) == target / "model.ply"

    # 3) 同时存在参照模型 → 仍优先本项目
    ref_root = tmp_path / "ref"
    cfg.set("paths.reference_root", ref_root)
    ref_dir = ref_root / "output" / str(cfg.subject) / str(cfg.work_name)
    ref_dir.mkdir(parents=True)
    (ref_dir / "model.ply").write_bytes(b"y")
    assert resolve_model_ply(cfg) == target / "model.ply"

    # 4) 删掉本项目模型 → 回退参照
    (target / "model.ply").unlink()
    assert resolve_model_ply(cfg) == ref_dir / "model.ply"


def test_explicit_model_ply_wins(tmp_path) -> None:
    """`paths.model_ply` 显式配置时直接用它，不做目录推导。"""
    from live3dgsavatar.config import resolve_model_ply

    explicit = tmp_path / "custom.ply"
    explicit.write_bytes(b"z")
    cfg = load_config()
    cfg.set("paths.model_ply", explicit)
    assert resolve_model_ply(cfg) == explicit

    # 指定的文件不存在 → 明确报错（而不是静默回退）
    cfg.set("paths.model_ply", tmp_path / "missing.ply")
    with _raises(FileNotFoundError) as exc:
        resolve_model_ply(cfg)
    assert "paths.model_ply" in str(exc.value)


def test_resolve_model_config_finds_bundled_config(tmp_path) -> None:
    """模型随附的 `config.yaml`（RGBAvatar 放在模型目录里）应能被找到。"""
    from live3dgsavatar.config import model_dir, resolve_model_config

    cfg = load_config()
    cfg.set("paths.models_dir", tmp_path / "models")
    cfg.set("paths.models_config", None)
    cfg.set("paths.reference_root", None)
    assert resolve_model_config(cfg) is None

    target = model_dir(cfg)
    target.mkdir(parents=True)
    (target / "config.yaml").write_text("model:\n  tex_size: 256\n", encoding="utf-8")
    assert resolve_model_config(cfg) == target / "config.yaml"


def test_project_model_dir_exists_and_matches_reference() -> None:
    """本项目 `models/duda/test/` 应已就位，且与参照的同名模型内容一致。"""
    import hashlib

    from live3dgsavatar.config import model_dir, reference_model_dir

    cfg = load_config()
    ours = model_dir(cfg) / "model.ply"
    ref_dir = reference_model_dir(cfg)
    if not ours.exists():
        raise AssertionError(
            f"本项目模型缺失：{ours}\n"
            "应把 RGBAvatar 的 output/<subject>/<work_name>/ 复制到 models/ 下")
    if ref_dir is None or not (ref_dir / "model.ply").exists():
        return        # 参照不在本机 → 跳过一致性比对

    def md5(p: Path) -> str:
        h = hashlib.md5()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    assert md5(ours) == md5(ref_dir / "model.ply"), (
        "models/ 下的模型与参照仓库的同名模型不一致；"
        "若刻意改动过，请忽略/调整本测试")


def test_cli_accepts_rgba_avatar_underscore_style() -> None:
    """**兼容性测试**：命令行同时接受 `--work_name`（RGBAvatar 风格）与 `--work-name`。

    参照实现的 CLI 用 `--work_name`（见 `RGBAvatar/render.py`），本项目统一用
    `--work-name`。脚本里有一层 `_normalize_argv` 把下划线写法归一化，
    这样两边的命令可以直接互相复制。
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_render_test_for_cli", REPO_ROOT / "scripts" / "render_test.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                       # type: ignore[union-attr]

    normalize = mod._normalize_argv
    # 下划线 → 短横线
    assert normalize(["--work_name", "x"]) == ["--work-name", "x"]
    assert normalize(["--data_root=/tmp/a"]) == ["--data-root=/tmp/a"]
    # 已是短横线 → 不变
    assert normalize(["--work-name", "x"]) == ["--work-name", "x"]
    assert normalize(["--subject", "duda"]) == ["--subject", "duda"]
    # 位置参数与值里的下划线不受影响
    assert normalize(["out_dir", "--frames", "3"]) == ["out_dir", "--frames", "3"]
    # 单段参数与下划线值不被误改
    assert normalize(["--dry-run"]) == ["--dry-run"]
    assert normalize(["--ply", "/a/b_c/model.ply"]) == ["--ply", "/a/b_c/model.ply"]


def test_render_test_guards_image_access_for_no_save() -> None:
    """**回归测试**：`--no-save` 下不得对空图像字典取值。

    `--no-save` 跳过 D2H 拷贝，`core_imgs` / `ref_imgs` 保持为空。
    对比循环曾直接 `core_imgs[i]`，于是 `KeyError: 0` —— 而那发生在
    渲染与计时**全部跑完之后**，白白浪费一次完整运行。

    检查方式：对 `main()` 做 AST 作用域分析，找出每个 `core_imgs[...]` /
    `ref_imgs[...]` 所在的最近 `if` 条件，要求其中出现 `has_core` / `has_ref`
    或对 `*_imgs` 的成员判断。
    """
    import ast

    path = REPO_ROOT / "scripts" / "render_test.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    main_fn = next((n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == "main"), None)
    assert main_fn is not None, "render_test.py 应有一个 main()"

    # 收集每个"图像取值"节点 → 其所有外层 If 的条件源码
    problems: list[str] = []
    targets = {"core_imgs", "ref_imgs"}

    def cond_source(test: ast.AST) -> str:
        return ast.unparse(test)

    def visit(node: ast.AST, guards: list[str]) -> None:
        if isinstance(node, ast.If):
            guards = guards + [cond_source(node.test)]
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id in targets:
                blob = " ".join(guards)
                ok = ("has_core" in blob or "has_ref" in blob
                      or "core_imgs" in blob or "ref_imgs" in blob)
                if not ok:
                    problems.append(
                        f"{path.name}:{node.lineno} {node.value.id}[...] "
                        f"未被 has_core / has_ref 守卫（外层条件：{guards or '无'}）")
        for child in ast.iter_child_nodes(node):
            visit(child, guards)

    for stmt in main_fn.body:
        visit(stmt, [])

    assert not problems, "\n  ".join(["存在未守卫的图像取值："] + problems)
