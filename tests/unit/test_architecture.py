"""架构一致性测试（**不需要 GPU**）。

把 `docs/ARCHITECTURE.md` 里声明的分层规则变成可执行的检查，
避免它们在后续迭代中被无声破坏。

规则来源：
- §2.3 依赖方向严格单向，`core/` 不得 import `data/` / `training/` / `app/` / `tracking/` / `runtime/`；
- §2.2 `ext/`（现 `submodules/`）只被 `core/render/` 引用；
- §3.1 `core/` 不做文件 I/O 与参数解析。
"""

from __future__ import annotations

import ast
from pathlib import Path

from support import REPO_ROOT

SRC = REPO_ROOT / "src" / "live3dgsavatar"
CORE = SRC / "core"
DATA = SRC / "data"

# core/ 的允许依赖（子包 + 标准库 + 第三方）
CORE_FORBIDDEN_PREFIXES = (
    "live3dgsavatar.data",
    "live3dgsavatar.training",
    "live3dgsavatar.app",
    "live3dgsavatar.tracking",
    "live3dgsavatar.runtime",
    "live3dgsavatar.streaming",
    "live3dgsavatar.evaluation",      # 评估需要数据集，属于上层
)

# 唯一允许接触 CUDA 扩展的位置
CUDA_EXTENSION_PACKAGES = ("diff_gaussian_rasterization",)
# CUDA 扩展的允许位置。
#
# ⚠️ 这条规则的**本意是约束 `core/`**：`core/` 只应在这两处接触 CUDA 扩展，
#    这样替换/升级扩展时改动面可控。
#    `data/reference.py` 是**参照实现的适配层**（给等价门与渲染测试用），
#    它本来就要用参照的 `linear_blending`，且不在 `core/` 的运行路径上，
#    因此明确列入白名单而不是放宽规则。
CUDA_ALLOWED_FILES = {
    CORE / "render" / "rasterizer.py",       # 光栅化封装
    CORE / "deform" / "blend.py",            # 可选的 CUDA linear_blending 加速路径
    DATA / "reference.py",                   # 参照实现适配层（非 core 运行路径）
}


def _iter_modules() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts)


def _imports(path: Path) -> list[tuple[int, str]]:
    """返回 `(行号, 模块名)` 列表，覆盖 `import x` 与 `from x import y`。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                # 相对导入：解析成绝对包名
                parts = list(path.relative_to(SRC).with_suffix("").parts)[:-1]
                ups = node.level - 1
                base = parts[: len(parts) - ups] if ups else parts
                module = "live3dgsavatar." + ".".join(base + ([node.module] if node.module else []))
                found.append((node.lineno, module))
            elif node.module:
                found.append((node.lineno, node.module))
    return found


def test_core_does_not_import_upper_layers() -> None:
    """分层规则：`core/` 不得依赖 data / training / app / tracking / runtime / streaming。"""
    violations: list[str] = []
    for path in sorted(CORE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for lineno, module in _imports(path):
            if module.startswith(CORE_FORBIDDEN_PREFIXES):
                rel = path.relative_to(SRC)
                violations.append(f"{rel}:{lineno} → {module}")
    assert not violations, "core/ 依赖了上层模块：\n  " + "\n  ".join(violations)


def test_cuda_extension_only_imported_where_allowed() -> None:
    """CUDA 扩展只允许在**白名单**里出现（见 `CUDA_ALLOWED_FILES`）。

    这条规则的意义：`submodules/` 里的扩展与 CUDA/torch 版本强耦合，
    一旦 import 散落到各处，替换或升级的成本会迅速失控。

    核心约束是 **`core/` 只在这两处接触扩展**；`data/reference.py` 是
    参照实现适配层，明确列入白名单（它不在 `core/` 的运行路径上）。
    """
    violations: list[str] = []
    for path in _iter_modules():
        for lineno, module in _imports(path):
            root = module.split(".")[0]
            if root in CUDA_EXTENSION_PACKAGES and path not in CUDA_ALLOWED_FILES:
                rel = path.relative_to(SRC)
                violations.append(f"{rel}:{lineno} → {module}")
    assert not violations, (
        "CUDA 扩展被允许范围之外的文件导入：\n  " + "\n  ".join(violations)
        + f"\n（允许位置：{sorted(str(p.relative_to(SRC)) for p in CUDA_ALLOWED_FILES)}）"
    )


def test_core_has_no_file_io_or_argparse() -> None:
    """`core/` 只做算法，不做 I/O 与参数解析（I/O 仅限 `core/io/` 的序列化）。"""
    forbidden = {"argparse", "sys"}
    io_allowed = {CORE / "io" / "ply.py"}
    violations: list[str] = []

    for path in sorted(CORE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for lineno, module in _imports(path):
            if module in forbidden:
                violations.append(f"{path.relative_to(SRC)}:{lineno} → import {module}")
        if path not in io_allowed and path != CORE / "io" / "__init__.py":
            for lineno, module in _imports(path):
                if module in ("plyfile", "json"):
                    violations.append(f"{path.relative_to(SRC)}:{lineno} → import {module}")
    assert not violations, (
        "core/ 出现了 I/O 或参数解析：\n  " + "\n  ".join(violations)
        + f"\n（序列化只允许在 {sorted(str(p.relative_to(SRC)) for p in io_allowed)}）")


def test_no_torch_cuda_calls_in_core_module_scope() -> None:
    """`core/` 不得在**模块作用域**调用 `torch.cuda.*`。

    允许在函数/方法内部使用（例如训练态显式要求 CUDA），但模块导入时不得触碰设备，
    否则「导入 core」就会在无 GPU 环境下失败，CPU 侧的测试将全部无法运行。

    实现说明：必须**剪掉函数与类的定义体**，只检查真正在导入期执行的语句。
    直接 `ast.walk(module)` 会穿透到方法体内部，产生误报。
    """
    violations: list[str] = []

    def module_level_calls(node: ast.AST):
        """只遍历导入期会执行的节点：跳过函数/类定义体。"""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue                      # 定义体不在导入期执行
            yield child
            yield from module_level_calls(child)

    for path in sorted(CORE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for sub in module_level_calls(tree):
            if (isinstance(sub, ast.Attribute)
                    and isinstance(sub.value, ast.Attribute)
                    and sub.value.attr == "cuda"
                    and isinstance(sub.value.value, ast.Name)
                    and sub.value.value.id == "torch"):
                violations.append(f"{path.relative_to(SRC)}:{sub.lineno} → torch.cuda.{sub.attr}")

    # 类体内的默认参数/装饰器也属于导入期，但上述遍历已覆盖（ClassDef 被剪掉的是其 body，
    # 如果将来出现类属性形式的 torch.cuda 调用，这里会漏检；用下面的兜底断言弥补）
    assert not violations, (
        "core/ 在模块作用域调用了 torch.cuda：\n  " + "\n  ".join(violations))


def test_importing_core_does_not_require_gpu() -> None:
    """健全性检查：导入整个 `core` 包不得触发任何设备访问。"""
    import importlib

    for name in ("live3dgsavatar.core", "live3dgsavatar.core.avatar",
                 "live3dgsavatar.core.deform", "live3dgsavatar.core.render",
                 "live3dgsavatar.core.io", "live3dgsavatar.core.types"):
        importlib.import_module(name)


def test_flame_dtype_is_not_overridden() -> None:
    """**回归测试**：不得覆盖 `FlameConfig.dtype`。

    `FLAMEDataset` 把姿态/形状参数存为 **float64**，而 `FLAME` 的
    `v_template` / `shapedirs` 跟随 `cfg.dtype`。把它改成 float32 会让
    `template_vertices(float32) + blend_shapes(betas(float64), ...)`
    抛 `expected scalar type Double but found Float`。

    参照仓库的 `FlameConfig` 默认 float64，故两边一致、从不暴露该问题。
    本项目曾在脚本侧改成 float32 而踩坑，此处用静态检查钉住。
    """
    import re

    offenders: list[str] = []
    for path in sorted(REPO_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts or "submodules" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "FlameConfig" not in text:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            code = line.split("#", 1)[0]          # 跳过注释（含本测试自身的文档示例）
            if re.search(r"\bcfg\.dtype\s*=", code) or \
                    re.search(r"FlameConfig\([^)]*dtype", code):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "不得覆盖 FLAME 的 dtype（会导致 Double/Float 混算）：\n  " + "\n  ".join(offenders))


def test_modules_that_unpickle_flame_import_compat() -> None:
    """**回归测试**：要 `pickle.load` FLAME 资产的模块必须显式导入 `compat`。

    FLAME 的 pickle 里含 chumpy 对象，而 chumpy 在新版 numpy / Python 下有
    两个独立的兼容问题（numpy 2.x 的标量别名、Python 3.11 移除的
    `inspect.getargspec`）。补丁在 `live3dgsavatar.compat` 里，**导入即生效**。

    ⚠️ 曾经的做法是"包初始化时急切施加补丁"，靠 `import live3dgsavatar` 触发。
    那样有代价：任何导入本包的模块都会被拖去 import chumpy ——
    桥接进程（只需要纯 `struct` 打包）在 RDF 环境里因此直接崩掉。
    现在改为**按需导入**，这条测试守的就是"按需"这件事：需要的地方不能忘。

    注意 `tracking/rdf_wire.py` 这类纯逻辑模块**不应该**导入 compat ——
    反向的约束由 `test_pure_modules_do_not_pull_in_chumpy` 守。
    """
    must_have = [
        REPO_ROOT / "src" / "live3dgsavatar" / "data" / "flame_topology.py",
        REPO_ROOT / "src" / "live3dgsavatar" / "data" / "flame_model.py",
        REPO_ROOT / "src" / "live3dgsavatar" / "data" / "checkpoint_loader.py",
    ]
    for path in must_have:
        assert path.exists(), f"缺少文件：{path}"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        # 三种写法都要认：`from .. import compat`（本项目实际用的）、
        # `from ..compat import x`、`import a.b.compat`
        found = any(
            (isinstance(n, ast.ImportFrom)
             and (n.module or "").split(".")[-1] == "compat")
            or (isinstance(n, ast.ImportFrom)
                and any(a.name == "compat" for a in n.names))
            or (isinstance(n, ast.Import)
                and any(a.name.split(".")[-1] == "compat" for a in n.names))
            for n in tree.body
        )
        assert found, (
            f"{path.name} 必须在模块层导入 compat（否则第一次 unpickle FLAME 时"
            "补丁尚未生效，会抛 chumpy 的 ImportError/AttributeError）")


def test_pure_modules_do_not_pull_in_chumpy() -> None:
    """纯逻辑模块导入时**不得**触发 chumpy —— 否则桥接进程会崩。

    真实事故：桥接脚本（跑在 RDF 的 py3.11 环境，那里的 chumpy 需要另一种补丁）
    只 import 了 `tracking.rdf_wire`，却因为包初始化急切施加补丁而崩在
    `inspect.getargspec`。
    """
    import subprocess
    import sys as _sys

    code = (
        "import sys\n"
        "sys.path.insert(0, r'{}')\n"
        "from live3dgsavatar.tracking import rdf_wire\n"
        "assert 'chumpy' not in sys.modules, 'chumpy was imported'\n"
        "assert 'live3dgsavatar.compat' not in sys.modules, 'compat was imported'\n"
        "print('ok', rdf_wire.VERSION)\n"
    ).format(SRC)
    r = subprocess.run([_sys.executable, "-c", code], capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, (
        "纯逻辑模块把 chumpy 拖进来了：\n"
        f"stdout={r.stdout[-400:]}\nstderr={r.stderr[-600:]}")
def test_no_undefined_names_across_project() -> None:
    """**静态检查**：全项目不得有未定义名字。

    `python -m py_compile` 只查语法，查不出「某个分支用了未赋值的变量」；
    而 `--dry-run` 只覆盖部分代码路径。`build_reference` 那种只在 GPU 上跑的
    分支，若引用了未定义变量，会在**用户执行时**才炸（曾如此：
    `_print_summary` 里的 `c` / `r` 未定义，`--sweep-batch` 用 `nargs="*"` 拿到空列表）。

    这里用 AST 收集模块级绑定、用 `symtable` 做词法/闭包解析，二者结合定位。
    """
    import ast
    import builtins
    import symtable

    def module_bindings(tree) -> set[str]:
        names: set[str] = set()
        for n in ast.walk(tree):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    names.add((a.asname or a.name).split(".")[0])
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                names.add(n.id)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(n.name)
            elif isinstance(n, ast.arg):
                names.add(n.arg)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                names.add(n.name)
        return names

    files = (list((REPO_ROOT / "src").rglob("*.py"))
             + list((REPO_ROOT / "tests").rglob("*.py"))
             + list((REPO_ROOT / "scripts").glob("*.py")))

    problems: list[str] = []
    for path in sorted(files):
        if "__pycache__" in str(path):
            continue
        src = path.read_text(encoding="utf-8")
        mod = module_bindings(ast.parse(src, filename=str(path)))
        table = symtable.symtable(src, str(path), "exec")

        def walk(t):
            for sym in t.get_symbols():
                name = sym.get_name()
                if (not sym.is_referenced() or sym.is_assigned()
                        or sym.is_parameter() or sym.is_imported() or sym.is_local()):
                    continue
                if name in mod or hasattr(builtins, name) or name == "__file__":
                    continue
                problems.append(
                    f"{path.relative_to(REPO_ROOT)}:{t.get_lineno()} "
                    f"[{t.get_name()}] 未定义 {name!r}")
            for child in t.get_children():
                walk(child)

        walk(table)

    assert not problems, "存在未定义名字：\n  " + "\n  ".join(problems)


def test_call_sites_match_function_signatures() -> None:
    """**静态检查**：对本项目模块级函数的调用，其关键字参数必须存在于签名中。

    这类错误（参数名写错、重构时改了签名）只在**运行时**才炸，而
    `py_compile` 查不出、`--dry-run` 也未必覆盖到那条分支。
    曾真实发生：`load_scene` 的 `device` 参数在重构中被删掉，
    函数体里仍在用，直到用户执行渲染测试才 `NameError`。

    为避免误报，只检查：
    - **模块级**函数（排除类方法，避免 `to(device=)` / `to(dtype=)` 这类同名冲突）
    - 以裸名 `f(...)` 调用的（排除 `obj.method(...)`）
    - 名字在本项目内唯一

    Args:
        无
    """
    import ast
    import builtins

    def project_files():
        for sub in ("src", "tests", "scripts"):
            for f in (REPO_ROOT / sub).rglob("*.py"):
                if "__pycache__" not in str(f):
                    yield f

    files = list(project_files())

    # 索引模块级函数名 → 参数集合（仅名字唯一的才检查）
    by_name: dict[str, list[set[str]]] = {}
    has_var_kw: dict[str, bool] = {}
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            a = node.args
            names = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
            by_name.setdefault(node.name, []).append(names)
            has_var_kw[node.name] = has_var_kw.get(node.name, False) or a.kwarg is not None

    builtin_names = set(dir(builtins))
    problems: list[str] = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            name = node.func.id
            if name in builtin_names or name not in by_name:
                continue
            if len(by_name[name]) != 1 or has_var_kw.get(name):
                continue
            known = next(iter(by_name[name]))
            passed = {k.arg for k in node.keywords if k.arg is not None}
            unknown = passed - known
            if unknown:
                problems.append(
                    f"{path.relative_to(REPO_ROOT)}:{node.lineno} "
                    f"{name}(...) 传入未知参数 {sorted(unknown)}；"
                    f"签名参数 {sorted(known)}")

    assert not problems, "存在与签名不匹配的调用：\n  " + "\n  ".join(problems)


def test_no_tautological_assertions() -> None:
    """**静态检查**：不得出现恒真/恒假的断言。

    形如 `assert x or True, ""` 或 `assert True` 的断言是**假的安全感** ——
    看起来在检查，实际永不失败。曾真实残留一条。
    """
    import ast

    offenders: list[str] = []
    for sub in ("src", "tests", "scripts"):
        for path in sorted((REPO_ROOT / sub).rglob("*.py")):
            if "__pycache__" in str(path):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assert):
                    continue
                test = node.test
                # `... or True` / `True and ...`
                if (isinstance(test, ast.BoolOp)
                        and any(isinstance(v, ast.Constant) and v.value is True
                                for v in test.values)
                        and isinstance(test.op, (ast.Or, ast.And))):
                    offenders.append(
                        f"{path.relative_to(REPO_ROOT)}:{node.lineno} "
                        f"恒真断言（含 True 的 {type(test.op).__name__}）")
                elif isinstance(test, ast.Constant) and test.value is True:
                    offenders.append(
                        f"{path.relative_to(REPO_ROOT)}:{node.lineno} `assert True`")

    assert not offenders, "存在恒真断言：\n  " + "\n  ".join(offenders)


def test_pyproject_declares_package_layout() -> None:
    """打包配置必须声明 src 布局与 CLI 入口。"""
    import re

    path = REPO_ROOT / "pyproject.toml"
    assert path.exists(), (
        "缺少 pyproject.toml —— 项目不可安装，"
        "`python -m live3dgsavatar.app` 会 ModuleNotFoundError")

    text = path.read_text(encoding="utf-8")

    # src 布局
    assert 'package-dir = { "" = "src" }' in text, "应声明 src 布局"
    assert 'where = ["src"]' in text, "应声明 package 查找目录"

    # console script 入口指向真实存在的函数
    m = re.search(r'\[project\.scripts\]\s*\n(\w+)\s*=\s*"([^"]+)"', text)
    assert m, "应声明 [project.scripts] 入口"
    target = m.group(2)
    module_path, _, func = target.partition(":")
    assert module_path and func, f"入口格式应为 module:func，实际 {target!r}"

    rel = Path(*module_path.split(".")).with_suffix(".py")
    mod_file = REPO_ROOT / "src" / rel
    assert mod_file.exists(), f"入口模块不存在：{mod_file}"
    src = mod_file.read_text(encoding="utf-8")
    assert f"def {func}(" in src, f"{mod_file.name} 里没有 def {func}()"


def test_setup_env_installs_project() -> None:
    """`setup_env.sh` 必须安装**本项目自身**（可编辑、--no-deps）。

    与 `test_pyproject_declares_package_layout` 配套：光有 pyproject 而没人装它，
    `python -m live3dgsavatar.app` 仍然不可用。

    ⚠️ 断言必须**具体到那一条命令**。初版只查 `"-e "` 与 `"--no-deps"` 是否出现，
    但文件里 `pip install -e "$SUBMODULES/diff-gaussian-rasterization" --no-build-isolation`
    等步骤也含这些片段，于是**删掉本项目的安装步骤测试仍然通过** —— 假保证。
    """
    text = (REPO_ROOT / "setup_env.sh").read_text(encoding="utf-8")

    # 安装本项目：必须是 `-e "$REPO_ROOT"`，且带 --no-deps
    lines = [ln.strip() for ln in text.splitlines()]
    project_install = [
        ln for ln in lines
        if ln.startswith("pip install") and "$REPO_ROOT" in ln and " -e " in ln
    ]
    assert project_install, (
        'setup_env.sh 缺少安装本项目的命令（应形如 '
        '`pip install --no-deps --no-build-isolation -e "$REPO_ROOT"`）')
    cmd = project_install[0]
    assert "--no-deps" in cmd, (
        "安装本项目必须加 --no-deps：torch 与三个 CUDA 扩展有自己的安装顺序"
        "（见 docs/ENVIRONMENT.md §2），让 pip 顺着 pyproject 解析会与它们打架")

    # 安装后立刻自检导入，而不是等 run_gui.sh 才 ModuleNotFoundError
    assert "import live3dgsavatar" in text, "setup_env.sh 应在安装后自检导入"


def test_training_does_not_depend_on_data_or_app() -> None:
    """**分层**：`training/` 只能向下依赖 `core/`（与标准库/torch）。

    为什么这条重要：在线更新的观测可能来自**数据集**，也可能来自**摄像头**。
    如果 `training/` 直接 import `data/`，那么"摄像头来源"就得复制一份更新逻辑；
    反之若它 import `app/`，测试与脚本就无法脱离服务端使用它。

    观测通过 `FrameObservation` / `BatchObservation` 传入 —— 这样两种来源
    共用同一份更新实现（见 `training/observations.py`）。
    """
    training = SRC / "training"
    if not training.is_dir():
        return
    forbidden = (
        "live3dgsavatar.data",
        "live3dgsavatar.app",
        "live3dgsavatar.tracking",
        "live3dgsavatar.runtime",
        "live3dgsavatar.streaming",
    )
    violations: list[str] = []
    for path in sorted(training.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for lineno, module in _imports(path):
            if module.startswith(forbidden):
                rel = path.relative_to(SRC)
                violations.append(f"{rel}:{lineno} → {module}")
    assert not violations, (
        "training/ 依赖了不该依赖的上层（观测应通过 FrameObservation 传入）：\n  "
        + "\n  ".join(violations))


def test_app_package_imports_without_broken_symbols() -> None:
    """**回归测试**：`live3dgsavatar.app` 必须能真正导入。

    曾经 `session.py` 引用了已删除的 `reference_model_dir`，而
    `app/__init__.py` 把任何 `ImportError` 都当成"依赖未安装"，
    于是 `python -m live3dgsavatar.app` 报的是「请先运行 setup_env.sh」——
    把**代码 bug** 伪装成了环境问题，排查方向完全跑偏。

    这里直接检查：导入成功，且模块里没有残留的未定义引用。
    """
    import importlib

    mod = importlib.import_module("live3dgsavatar.app")
    assert hasattr(mod, "protocol")
    # 若 server 可导入（依赖齐全），则 create_app 必须是真函数而非兜底桩
    import live3dgsavatar.app.session as session  # noqa: F401


def test_app_cli_help_works() -> None:
    """`python -m live3dgsavatar.app --help` 必须能跑（不需要 GPU/模型）。"""
    import subprocess
    import sys as _sys

    r = subprocess.run(
        [_sys.executable, "-m", "live3dgsavatar.app", "--help"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120,
        env={**__import__("os").environ, "PYTHONPATH": str(SRC)})
    assert r.returncode == 0, (
        f"--help 应成功，实际 {r.returncode}\nstdout={r.stdout[-500:]}\n"
        f"stderr={r.stderr[-800:]}")
    assert "--subject" in r.stdout


def test_scripts_with_main_have_entry_point_guard() -> None:
    """**回归测试**：定义了 `main()` 的脚本必须有 `if __name__ == "__main__"`。

    真实事故：重写 `rdf_bridge.py` 的 `main()` 时把文件尾部的入口判断一起截掉了。
    脚本从此**什么都不做、退出码 0、没有任何输出** —— 看起来像"跑完了"，
    实际上 `main()` 从没被调用。这种失效没有任何报错，只能靠静态检查发现。
    """
    import ast

    for path in sorted((REPO_ROOT / "scripts").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        has_main = any(
            isinstance(n, ast.FunctionDef) and n.name == "main" for n in tree.body)
        if not has_main:
            continue
        guarded = any(
            isinstance(n, ast.If)
            and isinstance(n.test, ast.Compare)
            and isinstance(n.test.left, ast.Name)
            and n.test.left.id == "__name__"
            for n in tree.body
        )
        assert guarded, (
            f"{path.name} 定义了 main() 却没有 `if __name__ == \"__main__\"` 入口 —— "
            "运行它会静默退出（0，无输出），main() 永远不会被调用")


def test_training_does_not_depend_on_evaluation() -> None:
    """`training/` 不得 import `evaluation/`。

    评估需要数据集与场景，而训练层必须能脱离数据集使用（摄像头来源同样要能训练）。
    训练只需要一个 `evaluate` 回调（`ValidationHook`），由脚本把评估实现注入。
    """
    training = SRC / "training"
    offenders: list[str] = []
    for path in sorted(training.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for lineno, module in _imports(path):
            if module.startswith("live3dgsavatar.evaluation"):
                offenders.append(f"{path.relative_to(SRC)}:{lineno} → {module}")
    assert not offenders, "training/ 不应依赖 evaluation/：\n  " + "\n  ".join(offenders)


def test_torch_attributes_exist() -> None:
    """`torch.xxx` 用到的属性必须真的存在于当前 torch。

    为什么值得查：我们无法在本机跑 GPU 路径，像 `torch.flatnonzero`（不存在，
    应写 `nonzero`）这种错误只能等到用户机器上才炸。这里静态扫一遍属性名，
    在没有 GPU 的机器上也能拦住同类问题。
    """
    import re

    try:
        import torch
    except ImportError:                                # pragma: no cover
        return

    # 负向断言排除 `nvdiffrast.torch.xxx` 这类"别的包的 torch 子模块"
    pattern = re.compile(r"(?<![\w.])torch\.([A-Za-z_][A-Za-z0-9_]*)")
    missing: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            name = match.group(1)
            if not hasattr(torch, name):
                line = text[:match.start()].count("\n") + 1
                missing.append(f"{path.relative_to(SRC)}:{line} → torch.{name}")
    assert not missing, "使用了不存在的 torch 属性：\n  " + "\n  ".join(missing)


def _use_before_assignment(source: str) -> list[str]:
    """找出"同一个语句块内先用后赋值"的名字（Python 会抛 `UnboundLocalError`）。

    只比较**同一语句块**内的先后顺序：跨分支/跨循环的比较会产生大量误报
    （例如 `if` 分支里赋值、另一分支里使用是合法的）。这正是我们踩过的模式 ——
    `find_defects.py` 的 `hit_rate` 在函数体里先被使用、几十行后才赋值。
    """
    import ast

    tree = ast.parse(source)
    problems: list[str] = []

    def binds(stmt: ast.stmt) -> set[str]:
        names: set[str] = set()
        for node in ast.walk(stmt):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
        return names

    def own_loads(stmt: ast.stmt) -> list[tuple[str, int]]:
        """只取该语句**自身表达式**里的读取，嵌套语句块交给递归。

        推导式（`[f(i) for i in xs]`）与 lambda 的参数名在**语句内部**绑定，
        不算"先用后赋值"，必须排除，否则会产生成片误报。
        """
        inner: set[str] = set()
        for node in ast.walk(stmt):
            if isinstance(node, ast.comprehension):
                for target in ast.walk(node.target):
                    if isinstance(target, ast.Name):
                        inner.add(target.id)
            elif isinstance(node, ast.Lambda):
                for arg in (list(node.args.args) + list(node.args.kwonlyargs)
                            + list(node.args.posonlyargs)):
                    inner.add(arg.arg)

        out: list[tuple[str, int]] = []
        for node in ast.iter_child_nodes(stmt):
            # 子语句块（含 `except` handler）由 `nested_bodies` 递归处理
            if isinstance(node, (ast.stmt, ast.ExceptHandler)):
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load)
                        and sub.id not in inner):
                    out.append((sub.id, sub.lineno))
        return out

    def nested_bodies(stmt: ast.stmt) -> list[list[ast.stmt]]:
        bodies: list[list[ast.stmt]] = []
        for field in ("body", "orelse", "finalbody"):
            body = getattr(stmt, field, None)
            if isinstance(body, list) and body and isinstance(body[0], ast.stmt):
                # `orelse`/`finalbody` 只在复合语句上出现
                bodies.append(body)
        if isinstance(stmt, ast.Try):
            bodies.extend(handler.body for handler in stmt.handlers)
        return bodies

    def check_block(stmts: list[ast.stmt], bound: set[str], fn_name: str) -> None:
        later: set[str] = set()
        for stmt in stmts:
            later |= binds(stmt)
        for stmt in stmts:
            for name, lineno in own_loads(stmt):
                if name in bound:
                    continue
                if name in later or name in binds(stmt):
                    problems.append(
                        f"{fn_name}:{lineno} 使用了 {name!r}，但它在同一段里稍后才赋值")
            bound |= binds(stmt)
            for body in nested_bodies(stmt):
                check_block(body, set(bound), fn_name)

    for fn in [n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        params = {a.arg for a in list(fn.args.args) + list(fn.args.kwonlyargs)
                  + list(fn.args.posonlyargs)}
        if fn.args.vararg:
            params.add(fn.args.vararg.arg)
        if fn.args.kwarg:
            params.add(fn.args.kwarg.arg)
        declared = {n for node in ast.walk(fn)
                    if isinstance(node, (ast.Global, ast.Nonlocal))
                    for n in node.names}
        check_block(fn.body, params | declared, fn.name)
    return problems


def test_no_use_before_local_assignment() -> None:
    """全仓库不得有"函数内先用后赋值"（会抛 `UnboundLocalError`）。

    真实故障：`find_defects.py` 里 `hit_rate` 在第 272 行被使用、却在第 280 行
    才赋值。这类错误只在那条 GPU 分支上才炸，本机跑不到。
    """
    problems: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        problems.extend(f"{path.relative_to(SRC)}: {p}"
                        for p in _use_before_assignment(
                            path.read_text(encoding="utf-8")))
    assert not problems, "存在先用后赋值：\n  " + "\n  ".join(problems)


def test_checker_detects_known_case() -> None:
    """守卫自身的自检：确认它真的能抓到这个模式。"""
    src = "def f():\n    print(x)\n    x = 1\n"
    assert _use_before_assignment(src)
    ok = "def f():\n    x = 1\n    print(x)\n"
    assert not _use_before_assignment(ok)
