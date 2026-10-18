"""零依赖测试运行器 —— 不安装 pytest 也能跑全部测试。

设计：
- 测试模块里的 `test_*` 函数可被 pytest 直接收集（保持兼容）；
- 本运行器用极简方式调用它们，并处理项目用到的两种 pytest 特性：
  `@pytest.mark.parametrize` 与 `@pytest.fixture`；
- **不引入任何第三方测试依赖**（项目依赖保持干净）。

用法::

    python tests/run_tests.py                # 跑 tests/ 下全部
    python tests/run_tests.py tests/unit     # 只跑某个目录/文件
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import tempfile
import traceback
from pathlib import Path
from pathlib import Path
from types import ModuleType

TESTS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TESTS_ROOT.parent

# 运行器内置提供的夹具（与 pytest 同名同义）
BUILTIN_FIXTURES = {"tmp_path"}

# --------------------------------------------------------- 极简 pytest 替身 --


class _Mark:
    """支持 `@pytest.mark.parametrize("a,b", [(1,2), (3,4)])`。"""

    @staticmethod
    def parametrize(argnames: str, argvalues, **_: object):
        names = [n.strip() for n in argnames.split(",")]

        def decorator(func):
            cases = []
            for values in argvalues:
                vals = values if isinstance(values, (tuple, list)) else (values,)
                cases.append(dict(zip(names, vals)))
            func._parametrize = getattr(func, "_parametrize", []) + [cases]
            return func

        return decorator

    def __getattr__(self, name: str):  # 忽略 skip/slow 等未知 mark
        def _noop(*_a, **_k):
            def decorator(func):
                return func

            return decorator

        return _noop


class Skipped(Exception):
    """测试主动跳过（对应 pytest.skip）。运行器记为「跳过」而非「通过」。"""


def skip(reason: str = ""):
    """供测试模块调用：`from pytest import skip; skip("原因")`。"""
    raise Skipped(reason)


class _Fixture:
    def __init__(self, func, scope: str = "function") -> None:
        self.func = func
        self.scope = scope
        self.cache: dict[str, object] = {}

    def value(self, fixtures: dict[str, "_Fixture"]):
        key = self.func.__name__
        if self.scope == "module" and key in self.cache:
            return self.cache[key]
        kwargs = {
            p.name: fixtures[p.name].value(fixtures)
            for p in inspect.signature(self.func).parameters.values()
            if p.name in fixtures
        }
        result = self.func(**kwargs)
        if self.scope == "module":
            self.cache[key] = result
        return result


class _Raises:
    """零依赖的 `pytest.raises` 替身。

    支持两种写法::

        with pytest.raises(ValueError):
            ...

        with pytest.raises(ValueError, match="子串") as exc:
            ...
            assert "子串" in str(exc.value)
    """

    def __init__(self, exc_type, match=None) -> None:
        self.exc_type = exc_type
        self.match = match
        self.value = None

    def __enter__(self) -> "_Raises":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            raise AssertionError(
                f"期望抛出 {self.exc_type.__name__}，但没有抛出")
        if not issubclass(exc_type, self.exc_type):
            return False                      # 其它异常照常向上抛
        if self.match is not None:
            # 与 pytest 一致：`match` 是**正则**（re.search）。
            # 早先实现成子串匹配，导致 `match="\\[N,3\\]"` 这类写法静默不匹配。
            import re as _re
            if _re.search(self.match, str(exc)) is None:
                raise AssertionError(
                    f"异常信息不匹配 {self.match!r}：{exc}")
        self.value = exc
        return True


def _install_pytest_stub() -> None:
    """若环境未安装 pytest，注入最小替身，使测试模块可导入。"""
    if "pytest" in sys.modules:
        return
    try:
        import pytest  # noqa: F401, PLC0415

        return
    except ImportError:
        pass

    import types

    stub = types.ModuleType("pytest")

    def fixture(func=None, *, scope="function", **_):
        if func is None:  # @pytest.fixture(scope=...)
            return lambda f: f
        return func

    def raises(exc_type, match=None):
        """`pytest.raises` 的最小实现（支持 `as e` 与可选的 `match` 正则）。"""
        return _Raises(exc_type, match)

    stub.fixture = fixture
    stub.mark = _Mark()
    stub.approx = lambda v, **k: v
    stub.skip = skip
    stub.raises = raises
    sys.modules["pytest"] = stub
    # 同时提供顶层 `skip`，便于 `from pytest import skip`
    stub.Skipped = Skipped


# ------------------------------------------------------------------ 收集与执行 --


def _iter_test_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(p.rglob("test_*.py")))
        elif p.suffix == ".py":
            files.append(p)
    return files


def _load(path: Path, index: int) -> ModuleType:
    name = f"_dsh_test_{index}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(paths: list[Path]) -> int:
    _install_pytest_stub()
    for extra in (str(TESTS_ROOT), str(REPO_ROOT / "src")):
        if extra not in sys.path:
            sys.path.insert(0, extra)

    files = _iter_test_files(paths) if paths else _iter_test_files([TESTS_ROOT])
    if not files:
        print("没有找到 test_*.py")
        return 1

    n_pass = n_fail = n_skip = 0
    failures: list[tuple[str, str]] = []

    for idx, path in enumerate(files):
        rel = path.relative_to(REPO_ROOT)
        try:
            module = _load(path, idx)
        except Exception:
            n_fail += 1
            failures.append((str(rel), traceback.format_exc()))
            print(f"\n\033[1;31m[导入失败]\033[0m {rel}")
            continue

        fixtures = {
            name: _Fixture(obj, getattr(obj, "_pytestfixturefunction", None) and "module" or "function")
            for name, obj in vars(module).items()
            if callable(obj) and not name.startswith("test_")
        }
        # 需要 module 级缓存的夹具（参照相机模块只加载一次）
        for name, obj in vars(module).items():
            if name == "ref" and callable(obj):
                fixtures[name] = _Fixture(obj, scope="module")

        tests = [
            (name, obj) for name, obj in vars(module).items()
            if name.startswith("test_") and callable(obj)
        ]
        print(f"\n\033[1;36m{rel}\033[0m  ({len(tests)} 个测试)")

        for name, func in sorted(tests):
            sig = inspect.signature(func)
            # `_parametrize` 是「装饰器 → 用例组」的列表；这里展平。
            # 多层 parametrize 的真实语义是笛卡尔积，本运行器只做展平 ——
            # 够用，且比"悄悄少跑用例"安全（会在下方因未知夹具而显式失败）。
            _groups = getattr(func, "_parametrize", None) or []
            cases = [c for group in _groups for c in group] or [{}]
            # 被 `parametrize` 覆盖的形参不是夹具 —— 必须先排除它们，
            # 否则参数化测试会被误判成「未知夹具」而直接失败。
            parametrized = set()
            for case in cases:
                parametrized |= set(case)

            supported = set(fixtures) | BUILTIN_FIXTURES | parametrized
            unknown = [p for p in sig.parameters if p not in supported]
            if unknown:
                # 显式失败而非静默跳过：否则测试会「看起来通过」
                n_fail += 1
                msg = f"未知夹具 {unknown}；可用：{sorted(supported)}"
                failures.append((f"{rel}::{name}", msg))
                print(f"  \033[1;31m[失败]\033[0m {name}: {msg}")
                continue

            for case in cases:
                label = name + (
                    f"[{', '.join(f'{k}={v!r}' for k, v in case.items())}]" if case else "")
                with tempfile.TemporaryDirectory(prefix="dsh_test_") as tmpdir:
                    kwargs = {}
                    for pname in sig.parameters:
                        if pname in case:
                            kwargs[pname] = case[pname]
                        elif pname == "tmp_path":
                            kwargs[pname] = Path(tmpdir)
                        else:
                            kwargs[pname] = fixtures[pname].value(fixtures)
                    try:
                        func(**kwargs)
                    except Skipped as e:
                        n_skip += 1
                        print(f"  \033[1;33m[跳过]\033[0m {label}" + (f": {e}" if str(e) else ""))
                    except Exception as e:
                        n_fail += 1
                        failures.append((f"{rel}::{label}", traceback.format_exc()))
                        print(f"  \033[1;31m[失败]\033[0m {label}: {type(e).__name__}: {e}")
                    else:
                        n_pass += 1
                        print(f"  \033[1;32m[通过]\033[0m {label}")

    print(f"\n{'=' * 62}\n通过 {n_pass}，失败 {n_fail}，跳过 {n_skip}\n{'=' * 62}")
    for where, tb in failures:
        print(f"\n--- {where} ---\n{tb}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    targets = [Path(a).resolve() for a in sys.argv[1:]]
    raise SystemExit(run(targets))
