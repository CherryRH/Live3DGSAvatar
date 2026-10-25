"""按路径加载 `scripts/*.py` 的小工具（测试共用）。

脚本不是包的一部分（没有 `__init__.py`），测试要拿到它们的函数只能按文件
加载。这段逻辑在 5 个测试文件里各抄了一遍，集中到这里：

    from script_loader import load_script
    m = load_script("render_test.py")

模块名按脚本名缓存，重复加载返回同一个对象（脚本里的模块级常量不会被
重复求值，测试之间也不会互相污染）。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from support import REPO_ROOT  # 仓库根只在一处定义（tests/support.py）


def load_script(name: str):
    """加载 `scripts/<name>` 并返回模块对象（同名只加载一次）。"""
    key = f"_script_{Path(name).stem}"
    if key in sys.modules:
        return sys.modules[key]

    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(key, path)
    if spec is None or spec.loader is None:            # pragma: no cover
        raise ImportError(f"无法加载脚本：{path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod
