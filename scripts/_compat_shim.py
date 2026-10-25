"""共享兼容性补丁 —— 供 scripts/ 下的独立脚本使用。

**权威实现在 `src/live3dgsavatar/compat/__init__.py`**，本文件是它的精简副本：
`env_check.py` 必须在「未安装本项目、未设置 PYTHONPATH」的条件下也能运行，
因此不能依赖 `src/`。其余脚本都可以 `import live3dgsavatar`，直接用权威实现。

⚠️ 修改 `src/live3dgsavatar/compat/__init__.py` 中的补丁列表时，必须同步修改本文件
   （两个文件各有一份 `_NUMPY_ALIASES`）。

当前补丁（**两个独立的不兼容维度**）：

1. **numpy 2.x 移除了标量别名** —— `chumpy/__init__.py:11` 执行
       from numpy import bool, int, float, complex, object, unicode, str, nan, inf
   这些别名在 numpy 1.24 弃用、2.0 移除。
2. **Python 3.11 移除了 `inspect.getargspec`** —— `chumpy/ch.py:1203` 仍在用。
   这与 numpy 版本无关：3.10 下不出现，3.11 下即使 numpy 1.x 也会出现。

原因与移除条件见 docs/ENVIRONMENT.md §3.6。
"""

from __future__ import annotations

import warnings

# ⚠️ 过滤器必须在 import numpy **之前**装好：
# numpy 2.2 对已移除的别名做了 module __getattr__ 弃用转发，连 hasattr(np, "object")
# 这样的**读取**都会触发 FutureWarning，而告警的归属模块是 numpy 自己。
# chumpy 导入期会触发大量同类告警；补丁本身是有意为之，重复刷屏会掩盖真问题。
warnings.filterwarnings("ignore", category=FutureWarning, module=r"chumpy(\.|$)")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"chumpy(\.|$)")
# 探测 `hasattr(np, "object")` 本身就会触发 numpy 的弃用转发告警（归属模块是本文件）。
# 这是补丁的有意行为，精确按消息静音，不牵连其他告警。
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*np\.(object|str)\b.*")

import numpy as np  # noqa: E402

_NUMPY_ALIASES = {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}


def apply() -> tuple[str, ...]:
    """补齐 numpy 2.x 移除的标量别名。返回实际补齐的名称（幂等）。"""
    patched = []
    for name, builtin in _NUMPY_ALIASES.items():
        if not hasattr(np, name):
            setattr(np, name, builtin)
            patched.append(name)
    return tuple(patched)


def apply_inspect_getargspec() -> bool:
    """为 Python 3.11+ 补回 `inspect.getargspec`。返回是否真的补了。

    与 numpy 补丁是**两个独立维度**：3.11 下即使 numpy 1.x 也需要它。
    """
    import inspect

    if hasattr(inspect, "getargspec"):
        return False

    class _ArgSpec(tuple):
        __slots__ = ()

        def __new__(cls, args, varargs, keywords, defaults):
            return tuple.__new__(cls, (args, varargs, keywords, defaults))

        args = property(lambda self: self[0])
        varargs = property(lambda self: self[1])
        keywords = property(lambda self: self[2])
        defaults = property(lambda self: self[3])

    def getargspec(func):                      # noqa: ANN001
        spec = inspect.getfullargspec(func)
        return _ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    inspect.getargspec = getargspec
    return True


def verify_chumpy() -> tuple[bool, str]:
    """应用补丁后验证 chumpy 是否可用。返回 (是否可用, 说明)。"""
    apply()
    apply_inspect_getargspec()
    try:
        import chumpy  # noqa: F401, PLC0415
    except Exception as e:  # pragma: no cover
        return False, f"{type(e).__name__}: {e}"
    return True, "OK"
