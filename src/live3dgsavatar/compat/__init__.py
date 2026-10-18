"""第三方兼容性补丁。

导入本模块即生效（无副作用函数，只有导入期补丁）。

⚠️ **本模块是「按需」导入的，不是包初始化的一部分**：
只有真正要 `pickle.load` FLAME 资产或 `.frame` 的模块才需要它
（`data/flame_topology.py`、`data/flame_model.py`、`data/checkpoint_loader.py`）。

这样做是因为补丁会 import chumpy，而 chumpy 在不同环境下**可能本来就坏**
（它有两个独立的不兼容维度：numpy 2.x 的标量别名、Python 3.11 的
`inspect.getargspec`）。纯逻辑模块（如 `tracking/rdf_wire.py`）不该为此被拖累 ——
桥接进程就曾因此在 RDF 环境里直接崩掉。

新增补丁时请遵守：
1. 只做「最小必要」的别名/接口补齐，不改变第三方库的行为语义；
2. 必须幂等，可重复导入；
3. 在下方登记原因与可移除条件。
"""

from __future__ import annotations

import warnings

# ⚠️ 过滤器必须在 import numpy **之前**装好：
# numpy 2.2 对已移除的别名做了 module __getattr__ 弃用转发，连 hasattr(np, "object")
# 这样的**读取**都会触发 FutureWarning（归属模块是本文件/调用方，不是 chumpy）。
# 这是补丁的有意行为，按消息精确静音，不牵连其他告警。
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*np\.(object|str)\b.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"chumpy(\.|$)")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"chumpy(\.|$)")

import numpy as _np  # noqa: E402


# ---------------------------------------------------------------------------
# 补丁 1：numpy 2.x 移除的标量别名（chumpy 依赖）
#
# 原因：chumpy 0.70 的 `chumpy/__init__.py:11` 执行
#     from numpy import bool, int, float, complex, object, unicode, str, nan, inf
#   这些别名在 numpy 1.24 被弃用、2.0 被移除，因此 numpy>=2 下 chumpy
#   导入即抛 `ImportError: cannot import name 'int' from 'numpy'`。
#
# 为什么不等价于「把 numpy 降到 1.23.5」：
#   1) numpy==1.23.5 的 cp310 wheel 在 pip 当前解析策略下已无法获取
#      （官方与镜像 simple 索引中该文件仍存在，但 pip 不再选用；
#        退回源码包后编译失败：`ModuleNotFoundError: distutils.msvccompiler`），
#      实测 `pip download "numpy==1.23.5" --only-binary=:all:` 直接
#      `No matching distribution found`。
#   2) 该约束还会与 scipy>=1.14（要求 numpy>=1.23.5）等包形成脆弱交集。
#
# 补丁内容：仅在缺失时补齐别名。numpy 1.x 下 hasattr 为真，本补丁不生效。
#
# 可移除条件：chumpy 上游发布兼容 numpy 2.x 的版本后。
# ---------------------------------------------------------------------------
_NUMPY_ALIASES = {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}


def _patch_numpy_legacy_aliases() -> tuple[str, ...]:
    """补齐 numpy 2.x 移除的标量别名。返回实际补齐的名称。"""
    patched = []
    for name, builtin in _NUMPY_ALIASES.items():
        if not hasattr(_np, name):
            setattr(_np, name, builtin)
            patched.append(name)
    return tuple(patched)


patched_aliases = _patch_numpy_legacy_aliases()


# ---------------------------------------------------------------------------
# 补丁 2：`inspect.getargspec`（Python 3.11 移除）
#
# 原因：chumpy 0.70 的 `ch.py:1203` 用 `inspect.getargspec(func).args`。
#   该函数在 Python 3.11 被删除（PEP 594 的清理），因此 **Python 3.11+** 下
#   chumpy 导入即抛 `AttributeError: module 'inspect' has no attribute 'getargspec'`。
#   注意这与 numpy 版本无关 —— 是**另一个**不兼容维度：3.10 下不会出现，
#   3.11 下即使 numpy 1.x 也会出现。
#
# 补丁内容：仅在缺失时补一个基于 `getfullargspec` 的等价实现
#   （`getargspec` 的返回类型是 4 元组，取 `getfullargspec` 的前四项）。
#
# 可移除条件：chumpy 上游改用 `getfullargspec` 后。
# ---------------------------------------------------------------------------
class _ArgSpec(tuple):
    """`inspect.ArgSpec` 的替身。

    `ArgSpec` 本身也在 Python 3.11 被移除，所以不能复用它 ——
    直接给一个 4 元组，字段顺序与旧版一致（args, varargs, keywords, defaults）。
    """

    __slots__ = ()

    def __new__(cls, args, varargs, keywords, defaults):
        return tuple.__new__(cls, (args, varargs, keywords, defaults))

    args = property(lambda self: self[0])
    varargs = property(lambda self: self[1])
    keywords = property(lambda self: self[2])
    defaults = property(lambda self: self[3])


def _patch_inspect_getargspec() -> bool:
    """为 Python 3.11+ 补回 `inspect.getargspec`。返回是否真的补了。"""
    import inspect as _inspect

    if hasattr(_inspect, "getargspec"):
        return False

    def getargspec(func):                      # noqa: ANN001 - 与旧签名一致
        spec = _inspect.getfullargspec(func)
        return _ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    _inspect.getargspec = getargspec
    return True


patched_getargspec = _patch_inspect_getargspec()

# 自检：补丁必须真的让 chumpy 可用，否则本模块等同于没生效。
# 失败时立即报错，避免补丁与 chumpy 版本漂移后被静默忽略。
try:
    import chumpy as _chumpy  # noqa: F401, PLC0415
except Exception as _e:  # pragma: no cover
    raise ImportError(
        f"chumpy 兼容补丁未生效（numpy {_np.__version__}）：{type(_e).__name__}: {_e}\n"
        "请同步更新 src/live3dgsavatar/compat/__init__.py 与 scripts/_compat_shim.py，"
        "参见 docs/ENVIRONMENT.md §3.6"
    ) from _e

__all__ = ["patched_aliases", "patched_getargspec"]
