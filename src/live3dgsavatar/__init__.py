"""Live3DGSAvatar.

包初始化**刻意不做任何副作用**（不 import torch、不 import chumpy）。

兼容性补丁在 `live3dgsavatar.compat`，由**真正需要它的模块**按需导入：
只有要 `pickle.load` FLAME 资产或 `.frame` 的地方才需要（那些 pickle 里含
chumpy 对象）。见 `compat/__init__.py` 的说明。

为什么不在这里急切导入补丁：补丁会 import chumpy，而 chumpy 在不同环境下
可能本来就坏（两个独立的不兼容维度：numpy 2.x 的标量别名、Python 3.11 的
`inspect.getargspec`）。纯逻辑模块（如 `tracking/rdf_wire.py`）不该被牵连 ——
桥接进程曾因此在 RDF 环境里直接崩掉。
"""

__all__: list[str] = []
