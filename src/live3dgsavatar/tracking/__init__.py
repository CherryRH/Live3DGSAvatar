"""驱动参数求解：外部追踪器（RDF）→ 本项目可渲染的驱动参数与网格。

分层约定：`tracking/` 可依赖 `core/types` 与 `data/`，但**不得**被 `core/` 导入 ——
驱动来源（数据集 / 摄像头 / 桥接进程）是可替换的外部依赖。

模块：

- `param_map`   ：RDF 的 118 维 `x` → 129 维 `blend_weight`（纯函数）
- `align`       ：RDF 坐标空间 → 数据集坐标空间的相似变换（纯函数）
- `mesh_build`  ：补 60 个牙齿顶点（5023 → 5083）
- `rdf_wire`    ：桥接帧格式编解码（纯函数）
- `rdf_client`  ：TCP 客户端 + 最新帧槽位（后台线程）
- `live_source` ：观测 → `(mesh, blend_weight)`
"""

# ⚠️ **刻意不在这里 import 子模块**：`mesh_build` → `data.flame_model` →
#    `compat` → chumpy。桥接进程只需要 `rdf_wire`（纯 struct 打包），
#    不该因为"导入了 tracking 包"而被迫 import chumpy。
#    需要什么就 `from live3dgsavatar.tracking import rdf_wire`。

__all__ = ["align", "live_source", "mesh_build", "param_map",
           "rdf_client", "rdf_wire"]
