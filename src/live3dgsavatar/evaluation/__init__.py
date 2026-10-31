"""评估与缺陷定位：判断一个 avatar"哪里不行"。

分成三层，各自单一职责：

| 模块 | 职责 | 依赖 |
|---|---|---|
| `views.py` | 候选视角网格（可枚举、可去重） | 无 |
| `protocol.py` | 有真值 / 无真值的度量与报告结构 | `core.metrics` |
| （后续）`defects.py` | 缺陷信号与归因（覆盖率、floater、Δmap） | `core` + `data` |

**为什么独立成包**：评估要同时被脚本（`render_test.py`）、训练验证钩子和 GUI
使用，而它既不属于纯渲染（`core/` 不许碰数据集），也不属于优化（`training/`
必须能脱离数据集使用）。因此这里的依赖方向是 `evaluation → core (+ data)`，
且**任何上层模块都不被它反向依赖**。
"""

from __future__ import annotations

from .protocol import (EvaluationReport, summarize_no_reference,  # noqa: F401
                       summarize_quality)
from .defects import (DefectMap, alpha_hole_ratio,  # noqa: F401
                      attribute_to_faces, build_defect_map, coverage_counts,
                      coverage_defect, floater_offsets, hole_rate, per_texel,
                      refine_delta, sharpness_map)
from .expressions import (ExpressionSpec, axis_extremes,  # noqa: F401
                          cluster, cluster_frames, expression_matrix,
                          load_expressions, load_named, merge_specs)
from .selection import Candidate, CoverageResult, greedy_cover  # noqa: F401
from .views import ViewSpec, candidate_views, grid_size  # noqa: F401

__all__ = [
    "Candidate",
    "CoverageResult",
    "DefectMap",
    "EvaluationReport",
    "ExpressionSpec",
    "ViewSpec",
    "alpha_hole_ratio",
    "attribute_to_faces",
    "axis_extremes",
    "build_defect_map",
    "candidate_views",
    "cluster",
    "cluster_frames",
    "coverage_counts",
    "expression_matrix",
    "floater_offsets",
    "greedy_cover",
    "grid_size",
    "load_expressions",
    "load_named",
    "merge_specs",
    "per_texel",
    "coverage_defect",
    "hole_rate",
    "refine_delta",
    "sharpness_map",
    "summarize_no_reference",
    "summarize_quality",
]
