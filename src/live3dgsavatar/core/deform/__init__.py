"""变形算子：blend（权重→切空间高斯）、bind（切空间→世界空间）、tbn、binding 构建。"""

from .bind import Binding, MeshBinder, matrix_to_quaternion, quaternion_multiply
from .binding import build_binding, compute_uv_rast_info
from .blend import GaussianBlendField, linear_blending
from .tbn import compute_face_normal, compute_face_tbn

__all__ = [
    "Binding",
    "GaussianBlendField",
    "MeshBinder",
    "build_binding",
    "compute_face_normal",
    "compute_face_tbn",
    "compute_uv_rast_info",
    "linear_blending",
    "matrix_to_quaternion",
    "quaternion_multiply",
]
from .capacity import (allocate_bary, capacity_binding,  # noqa: F401
                       capacity_summary, face_area_uv, face_capacity)
