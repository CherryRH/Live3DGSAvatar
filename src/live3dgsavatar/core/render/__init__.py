"""渲染算子：光栅化封装与相机矩阵边界转换。"""

from .camera_utils import project_points, to_kernel_matrix
from .normals import normal_map, vertex_normals
from .pipeline import render_frame, render_gaussians
from .rasterizer import SimpleRasterizer

__all__ = [
    "SimpleRasterizer",
    "normal_map",
    "project_points",
    "render_frame",
    "render_gaussians",
    "to_kernel_matrix",
    "vertex_normals",
]
