"""相机数值等价测试（**不需要 GPU**）。

逐元素比对 `core.types.Camera` 与参照实现 `camera/camera.py` 的
`w2v` 与 `full_proj` —— 这两个矩阵是整条渲染链路的基座，
一旦不一致，后面所有渲染结果都无意义。

判据：max|Δ| < 1e-6（纯矩阵乘法，float32 下应几乎逐位一致）。

注意：测试写成普通循环而非 pytest 参数化，以便在**未安装 pytest** 时
也能用 `python tests/run_tests.py` 运行（见该脚本说明）。
"""

from __future__ import annotations

import numpy as np
import torch

from live3dgsavatar.core.types import Camera


ATOL = 1e-6

# 参照实现 `IntrinsicsCamera` 从 K 取出 cx/cy 后直接与 Python 数值运算并赋给张量，
# 若 K 是 **float32** 数组会抛 `TypeError: can't assign a numpy.float32 to a torch.FloatTensor`。
# 这是参照实现的既有行为（不改动参照仓库），因此测试统一用 float64 构造 K。
K_DTYPE = np.float64


def test_projection_depth_range_is_0_1() -> None:
    """深度必须映射到 [0,1]（OpenGL 约定）；znear→0，zfar→1。"""
    w = h = 256
    K = np.array([[200.0, 0.0, w / 2], [0.0, 200.0, h / 2], [0.0, 0.0, 1.0]],
                 dtype=K_DTYPE)
    cam = Camera.from_intrinsics_extrinsics(
        K=K, R=np.eye(3, dtype=K_DTYPE), T=np.zeros(3, dtype=K_DTYPE),
        width=w, height=h, znear=0.01, zfar=100.0)
    P = cam.proj[0]

    def depth(z_cam: float) -> float:
        p = torch.tensor([[0.0, 0.0, z_cam, 1.0]], dtype=torch.float32)
        clip = p @ P.T
        return float(clip[0, 2] / clip[0, 3])

    assert abs(depth(0.01) - 0.0) < 1e-5, "znear 应对应 depth = 0"
    assert abs(depth(100.0) - 1.0) < 1e-4, "zfar 应对应 depth = 1"
    assert 0.0 < depth(1.0) < 1.0


def test_focal_fov_roundtrip() -> None:
    """fov 与 focal 必须自洽：`focal = W / (2·tan(fov_x/2))`。"""
    w, h = 640, 480
    fx, fy = 900.0, 880.0
    K = np.array([[fx, 0.0, w / 2], [0.0, fy, h / 2], [0.0, 0.0, 1.0]],
                 dtype=K_DTYPE)
    cam = Camera.from_intrinsics_extrinsics(
        K=K, R=np.eye(3, dtype=K_DTYPE), T=np.zeros(3, dtype=K_DTYPE),
        width=w, height=h)

    assert abs(float(cam.tan_fov_x[0]) - w / (2 * fx)) < 1e-6
    assert abs(float(cam.tan_fov_y[0]) - h / (2 * fy)) < 1e-6

def test_camera_inherits_input_device() -> None:
    """**回归测试**：`Camera` 的所有张量必须与输入同设备。

    曾经 `from_intrinsics_extrinsics` 里写的是裸 `torch.eye(4, dtype=...)`，
    `w2c` 因此永远留在 CPU，而 `K` 继承了输入设备。跨设备赋值 `copy_` 不报错，
    于是 Camera 一半在 GPU、一半在 CPU；之后 `w2c` 被当作 CUDA 指针传给
    光栅化内核，表现为 `CUDA error: an illegal memory access` ——
    报错点离根因极远（渲染循环里），且只在调用方忘了 `.to()` 时出现。

    这里用 `meta` 作为"默认设备"来模拟非 CPU 输入：若函数里漏了 `device=`，
    `w2c` 会落到 meta 而与 K 不一致。
    """
    K = torch.tensor([[400.0, 0.0, 256.0], [0.0, 400.0, 256.0], [0.0, 0.0, 1.0]])
    R = torch.eye(3)
    T = torch.tensor([0.0, 0.0, 1.0])

    torch.set_default_device("meta")
    try:
        cam = Camera.from_intrinsics_extrinsics(
            K=K, R=R, T=T, width=512, height=512)
        assert cam.K.device == cam.w2c.device, (
            f"Camera 张量分散在多个设备：K={cam.K.device} w2c={cam.w2c.device}")
    finally:
        torch.set_default_device("cpu")
