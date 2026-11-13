"""跨层核心数据类型。

设计原则（见 docs/CONVENTIONS.md）：
- 所有张量 **batch-first**，第一维恒为 batch（单帧为 1）；
- `GaussianSet` 内**永远存放已激活的物理量**（opacity∈[0,1]、scaling>0、rotation 单位四元数）；
- 空间归属**显式标注**，不靠调用顺序隐含约定；
- 本模块只做数据与形状校验，**不做 I/O、不引入训练逻辑**。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal, Optional

import numpy as np
import torch

Space = Literal["tangent", "world"]


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


@dataclass
class GaussianSet:
    """一批可直接交给光栅化器的高斯。

    形状（B = batch，N = 高斯数）::

        xyz      [B, N, 3]
        rotation [B, N, 4]   单位四元数，WXYZ
        scaling  [B, N, 3]   正尺度（已 exp）
        opacity  [B, N, 1]   [0, 1]（已 sigmoid）
        color    [B, N, 1, 3] SH 0 阶系数（DC）

    Attributes:
        space: ``"tangent"`` 表示位于绑定面的切空间，``"world"`` 表示已变形到世界空间。
            只有 ``Binder`` 的输出是 ``"world"``。
    """

    xyz: torch.Tensor
    rotation: torch.Tensor
    scaling: torch.Tensor
    opacity: torch.Tensor
    color: torch.Tensor
    space: Space = "tangent"

    def __post_init__(self) -> None:
        tensors = {
            "xyz": (self.xyz, 3),
            "rotation": (self.rotation, 4),
            "scaling": (self.scaling, 3),
            "opacity": (self.opacity, 1),
        }
        for name, (t, last) in tensors.items():
            _check(t.ndim == 3 and t.shape[-1] == last,
                   f"GaussianSet.{name} 形状应为 [B, N, {last}]，实际 {tuple(t.shape)}")
        _check(self.color.ndim == 4 and self.color.shape[-1] == 3,
               f"GaussianSet.color 形状应为 [B, N, 1, 3]，实际 {tuple(self.color.shape)}")

        n = self.xyz.shape[1]
        for name, (t, _) in tensors.items():
            _check(t.shape[1] == n, f"GaussianSet.{name} 的 N={t.shape[1]} 与 xyz 的 N={n} 不一致")
        _check(self.color.shape[1] == n,
               f"GaussianSet.color 的 N={self.color.shape[1]} 与 xyz 的 N={n} 不一致")
        _check(self.color.shape[2] == 1,
               f"GaussianSet.color 应为单阶 SH（第 2 维为 1），实际 {self.color.shape[2]}")

        b = self.xyz.shape[0]
        for name, (t, _) in tensors.items():
            _check(t.shape[0] == b, f"GaussianSet.{name} 的 B 不一致")

        if self.space not in ("tangent", "world"):
            raise ValueError(f"GaussianSet.space 只能是 'tangent' 或 'world'，实际 {self.space!r}")

    # ---------------------------------------------------------------- 属性 --
    @property
    def batch_size(self) -> int:
        return self.xyz.shape[0]

    @property
    def num_gaussians(self) -> int:
        return self.xyz.shape[1]

    @property
    def device(self) -> torch.device:
        return self.xyz.device

    @property
    def dtype(self) -> torch.dtype:
        return self.xyz.dtype

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (self.xyz, self.rotation, self.scaling, self.opacity, self.color)

    # ---------------------------------------------------------------- 变换 --
    def to(self, device: torch.device | str) -> "GaussianSet":
        return replace(self, **{k: t.to(device) for k, t in self._named().items()})

    def detach(self) -> "GaussianSet":
        return replace(self, **{k: t.detach() for k, t in self._named().items()})

    def to_space(self, space: Space) -> "GaussianSet":
        """改变空间标注（不做任何几何变换），用于显式断言空间流转。"""
        return replace(self, space=space)

    def _named(self) -> dict[str, torch.Tensor]:
        return {"xyz": self.xyz, "rotation": self.rotation, "scaling": self.scaling,
                "opacity": self.opacity, "color": self.color}


@dataclass
class Mesh:
    """一批已变形的网格。

    ``faces`` / ``uv_faces`` / ``uvs`` 是拓扑与 UV，**对所有 batch 与所有帧恒定**，
    因此不带 batch 维。

    Attributes:
        verts: [B, V, 3] 世界空间顶点
        faces: [F, 3] int32 三角面
        uvs:   [Vuv, 2] 模板 UV
        uv_faces: [F, 3] int32 UV 三角面
    """

    verts: torch.Tensor
    faces: torch.Tensor
    uvs: torch.Tensor
    uv_faces: torch.Tensor

    def __post_init__(self) -> None:
        _check(self.verts.ndim == 3 and self.verts.shape[-1] == 3,
               f"Mesh.verts 形状应为 [B, V, 3]，实际 {tuple(self.verts.shape)}")
        for name in ("faces", "uv_faces"):
            t = getattr(self, name)
            _check(t.ndim == 2 and t.shape[-1] == 3,
                   f"Mesh.{name} 形状应为 [F, 3]，实际 {tuple(t.shape)}")
        _check(self.uvs.ndim == 2 and self.uvs.shape[-1] == 2,
               f"Mesh.uvs 形状应为 [Vuv, 2]，实际 {tuple(self.uvs.shape)}")

    @property
    def batch_size(self) -> int:
        return self.verts.shape[0]

    @property
    def num_vertices(self) -> int:
        return self.verts.shape[1]

    @property
    def num_faces(self) -> int:
        return self.faces.shape[0]

    def to(self, device: torch.device | str) -> "Mesh":
        return Mesh(self.verts.to(device), self.faces.to(device),
                    self.uvs.to(device), self.uv_faces.to(device))


@dataclass
class Camera:
    """针孔相机。**只存 K 与 w2c**，其余全部派生（见 docs/CONVENTIONS.md §2.3）。

    Attributes:
        K:     [B, 3, 3] 或 [3, 3] 内参
        w2c:   [B, 4, 4] 或 [4, 4] world→camera
        width / height: 图像尺寸
        znear / zfar: 裁剪面；深度被映射到 **[0, 1]**（OpenGL 约定）
    """

    K: torch.Tensor
    w2c: torch.Tensor
    width: int
    height: int
    znear: float = 0.01
    zfar: float = 100.0

    def __post_init__(self) -> None:
        if self.K.ndim == 2:
            self.K = self.K.unsqueeze(0)
        if self.w2c.ndim == 2:
            self.w2c = self.w2c.unsqueeze(0)
        _check(self.K.ndim == 3 and self.K.shape[-2:] == (3, 3),
               f"Camera.K 形状应为 [B, 3, 3]，实际 {tuple(self.K.shape)}")
        _check(self.w2c.ndim == 3 and self.w2c.shape[-2:] == (4, 4),
               f"Camera.w2c 形状应为 [B, 4, 4]，实际 {tuple(self.w2c.shape)}")
        _check(self.K.shape[0] == self.w2c.shape[0],
               "Camera.K 与 Camera.w2c 的 B 不一致")

    # ------------------------------------------------------------- 基本量 --
    @property
    def batch_size(self) -> int:
        return self.w2c.shape[0]

    @property
    def device(self) -> torch.device:
        return self.w2c.device

    @property
    def fx(self) -> torch.Tensor:
        return self.K[:, 0, 0]

    @property
    def fy(self) -> torch.Tensor:
        return self.K[:, 1, 1]

    @property
    def cx(self) -> torch.Tensor:
        return self.K[:, 0, 2]

    @property
    def cy(self) -> torch.Tensor:
        return self.K[:, 1, 2]

    @property
    def fov_x(self) -> torch.Tensor:
        """[B]，弧度。`fov = 2·atan(W / (2·fx))`。"""
        return 2.0 * torch.atan(self.width / (2.0 * self.fx))

    @property
    def fov_y(self) -> torch.Tensor:
        """[B]，弧度。"""
        return 2.0 * torch.atan(self.height / (2.0 * self.fy))

    @property
    def proj(self) -> torch.Tensor:
        """[B, 4, 4] 投影矩阵（不含 view），深度映射到 [0, 1]。"""
        b = self.batch_size
        p = torch.zeros(b, 4, 4, dtype=self.w2c.dtype, device=self.device)
        w, h = float(self.width), float(self.height)
        n, f = float(self.znear), float(self.zfar)

        p[:, 0, 0] = 2.0 * self.fx / w
        p[:, 1, 1] = 2.0 * self.fy / h
        p[:, 0, 2] = -1.0 + 2.0 * (self.cx / w)
        p[:, 1, 2] = -1.0 + 2.0 * (self.cy / h)
        p[:, 3, 2] = 1.0
        p[:, 2, 2] = f / (f - n)
        p[:, 2, 3] = -(f * n) / (f - n)
        return p

    @property
    def full_proj(self) -> torch.Tensor:
        """[B, 4, 4] = proj @ w2c，一次性预乘，避免内核侧重复计算。"""
        return self.proj @ self.w2c

    @property
    def position(self) -> torch.Tensor:
        """[B, 3] 相机在世界空间的位置（w2c 的逆变换作用于原点）。"""
        r = self.w2c[:, :3, :3]
        t = self.w2c[:, :3, 3]
        return -(r.transpose(1, 2) @ t.unsqueeze(-1)).squeeze(-1)

    @property
    def tan_fov_x(self) -> torch.Tensor:
        """[B] `tan(fov_x / 2)`，直接喂给光栅化器。"""
        return torch.tan(self.fov_x * 0.5)

    @property
    def tan_fov_y(self) -> torch.Tensor:
        """[B] `tan(fov_y / 2)`。"""
        return torch.tan(self.fov_y * 0.5)

    # ------------------------------------------------------------- 构造器 --
    @classmethod
    def from_intrinsics_extrinsics(
        cls,
        K: torch.Tensor | "object",
        R: torch.Tensor | "object",
        T: torch.Tensor | "object",
        width: int,
        height: int,
        znear: float = 0.01,
        zfar: float = 100.0,
    ) -> "Camera":
        """从 OpenCV 风格的 `K, R, T`（world→camera 的旋转与平移）构造。

        与参照实现 `camera/camera.py::IntrinsicsCamera` 的换算式完全一致：

            rot = Rᵀ            # view→world 的旋转
            pos = -Rᵀ @ T       # 相机在世界中的位置
            w2c = [[rotᵀ, -rotᵀ @ pos], [0, 1]] = [[R, T], [0, 1]]
        """
        K_t = _as_tensor(K, dtype=torch.float32)
        R_t = _as_tensor(R, dtype=torch.float32)
        T_t = _as_tensor(T, dtype=torch.float32).reshape(-1)[:3]
        K_t = K_t.reshape(3, 3)
        R_t = R_t.reshape(3, 3)

        rot = R_t.transpose(0, 1)                 # view→world
        pos = -(rot @ T_t)                        # 相机位置
        # ⚠️ **必须继承输入的设备**。曾经写成 `torch.eye(4, dtype=...)`（无 device=），
        #    w2c 就永远留在 CPU；而 K 保留了输入设备。跨设备赋值 `copy_` 不会报错，
        #    于是 Camera 一半在 GPU、一半在 CPU。之后 w2c 被当作 CUDA 指针传给
        #    光栅化内核，表现为 `CUDA error: an illegal memory access` —— 
        #    报错点离根因极远，且只在"调用方忘了 .to()"时出现。
        w2c = torch.zeros(4, 4, dtype=torch.float32,
                          device=rot.device) + torch.eye(
            4, dtype=torch.float32, device=rot.device)
        w2c[:3, :3] = rot.transpose(0, 1)         # = R
        w2c[:3, 3] = T_t
        _ = pos  # 保留推导，便于对照参照实现；w2c 已由 R, T 直接给出
        return cls(K=K_t, w2c=w2c, width=width, height=height, znear=znear, zfar=zfar)

    @classmethod
    def from_fov(
        cls,
        fov_y: float,
        width: int,
        height: int,
        w2c: Optional[torch.Tensor] = None,
        znear: float = 0.01,
        zfar: float = 100.0,
    ) -> "Camera":
        """从竖直视场角构造（`PerspectiveCamera` 的等价形式）。

        `focal = H / (2·tan(fovy/2))`，再由 focal 反推 `fov_x` —— 与参照实现一致。
        """
        focal = height / (2.0 * math.tan(fov_y / 2.0))
        K = torch.tensor([[focal, 0.0, width / 2.0],
                          [0.0, focal, height / 2.0],
                          [0.0, 0.0, 1.0]], dtype=torch.float32)
        if w2c is None:
            w2c = torch.eye(4, dtype=torch.float32)
        return cls(K=K, w2c=w2c, width=width, height=height, znear=znear, zfar=zfar)

    def to(self, device: torch.device | str) -> "Camera":
        return Camera(self.K.to(device), self.w2c.to(device),
                      self.width, self.height, self.znear, self.zfar)


def _as_tensor(x, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    return torch.from_numpy(np.asarray(x, dtype="float32")).to(dtype)


@dataclass
class RenderOutput:
    """光栅化输出。

    ``est_color`` / ``est_weight`` / ``radii`` 仅在训练态（传入 ``target_image``）有意义，
    推理态为 ``None``。
    """

    color: torch.Tensor                      # [B, 3, H, W] 已叠背景
    alpha: torch.Tensor                      # [B, 1, H, W] 累积不透明度
    est_color: Optional[torch.Tensor] = None   # [B, N, 3]
    est_weight: Optional[torch.Tensor] = None  # [B, N]
    radii: Optional[torch.Tensor] = None       # [B, N] int32


@dataclass
class Frame:
    """一帧的全部输入。训练与推理共用。"""

    mesh: Mesh
    blend_weight: torch.Tensor                  # [B, D]
    camera: Camera
    image: Optional[torch.Tensor] = None        # [B, 3, H, W] GT，推理时为 None
    mask: Optional[torch.Tensor] = None         # [B, 1, H, W]

    @property
    def batch_size(self) -> int:
        return self.mesh.batch_size
