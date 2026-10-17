"""绑定：把切空间高斯搬到世界空间。

    xyz_world  = R @ xyz_tangent + Σ_i bary_i · v_i
    rot_world  = q(R) ⊗ rot_tangent

其中 `R = TBN[face_id]`，**列 j 是第 j 个基向量**（见 `tbn.py`），
`bary = binding_face_bary`。

## 为什么是 `R · x` 而不是 `Rᵀ · x`

`mesh_binding` 是**逐元素**算子：`R` 与 `Rᵀ` 都数学自洽，
**唯一有效判据是与参照逐位一致**（参照 `gaussian_deform_batch` 用
`binding_rotations @ gs.xyz`）。⚠️ 切勿改成 `Rᵀ`。

**易混淆点**：CUDA `face_tbn.cu` 里 `TBNs[idx] = transpose(mat3(t,b,n))`
（**行**为基），而 Python 侧的 `utils.compute_face_tbn` 与本项目 `tbn.py`
都是**列**为基 —— 两者互为转置，**切勿跨来源比对**。`mesh_binding` 内部那次
`transpose` 是针对**调用方传入布局**的修正，不能脱离"传入的是什么"来判断。

回归测试：`test_bind_matches_reference_translation`、
`test_mesh_binder_uses_R_not_transpose`。详见 `docs/MIGRATION.md` D 节。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..types import GaussianSet, Mesh
from .tbn import compute_face_tbn


def quaternion_multiply(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """四元数乘法（WXYZ）。形状按 batch 广播，等价于 `utils.quaternion_multiply`。"""
    vector = (
        p[..., None, 0] * q[..., 1:]
        + q[..., None, 0] * p[..., 1:]
        + torch.cross(p[..., 1:], q[..., 1:], dim=-1)
    )
    last = p[..., 0] * q[..., 0] - torch.sum(p[..., 1:] * q[..., 1:], dim=-1)
    return torch.cat((last[..., None], vector), dim=-1)


def matrix_to_quaternion(m: torch.Tensor) -> torch.Tensor:
    """旋转矩阵 → 单位四元数（WXYZ）。

    与 CUDA 的 `matrix_to_quaternion`、Python 的 `utils.matrix_to_quaternion`
    **逐位一致**（实测 `max|Δ| = 0`，见 `tests/unit/test_deform.py`）。

    实现要点：Shepperd 四分支，用「一次算四个候选 + 按判据选取」代替
    `torch.nonzero` 分支。`nonzero` 的输出形状依赖数据，会强制 GPU 同步
    （device→host 往返），在逐帧路径上是明显开销。

    ⚠️ **分支判据必须与参照一致**，不能图省事改成"选范数最大的候选"：
    虽然那在纯旋转下等价（且数值上更稳），但对**非正交矩阵**（真实 TBN 就是）
    会选到**不同的分支**，从而给出不同的四元数 —— 那就不再与参照逐位一致。

    判据沿用参照写法：在 `[diag0, diag1, diag2, trace]` 上取 `argmax`
    （`argmax` 取首个最大值，故平局时优先级为 diag0 > diag1 > diag2 > trace），
    再映射到候选顺序 `[trace, diag0, diag1, diag2]`。

    Args:
        m: `[..., 3, 3]`。不要求严格正交；此时结果为"最接近的"单位四元数。
    """
    if m.shape[-2:] != (3, 3):
        raise ValueError(f"期望 [..., 3, 3]，实际 {tuple(m.shape)}")
    shape = m.shape[:-2]
    f = m.reshape(-1, 3, 3)
    m00, m01, m02 = f[:, 0, 0], f[:, 0, 1], f[:, 0, 2]
    m10, m11, m12 = f[:, 1, 0], f[:, 1, 1], f[:, 1, 2]
    m20, m21, m22 = f[:, 2, 0], f[:, 2, 1], f[:, 2, 2]
    trace = m00 + m11 + m22
    one = torch.ones_like(trace)

    x = m21 - m12
    y = m02 - m20
    z = m10 - m01
    # 候选顺序与参照的分支顺序一致：[trace, diag0, diag1, diag2]
    cand = torch.stack([
        torch.stack([one + trace, x, y, z], dim=1),
        torch.stack([x, one - trace + 2 * m00, m01 + m10, m02 + m20], dim=1),
        torch.stack([y, m01 + m10, one - trace + 2 * m11, m12 + m21], dim=1),
        torch.stack([z, m02 + m20, m12 + m21, one - trace + 2 * m22], dim=1),
    ], dim=1)                                            # [n, 4, 4]

    # 判据与参照完全一致：argmax([diag0, diag1, diag2, trace])
    scores = (m00, m11, m22, trace)
    pick = torch.zeros_like(trace, dtype=torch.long)     # 默认第一个最大 → trace 分支
    best = trace
    for idx, value in ((1, m00), (2, m11), (3, m22)):
        # torch.argmax 取首个最大值 ⟹ 严格大于时才替换（保持 diag0>diag1>diag2>trace）
        take = value > best
        pick = torch.where(take, torch.full_like(pick, idx), pick)
        best = torch.where(take, value, best)
    assert len(scores) == 4

    quat = cand.gather(1, pick.view(-1, 1, 1).expand(-1, 1, 4)).squeeze(1)
    quat = torch.nn.functional.normalize(quat, dim=1)
    return quat.reshape(*shape, 4)


def _matrix_to_quaternion_branchwise(m: torch.Tensor) -> torch.Tensor:
    """旧实现（按迹/对角元选分支 + `nonzero`），**仅供测试作基准**，勿在生产路径使用。

    保留它是为了证明上面的向量化实现逐位等价（含四个分支各自被覆盖的情形）。
    """
    if m.shape[-2:] != (3, 3):
        raise ValueError(f"期望 [..., 3, 3]，实际 {tuple(m.shape)}")
    shape = m.shape[:-2]
    flat = m.reshape(-1, 3, 3)
    n = flat.shape[0]
    dtype, device = m.dtype, m.device

    diag = torch.stack([flat[:, 0, 0], flat[:, 1, 1], flat[:, 2, 2]], dim=1)
    trace = diag.sum(dim=1, keepdim=True)
    decision = torch.cat([diag, trace], dim=1)
    choices = decision.argmax(dim=1)

    quat = torch.empty(n, 4, dtype=dtype, device=device)
    idx1 = torch.nonzero(choices != 3, as_tuple=True)[0]
    idx2 = torch.nonzero(choices == 3, as_tuple=True)[0]

    if idx1.numel():
        i = choices[idx1]
        j = (i + 1) % 3
        k = (j + 1) % 3
        rows = torch.arange(i.numel(), device=device)
        fsel = flat[idx1]
        quat[idx1, i + 1] = 1 - trace[idx1, 0] + 2 * fsel[rows, i, i]
        quat[idx1, j + 1] = fsel[rows, j, i] + fsel[rows, i, j]
        quat[idx1, k + 1] = fsel[rows, k, i] + fsel[rows, i, k]
        quat[idx1, 0] = fsel[rows, k, j] - fsel[rows, j, k]

    if idx2.numel():
        fsel = flat[idx2]
        quat[idx2, 1] = fsel[:, 2, 1] - fsel[:, 1, 2]
        quat[idx2, 2] = fsel[:, 0, 2] - fsel[:, 2, 0]
        quat[idx2, 3] = fsel[:, 1, 0] - fsel[:, 0, 1]
        quat[idx2, 0] = 1 + trace[idx2, 0]

    quat = torch.nn.functional.normalize(quat, dim=1)
    return quat.reshape(*shape, 4)


@dataclass
class Binding:
    """高斯与模板三角面的绑定关系（由 UV 光栅化一次性确定，不参与训练）。

    Attributes:
        face_id:   [N] int64 每个高斯绑定的三角面下标（0-based）
        face_bary: [N, 3] 该面内的重心坐标 `(u, v, 1-u-v)`
        valid_mask: UV 图掩码。两种形状并存：
            - `[H*W]`（**构建时**，来自 UV 光栅化）：每个 texel 是否绑定高斯，
              可用于 `extract_texture` 重建 UV 图；
            - `[N]`（**从 .ply 加载时**）：文件只存有效 texel，故等价于全 True，
              此时无法还原 texel 位置（`extract_texture` 会显式报错）。
    """

    face_id: torch.Tensor
    face_bary: torch.Tensor
    valid_mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.face_id.ndim != 1:
            raise ValueError(f"face_id 形状应为 [N]，实际 {tuple(self.face_id.shape)}")
        if self.face_bary.shape != (self.face_id.shape[0], 3):
            raise ValueError(
                f"face_bary 形状应为 [N, 3]，实际 {tuple(self.face_bary.shape)}"
                f"（N = {self.face_id.shape[0]}）")
        if self.valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask 应为 bool")

    @property
    def num_gaussians(self) -> int:
        return self.face_id.shape[0]

    def to(self, device: torch.device | str) -> "Binding":
        return Binding(self.face_id.to(device), self.face_bary.to(device),
                       self.valid_mask.to(device))


class MeshBinder:
    """`Binder` 的实现：网格 + 绑定 → 世界空间高斯。

    对应 CUDA 的 `mesh_binding`。

    ⚠️ **面 TBN 不能跨帧缓存**：`TBN = f(face_vertices, face_uvs)` 中的
    `face_vertices` 来自**当前帧的 `mesh.verts`**，随帧变化；只有 UV 与拓扑固定。
    虽有缓存诱惑，但无法复用 —— `tests/unit/test_deform.py` 会抓住这类错误。
    """

    def __init__(self, binding: Binding) -> None:
        self.binding = binding
        # 缓存"绑定面索引"（纯模板依赖，与顶点位置无关）：
        # 避免每帧对全量 F 个面算 TBN，只算真正被高斯引用到的面。
        self._used_faces: torch.Tensor | None = None
        self._inv_index: torch.Tensor | None = None
        self._face_key: tuple | None = None

    def _face_index(self, mesh, face_id: torch.Tensor):
        """把 `face_id` 压缩成「去重后的面 + 反查索引」。

        返回 `(used, inv)`：
        - `used` `[Fu]`：真正被引用到的面下标（升序）
        - `inv`  `[N]` ：`face_id[n] == used[inv[n]]`

        这样只需对 `Fu` 个面算 TBN，而不是全量 `F` 个。
        纯模板依赖（只看 `face_id` 与面数），可跨帧缓存。
        """
        key = (int(face_id.shape[0]), int(mesh.num_faces), int(face_id.data_ptr()),
               int(face_id.max()) if face_id.numel() else -1)
        if self._used_faces is None or self._face_key != key:
            used, inv = torch.unique(face_id, return_inverse=True)
            self._used_faces = used
            self._inv_index = inv
            self._face_key = key
        assert self._used_faces is not None and self._inv_index is not None
        return self._used_faces, self._inv_index

    def bind(self, gaussians: GaussianSet, mesh: Mesh) -> GaussianSet:
        if gaussians.space != "tangent":
            raise ValueError(
                f"MeshBinder 的输入必须是切空间高斯，实际 space={gaussians.space!r}")
        if gaussians.batch_size != mesh.batch_size:
            raise ValueError(
                f"batch 不一致：高斯 {gaussians.batch_size}，网格 {mesh.batch_size}")

        face_id = self.binding.face_id
        bary = self.binding.face_bary.to(dtype=mesh.verts.dtype, device=mesh.verts.device)

        # 索引合法性：**负索引会被 PyTorch 当作「从末尾数」而静默取到错误的三角面**，
        # 越界索引同样不会报错。必须在 gather 之前拦截。
        n_faces = mesh.num_faces
        if face_id.numel():
            lo, hi = int(face_id.min()), int(face_id.max())
            if lo < 0 or hi >= n_faces:
                raise ValueError(
                    f"binding.face_id 超出网格面数范围 [{lo}, {hi}]，"
                    f"合法区间为 [0, {n_faces - 1}]。"
                    "（负值会被当作从末尾索引，越界值会静默给出错误结果）")

        # 只对**被高斯引用到的面**计算 TBN。
        # 绑定索引是纯模板依赖（与顶点位置无关），可安全缓存；
        # 而 TBN 本身依赖每帧的 `mesh.verts`，必须逐帧重算。
        used, inv = self._face_index(mesh, face_id)
        tri_used = mesh.verts[:, mesh.faces[used]]                  # [B, Fu, 3, 3]
        tbn_used = compute_face_tbn(tri_used, mesh.uvs[mesh.uv_faces[used]])

        binding_rot = tbn_used[:, inv]                              # [B, N, 3, 3]
        binding_tri = tri_used[:, inv]                              # [B, N, 3, 3]
        offset = (binding_tri * bary.unsqueeze(0).unsqueeze(-1)).sum(dim=-2)  # [B, N, 3]

        # ⚠️ 用 R（不是 Rᵀ）：与参照 `gaussian_deform_batch` 一致。
        #    判据是"与参照逐位一致"，不是自证；两条独立证据见模块 docstring。
        xyz = (binding_rot @ gaussians.xyz.unsqueeze(-1)).squeeze(-1) + offset

        rot = quaternion_multiply(matrix_to_quaternion(binding_rot), gaussians.rotation)

        return GaussianSet(
            xyz=xyz,
            rotation=rot,
            scaling=gaussians.scaling,
            opacity=gaussians.opacity,
            color=gaussians.color,
            space="world",
        )
