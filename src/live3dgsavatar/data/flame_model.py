"""FLAME 最小前向：从 3DMM 参数算出网格顶点。

**为什么自己写**：渲染"参照格式"的数据集时需要把 `checkpoint/` 里的
3DMM 参数变成网格顶点。原实现依赖一个 925 行的 FLAME 封装（含 chumpy、
landmark、多套 forward 变体），本项目只需要**一条**路径：
shape + expression + neck/jaw/eye 旋转 +  eyelids → 顶点。

因此这里只保留 LBS 所需的数学，不引入任何额外依赖（除 torch/numpy/pickle）。

## 与 RGBAvatar 的等价性

数学与 `submodules/flame` 的实现一致（同一份 `generic_model.pkl`，
同样的关节树与 posedirs 约定），差异仅在于：
- 不计算 landmark、不做 transfer、不暴露未使用的参数组合
- 牙齿顶点由 `flame_topology` 提供（同样 60 个，同样的蒙皮分配）
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

# ⚠️ 必须在 pickle.load 之前施加补丁：FLAME 资产与 `.frame` 里含
#    chumpy 对象，而 chumpy 在新版 numpy / Python 下需要别名补齐。
from .. import compat  # noqa: F401, E402  （导入即生效）

from .flame_topology import NUM_TEETH_VERTS


def rodrigues(rot: torch.Tensor) -> torch.Tensor:
    """轴角 `[..., 3]` → 旋转矩阵 `[..., 3, 3]`（与 FLAME 的 `batch_rodrigues` 同式）。"""
    theta = rot.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    k = rot / theta
    zero = torch.zeros_like(theta[..., 0])
    K = torch.stack([
        torch.stack([zero, -k[..., 2], k[..., 1]], dim=-1),
        torch.stack([k[..., 2], zero, -k[..., 0]], dim=-1),
        torch.stack([-k[..., 1], k[..., 0], zero], dim=-1),
    ], dim=-2)
    eye = torch.eye(3, dtype=rot.dtype, device=rot.device).expand_as(K)
    sin = torch.sin(theta)[..., None]
    cos = torch.cos(theta)[..., None]
    return eye + sin * K + (1.0 - cos) * (K @ K)


def _rigid_transform(
    rot_mats: torch.Tensor,   # [B, J, 3, 3]
    joints: torch.Tensor,     # [B, J, 3]
    parents: torch.Tensor,    # [J]
) -> tuple[torch.Tensor, torch.Tensor]:
    """沿关节树累积变换，返回 `(J_transformed, A)`，`A` 为 `[B, J, 4, 4]`。"""
    b, j = rot_mats.shape[0], rot_mats.shape[1]
    device, dtype = rot_mats.device, rot_mats.dtype

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms = torch.zeros(b, j, 4, 4, device=device, dtype=dtype)
    transforms[:, :, :3, :3] = rot_mats
    transforms[:, :, :3, 3] = rel_joints
    transforms[:, :, 3, 3] = 1.0

    # 按父节点顺序累积（kintree 保证父索引小于子索引）
    for i in range(1, j):
        transforms[:, i] = transforms[:, parents[i]] @ transforms[:, i]

    A = transforms.clone()
    A[:, :, :3, 3] -= torch.matmul(
        transforms[:, :, :3, :3], joints.unsqueeze(-1)).squeeze(-1)
    return transforms[:, :, :3, 3], A


class FlameModel:
    """最小 FLAME：资产加载一次，`forward` 只做 LBS。

    Attributes:
        v_template: `[V, 3]`（含 60 个牙齿顶点）
        shapedirs:  `[V, 3, 400]`（前 300 为 shape、后 100 为 expression）
        posedirs:   `[V*3, 36]` → 内部转成 `[36, V*3]`
        lbs_weights: `[V, 5]`
        J_regressor: `[5, V]`
        parents:    `[5]`
    """

    def __init__(self, flame_dir: Path, dtype: torch.dtype = torch.float64) -> None:
        self._flame_dir = Path(flame_dir)
        geom = Path(flame_dir) / "generic_model.pkl"
        if not geom.exists():
            raise FileNotFoundError(f"缺少 FLAME 模型：{geom}")
        with open(geom, "rb") as fh:
            data = pickle.load(fh, encoding="latin1")

        v_template = torch.from_numpy(np.asarray(data["v_template"])).to(dtype)
        shapedirs = torch.from_numpy(np.asarray(data["shapedirs"])).to(dtype)
        posedirs = torch.from_numpy(np.asarray(data["posedirs"])).to(dtype)
        weights = torch.from_numpy(np.asarray(data["weights"])).to(dtype)
        j_reg = data["J_regressor"]
        j_reg = torch.from_numpy(np.asarray(j_reg.todense() if hasattr(j_reg, "todense")
                                            else j_reg)).to(dtype)
        kintree = np.asarray(data["kintree_table"])

        # ⚠️ `kintree_table[0]` 是**父节点**索引（[1] 是子节点列表，用错会让关节树错位）
        self.parents = torch.from_numpy(kintree[0].astype(np.int64))
        self.parents[0] = -1
        self.v_template = v_template
        # `posedirs` 存成 [V,3,P]。参照的布局是：[V*3, P] 转置 → [P, V*3]。
        # 随后牙齿扩展要在 (J,9,V,3) 视图下补零**再展平**，
        # 直接对 [P, V*3] 拼零会得到错误的元素顺序。
        self.posedirs = posedirs.reshape(-1, posedirs.shape[-1]).T.contiguous()
        self.shapedirs = shapedirs
        self.lbs_weights = weights
        self.J_regressor = j_reg

        # ---- 牙齿扩展（与 RGBAvatar 的 add_teeth 一致）----
        n_orig = v_template.shape[0]
        self.v_template = torch.cat(
            [v_template, self._teeth_verts(v_template, dtype)], dim=0)
        # 牙齿的 **shape 方向**复制自嘴唇（否则张嘴时牙齿不跟随）；
        # ⚠️ **只复制 shape 列（前 300）**，表达式列保持零 ——
        #    牙齿不受表情基影响（上/下牙各跟随 neck/jaw 的刚体变换）。
        n_shape = 300
        teeth_sd = torch.zeros(NUM_TEETH_VERTS, 3, shapedirs.shape[-1], dtype=dtype)
        up_idx, lo_idx = self._lip_rings()
        mean_sd = (shapedirs[up_idx, :, :n_shape] + shapedirs[lo_idx, :, :n_shape]) / 2
        for rows in (torch.arange(0, 15), torch.arange(15, 30),
                     torch.arange(30, 45), torch.arange(45, 60)):
            teeth_sd[rows, :, :n_shape] = mean_sd
        self.shapedirs = torch.cat([shapedirs, teeth_sd], dim=0)
        # 牙齿扩展：在 (J-1, 9, V, 3) 视图下补零再展平（与参照逐位一致）
        n_j = self.parents.shape[0] - 1
        pd = self.posedirs.reshape(n_j, 9, n_orig, 3)
        pd = torch.cat([pd, torch.zeros(n_j, 9, NUM_TEETH_VERTS, 3, dtype=dtype)], dim=2)
        self.posedirs = pd.reshape(n_j * 9, (n_orig + NUM_TEETH_VERTS) * 3)

        lbsw = torch.cat(
            [weights, torch.zeros(NUM_TEETH_VERTS, weights.shape[-1], dtype=dtype)], dim=0)
        # ⚠️ 牙齿顶点的排列是**根+边缘交错**，不是"前 30 个是上牙"：
        #    [0:15]=上根  [15:30]=下根  [30:45]=上边缘  [45:60]=下边缘
        #    上牙 = 上根 + 上边缘，跟随 neck；下牙 = 下根 + 下边缘，跟随 jaw。
        upper = torch.cat([torch.arange(0, 15), torch.arange(30, 45)]) + n_orig
        lower = torch.cat([torch.arange(15, 30), torch.arange(45, 60)]) + n_orig
        # 与参照一致：**累加**（牙齿顶点原本权重为 0，累加后为 1）
        lbsw[upper, 1] += 1.0
        lbsw[lower, 2] += 1.0
        self.lbs_weights = lbsw

        self.J_regressor = torch.cat(
            [j_reg, torch.zeros(j_reg.shape[0], NUM_TEETH_VERTS, dtype=dtype)], dim=1)

    @staticmethod
    def _lip_rings() -> tuple[torch.Tensor, torch.Tensor]:
        """上下嘴唇外环的顶点索引（牙齿几何与 shape 方向的来源）。"""
        return (torch.tensor([1713, 1715, 1716, 1735, 1696, 1694, 1657,
                              3543, 2774, 2811, 2813, 2850, 2833, 2832, 2830]),
                torch.tensor([1576, 1577, 1773, 1774, 1795, 1802, 1865,
                              3503, 2948, 2905, 2898, 2881, 2880, 2713, 2712]))

    @staticmethod
    def _teeth_verts(v_template: torch.Tensor, dtype) -> torch.Tensor:
        """由嘴唇顶点推导 60 个牙齿顶点（与 `add_teeth` 同一算法）。"""
        up_idx, lo_idx = FlameModel._lip_rings()
        vu, vl = v_template[up_idx], v_template[lo_idx]
        mean_dist = (vu - vl).norm(dim=-1, keepdim=True).mean()
        mid = (vu + vl) / 2
        mid[:, 1] = mid[:, 1:2].mean(dim=0, keepdim=True)
        mid[:, 2] -= mean_dist * 2.0
        up_edge = mid.clone() - torch.tensor([[0, mean_dist, 0]], dtype=dtype) * 1.0
        up_root = up_edge + torch.tensor([[0, mean_dist, 0]], dtype=dtype) * 1.0
        lo_edge = mid.clone() + torch.tensor([[0, mean_dist, 0]], dtype=dtype) * 1.0
        lo_edge -= torch.tensor([[0, 0, mean_dist]], dtype=dtype) * 0.4
        lo_root = lo_edge - torch.tensor([[0, mean_dist, 0]], dtype=dtype) * 1.0
        return torch.cat([up_root, lo_root, up_edge, lo_edge], dim=0)

    def forward(
        self,
        shape: torch.Tensor,        # [B, 300]
        expression: torch.Tensor,   # [B, 100]
        neck: torch.Tensor,         # [B, 3, 3]  旋转矩阵
        jaw: torch.Tensor,          # [B, 3, 3]
        eyes: torch.Tensor,         # [B, 2, 3, 3]
        eyelids: torch.Tensor | None = None,   # [B, 2]
        root: torch.Tensor | None = None,      # [B, 3, 3]，默认单位阵
    ) -> torch.Tensor:
        """返回世界前空间下的顶点 `[B, V, 3]`（V = 5023+60）。"""
        b = shape.shape[0]
        dtype, device = self.v_template.dtype, self.v_template.device
        if root is None:
            root = torch.eye(3, dtype=dtype, device=device).expand(b, 3, 3)

        betas = torch.cat([shape.to(dtype), expression.to(dtype)], dim=1)
        rot = torch.cat([root.reshape(b, 1, 3, 3), neck.reshape(b, 1, 3, 3),
                         jaw.reshape(b, 1, 3, 3), eyes.reshape(b, 2, 3, 3)], dim=1)

        v = self.v_template.unsqueeze(0).expand(b, -1, -1).clone()
        if eyelids is not None:
            v = v + self._eyelid_dirs(device, dtype)[0] * eyelids[:, 0:1, None]
            v = v + self._eyelid_dirs(device, dtype)[1] * eyelids[:, 1:2, None]

        # shapedirs: [V, 3, D] → [V*3, D]，与 betas [B, D] 相乘
        n_dirs = betas.shape[1]
        sd = self.shapedirs[..., :n_dirs].reshape(-1, n_dirs)
        v_shaped = v + torch.matmul(betas, sd.T).view(b, -1, 3)
        joints = torch.matmul(self.J_regressor, v_shaped)

        pose_feature = (rot[:, 1:] - torch.eye(3, dtype=dtype, device=device)).reshape(b, -1)
        v_posed = v_shaped + torch.matmul(pose_feature, self.posedirs).view(b, -1, 3)

        _, A = _rigid_transform(rot, joints, self.parents)
        T = torch.matmul(self.lbs_weights.unsqueeze(0).expand(b, -1, -1),
                         A.reshape(b, A.shape[1], 16)).view(b, -1, 4, 4)
        homo = torch.cat([v_posed, torch.ones(b, v_posed.shape[1], 1,
                                              dtype=dtype, device=device)], dim=2)
        return torch.matmul(T, homo.unsqueeze(-1))[:, :, :3, 0]

    def joints(self, shape: torch.Tensor, expression: torch.Tensor) -> torch.Tensor:
        """零姿态下的关节位置 `[B, 5, 3]`（shape+exp 决定）。"""
        betas = torch.cat([shape.to(self.v_template.dtype),
                           expression.to(self.v_template.dtype)], dim=1)
        n_dirs = betas.shape[1]
        sd = self.shapedirs[..., :n_dirs].reshape(-1, n_dirs)
        v_shaped = self.v_template.unsqueeze(0) + torch.matmul(betas, sd.T
                                                              ).view(betas.shape[0], -1, 3)
        return torch.matmul(self.J_regressor, v_shaped)

    def _eyelid_dirs(self, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """眼睑位移向量（`l/r_eyelid.npy`，形状 `[V,3]`），补零到牙齿顶点数。"""
        from .flame_topology import asset_paths  # noqa: PLC0415

        if not hasattr(self, "_eyelids"):
            geom = Path(asset_paths(self._flame_dir)[0]).parent
            dirs = []
            n_base = self.v_template.shape[0] - NUM_TEETH_VERTS
            for name in ("l_eyelid", "r_eyelid"):
                arr = np.load(geom / f"{name}.npy")           # [5023, 3]
                t = torch.from_numpy(arr).to(dtype)
                pad = torch.zeros(NUM_TEETH_VERTS, 3, dtype=dtype)
                dirs.append(torch.cat([t, pad], dim=0).to(device)[None])
            self._eyelids = tuple(dirs)
        return self._eyelids
