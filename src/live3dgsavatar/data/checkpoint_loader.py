"""加载"参照格式"的数据集：`<subject>/{images,checkpoint}/`。

每帧一个 `.frame` 文件（torch pickle），含 FLAME 参数与相机。
它**没有**现成的网格顶点，所以需要跑一次 FLAME 前向 —— 由本项目的
`flame_model.FlameModel` 完成（不依赖任何外部仓库）。

## 与训练侧的一致性

`checkpoint` 里的参数是**训练时**用的那套。这里必须复现训练侧的两步处理，
否则渲染出的网格会与模型训练时的网格不一致：

1. **neck 重定向**：`global_rots` 存的是根旋转，但训练侧把它转到 neck 关节，
   并相应修正平移（`transform_flame_params`）。
2. **全局变换**：`verts = R · verts + t`，`R`/`t` 来自 `opencv` 相机外参。

## 129 维驱动参数

```
[  0:100] 表情          flame.exp
[100:106] 头姿 rot6d    global_rots 的前两行
[106:112] 下颌 rot6d    jaw
[112:124] 眼球 rot6d    eyes（两组）
[124:126] 眼睑          eyelids
[126:129] 平移          global_transls
```
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .flame_model import FlameModel


def _rot6d_from_matrix(m: torch.Tensor) -> torch.Tensor:
    """`[..., 3, 3]` → `[..., 6]`，取**前两行**（与 `rotation_6d_to_matrix` 互逆）。"""
    return m[..., :2, :].reshape(*m.shape[:-2], 6)


def _matrix_from_rot6d(d6: torch.Tensor) -> torch.Tensor:
    """`[..., 6]` → `[..., 3, 3]`，Gram-Schmidt 造正交矩阵（行向量为基）。"""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


@dataclass
class CheckpointFrame:
    """一帧的参数（世界空间网格延迟计算）。"""

    shape: torch.Tensor        # [300] float64
    expression: torch.Tensor   # [100]
    jaw: torch.Tensor          # [6]  rot6d
    eyes: torch.Tensor         # [12] rot6d（两组）
    eyelids: torch.Tensor      # [2]
    global_rot: torch.Tensor   # [3,3]
    global_transl: torch.Tensor  # [3]


class CheckpointDataset:
    """读 `checkpoint/` 目录，按需跑 FLAME 前向得到网格顶点。

    只在**首次访问某帧**时计算并缓存；不预计算全部帧（254 帧 × 5083 顶点
    在 CPU 上要几秒，而 GUI 通常只播一部分）。
    """

    def __init__(self, data_dir: Path, dtype: torch.dtype = torch.float64) -> None:
        self.root = Path(data_dir)
        ckpt = self.root / "checkpoint"
        if not ckpt.is_dir():
            raise FileNotFoundError(f"缺少 checkpoint 目录：{ckpt}")
        self.ckpt_dir = ckpt
        # 与 images/ 用同一套排序，保证帧序一致
        self.files = sorted(p for p in ckpt.iterdir() if p.is_file())
        if not self.files:
            raise ValueError(f"{ckpt} 里没有帧文件")
        self.dtype = dtype

        first = self._read_raw(self.files[0])
        self.camera_K = torch.from_numpy(np.asarray(first["opencv"]["K"])[0]).to(dtype)
        self.img_size = tuple(int(x) for x in np.asarray(first["img_size"]).ravel())
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        # 相机外参：与训练侧一致 —— `diag(1,-1,-1,1)` 再把 opencv.t 放进平移位
        extri = torch.eye(4, dtype=dtype)
        extri[0, 0] = extri[1, 1] = extri[2, 2] = 1.0
        extri[1, 1] = -1.0
        extri[2, 2] = -1.0
        extri[:3, 3] = torch.from_numpy(
            np.asarray(first["opencv"]["t"], dtype=np.float64)[0]).to(dtype)
        self.camera_extri = extri
        self.camera_R = extri[:3, :3]
        self.camera_T = extri[:3, 3]

    @staticmethod
    def _read_raw(path: Path) -> dict:
        """`.frame` 是 torch 的序列化格式（含 chumpy 对象），必须用 `torch.load`。"""
        return torch.load(path, weights_only=False, map_location="cpu")

    def __len__(self) -> int:
        return len(self.files)

    def _load_params(self, idx: int) -> CheckpointFrame:
        import torch as _t

        d = self._read_raw(self.files[idx])
        fl = d["flame"]
        shape = _t.from_numpy(np.asarray(fl["shape"], np.float64)[0])
        exp = _t.from_numpy(np.asarray(fl["exp"], np.float64)[0])
        jaw6 = _t.from_numpy(np.asarray(fl["jaw"], np.float64)[0])
        eyes12 = _t.from_numpy(np.asarray(fl["eyes"], np.float64)[0])
        eyelids = _t.from_numpy(np.asarray(fl["eyelids"], np.float64)[0])

        # 全局位姿 = inv(camera_extri) @ [opencv.R | opencv.t]
        # （把每帧位姿变换到**第 0 帧相机**坐标系，与训练侧一致）
        cam_R = _t.from_numpy(np.asarray(d["opencv"]["R"], np.float64)[0])
        cam_t = _t.from_numpy(np.asarray(d["opencv"]["t"], np.float64)[0])
        xf = _t.eye(4, dtype=_t.float64)
        xf[:3, :3] = cam_R
        xf[:3, 3] = cam_t
        xf = _t.linalg.inv(self.camera_extri) @ xf
        return CheckpointFrame(shape=shape, expression=exp, jaw=jaw6, eyes=eyes12,
                               eyelids=eyelids, global_rot=xf[:3, :3],
                               global_transl=xf[:3, 3])

    def mesh_and_weight(self, idx: int, flame: FlameModel) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 `(mesh [V,3] float32, blend_weight [129] float32)`。

        复现训练侧的处理顺序：

        1. FLAME 前向（root 为单位阵）→ 零全局姿态下的顶点
        2. neck 重定向 + 全局变换 `v' = R·v + t`
        """
        cached = self._cache.get(idx)
        if cached is not None:
            return cached

        p = self._load_params(idx)
        d = p.shape.dtype
        eye3 = torch.eye(3, dtype=d)
        jaw_m = _matrix_from_rot6d(p.jaw.reshape(1, 6))[0]
        eyes_m = _matrix_from_rot6d(p.eyes.reshape(2, 6))

        # 归一化 rot6d（供 FLAME 前向与 blend_weight 共用）
        jaw6n = _rot6d_from_matrix(_matrix_from_rot6d(p.jaw.reshape(1, 6))[0])
        eyes6n = _rot6d_from_matrix(_matrix_from_rot6d(p.eyes.reshape(2, 6))).reshape(-1)

        # ---- neck 重定向（与训练侧 transform_flame_params 同式）----
        # 训练侧把"全局旋转"从 root 挪到 neck 关节，root 变为单位阵，
        # 并相应修正平移： t' = R·neck_j - neck_j + t
        neck_j = flame.joints(p.shape[None], p.expression[None])[0][1]
        transl = p.global_rot @ neck_j - neck_j + p.global_transl

        # 1) FLAME 前向：root=单位阵，neck=原全局旋转。
        #    ⚠️ jaw/eyes 传**归一化后**的矩阵（与训练侧一致：那里也是
        #       `rotation_6d_to_matrix(原始6D)` 的结果）。
        verts = flame.forward(
            p.shape[None], p.expression[None], p.global_rot[None],
            _matrix_from_rot6d(jaw6n.reshape(1, 6))[0][None],
            _matrix_from_rot6d(eyes6n.reshape(2, 6))[None],
            p.eyelids[None])[0]

        # 2) 全局变换：重定向后 `global_rots` 变成**单位阵**（旋转已在 neck 里），
        #    因此这里只加平移（与训练侧 precompute_bind_mesh 等价）。
        verts = verts + transl

        # 3) 129 维驱动参数
        #    ⚠️ 下颌/眼球段用**归一化后**的 rot6d，而不是 checkpoint 里的原始值：
        #    checkpoint 存的是网络原始 6D 输出（未正交），而训练侧的 `blend_weight`
        #    取自 `rotation_6d_to_matrix(...)` 的**前两行**（已 Gram-Schmidt 正交化）。
        #    直接用原值会让下颌/眼球与训练时不一致。
        #    头姿段用**原始全局旋转**（训练侧取自交换后的 `neck_poses`）。
        w = torch.cat([
            p.expression,                    # 100 表情
            _rot6d_from_matrix(p.global_rot),  # 6  头姿（原全局旋转）
            jaw6n,                           # 6  下颌（归一化后）
            eyes6n,                          # 12 眼球（归一化后）
            p.eyelids,                       # 2  眼睑
            transl,                          # 3  平移（重定向后）
        ]).to(torch.float32)
        out = (verts.to(torch.float32), w)
        self._cache[idx] = out
        return out
