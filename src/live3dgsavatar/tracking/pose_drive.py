#!/usr/bin/env python
"""把任意帧的头"转"到目标角度：**直接对烘焙 mesh 做绕颈关节的蒙皮旋转**。

为什么不是相机绕转：数据集里的"大角度"是**头转、肩/胸不动**。绕头心转相机
会把胸口一起转走（与 GT 差约 12 dB），并在颈/胸边界教出错误的"背景"目标。

为什么不是"重算 FLAME 再配准"：烘焙数据走的是 `teeth(rdf.verts)`，而重算
FLAME + Kabsch 只能贴到 ~3 mm（p99 1–2 cm），`turn(idx, 0)` 都不是恒等。
这里改成**纯几何**做法，闭环天然成立：

    v' = v + w_h · (D·(v − J) + J − v)

- `w_h` 是每个顶点的头骨权重（`1 − root 权重`，来自 FLAME 的 LBS 权重；
  下颌/眼球是颈的子节点，一起转），
- `J` 是颈关节在**数据集坐标**里的位置（用 RDF 的 identity/pose 经一次刚体
  映射得到；没有 RDF 时退回"过渡带顶点的质心"），
- `D` 是目标相对当前头姿的**世界系增量旋转**。

`D = I` 时 `v' = v` 严格成立，所以 `turn(idx, 0)` 与烘焙 mesh 逐位一致；
`w_h = 0` 的胸/肩顶点也保持不动。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..data.flame_model import FlameModel
from ..data.pose import rot6d_to_matrix
from .param_map import matrix_to_rot6d, rotvec_to_matrix, split_pose

#: 数据集里有第二套 RDF 资产时优先用（跟踪就是 2023 做的）
DEFAULT_FLAME_DIR = Path("data/FLAME2023")


def _kabsch(A: torch.Tensor, B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """求 `R,t` 使 `R·A + t ≈ B`（刚体，无缩放）。"""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vt.T @ U.T))
    R = Vt.T @ torch.diag(torch.tensor([1.0, 1.0, d], dtype=A.dtype)) @ U.T
    return R, cb - R @ ca


@dataclass
class TurnedFrame:
    """一次"转头"的结果（都在数据集空间）。"""

    mesh: torch.Tensor      # [V,3] float32
    weight: torch.Tensor    # [129] float32
    head: torch.Tensor      # [3,3] float32，实际头姿


class PoseDriver:
    """把数据集某一帧的头姿转 `dyaw/dpitch`，返回可直接渲染的 mesh/weight。"""

    def __init__(self, data_dir: Path, flame_dir: Path | str | None = None,
                 device: str = "cpu") -> None:
        from .tracked import RDF_TRACKING_FILE

        self.data_dir = Path(data_dir)
        baked = self.data_dir / "render_inputs.npz"
        if not baked.is_file():
            raise FileNotFoundError(f"缺少烘焙数据：{baked}")
        with np.load(baked, allow_pickle=True) as d:
            self.mesh = torch.from_numpy(d["mesh"].copy()).float()
            self.weight = torch.from_numpy(d["blend_weight"].copy()).float()
        self.device = torch.device(device)

        # FLAME 只用来拿 LBS 权重（拓扑）与（可选的）颈关节位置，不参与顶点重建。
        flame_dir = Path(flame_dir) if flame_dir else DEFAULT_FLAME_DIR
        if not (flame_dir / "generic_model.pkl").is_file() and not (
                flame_dir / "flame2023.pkl").is_file():
            flame_dir = Path("data/FLAME2020")
        self.flame = FlameModel(flame_dir, dtype=torch.float64)
        # 头骨权重：FLAME 的 5 个关节里 0 是 root（胸/颈根），其余（neck/jaw/eyes）
        # 都随头转；牙齿的权重在 FlameModel 里已经补好。
        lbs = self.flame.lbs_weights.float()
        self.head_weight = (1.0 - lbs[:, 0]).clamp(0.0, 1.0)
        if int(self.mesh.shape[1]) != int(self.head_weight.shape[0]):
            raise ValueError(
                f"mesh 顶点数 {self.mesh.shape[1]} 与 FLAME 拓扑 "
                f"{self.head_weight.shape[0]} 不一致，无法做蒙皮转头")

        self.rdf_path = self.data_dir / RDF_TRACKING_FILE
        self._x = self._identity = None
        if self.rdf_path.is_file():
            with np.load(self.rdf_path, allow_pickle=True) as d:
                self._x = torch.from_numpy(
                    np.asarray(d["x"], dtype=np.float32))
                self._identity = torch.from_numpy(
                    np.asarray(d["identity"], dtype=np.float32)).reshape(-1)
            if self._x.shape[0] != self.mesh.shape[0]:
                self._x = self._identity = None   # 帧数对不上就退回几何兜底

        self._joint: dict[int, torch.Tensor] = {}
        self._cache: dict[tuple[int, float, float], TurnedFrame] = {}

    # ------------------------------------------------------------ 基础 --

    @property
    def frames(self) -> int:
        return int(self.mesh.shape[0])

    def base_index(self, train_only: int | None = None) -> int:
        """最正脸的帧（可用 `train_only` 限制在前段训练帧内）。"""
        from ..data.pose import head_yaw

        yaw = np.abs(head_yaw(self.weight.numpy()))
        n = self.frames if train_only is None else max(1, int(train_only))
        return int(np.argmin(yaw[:n]))

    def head_matrix(self, idx: int) -> torch.Tensor:
        """该帧的头姿（烘焙驱动里的 `[100:106]` rot6d，与训练同一口径）。"""
        return torch.from_numpy(
            rot6d_to_matrix(self.weight[idx:idx + 1, 100:106].numpy())[0]).float()

    # ------------------------------------------------------------ 颈关节 --

    def joint(self, idx: int) -> torch.Tensor:
        """颈关节在数据集坐标里的位置（按帧缓存）。"""
        hit = self._joint.get(int(idx))
        if hit is not None:
            return hit
        J = None
        if self._x is not None:
            J = self._joint_from_rdf(int(idx))
        if J is None:
            J = self._joint_from_weights(int(idx))
        self._joint[int(idx)] = J
        return J

    def _joint_from_rdf(self, idx: int) -> torch.Tensor | None:
        """用 RDF 的 identity/exp 算颈关节，再用刚体映射落到数据集坐标。"""
        try:
            p = split_pose(self._x[idx:idx + 1])
            neck_j = self.flame.joints(
                self._identity[None], self._x[idx:idx + 1, :100])[0][1]
            head = (rotvec_to_matrix(p["global"])
                    @ rotvec_to_matrix(p["neck"]))[0]
            jaw = rotvec_to_matrix(p["jaw"]).double()
            eyes = torch.stack([rotvec_to_matrix(p["eye_l"]),
                                rotvec_to_matrix(p["eye_r"])], dim=1).double()
            v = self.flame.forward(
                self._identity[None].double(),
                self._x[idx:idx + 1, :100].double(), head.double()[None],
                jaw, eyes, None)[0].float()
            R, t = _kabsch(v, self.mesh[idx])
            return (R @ neck_j.float() + t).float()
        except Exception:
            return None

    def _joint_from_weights(self, idx: int) -> torch.Tensor:
        """兜底：取"头/胸权重各半"的过渡带质心当关节锚点。"""
        w = self.head_weight
        band = (w > 0.35) & (w < 0.65)
        if bool(band.any()):
            return self.mesh[idx][band].mean(dim=0)
        return self.mesh[idx].mean(dim=0)

    # ------------------------------------------------------------ 转头 --

    def turn(self, idx: int, dyaw: float = 0.0, dpitch: float = 0.0,
             Rf: torch.Tensor | None = None,
             tf: torch.Tensor | None = None) -> TurnedFrame:
        """把第 `idx` 帧的头转 `(dyaw, dpitch)`。

        `Rf/tf` 仅为兼容旧调用保留，已不使用（不再做 FLAME 重建）。
        """
        key = (int(idx), round(float(dyaw), 2), round(float(dpitch), 2))
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        from ..core.camera import rotation_x, rotation_y

        D = ((rotation_x(float(dpitch)) @ rotation_y(float(dyaw)))[:3, :3]
             .double())
        v = self.mesh[int(idx)].double()
        w = self.head_weight.double().unsqueeze(1)
        weight = self.weight[int(idx)].clone()
        if bool(torch.allclose(D, torch.eye(3, dtype=D.dtype))):
            verts = v                                   # 恒等：逐位不动
            head = self.head_matrix(int(idx))
        else:
            J = self.joint(int(idx)).double()
            verts = v + w * ((D @ (v - J).T).T + J - v)
            head = (D @ self.head_matrix(int(idx)).double()).float()
            weight[100:106] = torch.from_numpy(
                matrix_to_rot6d(head.float()).numpy()).float()
        out = TurnedFrame(mesh=verts.float().to(self.device),
                          weight=weight.to(self.device), head=head.float())
        if len(self._cache) > 512:
            self._cache.clear()
        self._cache[key] = out
        return out
