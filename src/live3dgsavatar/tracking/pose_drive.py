#!/usr/bin/env python
"""按 **RDF 的 identity + pose** 驱动 FLAME：把任意帧的头"转"到目标角度。

为什么不是相机绕转：数据集里的"大角度"是**头转、肩/胸不动**。绕头心转相机
会把胸口一起转走（与 GT 差约 12 dB），并在颈/胸边界教出错误的"背景"目标。
这里的语义与生产驱动完全一致：相机不动，只改 FLAME 的头姿。

数据来源（都是我们自己 RDF 跟踪的产物）::

    datasets/<obj>/rdf_tracking.npz   x(T,118) + identity(300)
    datasets/<obj>/render_inputs.npz   mesh(T,5083,3) + blend_weight(T,129)

`FLAME 空间 -> 数据集空间` 的刚体映射用底帧估一次（Kabsch），之后每个目标角度
只做一次 FLAME 前向 + 刚体变换。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..data.flame_model import FlameModel
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
        rdf = self.data_dir / RDF_TRACKING_FILE
        if not rdf.is_file():
            raise FileNotFoundError(f"缺少 RDF 跟踪结果：{rdf}")
        baked = self.data_dir / "render_inputs.npz"
        if not baked.is_file():
            raise FileNotFoundError(f"缺少烘焙数据：{baked}")
        with np.load(rdf, allow_pickle=True) as d:
            self.x = torch.from_numpy(np.asarray(d["x"], dtype=np.float32))
            self.identity = torch.from_numpy(
                np.asarray(d["identity"], dtype=np.float32)).reshape(-1)
        with np.load(baked, allow_pickle=True) as d:
            self.mesh = torch.from_numpy(d["mesh"].copy()).float()
            self.weight = torch.from_numpy(d["blend_weight"].copy()).float()
        if self.x.shape[0] != self.mesh.shape[0]:
            raise ValueError(
                f"RDF 帧数 {self.x.shape[0]} 与数据 {self.mesh.shape[0]} 不一致")
        self.device = torch.device(device)
        flame_dir = Path(flame_dir) if flame_dir else DEFAULT_FLAME_DIR
        if not (flame_dir / "generic_model.pkl").is_file() and not (
                flame_dir / "flame2023.pkl").is_file():
            flame_dir = Path("data/FLAME2020")
        self.flame = FlameModel(flame_dir, dtype=torch.float64)
        self._cache: dict[tuple[int, float, float], TurnedFrame] = {}
        self._rigid: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

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
        p = split_pose(self.x[idx:idx + 1])
        return (rotvec_to_matrix(p["global"]) @ rotvec_to_matrix(p["neck"]))[0]

    def _build(self, idx: int, head: torch.Tensor) -> torch.Tensor:
        p = split_pose(self.x[idx:idx + 1])
        jaw = rotvec_to_matrix(p["jaw"]).double()
        eyes = torch.stack([rotvec_to_matrix(p["eye_l"]),
                            rotvec_to_matrix(p["eye_r"])], dim=1).double()
        v = self.flame.forward(
            self.identity[None].double(), self.x[idx:idx + 1, :100].double(),
            head.double()[None], jaw, eyes, None)
        return v[0].float()

    def rigid_map(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """该帧的 `FLAME 空间 -> 数据集空间` 刚体映射（按帧缓存）。"""
        hit = self._rigid.get(int(idx))
        if hit is None:
            hit = _kabsch(self._build(idx, self.head_matrix(idx)), self.mesh[idx])
            if len(self._rigid) > 4096:
                self._rigid.clear()
            self._rigid[int(idx)] = hit
        return hit

    # ------------------------------------------------------------ 转头 --

    def turn(self, idx: int, dyaw: float = 0.0, dpitch: float = 0.0,
             Rf: torch.Tensor | None = None,
             tf: torch.Tensor | None = None) -> TurnedFrame:
        key = (int(idx), round(float(dyaw), 2), round(float(dpitch), 2))
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        from ..core.camera import rotation_x, rotation_y

        R = ((rotation_x(float(dpitch)) @ rotation_y(float(dyaw)))[:3, :3]
             .double())
        head = R @ self.head_matrix(idx).double()
        if Rf is None or tf is None:
            Rf, tf = self.rigid_map(idx)
        verts = ((Rf @ self._build(idx, head).T).T + tf).float()
        weight = self.weight[idx].clone()
        weight[100:106] = torch.from_numpy(
            matrix_to_rot6d(head.float()).numpy()).float()
        out = TurnedFrame(mesh=verts.to(self.device), weight=weight.to(self.device),
                          head=head.float())
        if len(self._cache) > 512:
            self._cache.clear()
        self._cache[key] = out
        return out
