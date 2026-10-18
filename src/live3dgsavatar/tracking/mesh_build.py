"""把 RDF 的 5023 顶点补成 5083 顶点（补 60 个牙齿顶点）。

## 为什么必须补

RDF 用的是**原生 FLAME**（5023 顶点、9976 个面），而本项目的模型是在
带牙齿扩展的网格上训练的（5083 顶点、10032 个面）—— `model.ply` 里的
`face_id` 上界 10031 就来自后者。少 60 个顶点会让绑定到牙齿面的
**1408 个高斯**（2.33%）指向不存在的顶点。

## 为什么不能直接复现 `add_teeth`

RGBAvatar 的 `add_teeth` 在**模板空间**算出牙齿位置，然后让它们跟着
LBS 走（上牙绑 neck、下牙绑 jaw，权重为 1）。RDF 只给**已摆好姿态**的
顶点，我们拿不到它的 canonical 网格，也无法用我们的 FLAME2020 去算
（shape 基与 2023 几乎正交）。

## 本模块的做法：用嘴唇做刚性锚定

`add_teeth` 的位置完全由 15 个上唇顶点与 15 个下唇顶点推导。这两圈点
在 posed 网格里是已知的，于是：

1. 在**我们的模板**上算出 canonical 的牙齿位置与两圈嘴唇位置
2. 每帧用 Kabsch 求"canonical 嘴唇 → posed 嘴唇"的刚体变换（上/下各一个）
3. 把对应组的牙齿按该变换摆到 posed 空间

上唇圈本来就跟着上颌走、下唇圈跟着下颌走，所以张嘴时下牙会随下颌旋转，
而不是只平移 —— 这是纯平移近似做不到的。

**误差来源**：模板与 RDF 的 identity 不同（模板差异约 6.6e-3），
以及嘴唇皮肤本身是软混合。因此实现里提供了与真值对比的验证入口
（`scripts/calibrate_rdf.py` 会报告牙齿误差）。
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..data.flame_model import FlameModel

#: RDF（原生 FLAME）的顶点数。
NUM_BASE_VERTS = 5023

#: 牙齿顶点数（上 30 + 下 30）。
NUM_TEETH = 60

#: 本项目渲染需要的顶点数。
NUM_FULL_VERTS = NUM_BASE_VERTS + NUM_TEETH

#: 牙齿顶点在各组内的下标范围（与 `add_teeth` 的排列一致）：
#: [0:15] 上根  [15:30] 下根  [30:45] 上边缘  [45:60] 下边缘
_TEETH_UPPER = torch.cat([torch.arange(0, 15), torch.arange(30, 45)])
_TEETH_LOWER = torch.cat([torch.arange(15, 30), torch.arange(45, 60)])


def kabsch(src: torch.Tensor, dst: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """最小二乘刚体变换：求 `R, t` 使 `dst ≈ R·src + t`（**不含缩放、不含反射**）。

    牙齿大小是固定的，允许缩放会得到"张闭嘴时牙齿变大变小"的假解。
    """
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"需要两个同形状 [N,3]，实际 {tuple(src.shape)} vs {tuple(dst.shape)}")

    a = src.to(torch.float64)
    b = dst.to(torch.float64)
    mu_a, mu_b = a.mean(0), b.mean(0)
    cov = (b - mu_b).T @ (a - mu_a)

    u, _, vt = torch.linalg.svd(cov)
    sign = torch.ones(3, dtype=torch.float64)
    if float(torch.det(u) * torch.det(vt)) < 0:
        sign[-1] = -1.0
    rot = u @ torch.diag(sign) @ vt
    return rot, mu_b - rot @ mu_a


class TeethBuilder:
    """由 canonical 牙齿 + 两圈嘴唇锚点，把牙齿摆到 posed 网格上。

    构造一次即可复用（canonical 数据与 FLAME 资产无关地缓存）。
    """

    def __init__(self, flame_dir: Path) -> None:
        model = FlameModel(flame_dir, dtype=torch.float64)
        up_idx, lo_idx = FlameModel._lip_rings()

        self.teeth_canonical = FlameModel._teeth_verts(
            model.v_template[:NUM_BASE_VERTS], torch.float64)
        self.anchor_up_idx = up_idx.clone()
        self.anchor_lo_idx = lo_idx.clone()
        self.anchor_up_canonical = model.v_template[up_idx].to(torch.float64)
        self.anchor_lo_canonical = model.v_template[lo_idx].to(torch.float64)

        if tuple(self.teeth_canonical.shape) != (NUM_TEETH, 3):
            raise AssertionError(
                f"牙齿顶点数应为 {NUM_TEETH}，实际 {tuple(self.teeth_canonical.shape)}")

    def build(self, posed_base: torch.Tensor) -> torch.Tensor:
        """`[N, 5023, 3]`（或单帧 `[5023,3]`）→ `[N, 5083, 3]`。"""
        single = posed_base.ndim == 2
        if single:
            posed_base = posed_base[None]
        if posed_base.shape[1] != NUM_BASE_VERTS:
            raise ValueError(
                f"期望 {NUM_BASE_VERTS} 个基础顶点，实际 {posed_base.shape[1]}")

        out = []
        for frame in posed_base.to(torch.float64):
            teeth = torch.empty(NUM_TEETH, 3, dtype=torch.float64)
            for rows, anchor_idx, anchor_c in (
                    (_TEETH_UPPER, self.anchor_up_idx, self.anchor_up_canonical),
                    (_TEETH_LOWER, self.anchor_lo_idx, self.anchor_lo_canonical)):
                rot, trans = kabsch(anchor_c, frame[anchor_idx])
                teeth[rows] = self.teeth_canonical[rows] @ rot.T + trans
            out.append(teeth)

        stacked = torch.stack(out)
        full = torch.cat([posed_base.to(torch.float64), stacked], dim=1)
        full = full.to(posed_base.dtype)
        return full[0] if single else full
