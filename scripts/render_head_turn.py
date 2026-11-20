#!/usr/bin/env python
"""按 **FLAME 头姿** 合成"转头"视角并渲染（GT 语义，不是相机绕转）。

为什么要这个：数据集里的"大角度"是**头转、肩/胸不动**；而 `orbit_w2c`
是绕头心转相机，会把胸口一起转走（与 GT 差 ~12 dB）。这里改成：

1. 从 `checkpoint/` 取一帧的 FLAME 参数（形状/表情/下颌/眼球/眼睑/头姿/平移）；
2. 把头姿绕颈关节转 `Δyaw`（`R_new = R_y(Δ) @ R_head`，平移按颈关节补偿）；
3. FLAME 前向重建网格，再映射回**数据集坐标**（用该帧的刚体拟合 `Rf, tf`）；
4. 同步更新 `blend_weight` 的头姿段 `[100:106]` 与平移段 `[126:129]`，渲染。

用法::

    python scripts/render_head_turn.py --subject bala --work-name densify \
        --yaws "40,60,80" --out output/diag/headturn
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import live3dgsavatar  # noqa: F401,E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="FLAME 头姿合成视角 + 渲染")
    p.add_argument("--subject", default=None)
    p.add_argument("--work-name", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--base-frame", type=int, default=None,
                   help="用哪一帧的形状/表情作底（默认训练段最正脸的帧）")
    p.add_argument("--yaws", required=True, help="逗号分隔的相对 yaw（度），如 40,60,80")
    p.add_argument("--flame-dir", type=Path, default=Path("data/FLAME2020"))
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args(argv)


def _kabsch(A: torch.Tensor, B: torch.Tensor):
    """求 R,t 使 R·A + t ≈ B（用于 loader 空间 -> 数据集空间）。"""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vt.T @ U.T))
    D = torch.diag(torch.tensor([1.0, 1.0, d]))
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def main(argv=None) -> int:
    args = parse_args(argv)
    from PIL import Image

    from live3dgsavatar.config import load_config
    from live3dgsavatar.core.avatar import AvatarConfig
    from live3dgsavatar.core.camera import rotation_y
    from live3dgsavatar.core.io import load_ply
    from live3dgsavatar.core.render import SimpleRasterizer, render_frame
    from live3dgsavatar.core.types import Camera, Mesh
    from live3dgsavatar.data.checkpoint_loader import (CheckpointDataset,
                                                       _matrix_from_rot6d,
                                                       _rot6d_from_matrix)
    from live3dgsavatar.data.datasets import resolve_dataset_dir
    from live3dgsavatar.data.flame_model import FlameModel
    from live3dgsavatar.data.pose import head_yaw
    from live3dgsavatar.data.scene import load_scene

    cfg = load_config()
    subject = args.subject or str(cfg.get("subject") or "")
    if not subject:
        raise SystemExit("[error] 需要 --subject")
    cfg.set("subject", subject)
    work = args.work_name or str(cfg.get("work_name") or "")
    if work:
        cfg.set("work_name", work)
    data = resolve_dataset_dir(cfg, args.dataset or subject)
    scene = load_scene(cfg, frames=1, data_dir=data)
    ds = CheckpointDataset(data, dtype=torch.float64)
    flame = FlameModel(args.flame_dir, dtype=torch.float64)
    with np.load(data / "render_inputs.npz", allow_pickle=True) as z:
        M = torch.from_numpy(z["mesh"].copy()).float()
        W = torch.from_numpy(z["blend_weight"].copy()).float()
    n = M.shape[0]
    held = int(round(n * float(cfg.get("train.pipeline.test_ratio", 0.3))))
    yaw = np.abs(head_yaw(W.numpy()))
    base = int(args.base_frame) if args.base_frame is not None else int(
        np.argmin(yaw[: n - held]))
    ply = Path(str(cfg.get("paths.models_dir"))) / subject / work / "model.ply"
    if not ply.is_file():
        raise SystemExit(f"[error] 找不到模型：{ply}")
    avatar = load_ply(ply, config=AvatarConfig.from_yaml(ply.parent / "config.yaml"),
                      device="cuda")
    rasterizer = SimpleRasterizer()
    cam = Camera.from_intrinsics_extrinsics(
        K=scene.K, R=scene.R, T=scene.T,
        width=scene.width, height=scene.height).to("cuda")

    # loader 空间 -> 数据集空间的刚体映射（用底帧估一次）
    v0, _ = ds.mesh_and_weight(base, flame)
    Rf, tf = _kabsch(v0.float(), M[base])

    args.out.mkdir(parents=True, exist_ok=True)
    for deg in [float(v) for v in str(args.yaws).split(",") if str(v).strip()]:
        p = ds._load_params(base)
        Rd = torch.from_numpy(
            rotation_y(deg)[:3, :3].numpy().astype(np.float64)) @ p.global_rot
        jaw = _matrix_from_rot6d(
            _rot6d_from_matrix(_matrix_from_rot6d(p.jaw.reshape(1, 6))[0]
                               ).reshape(1, 6))
        eyes = _matrix_from_rot6d(
            _rot6d_from_matrix(_matrix_from_rot6d(p.eyes.reshape(2, 6))
                               ).reshape(2, 6))
        neck_j = flame.joints(p.shape[None], p.expression[None])[0][1]
        transl = Rd @ neck_j - neck_j + p.global_transl
        v = flame.forward(p.shape[None], p.expression[None], Rd[None],
                          jaw[None], eyes[None], p.eyelids[None])[0] + transl
        verts = ((Rf @ v.float().T).T + tf).unsqueeze(0).cuda()
        weight = W[base:base + 1].clone()
        weight[0, 100:106] = torch.from_numpy(
            _rot6d_from_matrix(Rd.float()).numpy()).float()
        weight[0, 126:129] = transl.float()
        mesh = Mesh(verts=verts, faces=scene.faces.to("cuda"),
                    uvs=scene.uvs.to("cuda"), uv_faces=scene.uv_faces.to("cuda"))
        with torch.no_grad():
            out = render_frame(avatar, rasterizer, mesh, weight.to("cuda"), cam,
                               torch.zeros(1, 3, 1, 1, device="cuda"))
        img = (out.color[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
               * 255.0).astype(np.uint8)
        Image.fromarray(img).save(args.out / f"yaw{deg:+.0f}.png")
        print(f"  yaw {deg:+.0f}° -> {args.out / f'yaw{deg:+.0f}.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
