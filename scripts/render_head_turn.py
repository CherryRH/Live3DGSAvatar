#!/usr/bin/env python
"""按 **RDF 的 identity + pose** 驱动 FLAME，渲染"转头"视角（不依赖 checkpoint）。

GT 的"大角度"是头转、肩/胸不动；`orbit_w2c` 是绕头心转相机，会把胸口一起
转走。这里用我们自己的 RDF 跟踪结果（`render_inputs.rdf.npz` 的 `identity`
与 `x`）驱动 FLAME2023，只改头姿，再映射回数据集坐标渲染。

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
    p = argparse.ArgumentParser(description="RDF 头姿驱动 + 渲染")
    p.add_argument("--subject", default=None)
    p.add_argument("--work-name", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--base-frame", type=int, default=None,
                   help="以哪一帧的表情/身份为底（默认训练段最正脸的帧）")
    p.add_argument("--yaws", required=True, help="相对 yaw（度），如 40,60,80")
    p.add_argument("--flame-dir", type=Path, default=Path("data/FLAME2023"))
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args(argv)


def _kabsch(A: torch.Tensor, B: torch.Tensor):
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vt.T @ U.T))
    R = Vt.T @ torch.diag(torch.tensor([1.0, 1.0, d])) @ U.T
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
    from live3dgsavatar.data.datasets import resolve_dataset_dir
    from live3dgsavatar.data.flame_model import FlameModel
    from live3dgsavatar.data.pose import head_yaw
    from live3dgsavatar.data.scene import load_scene
    from live3dgsavatar.tracking.param_map import (matrix_to_rot6d, rotvec_to_matrix,
                                                   split_pose)

    cfg = load_config()
    subject = args.subject or str(cfg.get("subject") or "")
    if not subject:
        raise SystemExit("[error] 需要 --subject")
    cfg.set("subject", subject)
    work = args.work_name or str(cfg.get("work_name") or "")
    if work:
        cfg.set("work_name", work)
    data = resolve_dataset_dir(cfg, args.dataset or subject)
    rdf_path = data / "render_inputs.rdf.npz"
    if not rdf_path.is_file():
        raise SystemExit(
            f"[error] 缺少 {rdf_path}（RDF 的 identity/pose）。\n"
            "        先跑 scripts/rdf_dump.py（realdenseface 环境）并保留结果。")
    with np.load(rdf_path, allow_pickle=True) as d:
        x = torch.from_numpy(np.asarray(d["x"], dtype=np.float32))
        identity = torch.from_numpy(
            np.asarray(d["identity"], dtype=np.float32)).reshape(-1)
    with np.load(data / "render_inputs.npz", allow_pickle=True) as d:
        M = torch.from_numpy(d["mesh"].copy()).float()
        W = torch.from_numpy(d["blend_weight"].copy()).float()
    n = M.shape[0]
    if x.shape[0] != n:
        raise SystemExit(f"[error] RDF 帧数 {x.shape[0]} 与数据集 {n} 不一致")
    held = int(round(n * float(cfg.get("train.pipeline.test_ratio", 0.3))))
    yaw = np.abs(head_yaw(W.numpy()))
    base = int(args.base_frame) if args.base_frame is not None else int(
        np.argmin(yaw[: n - held]))

    scene = load_scene(cfg, frames=1, data_dir=data)
    flame = FlameModel(args.flame_dir, dtype=torch.float64)
    pose = split_pose(x[base:base + 1])
    jaw = rotvec_to_matrix(pose["jaw"]).double()
    eyes = torch.stack([rotvec_to_matrix(pose["eye_l"]),
                        rotvec_to_matrix(pose["eye_r"])], dim=1).double()
    head0 = (rotvec_to_matrix(pose["global"]) @
             rotvec_to_matrix(pose["neck"])).double()

    def build(head_m: torch.Tensor) -> torch.Tensor:
        v = flame.forward(identity[None].double(), x[base:base + 1, :100].double(),
                          head_m[None], jaw, eyes, None)
        return v[0].float()

    # 刚体映射：FLAME 空间 -> 数据集空间（用底帧估一次）
    Rf, tf = _kabsch(build(head0), M[base])

    ply = Path(str(cfg.get("paths.models_dir"))) / subject / work / "model.ply"
    avatar = load_ply(ply, config=AvatarConfig.from_yaml(ply.parent / "config.yaml"),
                      device="cuda")
    rasterizer = SimpleRasterizer()
    cam = Camera.from_intrinsics_extrinsics(
        K=scene.K, R=scene.R, T=scene.T,
        width=scene.width, height=scene.height).to("cuda")

    args.out.mkdir(parents=True, exist_ok=True)
    for deg in [float(v) for v in str(args.yaws).split(",") if str(v).strip()]:
        head_m = (torch.from_numpy(rotation_y(deg)[:3, :3].numpy().astype(np.float64))
                  @ head0)
        verts = ((Rf @ build(head_m).T).T + tf).unsqueeze(0).to("cuda")
        weight = W[base:base + 1].clone()
        weight[0, 100:106] = torch.from_numpy(
            matrix_to_rot6d(head_m.float()).numpy()).float()
        weight[0, 106:112] = torch.from_numpy(
            matrix_to_rot6d(jaw.float()).numpy()).float()
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
