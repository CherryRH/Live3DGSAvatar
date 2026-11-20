#!/usr/bin/env python
"""按 **蒙皮旋转**（PoseDriver）渲染"转头"视角（不依赖 checkpoint）。

GT 的"大角度"是头转、肩/胸不动；`orbit_w2c` 是绕头心转相机，会把胸口一起
转走。这里用我们自己的 RDF 跟踪结果（`rdf_tracking.npz` 的 `identity`
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


def main(argv=None) -> int:
    args = parse_args(argv)
    from PIL import Image

    from live3dgsavatar.config import load_config
    from live3dgsavatar.core.avatar import AvatarConfig
    from live3dgsavatar.core.io import load_ply
    from live3dgsavatar.core.render import SimpleRasterizer, render_frame
    from live3dgsavatar.core.types import Camera, Mesh
    from live3dgsavatar.data.datasets import resolve_dataset_dir
    from live3dgsavatar.data.scene import load_scene
    from live3dgsavatar.tracking.pose_drive import PoseDriver

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
    driver = PoseDriver(data, flame_dir=args.flame_dir, device="cpu")
    n = driver.frames
    held = int(round(n * float(cfg.get("train.pipeline.test_ratio", 0.3))))
    base = (int(args.base_frame) if args.base_frame is not None
            else driver.base_index(train_only=n - held))

    ply = Path(str(cfg.get("paths.models_dir"))) / subject / work / "model.ply"
    if not ply.is_file():
        raise SystemExit(f"[error] 找不到模型：{ply}")
    avatar = load_ply(ply, config=AvatarConfig.from_yaml(ply.parent / "config.yaml"),
                      device="cuda")
    rasterizer = SimpleRasterizer()
    cam = Camera.from_intrinsics_extrinsics(
        K=scene.K, R=scene.R, T=scene.T,
        width=scene.width, height=scene.height).to("cuda")

    args.out.mkdir(parents=True, exist_ok=True)
    for deg in [float(v) for v in str(args.yaws).split(",") if str(v).strip()]:
        turned = driver.turn(base, deg, 0.0)
        mesh = Mesh(verts=turned.mesh.unsqueeze(0).cuda(),
                    faces=scene.faces.to("cuda"), uvs=scene.uvs.to("cuda"),
                    uv_faces=scene.uv_faces.to("cuda"))
        with torch.no_grad():
            out = render_frame(avatar, rasterizer, mesh,
                               turned.weight.unsqueeze(0).cuda(), cam,
                               torch.zeros(1, 3, 1, 1, device="cuda"))
        img = (out.color[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
               * 255.0).astype(np.uint8)
        Image.fromarray(img).save(args.out / f"yaw{deg:+.0f}.png")
        print(f"  yaw {deg:+.0f}° -> {args.out / f'yaw{deg:+.0f}.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
