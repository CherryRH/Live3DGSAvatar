#!/usr/bin/env python
"""预计算每帧 FLAME 可见表面 mask，供 hard_surface 训练直接加载。

训练时每步用 nvdiffrast 重算 surface mask 会逐步占用显存并 OOM；
这个脚本把 mask 一次性算好存到 `datasets/<name>/surface_masks.npz`。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="预计算 FLAME surface masks")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--size", type=int, default=256, help="光栅化分辨率，默认 256")
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from live3dgsavatar.config import load_config
    from live3dgsavatar.data.scene import load_scene
    from live3dgsavatar.core.types import Camera
    from live3dgsavatar.core.render.faceid import face_id_map
    import nvdiffrast.torch as dr

    if not (args.data_dir / "render_inputs.npz").is_file():
        raise SystemExit(f"[error] 不是数据集：{args.data_dir}")
    out = args.out or (args.data_dir / "surface_masks.npz")
    if out.exists() and not args.force:
        raise SystemExit(f"[error] 已存在：{out}（加 --force）")

    cfg = load_config()
    scene = load_scene(cfg, frames=-1, data_dir=args.data_dir, device="cuda")
    k = scene.K.clone()
    k[:2] *= float(args.size) / float(scene.width)
    # w2c 必须用数据相机，不能是单位阵
    w2c = torch.eye(4, device="cuda")
    w2c[:3, :3] = scene.R
    w2c[:3, 3] = scene.T
    cam = Camera(K=k, w2c=w2c.unsqueeze(0), width=args.size,
                 height=args.size).to("cuda")
    glctx = dr.RasterizeCudaContext()
    masks = {}
    n = scene.num_frames
    for i in range(n):
        verts = scene.frames[i]["mesh"].unsqueeze(0).to("cuda")
        ids = face_id_map(verts, scene.faces.to("cuda"), cam, glctx, flip_y=False)
        s = (ids >= 0).float()[None, None]
        s = torch.nn.functional.interpolate(
            s, size=(scene.height, scene.width), mode="nearest")
        masks[f"{i:05d}"] = (s[0, 0] > 0.5).cpu().numpy()
        if (i + 1) % 500 == 0 or i + 1 == n:
            print(f"  {i + 1}/{n}", flush=True)
    np.savez_compressed(out, **masks)
    print(f"写出：{out}（{n} 帧）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
