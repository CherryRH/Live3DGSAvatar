#!/usr/bin/env python
"""按硬表面渲染残差初始化自由层（不按语义分区）。

对训练集抽样渲染硬表面，统计每个 FLAME 面的颜色残差；把残差映射到面上的
高斯。只选残差超过阈值的可见高斯，作为自由层初始位置。
"""
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
import numpy as np, torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import live3dgsavatar  # noqa: F401


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--subject", default="bala")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--frames", type=int, default=24)
    p.add_argument("--top-k", type=int, default=12000)
    p.add_argument("--residual-threshold", type=float, default=0.03)
    p.add_argument("--opacity", type=float, default=0.02)
    p.add_argument("--scale-factor", type=float, default=0.85)
    args = p.parse_args()
    from PIL import Image
    from live3dgsavatar.config import load_config
    from live3dgsavatar.data.scene import load_scene
    from live3dgsavatar.core.avatar import AvatarConfig
    from live3dgsavatar.core.io import load_ply
    from live3dgsavatar.core.types import Camera, Mesh
    from live3dgsavatar.core.render import SimpleRasterizer, render_frame
    from live3dgsavatar.core.render.faceid import face_id_map
    import nvdiffrast.torch as dr

    cfg = load_config(); cfg.set("subject", args.subject)
    scene = load_scene(cfg, frames=-1, data_dir=Path("datasets") / args.subject,
                       device="cuda")
    avatar = load_ply(args.model, config=AvatarConfig.from_yaml(args.model.parent / "config.yaml"),
                      device="cuda")
    raster = SimpleRasterizer(); glctx = dr.RasterizeCudaContext()
    cam = Camera.from_intrinsics_extrinsics(K=scene.K, R=scene.R, T=scene.T,
                                            width=scene.width, height=scene.height).to("cuda")
    bg = torch.zeros(1, 3, 1, 1, device="cuda")
    n = int(avatar.num_gaussians); nf = int(scene.faces.shape[0])
    fe = np.zeros(nf, np.float64); fn = np.zeros(nf, np.float64)
    idxs = np.linspace(0, len(scene.frames) - 1, min(args.frames, len(scene.frames))).astype(int)
    for k, idx in enumerate(idxs):
        frame = scene.frames[int(idx)]
        mesh = Mesh(verts=frame["mesh"].unsqueeze(0), faces=scene.faces,
                    uvs=scene.uvs, uv_faces=scene.uv_faces)
        with torch.no_grad():
            out = render_frame(avatar, raster, mesh, frame["blend_weight"].unsqueeze(0),
                               cam, bg)
            ids = face_id_map(mesh.verts, scene.faces, cam, glctx, flip_y=False)
            if ids.ndim == 3:
                ids = ids[0]
            rgb = np.asarray(Image.open(f"datasets/{args.subject}/images/{int(idx):05d}.png").convert("RGBA"), np.float32)
            gt = rgb[..., :3].transpose(2, 0, 1) / 255.0
            fg = rgb[..., 3] > 127
            pred = out.color[0].detach().cpu().numpy()
            err = np.abs(pred - gt).mean(0)
            ids_np = ids.detach().cpu().numpy().reshape(-1)
            err = err.reshape(-1)
            fg = fg.reshape(-1)
            ok = (ids_np >= 0) & fg
            np.add.at(fe, ids_np[ok], err[ok]); np.add.at(fn, ids_np[ok], 1.0)
        if (k + 1) % 8 == 0:
            print(f"  {k+1}/{len(idxs)}")
    face_score = np.divide(fe, fn, out=np.zeros_like(fe), where=fn > 0)
    gscore = face_score[avatar.binding_face_id.detach().cpu().numpy()]
    gcover = fn[avatar.binding_face_id.detach().cpu().numpy()]
    observed = gcover > 0
    cand = observed & (gscore >= args.residual_threshold)
    if not cand.any():
        print("阈值过高，退化为可见高残差 top-k")
        cand = observed
    order = np.argsort(gscore)[::-1]
    chosen = [int(i) for i in order if cand[i]][:args.top_k]
    if not chosen:
        raise SystemExit("没有可选高斯")
    avatar.init_free_layer(torch.tensor(chosen, dtype=torch.long),
                           opacity=args.opacity, scale_factor=args.scale_factor)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    avatar.save_free_layer(args.out)
    meta = {"model": str(args.model), "frames": len(idxs),
            "threshold": args.residual_threshold, "top_k": args.top_k,
            "chosen": len(chosen), "mean_score": float(gscore[chosen].mean())}
    args.out.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"自由层 {len(chosen)}（阈值 {args.residual_threshold:g}，可见 {int(observed.sum())}），{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
