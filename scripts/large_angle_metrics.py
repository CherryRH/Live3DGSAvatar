#!/usr/bin/env python
"""大角度质量指标：没有 GT 时，用"头转视角"的几何/一致性量来量化。

留出集几乎是正脸（bala p50 yaw 0.9°），所以大角度质量不能只看留出集 PSNR。
这里对每个模型按目标 yaw 渲染头转视图（相机不动、只转头），统计：

- `coverage`：FLAME 表面内 alpha>0.5 的比例（几何覆盖）
- `hole`：表面内 alpha<0.5 的比例（空洞）
- `outside`：实心像素里落在 FLAME 表面外的比例（头发/肩/漂浮，越大越"糊"）
- `haze`：白底上半透明暗像素占比（轮廓外薄雾）
- `sharp`：表面内平均梯度（0–255），并给出相对正脸的比值
- `area`：剪影面积（像素），用于看跨角度稳定性
- `id_cos`：与正脸渲染的 ArcFace 余弦相似度（insightface；检不到脸则 None）

用法::

    python scripts/large_angle_metrics.py --subject bala \
        --work-names base densify --yaws "0,15,30,45,60"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import live3dgsavatar  # noqa: F401,E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="大角度质量（头转视角、无 GT 指标）")
    p.add_argument("--subject", default=None)
    p.add_argument("--work-names", nargs="+", required=True)
    p.add_argument("--yaws", default="0,15,30,45,60")
    p.add_argument("--base-frame", type=int, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--no-id", action="store_true", help="不算 ArcFace 相似度")
    return p.parse_args(argv)


def _face_embedder():
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(320, 320))
        return app
    except Exception:                                  # pragma: no cover
        return None


def _embed(app, image: np.ndarray):
    if app is None:
        return None
    faces = app.get(image)
    if not faces:
        return None
    return faces[0].normed_embedding


def main(argv=None) -> int:
    args = parse_args(argv)
    from PIL import Image

    from live3dgsavatar.config import load_config
    from live3dgsavatar.core.avatar import AvatarConfig
    from live3dgsavatar.core.io import load_ply
    from live3dgsavatar.core.render import SimpleRasterizer, render_frame
    from live3dgsavatar.core.render.faceid import face_id_map
    from live3dgsavatar.core.types import Camera, Mesh
    from live3dgsavatar.data.datasets import resolve_dataset_dir
    from live3dgsavatar.data.scene import load_scene
    from live3dgsavatar.tracking.pose_drive import PoseDriver

    cfg = load_config()
    subject = args.subject or str(cfg.get("subject") or "")
    if not subject:
        raise SystemExit("[error] 需要 --subject")
    cfg.set("subject", subject)
    data = resolve_dataset_dir(cfg, subject)
    scene = load_scene(cfg, frames=1, data_dir=data)
    driver = PoseDriver(data, device="cpu")
    n = driver.frames
    held = int(round(n * float(cfg.get("train.pipeline.test_ratio", 0.3))))
    base = (int(args.base_frame) if args.base_frame is not None
            else driver.base_index(train_only=n - held))
    yaws = [float(v) for v in str(args.yaws).split(",") if str(v).strip()]
    rasterizer = SimpleRasterizer()
    cam = Camera.from_intrinsics_extrinsics(
        K=scene.K, R=scene.R, T=scene.T,
        width=scene.width, height=scene.height).to("cuda")
    app = None if args.no_id else _face_embedder()

    import nvdiffrast.torch as dr
    glctx = dr.RasterizeCudaContext()
    out = {"subject": subject, "base_frame": int(base), "models": {}}
    for work in args.work_names:
        ply = Path(str(cfg.get("paths.models_dir"))) / subject / work / "model.ply"
        if not ply.is_file():
            raise SystemExit(f"[error] 找不到模型：{ply}")
        avatar = load_ply(ply, config=AvatarConfig.from_yaml(
            ply.parent / "config.yaml"), device="cuda")
        rows = {}
        ref_embed = None
        for yaw in yaws:
            turned = driver.turn(base, yaw, 0.0)
            mesh = Mesh(verts=turned.mesh.unsqueeze(0).cuda(),
                        faces=scene.faces.to("cuda"),
                        uvs=scene.uvs.to("cuda"),
                        uv_faces=scene.uv_faces.to("cuda"))
            with torch.no_grad():
                res = render_frame(avatar, rasterizer, mesh,
                                   turned.weight.unsqueeze(0).cuda(), cam,
                                   torch.zeros(1, 3, 1, 1, device="cuda"))
                ids = face_id_map(turned.mesh.unsqueeze(0).cuda(),
                                  scene.faces.to("cuda"), cam, glctx,
                                  flip_y=False)
            ids = ids[0] if ids.ndim == 3 else ids
            surf = (ids >= 0).cpu().numpy()
            alpha = res.alpha[0, 0].cpu().numpy()
            color = res.color[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            rgb = (color * 255).astype(np.uint8)
            grey = color.mean(-1)
            white = color + (1.0 - alpha[..., None])
            haze = float(((alpha > 0.05) & (alpha < 0.9)
                          & (white.mean(-1) < 0.9)).mean())
            filled = alpha > 0.5
            gy, gx = np.gradient(grey)
            # 0–255 尺度，和报告的 sharpeness 口径一致
            sharp = (float(np.hypot(gx, gy)[surf].mean()) * 255.0
                     if surf.any() else 0.0)
            emb = _embed(app, rgb) if app is not None else None
            if yaw == 0:
                ref_embed = emb
            rows[f"{yaw:+.0f}"] = {
                "coverage": float((alpha[surf] > 0.5).mean()) if surf.any() else 0.0,
                "hole": float((alpha[surf] < 0.5).mean()) if surf.any() else 0.0,
                "outside": float((filled & ~surf).sum()) / max(1, int(filled.sum())),
                "haze": haze,
                "sharp": sharp,
                "area": int(filled.sum()),
                "id_cos": (None if emb is None or ref_embed is None
                           else float(np.dot(emb, ref_embed))),
            }
        # 相对正脸的锐度比
        ref_sharp = rows.get("+0", {}).get("sharp") or 0.0
        for k, v in rows.items():
            v["sharp_rel"] = (v["sharp"] / ref_sharp) if ref_sharp > 0 else None
        out["models"][work] = rows

    print(f"{'model':12s} {'yaw':>5s} {'cover':>6s} {'hole':>6s} {'outside':>8s} "
          f"{'haze':>6s} {'sharp':>6s} {'sharp_rel':>9s} {'id_cos':>7s}")
    for work, rows in out["models"].items():
        for k, v in rows.items():
            idc = "-" if v["id_cos"] is None else f"{v['id_cos']:.3f}"
            rel = "-" if v["sharp_rel"] is None else f"{v['sharp_rel']:.3f}"
            print(f"{work:12s} {k:>5s} {v['coverage']:6.3f} {v['hole']:6.3f} "
                  f"{v['outside']:8.3f} {v['haze']:6.3f} {v['sharp']:6.2f} "
                  f"{rel:>9s} {idc:>7s}")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
