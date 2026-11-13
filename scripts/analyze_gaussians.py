#!/usr/bin/env python
"""模型解剖 + 推理时消融：容量都用在哪了、飞点/雾到底在多大程度上承重。

不做训练、不改模型格式。输出三样东西：

1. `anatomy.png`：不透明度 / 离面距离 / 尺度分布，并给出"死点"占比；
2. `ablation.png`：按离面距离、不透明度分层把高斯压到 0，比较正面（有 GT）
   与 ±20/40°（暴露缺陷）的变化；
3. `report.json`：逐变体的 PSNR / hole / spill / spike / 前景像素。

用途：回答"这些点是不是在承重、去掉后大角度是否真的变好"。**不追通过率**，
先看图再决定要不要改模型。

用法：
```bash
python scripts/analyze_gaussians.py --subject biden --work-name train1 \\
    --dataset biden --frame 1901 --yaws 0 20 40
```
产物：`<output>/analyze/<subject>/<work>/<dataset>/{anatomy.png,ablation.png,report.json}`。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import live3dgsavatar  # noqa: F401


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="高斯解剖与飞点消融")
    p.add_argument("--subject", default=None)
    p.add_argument("--work-name", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--frame", type=int, default=0, help="参考帧（用它的几何做消融）")
    p.add_argument("--yaws", type=float, nargs="+", default=[0.0, 20.0, 40.0])
    p.add_argument("--pitch", type=float, default=0.0)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--offsets", type=float, nargs="+", default=[8.0, 15.0, 25.0],
                   help="离面距离阈值（毫米），每个阈值一个消融变体")
    p.add_argument("--opacity", type=float, default=0.05,
                   help="低于该不透明度的高斯作为雾去掉")
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


def _erode(mask, radius: int = 2):
    import numpy as np
    from scipy.ndimage import binary_erosion

    m = np.asarray(mask, bool)
    if radius <= 0 or not m.any():
        return m
    return binary_erosion(m, iterations=int(radius))


def _masked_psnr(pred, gt, mask) -> float:
    from live3dgsavatar.core.metrics import psnr_masked

    v = psnr_masked(pred, gt, mask)
    return float("nan") if v is None else float(v)


def _bars(tile, values, labels, title, color=(70, 110, 200)):
    """在 PIL 图上画一个简单竖条图（不依赖 matplotlib）。"""
    from PIL import ImageDraw

    w, h = tile.size
    d = ImageDraw.Draw(tile)
    d.rectangle([0, 0, w, h], fill=(250, 250, 250))
    d.text((10, 6), title, fill=(20, 20, 20))
    if not values:
        return
    top = max(1e-9, max(values))
    left, right, bottom, top_y = 40, w - 12, h - 34, 28
    bw = (right - left) / max(1, len(values))
    for i, (v, lab) in enumerate(zip(values, labels)):
        x0 = left + i * bw + bw * 0.15
        x1 = left + (i + 1) * bw - bw * 0.15
        y0 = bottom - (bottom - top_y) * (v / top)
        d.rectangle([x0, y0, x1, bottom], fill=color)
        d.text((x0, bottom + 3), str(lab), fill=(60, 60, 60))
    d.text((6, top_y), f"max {top:.0f}", fill=(120, 120, 120))


def _render(avatar, rasterizer, mesh, weight, cam, bg, remove=None):
    """渲染一帧；`remove` 是要压到 0 的高斯 bool 掩码。"""
    import numpy as np
    import torch

    from live3dgsavatar.core.render import render_frame

    orig = avatar.opacity.detach().clone()
    try:
        if remove is not None and bool(np.any(remove)):
            avatar.opacity.data[torch.from_numpy(np.asarray(remove)).to(
                avatar.opacity.device)] = -30.0
        out = render_frame(avatar, rasterizer, mesh, weight, cam, bg)
    finally:
        avatar.opacity.data.copy_(orig)
    rgb = (out.color[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
           * 255.0).astype(np.uint8)
    return out, rgb


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont

    from live3dgsavatar.config import (load_config, output_dir_for,
                                       resolve_model_ply)
    from live3dgsavatar.core.camera import orbit_w2c
    from live3dgsavatar.core.io import load_ply
    from live3dgsavatar.core.io.ply import read_embedded_config
    from live3dgsavatar.core.render import SimpleRasterizer, render_frame
    from live3dgsavatar.core.render.faceid import detect_flip, face_id_map
    from live3dgsavatar.core.types import Camera, Mesh
    from live3dgsavatar.data.datasets import resolve_dataset_dir
    from live3dgsavatar.data.scene import load_scene
    from live3dgsavatar.evaluation.anatomy import (
        dead_mask,
        faces_without_gaussians,
        histogram_counts,
        summarize,
    )
    from live3dgsavatar.evaluation.defects import floater_offsets

    cfg = load_config()
    subject = args.subject or str(cfg.get("subject") or "")
    work = args.work_name or str(cfg.get("work_name") or "")
    if not (subject and work):
        raise SystemExit("[error] 需要 --subject 与 --work-name")
    cfg.set("subject", subject)
    cfg.set("work_name", work)
    dataset = args.dataset or subject
    data_dir = resolve_dataset_dir(cfg, dataset)
    ply = resolve_model_ply(cfg)
    out_dir = Path(args.out or output_dir_for(
        cfg, "analyze", subject, work, dataset))
    out_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise SystemExit("[error] 需要 GPU（nvdiffrast）；--no-ablation 仍需拿离面距离")
    import nvdiffrast.torch as dr

    device = args.device
    glctx = dr.RasterizeCudaContext()
    scene = load_scene(cfg, frames=-1, data_dir=data_dir)
    idx = min(max(int(args.frame), 0), scene.num_frames - 1)
    avatar = load_ply(ply, config=read_embedded_config(ply), device=device)
    rasterizer = SimpleRasterizer()
    faces = scene.faces.to(torch.int32)
    bg = torch.tensor([float(x) for x in cfg.get("render.background", [0, 0, 0])],
                      dtype=torch.float32, device=device)

    w2c = torch.eye(4, dtype=torch.float32)
    w2c[:3, :3] = scene.R.detach().cpu()
    w2c[:3, 3] = scene.T.detach().cpu()

    frame = scene.frames[idx]
    verts = frame["mesh"].unsqueeze(0).to(device)
    weight = frame["blend_weight"].unsqueeze(0).to(device)
    mesh = Mesh(verts=verts, faces=scene.faces, uvs=scene.uvs,
                uv_faces=scene.uv_faces)

    # ---- 1) 解剖 ----
    opacity = torch.sigmoid(avatar.opacity.detach()).reshape(-1).cpu().numpy()
    scale_mm = (torch.exp(avatar.scaling.detach()).max(dim=1).values
                * 1000.0).cpu().numpy()
    with torch.no_grad():
        gs0 = avatar.deform(mesh, weight)
        off_mm = (floater_offsets(gs0.xyz[0], verts[0], scene.faces,
                                  avatar.binding_face_id,
                                  avatar.binding_face_bary) * 1000.0
                  ).detach().cpu().numpy()
    n_faces = int(scene.faces.shape[0])
    missing = faces_without_gaussians(
        avatar.binding_face_id.detach().cpu().numpy(), n_faces)
    dead = dead_mask(opacity)
    coverage = None
    mass_path = output_dir_for(cfg, "defects", subject, work, dataset) / "mass.npz"
    if mass_path.is_file():
        with np.load(mass_path, allow_pickle=True) as d:
            if "coverage" in d.files:
                coverage = np.asarray(d["coverage"])

    active = opacity >= 0.5
    stats = {
        "n_gaussians": int(opacity.size),
        "n_faces": n_faces,
        "dead_share": float(dead.mean()),
        "dead_n": int(dead.sum()),
        "active_share": float(active.mean()),
        "offset_mm_active": summarize(off_mm[active]),
        "faces_without_gaussians": int(missing.sum()),
        "faces_without_gaussians_share": float(missing.mean()),
        "opacity": {"summary": summarize(opacity),
                    "hist": histogram_counts(
                        opacity, [0.0, 0.01, 0.1, 0.5, 0.9, 1.0001])},
        "offset_mm": {"summary": summarize(off_mm),
                      "hist": histogram_counts(
                          off_mm, [0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 1e9])},
        "scale_mm": {"summary": summarize(scale_mm),
                     "hist": histogram_counts(
                         scale_mm, [0.0, 0.2, 0.5, 1.0, 2.0, 5.0, 1e9])},
    }
    if coverage is not None:
        stats["coverage"] = {"zero": float((coverage == 0).mean()),
                             "lt_50": float((coverage < 50).mean())}
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    # ---- 2) 消融 ----
    rows = []
    panels: dict[tuple[str, float], np.ndarray] = {}
    variants = [("full", None)]
    for th in args.offsets:
        variants.append((f"off>{th:g}mm", off_mm > float(th)))
    variants.append((f"opacity<{args.opacity:g}", opacity < float(args.opacity)))

    center = verts[0].mean(dim=0).detach().cpu()
    views = []
    for yaw in args.yaws:
        if abs(float(yaw)) < 1e-6 and abs(float(args.pitch)) < 1e-6:
            view_w2c = w2c
        else:
            view_w2c = orbit_w2c(w2c, center, float(yaw), float(args.pitch))
        cam = Camera(K=scene.K, w2c=view_w2c.unsqueeze(0).to(device),
                     width=scene.width, height=scene.height).to(device)
        views.append((float(yaw), cam))

    gt_path = data_dir / "images" / f"{idx:05d}.png"
    gt = mask_gt = None
    if gt_path.is_file():
        rgba = np.asarray(Image.open(gt_path).convert("RGBA"))
        gt = rgba[..., :3].copy()
        gt[rgba[..., 3] <= 127] = 0
        mask_gt = rgba[..., 3] > 127

    flip = None
    for yaw, cam in views:
        out0 = render_frame(avatar, rasterizer, mesh, weight, cam, bg)
        a0 = out0.alpha[0, 0].detach().cpu().numpy()
        if flip is None:
            flip, _ = detect_flip(verts, faces, cam, glctx, a0)
        ids_np = face_id_map(verts, faces, cam, glctx,
                             flip_y=flip).detach().cpu().numpy()
        surface = ids_np >= 0
        interior = _erode(surface, 2)
        variant_rows = []
        for name, rm in variants:
            out, rgb = _render(avatar, rasterizer, mesh, weight, cam, bg, rm)
            alpha = out.alpha[0, 0].detach().cpu().numpy()
            filled = alpha > 0.5
            hole = float((alpha[surface] < 0.5).mean()) if surface.any() else 0.0
            hole_in = (float((alpha[interior] < 0.5).mean())
                       if interior.any() else 0.0)
            spill = (float((filled & ~surface).sum() / max(1, int(filled.sum())))
                     if filled.any() else 0.0)
            gy, gx = np.gradient(alpha.astype(np.float64))
            spike = (float(np.hypot(gx, gy)[interior].mean())
                     if interior.any() else 0.0)
            psnr = (_masked_psnr(rgb, gt, mask_gt)
                    if gt is not None and abs(yaw) < 1e-6 else None)
            variant_rows.append({
                "variant": name, "yaw": yaw,
                "n_removed": int(np.asarray(rm).sum()) if rm is not None else 0,
                "psnr_front": psnr, "hole": round(hole, 4),
                "hole_in": round(hole_in, 4), "spill": round(spill, 4),
                "spike": round(spike, 5), "fg_px": int(filled.sum())})
            panels[(name, yaw)] = rgb
        rows.extend(variant_rows)
        print(f"  yaw {yaw:+g}°: " + "  ".join(
            f"{r['variant']} psnr={r['psnr_front']} hole={r['hole']} "
            f"spill={r['spill']} fg={r['fg_px']}" for r in variant_rows),
              flush=True)

    # ---- 3) 拼图 ----
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:                                      # pragma: no cover
        font = ImageFont.load_default()
    cols = ["GT"] + [v[0] for v in variants]
    pad, label_h, w = 6, 26, scene.width
    sheet = Image.new("RGB", (len(cols) * (w + pad) + pad,
                              len(views) * (w + label_h + pad) + pad),
                      (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for ci, name in enumerate(cols):
        d.text((pad + ci * (w + pad) + 4, 4), name, fill=(0, 0, 0), font=font)
    for ri, (yaw, _cam) in enumerate(views):
        y0 = pad + ri * (w + label_h + pad) + label_h
        for ci, name in enumerate(cols):
            x0 = pad + ci * (w + pad)
            if name == "GT":
                img = (gt if gt is not None and abs(yaw) < 1e-6
                       else np.zeros((w, w, 3), np.uint8))
            else:
                img = panels[(name, yaw)]
            sheet.paste(Image.fromarray(img), (x0, y0))
        d.text((4, y0 + 4), f"yaw {yaw:+g}", fill=(0, 0, 0), font=font)
    sheet.save(out_dir / "ablation.png")

    # ---- 4) 分布图 ----
    tiles = (
        (stats["opacity"]["hist"], "opacity distribution", (70, 110, 200)),
        (stats["offset_mm"]["hist"], "offset from surface (mm)", (200, 120, 60)),
        (stats["scale_mm"]["hist"], "gaussian scale (mm)", (80, 160, 100)),
    )
    anatomy = Image.new("RGB", (3 * 520, 230), (255, 255, 255))
    for i, (h, title, color) in enumerate(tiles):
        tile = Image.new("RGB", (520, 220), (250, 250, 250))
        _bars(tile, h["counts"],
              [f"{a:g}" for a in h["edges"][:-1]], title, color)
        anatomy.paste(tile, (i * 520, 5))
    anatomy.save(out_dir / "anatomy.png")

    payload = {"subject": subject, "work_name": work, "dataset": dataset,
               "frame": idx, "yaws": [v[0] for v in views],
               "stats": stats, "rows": rows}
    (out_dir / "report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n产物：{out_dir}")
    print("  anatomy.png / ablation.png / report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
