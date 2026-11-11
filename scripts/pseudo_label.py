#!/usr/bin/env python
"""几何条件扩散生成伪标签：在"有表面、模型没铺满"的区域补出可信纹理。

这是 GAF（CVPR 2025）思路的**轻量版**：不训练任何网络，用冻结的
SD + Normal ControlNet 在低强度下重绘缺陷视角，再用几何与一致性检查过滤，
通过的图作为微调目标（落到 `<缺陷目录>/refined/`，交给 `refine_from_images.py`）。

## 与 GAF 的差别（以及为什么这样取舍）

| | GAF | 这里 |
|---|---|---|
| 扩散先验 | 自训练的多视角头部扩散（跨视角注意力） | 公开 SD1.5 + ControlNet-normal（单视角） |
| 条件 | FLAME 法线 + 输入图 VAE latent | FLAME 法线 + 低强度 img2img（渲染图做初始化） |
| 用法 | 迭代去噪图当伪 GT | 只采信**筛选通过**的图 |

单视角没有跨视角耦合，所以一致性必须靠**筛选**补回来：
轮廓 IoU、可信区域不变、洞内结构非平坦、两次采样一致性。

## 用法

```bash
# 只生成伪标签（需要 diffusers；首次会下载 SD1.5 与 ControlNet 权重）
python scripts/pseudo_label.py --subject <对象> --work-name <版本> \
    --defects output/defects/<对象>/<版本>/<数据集> \
    --install-refined

# 只渲染法线与掩码（不需要 diffusers，可先检查几何是否对齐）
python scripts/pseudo_label.py ... --dry-run
```

产物：`<缺陷目录>/pseudo/{accepted/,debug/,report.json}`。
`--install-refined` 会把通过的图复制到 `<缺陷目录>/refined/`，
之后 `bash enhance.sh ... --skip-syncfix` 即可直接微调。
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
    p = argparse.ArgumentParser(description="几何条件扩散生成伪标签")
    p.add_argument("--subject", default=None)
    p.add_argument("--work-name", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--defects", type=Path, default=None,
                   help="缺陷目录（含 selected.json）")
    p.add_argument("--out", type=Path, default=None,
                   help="输出目录（默认 <缺陷目录>/pseudo）")
    p.add_argument("--install-refined", action="store_true",
                   help="把通过筛选的图复制到 <缺陷目录>/refined/")
    p.add_argument("--limit", type=int, default=0, help="只处理前 N 个视角（调试）")
    p.add_argument("--seed", type=int, default=0)

    # ---- 扩散 ----
    p.add_argument("--base-model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--controlnet", default="lllyasviel/control_v11p_sd15_normalbae")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--strength", type=float, default=0.40,
                   help="img2img 强度；越大改得越多（也越容易改坏）")
    p.add_argument("--guidance", type=float, default=5.5)
    p.add_argument("--prompt",
                   default="professional photo of a human head, natural skin "
                           "texture, fine detail, sharp focus")
    p.add_argument("--negative",
                   default="blurry, smooth plastic, deformed, cartoon, "
                           "extra eyes, low resolution, watermark")

    # ---- 掩码与筛选 ----
    p.add_argument("--normal-convention", choices=("opengl", "camera"),
                   default="opengl",
                   help="法线编码约定（ControlNet 用 opengl；camera 仅用于对比调试）")
    p.add_argument("--mask-dilate", type=int, default=6,
                   help="洞掩码向外扩多少像素（防止接缝）")
    p.add_argument("--trusted-tol", type=float, default=2.0,
                   help="可信区域允许的平均改动（0-255）")
    p.add_argument("--min-iou", type=float, default=0.90,
                   help="伪标签前景与 FLAME 轮廓的最小 IoU")
    p.add_argument("--min-std", type=float, default=3.0,
                   help="洞内像素标准差下限（太平坦说明糊了）")
    p.add_argument("--consistency", type=int, default=2,
                   help="每个视角采样几次做一致性检查（1 = 关闭）")
    p.add_argument("--min-consistency", type=float, default=0.60,
                   help="多次采样在洞内的 SSIM 下限")
    p.add_argument("--dry-run", action="store_true",
                   help="只渲染 RGB/法线/掩码，不跑扩散（无需 diffusers）")
    p.add_argument("--force", action="store_true", help="覆盖已有产物")
    return p.parse_args(argv)


# ------------------------------------------------------------------ 工具 --


def dilate(mask, radius: int):
    """二值膨胀（纯 numpy，圆形结构元近似为方形，够用）。"""
    import numpy as np

    m = np.asarray(mask, bool)
    if radius <= 0 or not m.any():
        return m
    out = m.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx > radius * radius:
                continue
            out |= np.roll(np.roll(m, dy, axis=0), dx, axis=1)
    return out


def _erode(mask, radius: int):
    """二值腐蚀（与 dilate 对称的方形近似）。"""
    import numpy as np

    m = np.asarray(mask, bool)
    if radius <= 0 or not m.any():
        return m
    out = m.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx > radius * radius:
                continue
            out &= np.roll(np.roll(m, dy, axis=0), dx, axis=1)
    return out


def foreground_iou(a, b) -> float:
    import numpy as np

    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = int((a | b).sum())
    return float((a & b).sum()) / union if union else 1.0


def masked_ssim(a, b, mask) -> float:
    import numpy as np

    from live3dgsavatar.core.metrics import ssim

    m = np.asarray(mask, bool)
    if not m.any():
        return 1.0
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    # 只比较洞内：把外面填成同一张图，避免边界影响
    aa = a.copy()
    bb = b.copy()
    aa[~m] = a[~m]
    bb[~m] = a[~m]
    return float(ssim(aa.astype(np.uint8), bb.astype(np.uint8)))


def build_pipeline(args):
    """加载 SD + ControlNet inpainting 管线（fp16、冻结）。"""
    try:
        import torch
        from diffusers import (ControlNetModel,
                               StableDiffusionControlNetInpaintPipeline)
    except ImportError as e:  # pragma: no cover - 依赖缺失
        raise SystemExit(
            "[error] 需要 diffusers / transformers / accelerate。\n"
            "        先跑 bash setup_env.sh（requirements.txt 里已列出）。\n"
            f"        原始错误：{e}") from None

    dtype = torch.float16
    controlnet = ControlNetModel.from_pretrained(args.controlnet,
                                                 torch_dtype=dtype)
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        args.base_model, controlnet=controlnet, torch_dtype=dtype,
        safety_checker=None, requires_safety_checker=False)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


# ------------------------------------------------------------------ 主流程 --


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import numpy as np
    import torch
    from PIL import Image

    from live3dgsavatar.config import load_config, resolve_model_ply
    from live3dgsavatar.core.camera import orbit_w2c, roll_w2c
    from live3dgsavatar.core.io import load_ply
    from live3dgsavatar.core.io.ply import read_embedded_config
    from live3dgsavatar.core.render import SimpleRasterizer
    from live3dgsavatar.core.render.faceid import face_id_map
    from live3dgsavatar.core.render.normals import normal_map
    from live3dgsavatar.core.types import Camera, Mesh
    from live3dgsavatar.data.datasets import resolve_dataset_dir
    from live3dgsavatar.data.scene import load_scene

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
    defects_dir = Path(args.defects or (
        Path(str(cfg.get("paths.output_dir", "output"))) / "defects"
        / subject / work / dataset))
    selected = defects_dir / "selected.json"
    if not selected.is_file():
        raise SystemExit(f"[error] 找不到 {selected}（先跑 find_defects.py）")
    out_dir = Path(args.out or (defects_dir / "pseudo"))
    acc_dir = out_dir / "accepted"
    dbg_dir = out_dir / "debug"
    for d in (acc_dir, dbg_dir):
        d.mkdir(parents=True, exist_ok=True)

    payload = json.loads(selected.read_text(encoding="utf-8"))
    labels = list(payload["views"]["chosen"])
    if args.limit:
        labels = labels[: args.limit]
    print(f"模型  : {ply}")
    print(f"数据  : {data_dir}")
    print(f"缺陷  : {defects_dir}（{len(labels)} 个视角）")
    print(f"产物  : {out_dir}")

    device = "cuda"
    scene = load_scene(cfg, frames=-1, data_dir=data_dir)
    avatar = load_ply(ply, config=read_embedded_config(ply), device=device)
    rasterizer = SimpleRasterizer()
    from live3dgsavatar.evaluation.views import parse_view_label

    w2c_data = torch.eye(4, dtype=torch.float32)
    w2c_data[:3, :3] = scene.R.detach().cpu()
    w2c_data[:3, 3] = scene.T.detach().cpu()
    bg = torch.tensor([float(x) for x in cfg.get("render.background", [0, 0, 0])],
                      dtype=torch.float32, device=device)
    faces = scene.faces.to(torch.int32)
    import nvdiffrast.torch as dr
    glctx = dr.RasterizeCudaContext()

    pipe = None if args.dry_run else build_pipeline(args)
    records = []
    for label in labels:
        info = parse_view_label(label)
        idx = int(info["expr"][5:]) if info["expr"].startswith("frame") else 0
        idx = min(max(idx, 0), scene.num_frames - 1)
        frame = scene.frames[idx]
        center = frame["mesh"].mean(dim=0).detach().cpu()
        view = orbit_w2c(w2c_data, center, info["yaw"], info["pitch"])
        view = roll_w2c(view, info["roll"])
        cam = Camera(K=scene.K, w2c=view.unsqueeze(0).to(device),
                     width=scene.width, height=scene.height).to(device)
        mesh = Mesh(verts=frame["mesh"].unsqueeze(0), faces=scene.faces,
                    uvs=scene.uvs, uv_faces=scene.uv_faces)
        weight = frame["blend_weight"].unsqueeze(0).to(device)
        with torch.no_grad():
            gs = avatar.deform(mesh, weight)
            out = rasterizer.render(gs, cam, bg)
            ids = face_id_map(mesh.verts, faces, cam, glctx)
            normal, surf = normal_map(mesh.verts, faces, cam, glctx,
                                      convention=args.normal_convention)
        alpha = out.alpha[0, 0].detach().cpu().numpy()
        img = (out.color[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
               * 255.0).astype(np.uint8)
        surf_np = surf.detach().cpu().numpy()
        # 洞：网格说"有表面"但 alpha 不足
        hole = surf_np & (alpha < 0.5)
        if not bool(hole.any()):
            gy, gx = np.gradient(alpha.astype(np.float32))
            spur = np.hypot(gx, gy) > 0.08
            hole = surf_np & spur
        mask = dilate(hole, args.mask_dilate)
        normal_u8 = (normal.detach().cpu().numpy() * 255).astype(np.uint8)

        Image.fromarray(img).save(dbg_dir / f"{label}.render.png")
        Image.fromarray(normal_u8).save(dbg_dir / f"{label}.normal.png")
        Image.fromarray((mask * 255).astype(np.uint8)).save(
            dbg_dir / f"{label}.mask.png")
        # 叠加图：在渲染图上画出掩码边界，便于**直接检查是否上下颠倒**
        overlay = img.copy()
        edge = mask ^ _erode(mask, 2)
        overlay[edge] = np.array([255, 0, 0], np.uint8)
        Image.fromarray(overlay).save(dbg_dir / f"{label}.overlay.png")
        # 自检：法线掩码与面 id 掩码必须几乎一致（同一套光栅化）
        ids_np = ids.detach().cpu().numpy() >= 0
        agree = float((surf_np & ids_np).sum()) / max(1, int((surf_np | ids_np).sum()))
        if agree < 0.98:
            print(f"  [warn] {label}: 法线掩码与面 id 掩码一致性 {agree:.3f}"
                  "（若明显偏低说明光栅化方向不一致）", flush=True)
        if args.dry_run:
            records.append({"label": label, "hole_px": int(mask.sum()),
                            "status": "dry-run"})
            continue

        pseudo_list = []
        for k in range(max(1, args.consistency)):
            g = torch.Generator(device="cuda").manual_seed(args.seed + k)
            res = pipe(
                prompt=args.prompt, negative_prompt=args.negative,
                image=Image.fromarray(img),
                mask_image=Image.fromarray((mask * 255).astype(np.uint8)),
                control_image=Image.fromarray(normal_u8),
                strength=float(args.strength),
                num_inference_steps=int(args.steps),
                guidance_scale=float(args.guidance),
                generator=g).images[0]
            pseudo_list.append(np.asarray(res.convert("RGB"), np.float32))
        pseudo = pseudo_list[0]

        # ---- 筛选 ----
        reasons = []
        iou = inside = tdiff = cons = 0.0
        out_m = mask
        if out_m.any():
            gen = pseudo
            trusted = ~out_m
            if trusted.any():
                diff = np.abs(gen - img.astype(np.float32)).mean(axis=-1)
                tdiff = float(diff[trusted].mean())
                if tdiff > args.trusted_tol:
                    reasons.append(f"可信区域改动 {tdiff:.2f} > {args.trusted_tol}")
            gen_fg = gen.max(axis=-1) > 8
            iou = foreground_iou(gen_fg, surf_np)
            if iou < args.min_iou:
                reasons.append(f"轮廓 IoU {iou:.3f} < {args.min_iou}")
            inside = float(gen[out_m].std())
            if inside < args.min_std:
                reasons.append(f"洞内过平 std {inside:.2f} < {args.min_std}")
            if len(pseudo_list) > 1:
                cons = masked_ssim(pseudo_list[0], pseudo_list[1], out_m)
                if cons < args.min_consistency:
                    reasons.append(
                        f"多次采样不一致 SSIM {cons:.3f} < {args.min_consistency}")
        else:
            iou, inside, tdiff, cons = 1.0, 0.0, 0.0, 1.0
        accepted = not reasons
        if accepted:
            Image.fromarray(pseudo.astype(np.uint8)).save(
                acc_dir / f"{label}.png")
        records.append({"label": label, "accepted": accepted,
                        "reasons": reasons, "hole_px": int(mask.sum()),
                        "silhouette_iou": round(float(iou), 4),
                        "trusted_diff": round(float(tdiff), 3),
                        "hole_std": round(float(inside), 2),
                        "consistency": round(float(cons), 4)})
        print(f"  {label:40s} {'✓' if accepted else '✗ ' + '; '.join(reasons)}",
              flush=True)

    report = {"subject": subject, "work_name": work, "dataset": dataset,
              "defects": str(defects_dir), "dry_run": bool(args.dry_run),
              "gates": {"trusted_tol": args.trusted_tol, "min_iou": args.min_iou,
                        "min_std": args.min_std,
                        "min_consistency": args.min_consistency},
              "views": records}
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    ok = sum(1 for r in records if r.get("accepted"))
    print(f"\n完成：{len(records)} 个视角，通过 {ok} 个")
    print(f"产物：{out_dir}")

    if args.install_refined and ok:
        import shutil

        refined = defects_dir / "refined"
        refined.mkdir(parents=True, exist_ok=True)
        n = 0
        for r in records:
            if not r.get("accepted"):
                continue
            shutil.copyfile(acc_dir / f"{r['label']}.png",
                            refined / f"{r['label']}.png")
            n += 1
        print(f"已装入 refined/：{n} 张（之后 `bash enhance.sh ... --skip-syncfix`）")
    return 0 if (args.dry_run or ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
