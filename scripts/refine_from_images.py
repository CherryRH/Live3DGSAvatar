#!/usr/bin/env python
"""用外部精修图微调模型：缺陷加权 + 高频保持 + 留出真值选点。

这条路线对应"缺陷检测 → 外部精修 → 微调"的闭环。前一版能在留出集上拿到约
+0.9 dB，但也有已知问题：画面变软（高频能量 −13%）、没有防住精修图自身的偏差。
这一版针对这三点做了收紧：

1. **只在缺陷处监督**：权重图取 `|精修 − 渲染|`，前景内给下限、前景外为 0；
2. **保住高频**：额外加 SSIM 项（`fused_ssim`），抑制 L1 带来的整体平滑；
3. **按留出真值选点**：每若干步在留出段上算 PSNR 与锐度比，用
   `score = PSNR − λ·max(0, 0.85 − 锐度比)` 选最佳检查点，锐度掉太多会被扣分；
4. **精修图体检**：前景 IoU 太低或整体色差过大的图直接跳过，避免把坏样本当目标。

不改模型格式、不改数据。

## 用法

```bash
# 1) 找缺陷并渲染待修视角（同时写出 refs/，作为精修的参考图）
python scripts/find_defects.py --subject crh_1 --work-name run --dataset crh_1

# 2) 用外部精修器处理 <缺陷目录>/views/，输出到 <缺陷目录>/refined/

# 3) 微调
python scripts/refine_from_images.py --subject crh_1 --work-name run \\
    --steps 600 --promote --promote-name run_refined --force
```
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
    p = argparse.ArgumentParser(description="用精修图微调模型")
    p.add_argument("--subject", default=None)
    p.add_argument("--work-name", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--defects", type=Path, default=None)
    p.add_argument("--refined-dir", type=Path, default=None)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--refined-per-batch", type=int, default=4)
    p.add_argument("--real-per-batch", type=int, default=4)
    p.add_argument("--lr-color", type=float, default=2e-3)
    p.add_argument("--lr-opacity", type=float, default=1e-2)
    p.add_argument("--lr-scaling", type=float, default=2e-3)
    p.add_argument("--weight-floor", type=float, default=0.05)
    p.add_argument("--ssim-weight", type=float, default=0.2,
                   help="SSIM 项权重；用于抑制 L1 造成的整体平滑")
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-frames", type=int, default=32)
    p.add_argument("--sharp-guard", type=float, default=0.85,
                   help="锐度比低于该值时开始扣分")
    p.add_argument("--sharp-penalty", type=float, default=20.0,
                   help="锐度比每低 1.0 扣多少 dB")
    p.add_argument("--min-fg-iou", type=float, default=0.5,
                   help="精修图与渲染的前景 IoU 低于此值则跳过该视角")
    p.add_argument("--max-color-shift", type=float, default=60.0,
                   help="前景平均色差（0–255）超过此值则跳过")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--promote", action="store_true")
    p.add_argument("--promote-name", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def foreground_iou(a, b) -> float:
    """两个前景掩码的 IoU。"""
    import numpy as np

    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    union = int((a | b).sum())
    return float((a & b).sum()) / union if union else 1.0


def color_shift(render, refined, mask) -> float:
    """前景内的平均色差（0–255 尺度）。"""
    import numpy as np

    m = np.asarray(mask, bool)
    if not bool(m.any()):
        return 0.0
    a = np.asarray(render, np.float32)[m]
    b = np.asarray(refined, np.float32)[m]
    return float(np.abs(a - b).mean())


def composite_score(psnr: float, sharp_ratio: float, *, guard: float = 0.85,
                    penalty: float = 20.0) -> float:
    """`psnr − penalty·max(0, guard − 锐度比)`：锐度掉太多会被扣分。"""
    drop = max(0.0, float(guard) - float(sharp_ratio))
    return float(psnr) - float(penalty) * drop


def sharpness_ratio(render, gt, mask) -> float:
    """前景内平均梯度之比（渲染 / 真值）。"""
    import numpy as np

    m = np.asarray(mask, bool)
    if not bool(m.any()):
        return 1.0

    def grad(img):
        grey = np.asarray(img, np.float32).mean(axis=-1)
        gy, gx = np.gradient(grey)
        return float(np.hypot(gx, gy)[m].mean())

    ref = grad(gt)
    return float(grad(render)) / ref if ref > 0 else 1.0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import numpy as np

    from live3dgsavatar.config import load_config, resolve_model_ply
    from live3dgsavatar.data.datasets import resolve_dataset_dir
    from live3dgsavatar.evaluation.views import LABEL_RE, parse_view_label

    cfg = load_config()
    subject = args.subject or str(cfg.get("subject") or "")
    work = args.work_name or str(cfg.get("work_name") or "")
    if not (subject and work):
        raise SystemExit("[error] 需要 --subject 与 --work-name")
    cfg.set("subject", subject)
    cfg.set("work_name", work)
    dataset = args.dataset or subject
    try:
        data_dir = resolve_dataset_dir(cfg, dataset)
        ply = resolve_model_ply(cfg)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"[error] {e}") from None

    defects_dir = Path(args.defects or (
        Path(str(cfg.get("paths.output_dir", "output"))) / "defects"
        / subject / work / dataset))
    refined_dir = Path(args.refined_dir or (defects_dir / "refined"))
    selected = defects_dir / "selected.json"
    if not selected.is_file():
        raise SystemExit(f"[error] 找不到 {selected}（先跑 find_defects.py）")
    if not refined_dir.is_dir():
        if not args.dry_run:
            raise SystemExit(f"[error] 找不到精修图目录 {refined_dir}")
        print(f"  [warn] 精修图目录不存在（dry-run 继续）：{refined_dir}")
    labels = sorted(p.stem for p in refined_dir.glob("*.png")
                    if LABEL_RE.match(p.stem))

    with np.load(data_dir / "render_inputs.npz", allow_pickle=True) as d:
        total = int(d["mesh"].shape[0])
    ratio = float(cfg.get("train.pipeline.test_ratio", 0.1))
    held = max(1, int(round(total * ratio)))
    train_end = total - held

    print(f"模型    : {ply}")
    print(f"数据集  : {data_dir}（{dataset}，{total} 帧）")
    print(f"缺陷目录: {defects_dir}")
    print(f"精修图  : {len(labels)} 个")
    print(f"微调    : {args.steps} 步，每批 {args.refined_per_batch} 精修 + "
          f"{args.real_per_batch} 真实")
    if args.dry_run:
        print("\n[dry-run] 未执行")
        return 0

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("[error] 需要 GPU")

    import nvdiffrast.torch as dr
    from PIL import Image

    from live3dgsavatar.core.camera import orbit_w2c
    from live3dgsavatar.core.io import load_ply, save_ply
    from live3dgsavatar.core.io.ply import read_embedded_config
    from live3dgsavatar.core.metrics import psnr_masked
    from live3dgsavatar.core.render import SimpleRasterizer
    from live3dgsavatar.core.render.faceid import face_id_map
    from live3dgsavatar.core.types import Camera, Mesh
    from live3dgsavatar.data.models import models_root, promote_model
    from live3dgsavatar.data.scene import load_scene
    from live3dgsavatar.data.sources import Batch, ManifestSource
    from live3dgsavatar.evaluation import refine_weight_map
    from live3dgsavatar.training.objectives import Objective
    from live3dgsavatar.training.plans import GroupPlan, ParameterPlan
    from live3dgsavatar.training.trainer import LogHook, Trainer

    torch.manual_seed(args.seed)
    device = "cuda"
    glctx = dr.RasterizeCudaContext()
    scene = load_scene(cfg, frames=train_end, data_dir=data_dir)
    hold_scene = load_scene(cfg, frames=held, frame_offset=train_end,
                            data_dir=data_dir)
    avatar = load_ply(ply, config=read_embedded_config(ply), device=device)
    rasterizer = SimpleRasterizer()
    faces = scene.faces.to(torch.int32)
    w2c_data = torch.eye(4, dtype=torch.float32)
    w2c_data[:3, :3] = scene.R.detach().cpu()
    w2c_data[:3, 3] = scene.T.detach().cpu()
    bg = torch.tensor([float(x) for x in cfg.get("render.background", [0, 0, 0])],
                      dtype=torch.float32, device=device)

    def render_frame(frame, cam, weights):
        mesh = Mesh(verts=frame["mesh"].unsqueeze(0), faces=scene.faces,
                    uvs=scene.uvs, uv_faces=scene.uv_faces)
        with torch.no_grad():
            gs = avatar.deform(mesh, weights)
            return rasterizer.render(gs, cam, bg)

    # ---- 建构精修视角的观测 ----
    obs, skipped = [], []
    for label in labels:
        info = parse_view_label(label)
        idx = int(info["expr"][5:]) if info["expr"].startswith("frame") else 0
        idx = min(max(idx, 0), scene.num_frames - 1)
        frame = scene.frames[idx]
        center = frame["mesh"].mean(dim=0).detach().cpu()
        view = orbit_w2c(w2c_data, center, info["yaw"], info["pitch"])
        cam = Camera(K=scene.K, w2c=view.unsqueeze(0).to(device),
                     width=scene.width, height=scene.height).to(device)
        weight = frame["blend_weight"].unsqueeze(0).to(device)
        out = render_frame(frame, cam, weight)
        degraded = (out.color[0].clamp(0, 1).permute(1, 2, 0).detach()
                    .cpu().numpy() * 255.0).astype(np.float32)
        refined = np.asarray(Image.open(refined_dir / f"{label}.png")
                             .convert("RGB"), dtype=np.float32)
        if refined.shape != degraded.shape:
            skipped.append((label, "尺寸不一致"))
            continue
        fg = refined.max(axis=-1) > 8
        iou = foreground_iou(out.alpha[0, 0].detach().cpu().numpy() > 0.5, fg)
        shift = color_shift(degraded, refined, fg)
        if iou < args.min_fg_iou:
            skipped.append((label, f"前景 IoU {iou:.2f}"))
            continue
        if shift > args.max_color_shift:
            skipped.append((label, f"色差 {shift:.0f}"))
            continue
        wmap = refine_weight_map(degraded, refined, fg, floor=args.weight_floor)
        obs.append({
            "label": label, "mesh": frame["mesh"].detach().cpu(),
            "blend_weight": frame["blend_weight"].detach().cpu(),
            "image": torch.from_numpy(refined.transpose(2, 0, 1) / 255.0),
            "mask": torch.from_numpy(fg[None].astype(np.float32)),
            "weight_map": torch.from_numpy(wmap[None]),
            "K": scene.K.detach().cpu(), "w2c": cam.w2c[0].detach().cpu(),
        })
    if skipped:
        print(f"  [warn] 跳过 {len(skipped)} 张：" +
              "，".join(f"{l}（{r}）" for l, r in skipped[:3]) + "…")
    if not obs:
        raise SystemExit("[error] 没有可用的精修视角")

    # ---- 留出段评估 ----
    eval_idx = np.linspace(0, hold_scene.num_frames - 1,
                           min(args.eval_frames, hold_scene.num_frames)).astype(int)
    cam_hold = Camera(K=hold_scene.K, w2c=w2c_data.unsqueeze(0).to(device),
                      width=hold_scene.width, height=hold_scene.height).to(device)

    def evaluate_holdout() -> dict:
        psnrs, ratios = [], []
        for local in eval_idx:
            i = train_end + int(local)
            path = data_dir / "images" / f"{i:05d}.png"
            if not path.is_file():
                continue
            rgba = np.asarray(Image.open(path).convert("RGBA"))
            gt = rgba[..., :3].copy()
            gt[rgba[..., 3] <= 127] = 0
            mask = rgba[..., 3] > 127
            frame = hold_scene.frames[int(local)]
            out = render_frame(frame, cam_hold,
                               frame["blend_weight"].unsqueeze(0).to(device))
            pred = (out.color[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                    * 255.0).astype(np.uint8)
            v = psnr_masked(pred, gt, mask)
            if v is not None:
                psnrs.append(v)
                ratios.append(sharpness_ratio(pred, gt, mask))
        if not psnrs:
            return {"psnr": float("nan"), "sharp": float("nan"), "n": 0}
        return {"psnr": float(np.mean(psnrs)), "sharp": float(np.mean(ratios)),
                "n": len(psnrs)}

    before = evaluate_holdout()
    print(f"  微调前：PSNR {before['psnr']:.2f} dB，锐度比 "
          f"{before['sharp']:.3f}（n={before['n']}）")

    class _Mixed:
        def __len__(self):
            return len(obs) + len(real)

        def random_batch(self, batch_size: int):
            import random

            n_ref = min(args.refined_per_batch, batch_size)
            n_real = max(0, batch_size - n_ref)
            parts = []
            for _ in range(n_ref):
                it = random.choice(obs)
                parts.append(Batch(
                    mesh=it["mesh"].unsqueeze(0).to(device),
                    blend_weight=it["blend_weight"].unsqueeze(0).to(device),
                    image=it["image"].unsqueeze(0).to(device),
                    mask=it["mask"].unsqueeze(0).to(device),
                    weight_map=it["weight_map"].unsqueeze(0).to(device),
                    K=it["K"].unsqueeze(0).to(device),
                    w2c=it["w2c"].unsqueeze(0).to(device)))
            if n_real:
                b = real.random_batch(n_real)
                m = b.mask.to(device=device, dtype=torch.float32)
                parts.append(Batch(
                    mesh=b.mesh.to(device), blend_weight=b.blend_weight.to(device),
                    image=b.image.to(device), mask=m,
                    weight_map=torch.ones_like(m),
                    K=scene.K.unsqueeze(0).expand(n_real, 3, 3).to(device).clone(),
                    w2c=w2c_data.unsqueeze(0).expand(n_real, 4, 4).to(device).clone()))
            return Batch(
                mesh=torch.cat([p.mesh for p in parts]),
                blend_weight=torch.cat([p.blend_weight for p in parts]),
                image=torch.cat([p.image for p in parts]),
                mask=torch.cat([p.mask for p in parts]),
                weight_map=torch.cat([p.weight_map for p in parts]),
                K=torch.cat([p.K for p in parts]),
                w2c=torch.cat([p.w2c for p in parts]))

        def iter_batches(self, batch_size: int, shuffle: bool = False):
            for _ in range(4):
                yield self.random_batch(batch_size)

    real = ManifestSource(data_dir, split="train", test_ratio=ratio,
                          device="cpu", seed=args.seed)
    source = _Mixed()

    def render_fn(mesh_verts, weight, batch, iteration):
        from types import SimpleNamespace

        outs = []
        for i in range(batch.K.shape[0]):
            cam = Camera(K=batch.K[i].to(device), w2c=batch.w2c[i].to(device),
                         width=scene.width, height=scene.height).to(device)
            mesh = Mesh(verts=mesh_verts[i:i + 1].to(device),
                        faces=scene.faces.to(device), uvs=scene.uvs.to(device),
                        uv_faces=scene.uv_faces.to(device))
            gs = avatar.deform(mesh, None if weight is None
                               else weight[i:i + 1].to(device))
            outs.append(rasterizer.render(gs, cam, bg))
        return SimpleNamespace(
            color=torch.cat([o.color for o in outs]),
            alpha=torch.cat([o.alpha for o in outs]))

    class _Eval:
        """按"留出 PSNR − 锐度惩罚"选最佳检查点。"""

        def __init__(self, out_dir: Path):
            self.out_dir = out_dir
            self.best = -1e9
            self.history: list[dict] = []

        def on_start(self, trainer) -> None:
            pass

        def on_step(self, trainer, iteration, parts) -> None:
            if args.eval_every <= 0 or (iteration + 1) % args.eval_every != 0:
                return
            m = evaluate_holdout()
            score = composite_score(m["psnr"], m["sharp"],
                                    guard=args.sharp_guard,
                                    penalty=args.sharp_penalty)
            self.history.append({"iteration": iteration, "score": round(score, 3),
                                 "psnr": round(m["psnr"], 3),
                                 "sharp": round(m["sharp"], 3)})
            print(f"  [{iteration}] PSNR {m['psnr']:.2f}  锐度比 {m['sharp']:.3f}"
                  f"  score {score:.2f}", flush=True)
            if score > self.best and m["n"]:
                self.best = score
                save_ply(avatar, self.out_dir / "model_best.ply")
                print(f"       ↑ 刷新最佳 → model_best.ply", flush=True)

        def on_end(self, trainer) -> None:
            pass

    out_dir = defects_dir / "finetune"
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = ParameterPlan(groups={
        "f_dc": GroupPlan(lr=args.lr_color, max_steps=args.steps),
        "opacity": GroupPlan(lr=args.lr_opacity, max_steps=args.steps),
        "scaling": GroupPlan(lr=args.lr_scaling, max_steps=args.steps),
    })
    objective = Objective(l1=1.0, ssim=args.ssim_weight, mask_weight=0.0)
    evaluator = _Eval(out_dir)
    trainer = Trainer(
        source=source, plan=plan, objective=objective, render_fn=render_fn,
        params=avatar.parameter_groups(), iterations=args.steps,
        batch_size=args.refined_per_batch + args.real_per_batch,
        blend_start=0, hooks=[LogHook(every=max(1, args.steps // 10)), evaluator],
        device=device)
    report = trainer.run()

    save_ply(avatar, out_dir / "model.ply")
    after = evaluate_holdout()
    if (out_dir / "model_best.ply").is_file():
        # 用最佳检查点覆盖最终产物（若有）
        best = load_ply(out_dir / "model_best.ply",
                        config=read_embedded_config(out_dir / "model_best.ply"),
                        device=device)
        save_ply(best, out_dir / "model.ply")
    payload = {
        "model": str(ply), "steps": int(args.steps),
        "seconds": round(report.seconds, 1),
        "refined_views": len(obs), "skipped": skipped,
        "holdout_before": before, "holdout_after": after,
        "score_history": evaluator.history,
        "ssim_weight": args.ssim_weight, "sharp_guard": args.sharp_guard,
    }
    (out_dir / "report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n完成：{report.iterations} 步，{report.seconds:.1f} s")
    print(f"  留出 PSNR：{before['psnr']:.2f} → {after['psnr']:.2f} dB")
    print(f"  留出锐度比：{before['sharp']:.3f} → {after['sharp']:.3f}")
    print(f"  产物：{out_dir}/model.ply")

    if args.promote:
        name = args.promote_name or f"{work}_refined"
        target = models_root(cfg) / subject / name
        try:
            promote_model(source_ply=out_dir / "model.ply",
                          structure_config=ply.parent / "config.yaml",
                          report=out_dir / "report.json", target=target,
                          force=args.force)
        except (FileExistsError, FileNotFoundError) as e:
            print(f"[error] {e}", file=sys.stderr)
            return 1
        print(f"已装入：{target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
