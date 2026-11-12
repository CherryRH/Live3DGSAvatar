#!/usr/bin/env python
"""用外部精修图微调模型：缺陷加权 + 精修图体检 + 验证集选点。

这条路线对应"缺陷检测 → 外部精修 → 微调"的闭环。几条刻意的约束：

1. **只在缺陷处监督**：权重图取 `|精修 − 渲染|`，前景内给下限、前景外为 0；
2. **精修图体检**：前景 IoU 太低或整体色差过大的视角直接跳过，
   避免把坏样本当目标；
3. **按真值选点**：每若干步在**验证帧**上算 PSNR/SSIM/LPIPS/锐度比。
   默认 `--select-by last`：只要最后一版比最佳低不超过 `--quality-tol`，
   就保留最后一版（小角度留出集测不到大角度变化）；超容差或最后一版
   没过硬门时才回退到 `model_best`。锐度/SSIM/LPIPS 硬门仍然生效；
4. **真实帧锚定**：每批混入真实数据，且验证帧从锚定里剔除，避免自己考自己。

`--ssim-weight` 默认 0（单独调用时只做 L1）；后训练流程传 0.2 + LPIPS 0.1。
SSIM 已改为纯 torch 实现，不再依赖 `fused-ssim` 的 backward。

不改模型格式、不改数据。

## 用法

```bash
# 1) 找缺陷并渲染待修视角（同时写出 refs/，作为精修的参考图）
python scripts/find_defects.py --subject <对象> --work-name <版本> --dataset <对象>

# 2) 用外部精修器处理 <缺陷目录>/views/，输出到 <缺陷目录>/refined/

# 3) 微调（一般通过 scripts/iterative_repair.py 串起来跑）
python scripts/refine_from_images.py --subject <对象> --work-name <版本> \
    --steps 600 --promote --promote-name <版本>_refined --force
```
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    p.add_argument("--real-only", action="store_true",
                   help="消融：只用真实帧，不用任何修复图")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--refined-per-batch", type=int, default=4)
    p.add_argument("--real-per-batch", type=int, default=4)
    p.add_argument("--lr-color", type=float, default=2e-3)
    p.add_argument("--lr-opacity", type=float, default=1e-2)
    p.add_argument("--lr-scaling", type=float, default=2e-3)
    p.add_argument("--lr-xyz", type=float, default=0.0,
                   help="切空间位置偏移的学习率；0 = 不动几何（默认）。"
                        "几何档建议 1e-5 ~ 5e-5（600 步约 6–30 mm）")
    p.add_argument("--weight-floor", type=float, default=0.05)
    p.add_argument("--ssim-weight", type=float, default=0.0,
                   help="SSIM 项权重（默认 0 = 关闭）。开启需要 fused-ssim 的 CUDA "
                        "算子；若 backward 报 illegal memory access 就保持 0")
    p.add_argument("--lpips-weight", type=float, default=0.0,
                   help="LPIPS 项权重；GAF 的 L_img 用 0.1。默认 0")
    p.add_argument("--lpips-from", type=int, default=0,
                   help="第几步开始启用 LPIPS（默认 0 = 一开始就开）")
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-frames", type=int, default=64,
                   help="选点用的帧数（从训练段随机取，且这些帧不参与锚定）")
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--select-by", choices=("last", "quality", "psnr"),
                   default="last",
                   help="最终模型选择：last=最后一次，若比最佳低不超过 "
                        "--quality-tol 就用 last（默认）；quality/psnr=旧行为")
    p.add_argument("--quality-tol", type=float, default=0.01,
                   help="last 模式允许相对最佳下降的容差")
    p.add_argument("--psnr-weight", type=float, default=0.002,
                   help="quality 模式下 PSNR 的小权重（默认 0.002/dB）")
    p.add_argument("--eval-mode", choices=("angle", "random", "holdout", "both"),
                   default="angle",
                   help="angle=全数据集里挑大角度帧（默认，适合增强阶段）；"
                        "random=全数据集随机；holdout=末尾 30%%（仅供对照）；"
                        "both=选点用 angle、另外报告 holdout")
    p.add_argument("--anchor-mode", choices=("angle", "uniform"), default="angle",
                   help="锚定帧采样：angle=按角度加权（默认），uniform=均匀；"
                        "两者都从整个数据集里取，且排除选点帧")
    p.add_argument("--angle-percentile", type=float, default=60.0,
                   help="angle 模式下选点帧的 |yaw| 分位下限（默认 60%%）")
    p.add_argument("--sharp-guard", type=float, default=None,
                   help="锐度比的绝对下限；默认按'训练前锐度 − 容差'自适应")
    p.add_argument("--sharp-tolerance", type=float, default=0.03,
                   help="允许的锐度比下降幅度（相对训练前，默认 0.03）")
    p.add_argument("--ssim-tolerance", type=float, default=0.005,
                   help="允许的 SSIM 下降幅度（相对训练前，默认 0.005）")
    p.add_argument("--lpips-tolerance", type=float, default=0.01,
                   help="允许的 LPIPS 上升幅度（相对训练前，默认 0.01）")
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
    """`psnr − penalty·max(0, guard − 锐度比)`：锐度掉太多会被扣分。

    `guard` 可以按"训练前的锐度 − 容差"自适应传入（见 main），
    这样不同模型/数据集不必共用同一个绝对值。
    """
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


def build_plan(args):
    """按命令行参数组装参数计划：默认只动外观，`--lr-xyz > 0` 才动位置。"""
    from live3dgsavatar.training.plans import GroupPlan, ParameterPlan

    groups = {
        "f_dc": GroupPlan(lr=args.lr_color, max_steps=args.steps),
        "opacity": GroupPlan(lr=args.lr_opacity, max_steps=args.steps),
        "scaling": GroupPlan(lr=args.lr_scaling, max_steps=args.steps),
    }
    if args.lr_xyz > 0:
        # 位置是切空间偏移（米），600 步 × lr 的累积量级要控制在毫米级
        groups["xyz"] = GroupPlan(lr=float(args.lr_xyz), max_steps=args.steps)
    return ParameterPlan(groups=groups)


def build_objective(args):
    """损失：L1 + 可选 SSIM / LPIPS（GAF 的 L_img 权重是 0.2 / 0.1）。"""
    from live3dgsavatar.training.objectives import Objective

    return Objective(l1=1.0, ssim=args.ssim_weight,
                     lpips=args.lpips_weight,
                     lpips_from=int(args.lpips_from), mask_weight=0.0)


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
    defects_dir.mkdir(parents=True, exist_ok=True)
    if args.real_only:
        args.refined_per_batch = 0
        refined_dir = None
        labels = []
    else:
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
        weights_all = np.asarray(d["blend_weight"], dtype=np.float32)
    ratio = float(cfg.get("train.pipeline.test_ratio", 0.3))
    held = max(1, int(round(total * ratio)))
    train_end = total - held

    print(f"模型    : {ply}")
    print(f"数据集  : {data_dir}（{dataset}，{total} 帧）")
    print(f"缺陷目录: {defects_dir}")
    print(f"精修图  : {len(labels)} 个")
    print(f"微调    : {args.steps} 步，每批 {args.refined_per_batch} 精修 + "
          f"{args.real_per_batch} 真实"
          + ("（real-only 消融）" if args.real_only else ""))
    print(f"SSIM    : {args.ssim_weight:g}" +
          ("（关闭）" if args.ssim_weight <= 0 else ""))
    print(f"损失    : L1 1.0 + SSIM {args.ssim_weight:g}"
          f" + LPIPS {args.lpips_weight:g}" +
          ("" if (args.ssim_weight > 0 or args.lpips_weight > 0) else "（仅 L1）"))
    print(f"几何    : xyz lr {args.lr_xyz:g}" +
          ("（不动位置）" if args.lr_xyz <= 0 else "（位置也参与微调）"))
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
    for label in ([] if args.real_only else labels):
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
    if not obs and not args.real_only:
        raise SystemExit("[error] 没有可用的精修视角")

    # ---- 选点帧：不依赖预训练的切片方式 ----
    # 增强阶段不知道预训练是怎么切分的，也没必要沿用；这里按"姿态"挑帧：
    #   angle  = 全数据集里 |yaw| 较大的那部分（默认）
    #   random = 全数据集随机
    #   holdout= 末尾 30%（保留，仅用于与历史数字对照）
    from live3dgsavatar.data.pose import angle_weights, head_yaw


    yaw_all = head_yaw(weights_all)
    rng_eval = np.random.default_rng(args.eval_seed)
    n_eval = int(min(args.eval_frames, total))

    if args.eval_mode in ("angle", "both"):
        cut = float(np.percentile(np.abs(yaw_all), args.angle_percentile))
        pool = np.flatnonzero(np.abs(yaw_all) >= cut)
        pick = rng_eval.choice(pool, size=min(n_eval, len(pool)), replace=False)
        eval_ids_final = sorted(int(i) for i in pick)
        print(f"  选点集：|yaw| ≥ {cut:.1f}° 的帧中随机取 {len(eval_ids_final)} 帧"
              "（不参与锚定）")
    elif args.eval_mode == "random":
        pick = rng_eval.choice(total, size=n_eval, replace=False)
        eval_ids_final = sorted(int(i) for i in pick)
        print(f"  选点集：全数据集随机 {len(eval_ids_final)} 帧（不参与锚定）")
    else:  # holdout
        eval_ids_final = list(np.linspace(train_end, total - 1,
                                          min(n_eval, held)).astype(int))
        print(f"  选点集：末尾 30% 的 {len(eval_ids_final)} 帧（仅对照）")
    eval_ids = set(eval_ids_final)

    def _frame_for(i: int) -> dict:
        """按全局帧号取网格（训练段用 scene，留出段用 hold_scene）。"""
        if i < train_end:
            return scene.frames[i]
        return hold_scene.frames[i - train_end]

    def _camera_for(i: int):
        if i < train_end:
            return Camera(K=scene.K, w2c=w2c_data.unsqueeze(0).to(device),
                          width=scene.width, height=scene.height).to(device)
        return Camera(K=hold_scene.K, w2c=w2c_data.unsqueeze(0).to(device),
                      width=hold_scene.width, height=hold_scene.height).to(device)

    holdout_ids = list(range(train_end, total))

    def evaluate_frames(ids: list[int] | set[int]) -> dict:
        from live3dgsavatar.evaluation.protocol import (frame_metrics,
                                                       lpips_score)

        psnrs, ratios, ssims, lpipss = [], [], [], []
        for i in sorted(ids):
            path = data_dir / "images" / f"{i:05d}.png"
            if not path.is_file():
                continue
            rgba = np.asarray(Image.open(path).convert("RGBA"))
            gt = rgba[..., :3].copy()
            gt[rgba[..., 3] <= 127] = 0
            mask = rgba[..., 3] > 127
            frame = _frame_for(i)
            out = render_frame(frame, _camera_for(i),
                               frame["blend_weight"].unsqueeze(0).to(device))
            pred = (out.color[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                    * 255.0).astype(np.uint8)
            v = psnr_masked(pred, gt, mask)
            if v is not None:
                psnrs.append(v)
                ratios.append(sharpness_ratio(pred, gt, mask))
                m = frame_metrics(pred, gt, mask)
                if m.get("ssim") is not None:
                    ssims.append(float(m["ssim"]))
                if m.get("lpips") is not None:
                    lpipss.append(float(m["lpips"]))
        if not psnrs:
            return {"psnr": float("nan"), "sharp": float("nan"), "n": 0}
        out = {"psnr": float(np.mean(psnrs)), "sharp": float(np.mean(ratios)),
               "n": len(psnrs)}
        if ssims:
            out["ssim"] = float(np.mean(ssims))
        if lpipss:
            out["lpips"] = float(np.mean(lpipss))
        return out

    before = evaluate_frames(eval_ids_final)
    before_holdout = (evaluate_frames(holdout_ids[:args.eval_frames])
                      if args.eval_mode in ("holdout", "both") else None)
    print(f"  微调前：PSNR {before['psnr']:.2f} dB，锐度比 "
          f"{before['sharp']:.3f}（n={before['n']}）")
    if before_holdout:
        print(f"          留出段参照：PSNR {before_holdout['psnr']:.2f} dB，"
              f"锐度比 {before_holdout['sharp']:.3f}（n={before_holdout['n']}）")
    guard = (float(args.sharp_guard) if args.sharp_guard is not None
             else float(before["sharp"]) - float(args.sharp_tolerance))
    print(f"  锐度守卫：{guard:.3f}（{'绝对' if args.sharp_guard is not None else '相对容差'}）")
    # 感知指标也要守住：只按 PSNR 选点会挑出"更平滑"的检查点
    ssim_min = (float(before["ssim"]) - float(args.ssim_tolerance)
                if "ssim" in before else None)
    lpips_max = (float(before["lpips"]) + float(args.lpips_tolerance)
                 if "lpips" in before else None)
    if ssim_min is not None:
        print(f"  SSIM 下限：{ssim_min:.4f}（训练前 {before['ssim']:.4f}）")
    if lpips_max is not None:
        print(f"  LPIPS 上限：{lpips_max:.4f}（训练前 {before['lpips']:.4f}）")

    class _Mixed:
        def __len__(self):
            return len(obs) + len(real)

        def random_batch(self, batch_size: int):
            import random

            n_ref = 0 if args.real_only else min(args.refined_per_batch, batch_size)
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
                # 按姿态权重从整个数据集取锚定帧（angle 模式）
                idx = np.random.choice(anchor_ids, size=n_real,
                                       replace=True, p=anchor_probs)
                b = real.batch([int(i) for i in idx])
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

    # 锚定帧来自**整个数据集**（不看预训练切片），并排除选点帧
    real = ManifestSource(data_dir, split="all", device="cpu", seed=args.seed,
                          exclude=eval_ids or None)
    anchor_ids = np.array([i for i in range(total) if i not in eval_ids])
    if args.anchor_mode == "angle":
        w = angle_weights(yaw_all[anchor_ids])
        anchor_probs = w / w.sum()
    else:
        anchor_probs = None
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
            self.best = -1e9          # 当前评分（psnr 或 quality），不是 PSNR
            self.best_seconds: float | None = None
            self._t0: float | None = None
            self.history: list[dict] = []

        def on_start(self, trainer) -> None:
            self._t0 = time.perf_counter()

        def on_step(self, trainer, iteration, parts) -> None:
            if args.eval_every <= 0 or (iteration + 1) % args.eval_every != 0:
                return
            m = evaluate_frames(eval_ids_final)
            # 硬门：锐度 / SSIM / LPIPS 任一越界就**不接受**该检查点。
            # 只按 PSNR 排序时，模型会挑"更平滑"的解：PSNR 涨、SSIM/LPIPS 退。
            reasons = []
            if m["sharp"] < guard:
                reasons.append(f"锐度 {m['sharp']:.3f} < {guard:.3f}")
            if ssim_min is not None and m.get("ssim") is not None \
                    and m["ssim"] < ssim_min:
                reasons.append(f"SSIM {m['ssim']:.4f} < {ssim_min:.4f}")
            if lpips_max is not None and m.get("lpips") is not None \
                    and m["lpips"] > lpips_max:
                reasons.append(f"LPIPS {m['lpips']:.4f} > {lpips_max:.4f}")
            passed = not reasons and bool(m["n"])
            score = (m["psnr"] if args.select_by == "psnr" else
                     (m.get("ssim") or 0.0) - (m.get("lpips") or 1.0)
                     + args.psnr_weight * m["psnr"])  # last 也用 quality 评分
            self.history.append({
                "iteration": iteration, "passed": passed, "score": round(score, 4),
                "psnr": round(m["psnr"], 3), "sharp": round(m["sharp"], 3),
                "ssim": (round(m["ssim"], 4) if m.get("ssim") is not None else None),
                "lpips": (round(m["lpips"], 4) if m.get("lpips") is not None else None),
                "reasons": reasons})
            extra = ""
            if m.get("ssim") is not None:
                extra += f"  SSIM {m['ssim']:.4f}"
            if m.get("lpips") is not None:
                extra += f"  LPIPS {m['lpips']:.4f}"
            print(f"  [{iteration}] PSNR {m['psnr']:.2f}  锐度比 {m['sharp']:.3f}"
                  f"{extra}" + ("" if passed else f"  ✗ {', '.join(reasons)}"),
                  flush=True)
            if passed and score > self.best:
                self.best = score
                save_ply(avatar, self.out_dir / "model_best.ply")
                secs = (time.perf_counter() - self._t0
                        if self._t0 is not None else 0.0)
                self.best_seconds = secs
                print(f"       ↑ 刷新最佳 → model_best.ply（累计 {secs:.1f}s）",
                      flush=True)

        def on_end(self, trainer) -> None:
            pass

    out_dir = defects_dir / "finetune"
    out_dir.mkdir(parents=True, exist_ok=True)
    # 旧的最佳检查点必须清掉：否则本次若从未刷新，最后会把**上一轮**的模型当结果
    for stale in (out_dir / "model_best.ply", out_dir / "model.ply"):
        if stale.is_file():
            stale.unlink()
    plan = build_plan(args)
    objective = build_objective(args)
    evaluator = _Eval(out_dir)
    trainer = Trainer(
        source=source, plan=plan, objective=objective, render_fn=render_fn,
        params=avatar.parameter_groups(), iterations=args.steps,
        batch_size=args.refined_per_batch + args.real_per_batch,
        blend_start=0, hooks=[LogHook(every=max(1, args.steps // 10)), evaluator],
        device=device)
    report = trainer.run()

    save_ply(avatar, out_dir / "model.ply")
    print(f"  [save] 最终 model.ply（累计 {report.seconds:.1f}s）", flush=True)
    final_metrics = evaluate_frames(eval_ids_final)

    def _score(m):
        return (m["psnr"] if args.select_by == "psnr" else
                (m.get("ssim") or 0.0) - (m.get("lpips") or 1.0)
                + args.psnr_weight * m["psnr"])

    final_score = _score(final_metrics)
    best_score = evaluator.best
    final_entry = evaluator.history[-1] if evaluator.history else None
    final_passed = bool(final_entry and final_entry.get("passed"))
    best_ply = out_dir / "model_best.ply"
    if args.select_by in ("quality", "psnr"):
        use_best = True
    else:  # last：评分下降在容差内就保留最后一版
        use_best = (best_score is not None
                    and (final_score < best_score - args.quality_tol
                         or not final_passed))
    use_best = bool(use_best and best_ply.is_file())

    if use_best:
        best_model = load_ply(best_ply,
                              config=read_embedded_config(best_ply),
                              device=device)
        save_ply(best_model, out_dir / "model.ply")
        after = evaluate_frames(eval_ids_final)
        selected = "best"
        selected_score = best_score
        selected_seconds = evaluator.best_seconds
        print("  [save] 最后一版低于容差/未过门限，改用 model_best", flush=True)
    else:
        after = final_metrics
        selected = "last"
        selected_score = final_score
        selected_seconds = report.seconds

    no_pass = bool(evaluator.history) and not best_ply.is_file()
    if no_pass:
        print("\n  [warn] 没有检查点通过锐度/SSIM/LPIPS 门限；"
              "本次微调判为失败，不落地", flush=True)
    after_holdout = (evaluate_frames(holdout_ids[:args.eval_frames])
                     if before_holdout is not None else None)
    payload = {
        "model": str(ply), "steps": int(args.steps),
        "seconds": round(report.seconds, 1),
        "best_seconds": (round(selected_seconds, 1)
                         if selected_seconds is not None else None),
        "best_score": (round(best_score, 4)
                       if best_score is not None else None),
        "selected": selected,
        "selected_score": (round(selected_score, 4)
                           if selected_score is not None else None),
        "quality_tol": args.quality_tol,
        "refined_views": len(obs), "skipped": skipped,
        "eval_mode": args.eval_mode,
        "eval_frames": len(eval_ids_final),
        "before": before, "after": after,
        "holdout_before": before_holdout, "holdout_after": after_holdout,
        "score_history": evaluator.history,
        "ssim_weight": args.ssim_weight,
        "lpips_weight": args.lpips_weight, "lpips_from": args.lpips_from,
        "lr_xyz": args.lr_xyz, "sharp_guard": args.sharp_guard,
        "select_by": args.select_by, "psnr_weight": args.psnr_weight,
        "gates": {"sharp_min": guard, "ssim_min": ssim_min,
                  "lpips_max": lpips_max},
        "passed_any": any(h.get("passed") for h in evaluator.history),
    }
    (out_dir / "report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n完成：{report.iterations} 步，{report.seconds:.1f} s")
    print(f"  选择     ：{selected}"
          + (f"（score {selected_score:.4f}）" if selected_score is not None else ""))
    print(f"  选点集 PSNR：{before['psnr']:.2f} → {after['psnr']:.2f} dB")
    if "ssim" in after:
        print(f"  选点集 SSIM：{before.get('ssim', float('nan')):.4f} → "
              f"{after['ssim']:.4f}")
    if "lpips" in after:
        print(f"  选点集 LPIPS：{before.get('lpips', float('nan')):.4f} → "
              f"{after['lpips']:.4f}")
    print(f"  选点集锐度比：{before['sharp']:.3f} → {after['sharp']:.3f}")
    if after_holdout:
        print(f"  留出段 PSNR：{before_holdout['psnr']:.2f} → "
              f"{after_holdout['psnr']:.2f} dB（参照，不参与选点）")
    print(f"  产物：{out_dir}/model.ply")

    if no_pass:
        return 1

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
