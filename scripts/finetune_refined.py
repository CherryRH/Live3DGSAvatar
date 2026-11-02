#!/usr/bin/env python
"""用"精修图"做一次缺陷加权微调。

## 与热更新的关系

热更新（`online_update.py`）解决的是"换了摄像头/光照，颜色整体偏了"：只更新颜色，
同一份修正对所有帧共享。它对极角缺陷**不够用** —— 侧脸糊成一团、发际线空洞，
本质是那里没有足够的高斯覆盖，改颜色填不上。

所以这里沿用同一条训练循环，但换三样东西：

| | 热更新 | 本脚本 |
|---|---|---|
| 监督 | 实时观测帧 | SyncFix 精修图，**只在缺陷区加权** |
| 参数 | f_dc（+opacity）| f_dc + opacity + **scaling** |
| 权重 | 前景掩码 | 掩码 × **|精修 − 渲染| 的逐像素权重** |

## 输入

`find_defects.py` 的产物目录：`selected.json`、`refined/*.png`，以及数据集本身。
每个候选的（网格、驱动参数、相机）按 `selected.json` 重建，与精修图配对。

## 为什么混入真实帧

只用精修图监督等于拟合"增强器的风格"（CAP4D 那轮的教训）。所以每批固定混入真实帧
做锚定，验收也看**留出真值**：PSNR 不能掉。

## 用法

```bash
python scripts/finetune_refined.py --subject crh_1 --work-name run \
    --defects output/defects/crh_1/run/crh_1 --steps 600
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

from live3dgsavatar.evaluation.views import (  # noqa: E402
    LABEL_RE, parse_view_label)


def parse_label(label: str) -> dict:
    """兼容旧名：语义转给 `evaluation.views.parse_view_label`。"""
    return parse_view_label(label)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="用精修图做缺陷加权微调")
    p.add_argument("--subject", default=None)
    p.add_argument("--work-name", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--defects", type=Path, default=None,
                   help="find_defects.py 的产物目录（默认按约定推导）")
    p.add_argument("--refined-dir", type=Path, default=None,
                   help="精修图目录（默认 <defects>/refined）")
    p.add_argument("--from-selection", action="store_true",
                   help="只用在 selected.json 里的候选；默认用目录里所有能解析的精修图")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--refined-per-batch", type=int, default=4)
    p.add_argument("--real-per-batch", type=int, default=4)
    p.add_argument("--lr-color", type=float, default=2e-3)
    p.add_argument("--lr-opacity", type=float, default=1e-2)
    p.add_argument("--lr-scaling", type=float, default=2e-3)
    p.add_argument("--params", choices=("defect", "color", "color+opacity"),
                   default="defect",
                   help="更新哪些参数：defect=颜色+不透明度+尺度（默认），"
                        "color=只颜色，color+opacity=颜色+不透明度")
    p.add_argument("--eval-frames", type=int, default=-1,
                   help="留出集评估用多少帧；-1 表示全部（与训练时的验证口径一致）")
    p.add_argument("--weight-floor", type=float, default=0.05,
                   help="权重图的下限：避免只在极小区域监督导致梯度太稀")
    p.add_argument("--promote", action="store_true",
                   help="把微调后的模型装进 models/<对象>/<版本>_refined/")
    p.add_argument("--promote-name", default=None,
                   help="落地目录名（默认 <版本>_refined）")
    p.add_argument("--force", action="store_true", help="目标已存在时覆盖")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def parse_label(label: str) -> dict:
    """把候选标签拆成 `{yaw, pitch, roll, expr}`。"""
    m = LABEL_RE.match(label)
    if m is None:
        raise ValueError(f"无法解析候选标签：{label!r}")
    return {"yaw": float(m.group(1)), "pitch": float(m.group(2)),
            "roll": float(m.group(3)), "expr": m.group(4)}


def render_observation(avatar, rasterizer, scene, frame, camera, bg):
    """渲染一个观测（网格 + 该帧驱动参数 + 指定相机）。"""
    import torch

    from live3dgsavatar.core.types import Mesh

    mesh = Mesh(verts=frame["mesh"].unsqueeze(0),
                faces=scene.faces, uvs=scene.uvs, uv_faces=scene.uv_faces)
    with torch.no_grad():
        gs = avatar.deform(mesh, frame["blend_weight"].unsqueeze(0))
        return rasterizer.render(gs, camera, bg)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import numpy as np

    from live3dgsavatar.config import load_config, resolve_model_ply

    cfg = load_config()
    subject = args.subject or str(cfg.get("subject") or "")
    work = args.work_name or str(cfg.get("work_name") or "")
    if not (subject and work):
        raise SystemExit("[error] 需要 --subject 与 --work-name")
    cfg.set("subject", subject)
    cfg.set("work_name", work)
    dataset = args.dataset or subject

    from live3dgsavatar.data.datasets import resolve_dataset_dir

    try:
        data_dir = resolve_dataset_dir(cfg, dataset)
        ply = resolve_model_ply(cfg)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"[error] {e}") from None

    defects_dir = Path(args.defects or (
        Path(str(cfg.get("paths.output_dir", "output"))) / "defects"
        / subject / work / dataset))
    selected_path = defects_dir / "selected.json"
    refined_dir = Path(args.refined_dir or (defects_dir / "refined"))
    if not selected_path.is_file():
        raise SystemExit(f"[error] 找不到 {selected_path}（先跑 find_defects.py）")
    if not refined_dir.is_dir():
        if not args.dry_run:
            raise SystemExit(f"[error] 找不到精修图目录 {refined_dir}\n"
                             "        用 --refined-dir 指向 SyncFix 的输出目录")
        print(f"  [warn] 精修图目录不存在（dry-run 继续）：{refined_dir}")

    selection = json.loads(selected_path.read_text(encoding="utf-8"))
    # 标签自带 (yaw, pitch, roll, 基帧)，所以**目录里每张精修图都能独立重建观测**。
    # 默认用全部能解析的图（用户可能修了 16 张，而 selected.json 只记了当次选中的）；
    # `--from-selection` 用于严格只取当次选择。
    available = sorted(p.stem for p in refined_dir.glob("*.png")
                       if p.is_file() and LABEL_RE.match(p.stem))
    if args.from_selection:
        chosen = [c for c in selection["views"]["chosen"] if c in set(available)]
    else:
        chosen = available
    if not chosen:
        raise SystemExit(f"[error] {refined_dir} 里没有可解析的精修图")

    with np.load(data_dir / "render_inputs.npz", allow_pickle=True) as d:
        total = int(d["mesh"].shape[0])
    ratio = float(cfg.get("train.pipeline.test_ratio", 0.1))
    held = max(1, int(round(total * ratio)))
    train_end = total - held

    print(f"模型    : {ply}")
    print(f"数据集  : {data_dir}（{dataset}，{total} 帧）")
    print(f"缺陷目录: {defects_dir}")
    print(f"精修视角: {len(chosen)} 个"
          + ("（按 selected.json 过滤）" if args.from_selection else "（目录内全部）"))
    print(f"微调    : {args.steps} 步，每批 {args.refined_per_batch} 精修 + "
          f"{args.real_per_batch} 真实")

    if args.dry_run:
        print("\n[dry-run] 未执行训练")
        return 0

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("[error] 微调需要 GPU")

    from PIL import Image

    from live3dgsavatar.core.camera import orbit_w2c, roll_w2c
    from live3dgsavatar.core.io import load_ply, save_ply
    from live3dgsavatar.core.io.ply import read_embedded_config
    from live3dgsavatar.core.metrics import psnr_masked
    from live3dgsavatar.core.render import SimpleRasterizer
    from live3dgsavatar.core.types import Camera
    from live3dgsavatar.data.scene import load_scene
    from live3dgsavatar.data.sources import Batch, ManifestSource
    from live3dgsavatar.evaluation import refine_weight_map
    from live3dgsavatar.training.objectives import Objective
    from live3dgsavatar.training.plans import ParameterPlan
    from live3dgsavatar.training.trainer import LogHook, Trainer

    torch.manual_seed(args.seed)
    device = "cuda"
    scene = load_scene(cfg, frames=train_end, frame_offset=0, data_dir=data_dir)
    avatar = load_ply(ply, config=read_embedded_config(ply), device=device)
    rasterizer = SimpleRasterizer()

    w2c_data = torch.eye(4, dtype=torch.float32)
    w2c_data[:3, :3] = scene.R.detach().cpu()
    w2c_data[:3, 3] = scene.T.detach().cpu()
    K_data = scene.K.detach().cpu()
    bg = torch.tensor([float(x) for x in cfg.get("render.background", [0, 0, 0])],
                      dtype=torch.float32, device=device)

    def camera_at(center, yaw, pitch, roll):
        view = orbit_w2c(w2c_data, center, yaw, pitch)
        if roll:
            view = roll_w2c(view, roll)
        return Camera(K=scene.K, w2c=view.unsqueeze(0).to(device),
                      width=scene.width, height=scene.height).to(device)

    # ---- 重建精修观测 ----
    neutral = int(np.argmin(np.linalg.norm(
        np.asarray([f["blend_weight"].detach().cpu().numpy()[:100]
                    for f in scene.frames]), axis=1)))
    items = []
    skipped = []
    for label in chosen:
        path = refined_dir / f"{label}.png"
        if not path.is_file():
            skipped.append(label)
            continue
        info = parse_label(label)
        base = (int(info["expr"][5:]) if info["expr"].startswith("frame")
                else neutral)
        base = min(max(base, 0), scene.num_frames - 1)
        frame = scene.frames[base]
        center = frame["mesh"].mean(dim=0).detach().cpu()
        cam = camera_at(center, info["yaw"], info["pitch"], info["roll"])
        out = render_observation(avatar, rasterizer, scene, frame, cam, bg)

        degraded = (out.color[0].clamp(0, 1).permute(1, 2, 0).detach()
                    .cpu().numpy() * 255.0).astype(np.float32)
        refined = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        if refined.shape != degraded.shape:
            raise SystemExit(
                f"[error] {label} 尺寸 {refined.shape} 与渲染 {degraded.shape} 不一致")
        # 前景必须取**精修图**的，不能用渲染的 alpha：
        # 缺陷区恰恰是 alpha 很低（空洞）的地方，用 alpha 当掩码会把要修的地方排除。
        fg = refined.max(axis=-1) > 8
        if not bool(fg.any()):
            print(f"  [warn] {label} 的精修图几乎全黑，跳过")
            skipped.append(label)
            continue

        wmap = refine_weight_map(degraded, refined, fg, floor=args.weight_floor)

        items.append({
            "mesh": frame["mesh"].detach().cpu(),
            "blend_weight": frame["blend_weight"].detach().cpu(),
            "image": torch.from_numpy(refined.transpose(2, 0, 1) / 255.0),
            "mask": torch.from_numpy(fg[None].astype(np.float32)),
            "weight_map": torch.from_numpy(wmap[None]),
            "K": K_data.clone(),
            "w2c": cam.w2c[0].detach().cpu(),
            "label": label,
        })
    if skipped:
        print(f"  [warn] {len(skipped)} 个候选没有精修图，跳过：{skipped[:3]}…")
    if not items:
        raise SystemExit("[error] 没有任何可用的精修视角")
    print(f"  重建观测 {len(items)} 个")

    real = ManifestSource(data_dir, split="train", test_ratio=ratio,
                          device="cpu", seed=args.seed)

    class MixedSource:
        """精修视角 + 真实帧锚定，各自带相机与权重图。"""

        def __init__(self) -> None:
            self._n = len(items) + len(real)
            self._items = items
            self._real = real

        def __len__(self) -> int:
            return self._n

        def random_batch(self, batch_size: int):
            import random

            n_ref = min(args.refined_per_batch, batch_size)
            n_real = max(0, batch_size - n_ref)
            refs = [random.choice(self._items) for _ in range(n_ref)]
            parts = []
            for item in refs:
                parts.append(Batch(
                    mesh=item["mesh"].unsqueeze(0).to(device, torch.float32),
                    blend_weight=item["blend_weight"].unsqueeze(0).to(device, torch.float32),
                    image=item["image"].unsqueeze(0).to(device, torch.float32),
                    mask=item["mask"].unsqueeze(0).to(device, torch.float32),
                    weight_map=item["weight_map"].unsqueeze(0).to(device, torch.float32),
                    K=item["K"].unsqueeze(0).to(device, torch.float32),
                    w2c=item["w2c"].unsqueeze(0).to(device, torch.float32)))
            if n_real > 0:
                b = self._real.random_batch(n_real)     # 数据源在 CPU 上
                mask = b.mask.to(device=device, dtype=torch.float32)
                parts.append(Batch(
                    mesh=b.mesh.to(device, torch.float32),
                    blend_weight=b.blend_weight.to(device, torch.float32),
                    image=b.image.to(device, torch.float32), mask=mask,
                    # 真实帧不打折：权重全 1，且**必须与其它张量同设备**
                    weight_map=torch.ones_like(mask, dtype=torch.float32),
                    K=K_data.unsqueeze(0).expand(n_real, 3, 3).to(device).clone(),
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

    source = MixedSource()

    from live3dgsavatar.core.types import Mesh

    def render_fn(mesh_verts, weight, batch, iteration):
        cams = [Camera(K=batch.K[i].to(device), w2c=batch.w2c[i].to(device),
                       width=scene.width, height=scene.height).to(device)
                for i in range(batch.K.shape[0])]
        outs = []
        for i, cam in enumerate(cams):
            mesh = Mesh(verts=mesh_verts[i:i + 1].to(device),
                        faces=scene.faces.to(device), uvs=scene.uvs.to(device),
                        uv_faces=scene.uv_faces.to(device))
            gs = avatar.deform(mesh, None if weight is None
                               else weight[i:i + 1].to(device))
            outs.append(rasterizer.render(gs, cam, bg))
        import torch as _t

        color = _t.cat([o.color for o in outs])
        alpha = _t.cat([o.alpha for o in outs])
        from types import SimpleNamespace

        return SimpleNamespace(color=color, alpha=alpha)

    # ---- 验收：留出真值 ----
    # 留出段只载入一次（每帧重新 load_scene 会反复读 npz，慢且没必要）
    hold_scene = load_scene(cfg, frames=held, frame_offset=train_end,
                            data_dir=data_dir)
    cam_hold = Camera(K=hold_scene.K, w2c=w2c_data.unsqueeze(0).to(device),
                      width=hold_scene.width, height=hold_scene.height).to(device)

    def evaluate_holdout() -> dict:
        n_eval = hold_scene.num_frames if args.eval_frames < 0 else min(
            args.eval_frames, hold_scene.num_frames)
        idx = np.linspace(0, hold_scene.num_frames - 1, n_eval).astype(int)
        vals = []
        for local in idx:
            dataset_idx = train_end + int(local)
            path = data_dir / "images" / f"{dataset_idx:05d}.png"
            if not path.is_file():
                continue
            rgba = np.asarray(Image.open(path).convert("RGBA"))
            gt_rgb = rgba[..., :3].copy()
            gt_rgb[rgba[..., 3] <= 127] = 0
            mask = rgba[..., 3] > 127
            out = render_observation(avatar, rasterizer, hold_scene,
                                     hold_scene.frames[int(local)], cam_hold, bg)
            pred = (out.color[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                    * 255.0).astype(np.uint8)
            v = psnr_masked(pred, gt_rgb, mask)
            if v is not None:
                vals.append(v)
        if not vals:
            return {"psnr_holdout_mean": float("nan"), "n": 0}
        arr = np.asarray(vals, dtype=np.float64)
        return {"psnr_holdout_mean": float(arr.mean()),
                "psnr_holdout_median": float(np.median(arr)),
                "psnr_holdout_min": float(arr.min()),
                "n": int(arr.size)}

    def evaluate_refined(avatar_) -> dict:
        """把全部精修视角重新渲染一遍，与精修图逐帧比 PSNR。

        这是"对训练目标做了多少"的直接度量：它一定涨（我们就是在拟合这些图），
        所以它的用处不是证明好坏，而是**判断收敛程度**与**找出没fit上的视角**。
        真正的质量判据仍然是留出真值。
        """
        per: dict[str, float] = {}
        for item in items:
            frame = {"mesh": item["mesh"], "blend_weight": item["blend_weight"]}
            cam = Camera(K=item["K"].unsqueeze(0).to(device),
                         w2c=item["w2c"].unsqueeze(0).to(device),
                         width=scene.width, height=scene.height).to(device)
            out = render_observation(avatar_, rasterizer, scene, frame, cam, bg)
            pred = (out.color[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                    * 255.0).astype(np.uint8)
            target = (item["image"].permute(1, 2, 0).numpy() * 255.0
                      ).astype(np.uint8)
            mask = item["mask"][0].numpy() > 0.5
            v = psnr_masked(pred, target, mask)
            if v is not None:
                per[item["label"]] = float(v)
        if not per:
            return {"n": 0}
        arr = np.asarray(list(per.values()), dtype=np.float64)
        return {"per_view": {k: round(v, 2) for k, v in per.items()},
                "mean": float(arr.mean()), "median": float(np.median(arr)),
                "min": float(arr.min()), "n": int(arr.size)}

    before = evaluate_holdout()
    refined_before = evaluate_refined(avatar)
    print(f"  微调前留出 PSNR：{before['psnr_holdout_mean']:.2f} dB（n={before['n']}）")

    if args.params == "defect":
        plan = ParameterPlan.defect_finetune(
            lr_color=args.lr_color, lr_opacity=args.lr_opacity,
            lr_scaling=args.lr_scaling, max_steps=args.steps)
    elif args.params == "color+opacity":
        plan = ParameterPlan.appearance_only(lr_color=args.lr_color,
                                             lr_opacity=args.lr_opacity,
                                             with_opacity=True)
    else:
        plan = ParameterPlan.appearance_only(lr_color=args.lr_color,
                                             with_opacity=False)
    print(f"  更新参数: {sorted(plan.enabled)}")
    objective = Objective(l1=1.0, mask_weight=0.0)
    trainer = Trainer(
        source=source, plan=plan, objective=objective, render_fn=render_fn,
        params=avatar.parameter_groups(), iterations=int(args.steps),
        batch_size=int(args.refined_per_batch + args.real_per_batch),
        blend_start=0, hooks=[LogHook(every=max(1, args.steps // 10))],
        device=device)
    report = trainer.run()

    after = evaluate_holdout()
    refined_after = evaluate_refined(avatar)
    out_dir = defects_dir / "finetune"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_ply(avatar, out_dir / "model.ply")
    payload = {
        "subject": subject, "work_name": work, "dataset": dataset,
        "refined_views": len(items), "skipped": skipped,
        "steps": int(args.steps), "seconds": round(report.seconds, 1),
        "holdout_before": before, "holdout_after": after,
        "refined_before": refined_before, "refined_after": refined_after,
        "delta_psnr": round(after["psnr_holdout_mean"]
                            - before["psnr_holdout_mean"], 3),
        "last_parts": report.last_parts,
    }
    (out_dir / "report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n微调完成：{report.iterations} 步，{report.seconds:.1f} s")
    print(f"  留出 PSNR：{before['psnr_holdout_mean']:.2f} → "
          f"{after['psnr_holdout_mean']:.2f} dB（{payload['delta_psnr']:+.2f}）")
    if refined_after.get("n"):
        print(f"  精修图 PSNR：{refined_before['mean']:.2f} → "
              f"{refined_after['mean']:.2f} dB"
              f"（{refined_after['mean'] - refined_before['mean']:+.2f}，"
              f"n={refined_after['n']}）")
        # 哪个视角没拟合上：从最差开始列
        worst = sorted(refined_after["per_view"].items(), key=lambda kv: kv[1])[:3]
        print("    拟合最差的 3 个视角：" +
              "，".join(f"{k.split('_roll')[0]} {v:.1f} dB" for k, v in worst))
    print(f"  产物：{out_dir}/model.ply 与 report.json")
    if payload["delta_psnr"] < -0.1:
        print("  [warn] 留出集变差了 —— 建议降低 --lr-* 或减少步数后重跑")

    if args.promote:
        from live3dgsavatar.data.models import models_root, promote_model

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
        print(f"  全量评测：python scripts/render_test.py --subject {subject} "
              f"--work-name {name} --dataset {dataset} --frames -1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
