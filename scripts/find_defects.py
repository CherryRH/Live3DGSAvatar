#!/usr/bin/env python
"""找瑕疵 + 选视角：把评估层落成一条可执行的 GPU 流程。

## 它做什么

1. **覆盖统计**：在训练段的若干真实帧上渲染**面 id 图**，累计每个三角面被看到
   多少次 —— 由此得到每个高斯的"观测次数"，从未被观测的就是缺陷候选；
2. **候选评估**：在"视角网格 × 表情候选"上渲染，得到每个候选**可见**的高斯集合，
   同时记录渲染侧的缺陷（alpha 孔洞、飞点、锐度）；
3. **贪心减选**：用缺陷质量做集合覆盖，选出 `--budget` 个 (视角, 表情) 组合；
4. **落盘**：`selected.json`（选中的组合与边际收益）、`mass.npz`（逐高斯缺陷质量、
   覆盖次数，后续微调直接用）、`views/`（选中组合的渲染图，肉眼确认）。

精修不在这里做 —— 它贵且需要外部模型，本脚本的产出直接喂给精修与微调。

## 用法

```bash
python scripts/find_defects.py --subject crh_1 --work-name run --dataset crh_1
```

产物在 `output/defects/<对象>/<版本>/<数据集>/`。

⚠️ 需要 GPU（nvdiffrast）。`--dry-run` 可离线检查数据与网格规模。
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
    p = argparse.ArgumentParser(description="定位缺陷并挑选待精修视角")
    p.add_argument("--subject", default=None)
    p.add_argument("--work-name", default=None)
    p.add_argument("--dataset", default=None, help="评估哪个数据集（默认同 subject）")
    p.add_argument("--coverage-frames", type=int, default=80,
                   help="用多少帧真实数据统计覆盖")
    p.add_argument("--expr-count", type=int, default=None,
                   help="表情候选数；缺省 聚类 + 主轴极值")
    p.add_argument("--budget", type=int, default=16, help="选多少个待精修组合")
    p.add_argument("--min-gain-ratio", type=float, default=0.002,
                   help="单次新增低于总质量的这个比例就停（太小会把预算花在长尾）")
    p.add_argument("--rolls", type=float, nargs="+", default=None,
                   help="滚转角（默认只取 0）")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--no-save-images", action="store_true",
                   help="只算指标，不落渲染图")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


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

    out_dir = args.out or (Path(str(cfg.get("paths.output_dir", "output")))
                           / "defects" / subject / work / dataset)

    # ---- 候选（纯 CPU，先算出来打印）----
    from live3dgsavatar.evaluation import (axis_extremes, candidate_views,
                                           cluster_frames, expression_matrix,
                                           merge_specs)

    rolls = tuple(args.rolls) if args.rolls else (0.0,)
    views = candidate_views(rolls=rolls)

    with np.load(data_dir / "render_inputs.npz", allow_pickle=True) as d:
        weights = np.asarray(d["blend_weight"], dtype=np.float32)
        total = int(weights.shape[0])
    X = expression_matrix(weights)

    ratio = float(cfg.get("train.pipeline.test_ratio", 0.1))
    held = max(1, int(round(total * ratio)))
    train_end = total - held                     # 覆盖统计与表情候选只用训练段

    # 聚类取**真实帧**（姿态与表情自洽，且能回查原图）；主轴极值兜底没做过的方向
    specs = merge_specs(cluster_frames(X[:train_end],
                                       k=min(8, max(4, train_end // 200))),
                        axis_extremes(X[:train_end], k=6))
    if args.expr_count:
        specs = specs[: int(args.expr_count)]

    print(f"模型   : {ply}")
    print(f"数据集 : {data_dir}（{dataset}，{total} 帧）")
    print(f"训练段 : [0, {train_end - 1}]  留出段 {held} 帧")
    print(f"候选   : {len(views)} 视角 × {len(specs)} 表情 = "
          f"{len(views) * len(specs)} 组")
    print(f"预算   : {args.budget}")
    print(f"产物   : {out_dir}")

    if args.dry_run:
        print("\n[dry-run] 未执行渲染")
        return 0

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("[error] 需要 GPU（nvdiffrast）；--dry-run 可离线检查")

    import nvdiffrast.torch as dr

    from live3dgsavatar.core.camera import orbit_w2c
    from live3dgsavatar.core.io import load_ply
    from live3dgsavatar.core.io.ply import read_embedded_config
    from live3dgsavatar.core.render import SimpleRasterizer
    from live3dgsavatar.core.render.faceid import face_id_map
    from live3dgsavatar.core.types import Camera, Mesh
    from live3dgsavatar.data.scene import load_scene
    from live3dgsavatar.evaluation import (Candidate, alpha_hole_ratio,
                                           build_defect_map, coverage_counts,
                                           coverage_defect, floater_offsets,
                                           greedy_cover, hole_rate)

    torch.manual_seed(args.seed)
    device = "cuda"
    glctx = dr.RasterizeCudaContext()

    # 载入整个训练段（只读，约 150 MB），再均匀抽样做覆盖统计。
    # ⚠️ 不能只 load 前 n 帧再按全局下标取 —— 那样抽到的还是开头那一段。
    n_cov = max(1, min(int(args.coverage_frames), train_end))
    idx_cov = np.linspace(0, train_end - 1, n_cov).astype(int).tolist()
    scene = load_scene(cfg, frames=train_end, frame_offset=0, data_dir=data_dir)
    frames_cov = [scene.frames[i] for i in idx_cov]
    avatar = load_ply(ply, config=read_embedded_config(ply), device=device)
    rasterizer = SimpleRasterizer()

    w2c = torch.eye(4, dtype=torch.float32)
    w2c[:3, :3] = scene.R.detach().cpu()
    w2c[:3, 3] = scene.T.detach().cpu()
    K_cpu = scene.K.detach().cpu()
    bg = torch.tensor([float(x) for x in cfg.get("render.background", [0, 0, 0])],
                      dtype=torch.float32, device=device)

    face_id_binding = avatar.binding_face_id.detach().cpu().numpy()
    n_faces = int(face_id_binding.max()) + 1
    n_gauss = int(face_id_binding.size)
    print(f"  三角面 {n_faces}   高斯 {n_gauss}")

    # 从 .ply 载入的 Binding 只保留有效 texel（掩码退化成全 True），
    # 因此这里重做一次 UV 光栅化，恢复"高斯 → 展平 texel 下标"的映射。
    from live3dgsavatar.core.deform.binding import compute_uv_rast_info

    _, uv_face_id = compute_uv_rast_info(
        scene.uvs.to(device), scene.uv_faces.to(device),
        (int(avatar.config.tex_size), int(avatar.config.tex_size)), glctx)
    texel_index = uv_face_id.reshape(-1).gt(0).nonzero(as_tuple=True)[0].cpu().numpy()
    if texel_index.size != n_gauss:
        raise SystemExit(
            f"[error] UV 光栅化得到 {texel_index.size} 个有效 texel，"
            f"与模型的 {n_gauss} 个高斯不一致 —— 模型与数据集的 tex_size/UV 布局不匹配")

    faces = scene.faces.to(torch.int32)

    def mesh_of(frame):
        return frame["mesh"].unsqueeze(0).to(device)

    def camera_at(center, yaw, pitch, roll):
        view = orbit_w2c(w2c, center, yaw, pitch)
        if roll:
            from live3dgsavatar.core.camera import roll_w2c
            view = roll_w2c(view, roll)
        return Camera(K=scene.K, w2c=view.unsqueeze(0).to(device),
                      width=scene.width, height=scene.height).to(device)

    # ---- 1) 覆盖统计 ----
    print(f"\n[1/3] 覆盖统计（{len(frames_cov)} 帧）…", flush=True)
    fids = []
    for j, frame in enumerate(frames_cov):
        verts = mesh_of(frame)
        cam = Camera(K=scene.K, w2c=w2c.unsqueeze(0).to(device),
                     width=scene.width, height=scene.height).to(device)
        fids.append(face_id_map(verts, faces, cam, glctx).detach().cpu().numpy())
        if (j + 1) % 20 == 0:
            print(f"  {j + 1}/{len(frames_cov)}", flush=True)
    cov_face = coverage_counts(fids, n_faces)
    cov_gauss = cov_face[face_id_binding]                 # 每个高斯被看到几次

    # ---- 2) 候选评估 ----
    # 每个表情候选都有"自己那一帧"作为基帧：姿态/表情/配件天然自洽。
    # 主轴极值不是真实帧，只能套到中性帧上（少量候选，可接受）。
    neutral_i = int(np.argmin(np.linalg.norm(X[:train_end], axis=1)))
    print(f"\n[2/3] 候选评估（中性基帧 {neutral_i}）…", flush=True)

    def frame_for(spec):
        """表情候选对应的基帧：`frameXXXXX` 用真实帧，其余用中性帧。"""
        if spec.name.startswith("frame"):
            idx = int(spec.name[5:])
            return idx, scene.frames[idx]
        return neutral_i, scene.frames[neutral_i]

    def mesh_input(verts):
        return Mesh(verts=verts, faces=scene.faces.to(device),
                    uvs=scene.uvs.to(device), uv_faces=scene.uv_faces.to(device))

    candidates = []
    hole_face_hits = np.zeros(n_faces, dtype=np.float64)
    hole_face_seen = np.zeros(n_faces, dtype=np.float64)
    floater_sum = np.zeros(n_gauss, dtype=np.float64)
    floater_n = 0
    for vi, spec_view in enumerate(views):
        for si, expr in enumerate(specs):
            base_idx, base_frame = frame_for(expr)
            base_verts = mesh_of(base_frame)
            center_i = base_verts[0].mean(dim=0).detach().cpu()
            cam = camera_at(center_i, spec_view.yaw, spec_view.pitch, spec_view.roll)
            weight = base_frame["blend_weight"].unsqueeze(0).to(device)
            with torch.no_grad():
                gs = avatar.deform(mesh_input(base_verts), weight)
                out = rasterizer.render(gs, cam, bg)
                ids = face_id_map(base_verts, faces, cam, glctx)
                # 飞点：变形后的高斯应贴在**该帧**的表面附近
                offs = floater_offsets(gs.xyz[0].detach(), base_verts[0],
                                       scene.faces.to(device),
                                       avatar.binding_face_id,
                                       avatar.binding_face_bary)
            floater_sum += offs.detach().cpu().numpy().astype(np.float64)
            floater_n += 1
            ids_np = ids.detach().cpu().numpy()
            alpha = out.alpha[0, 0].detach().cpu().numpy()
            image = (out.color[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                     * 255.0).astype(np.uint8)
            # 孔洞：累计"该面被看到几次 / 其中有几次漏了"，最后算命中率
            holes = alpha_hole_ratio(alpha, ids_np, n_faces)
            hole_face_hits += (holes > 0.5).astype(np.float64)
            hole_face_seen += (np.bincount(ids_np[ids_np >= 0].reshape(-1),
                                           minlength=n_faces) > 0).astype(np.float64)

            visible_faces = np.unique(ids_np[ids_np >= 0])
            covers = np.flatnonzero(np.isin(face_id_binding, visible_faces))
            candidates.append(Candidate(
                label=f"{spec_view.label}_{expr.name}", covers=covers,
                payload={"view": spec_view.to_dict(), "expr": expr.name,
                         "base_frame": base_idx}))
        if (vi + 1) % 10 == 0:
            print(f"  {vi + 1}/{len(views)} 个视角", flush=True)

    # ---- 3) 缺陷质量与选择 ----
    print("\n[3/3] 缺陷质量与贪心选择 …", flush=True)
    offs_gauss = floater_sum / max(1, floater_n)
    hit_rate = hole_rate(hole_face_hits, hole_face_seen)   # 逐面命中率，先算

    # 三个信号都归到**高斯**上：
    #   覆盖缺失（饱和到 50 次，见 coverage_defect 的说明）+ 孔洞命中率
    cov_norm = coverage_defect(cov_gauss, saturate=50.0)
    hole_norm = np.clip(hit_rate[face_id_binding], 0.0, 1.0)
    med_off = float(np.median(offs_gauss)) if offs_gauss.size else 0.0
    flo_norm = np.clip((offs_gauss - med_off) / (med_off + 1e-6), 0.0, 1.0)
    mass = np.asarray(0.6 * cov_norm + 0.35 * hole_norm + 0.05 * flo_norm,
                      dtype=np.float64)

    # 候选池**能触达**的质量上限：有些缺陷（后脑、口腔内侧）在 ±40° 网格里
    # 根本看不到，再多的候选也覆盖不了 —— 这部分不是选择器的问题。
    reachable = np.zeros(mass.size, dtype=bool)
    for cand in candidates:
        reachable[cand.covers] = True
    reachable_mass = float(mass[reachable].sum())

    result = greedy_cover(candidates, mass, budget=int(args.budget),
                          min_gain_ratio=float(args.min_gain_ratio))

    # 逐面汇总：覆盖次数、孔洞、飞点（都用同一套面索引，便于横向比较）
    off_face = np.zeros(n_faces, dtype=np.float64)
    np.add.at(off_face, face_id_binding, offs_gauss)
    counts = np.bincount(face_id_binding, minlength=n_faces).astype(np.float64)
    nz = counts > 0
    off_face[nz] /= counts[nz]
    mass_summary = {
        "sum": round(float(mass.sum()), 1),
        "mean": round(float(mass.mean()), 4),
        "nonzero_share": round(float((mass > 1e-6).mean()), 4),
        "strong_share": round(float((mass > 0.3).mean()), 4),
    }
    defect = build_defect_map(
        per_face={"coverage": cov_face.astype(np.float64),
                  "hole_rate": hit_rate,
                  "floater": off_face},
        texel_face_id=avatar.binding_face_id,
        tex_size=int(avatar.config.tex_size),
        texel_index=texel_index,
        # 总览分用"平均覆盖不足"，而不是 1-mean/max（后者对长尾分布没有意义）
        score=float(cov_norm.mean()))

    # ---- 落盘 ----
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mass.npz").parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "mass.npz",
                        mass=mass.astype(np.float32),
                        coverage=cov_gauss.astype(np.int32),
                        floater=offs_gauss.astype(np.float32))
    payload = {
        "subject": subject, "work_name": work, "dataset": dataset,
        "model": str(ply), "budget": int(args.budget),
        "candidate_count": len(candidates),
        "views": result.to_dict(),
        "mass": mass_summary,
        "reachable_mass": round(reachable_mass, 1),
        "reachable_ratio": round(reachable_mass / max(1e-9, float(mass.sum())), 4),
        "defect": defect.summary(),
        "coverage": {"zero_share": round(float((cov_gauss == 0).mean()), 4),
                     "below_saturate_share": round(float((cov_gauss < 50).mean()), 4),
                     "median": float(np.median(cov_gauss))},
    }
    (out_dir / "selected.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if not args.no_save_images:
        vis_dir = out_dir / "views"
        vis_dir.mkdir(exist_ok=True)
        refs_dir = out_dir / "refs"
        for cand in result.chosen:
            spec_view = cand.payload["view"]
            base_verts = mesh_of(scene.frames[int(cand.payload["base_frame"])])
            center_i = base_verts[0].mean(dim=0).detach().cpu()
            cam = camera_at(center_i, spec_view["yaw"], spec_view["pitch"],
                            spec_view["roll"])
            weight = scene.frames[int(cand.payload["base_frame"])][
                "blend_weight"].unsqueeze(0).to(device)
            with torch.no_grad():
                gs = avatar.deform(mesh_input(base_verts), weight)
                out = rasterizer.render(gs, cam, bg)
            img = (out.color[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                   * 255.0).astype(np.uint8)
            from PIL import Image
            Image.fromarray(img).save(vis_dir / f"{cand.label}.png")

            # 顺带把该候选的基帧（真实图）存成参考图：精修时正好用它做 --ref_image
            base_idx = int(cand.payload["base_frame"])
            src_path = data_dir / "images" / f"{base_idx:05d}.png"
            if src_path.is_file():
                refs_dir.mkdir(exist_ok=True)
                dst = refs_dir / f"{base_idx:05d}.png"
                if not dst.exists():
                    rgba = np.asarray(Image.open(src_path).convert("RGBA"))
                    rgb = rgba[..., :3].copy()
                    rgb[rgba[..., 3] <= 127] = 0        # 与渲染同底（黑）
                    Image.fromarray(rgb).save(dst)

    print(f"\n候选可触达质量 {reachable_mass:.0f}/{mass.sum():.0f}"
          f"（{reachable_mass / max(1e-9, mass.sum()):.1%}）")
    print(f"选中 {len(result.chosen)} 组，覆盖缺陷质量 {result.covered_ratio:.1%}")
    for cand, gain in zip(result.chosen, result.gains):
        print(f"  {cand.label:40s} +{gain:.1f}")
    print(f"\n产物：{out_dir}")
    print(f"  selected.json  选中的组合与收益")
    print(f"  mass.npz       逐高斯缺陷质量（后续微调权重）")
    if not args.no_save_images:
        print(f"  views/         选中组合的渲染图（可直接作 SyncFix 的 --input_image）")
        print(f"  refs/          各候选基帧的真实图（作 --ref_image）")
        print("\n下一轮精修（把 degraded 与 refs 一起喂给 SyncFix）：")
        print(f"  python ~/Projects/SyncFix/inference.py \\\n"
              f"    --input_image {out_dir}/views \\\n"
              f"    --ref_image   {out_dir}/refs \\\n"
              f"    --model_path  <SyncFix 权重> --output_dir {out_dir}/refined \\\n"
              f"    --n_per_pass {len(result.chosen)} --ref_size 1 "
              f"--height 512 --width 512")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
