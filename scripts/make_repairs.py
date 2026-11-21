#!/usr/bin/env python
"""一次性生成修复图：按固定 yaw 渲染当前模型 -> SyncFix -> unsharp -> 输出目录。

按固定间隔（如 5°）逐个视角渲染、修复，供加密管线做**初始化**监督。
视角是"头转"（`tracking.pose_drive`），不是绕头心转相机。

产物（syncfix_label 需要的 defects 目录结构）::

    <out>/views/yaw+45_pitch+0_roll+0_frame02104.png   # 待修复的渲染图
    <out>/refs/*.png                                    # 身份参考图（真实帧）
    <out>/selected.json                                 # 本次视角清单
    <out>/syncfix/*.png                                 # SyncFix 输出
    --out-dir/*.png                                     # unsharp 后的最终标签
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import live3dgsavatar  # noqa: F401,E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="按固定头姿渲染 + SyncFix 生成修复图")
    p.add_argument("--subject", default=None)
    p.add_argument("--work-name", default=None,
                   help="要渲染的模型版本名（models/<subject>/<work-name>）")
    p.add_argument("--model", type=Path, default=None,
                   help="直接给 model.ply；给定时忽略 --work-name")
    p.add_argument("--dataset", default=None)
    p.add_argument("--yaws", required=True,
                   help="逗号分隔的相对 yaw（度），如 45,-45；只做 pitch 时给 0")
    p.add_argument("--pitches", default="0",
                   help="逗号分隔的相对 pitch（度），如 -30,...,30；与 yaws 做叉积")
    p.add_argument("--base-frame", default="auto",
                   help="基帧：auto=训练段里 |yaw| 最小的帧；或给帧号")
    p.add_argument("--refs", type=Path, default=None,
                   help="参考图目录（真实帧）；缺省用 datasets/<obj>/repair/refs")
    p.add_argument("--out", type=Path, required=True,
                   help="中间目录（views/ refs/ selected.json syncfix/）")
    p.add_argument("--repair-root", type=Path, default=None,
                   help="同时恢复数据集的 repair/ 结构：views/ refs/ masks/ repairs/ + manifest")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="最终修复图目录（unsharp 后）；缺省 <out>/final")
    p.add_argument("--syncfix-repo", type=Path,
                   default=Path("/home/test/Projects/SyncFix"))
    p.add_argument("--syncfix-python", default="syncfix")
    p.add_argument("--syncfix-model", type=Path,
                   default=Path("/home/test/Projects/SyncFix/models/paper"))
    p.add_argument("--ref-size", type=int, default=3)
    # SyncFix 的 inference 会先把 refs 截断到 ref_size，再按 batch 切片：
    # 如果一次喂 N 张、n_per_pass < N，只有第一个 batch 有参考图。所以这里
    # 按 batch 分组、每组单独调用一次，保证每组都带参考。
    p.add_argument("--syncfix-batch", type=int, default=4)
    p.add_argument("--sharpen", type=float, default=0.35)
    p.add_argument("--no-matte", action="store_true",
                   help="跳过 BiRefNet 抠图（默认做，输出 RGBA）")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    import torch
    from PIL import Image

    from live3dgsavatar.config import load_config, resolve_model_ply
    from live3dgsavatar.core.avatar import AvatarConfig
    from live3dgsavatar.core.camera import orbit_w2c_matrix, rotation_y
    from live3dgsavatar.core.io import load_ply
    from live3dgsavatar.core.render import SimpleRasterizer, render_frame
    from live3dgsavatar.core.types import Camera, Mesh
    from live3dgsavatar.data.datasets import resolve_dataset_dir
    from live3dgsavatar.data.pose import head_yaw, rot6d_to_matrix
    from live3dgsavatar.data.scene import load_scene

    cfg = load_config()
    subject = args.subject or str(cfg.get("subject") or "")
    if not subject:
        raise SystemExit("[error] 需要 --subject 或配置里的 subject")
    cfg.set("subject", subject)
    work = args.work_name or str(cfg.get("work_name") or "")
    if work:
        cfg.set("work_name", work)
    data_dir = resolve_dataset_dir(cfg, args.dataset or subject)
    ply = args.model.expanduser() if args.model is not None else resolve_model_ply(cfg)
    if not Path(ply).is_file():
        raise SystemExit(f"[error] 找不到模型：{ply}")

    yaws = [float(v) for v in str(args.yaws).split(",") if str(v).strip()]
    pitches = [float(v) for v in str(args.pitches).split(",") if str(v).strip()]
    if not yaws:
        raise SystemExit("[error] --yaws 为空")

    with np.load(data_dir / "render_inputs.npz", allow_pickle=True) as d:
        meshes = torch.from_numpy(d["mesh"].copy()).float()
        weights = torch.from_numpy(d["blend_weight"].copy()).float()
    total = int(meshes.shape[0])

    yaw_all = np.abs(head_yaw(weights.numpy()))
    held = max(1, int(round(total * float(cfg.get("train.pipeline.test_ratio", 0.3)))))
    train_end = total - held
    if str(args.base_frame).lower() == "auto":
        base_idx = int(np.argmin(yaw_all[:train_end]))
    else:
        base_idx = int(args.base_frame)
    print(f"  数据集 : {data_dir}（{total} 帧，训练段 {train_end}）")
    print(f"  模型   : {ply}")
    print(f"  基帧   : {base_idx:05d}（|yaw| {yaw_all[base_idx]:.1f}°）")

    device = "cuda"
    if not torch.cuda.is_available():
        raise SystemExit("[error] 需要 GPU（nvdiffrast）")
    scene = load_scene(cfg, frames=1, data_dir=data_dir)
    avatar = load_ply(Path(ply),
                      config=AvatarConfig.from_yaml(Path(ply).parent / "config.yaml"),
                      device=device)
    rasterizer = SimpleRasterizer()

    # 头转（pose 驱动）：语义与 GT 一致；没有 RDF 跟踪结果时退回相机绕转
    cam_fixed = Camera.from_intrinsics_extrinsics(
        K=scene.K, R=scene.R, T=scene.T,
        width=scene.width, height=scene.height).to(device)
    try:
        from live3dgsavatar.tracking.pose_drive import PoseDriver
        driver = PoseDriver(data_dir, device=device)
        print("  视角   : FLAME 头姿驱动（头转、胸不动）")
    except Exception as exc:
        driver = None
        print(f"  视角   : 相机绕转（pose 驱动不可用：{exc}）")

    w2c = torch.eye(4, dtype=torch.float32)
    w2c[:3, :3] = torch.as_tensor(scene.R).detach().cpu()
    w2c[:3, 3] = torch.as_tensor(scene.T).detach().cpu()
    base_verts = meshes[base_idx:base_idx + 1].to(device)
    weight0 = weights[base_idx:base_idx + 1].to(device)
    center = base_verts[0].mean(dim=0).detach().cpu()
    head_R = torch.from_numpy(
        rot6d_to_matrix(weights[base_idx:base_idx + 1, 100:106].numpy())[0]).float()

    out = args.out
    views_dir = out / "views"
    refs_dir = out / "refs"
    if not args.dry_run:
        views_dir.mkdir(parents=True, exist_ok=True)
        refs_dir.mkdir(parents=True, exist_ok=True)
        for stale in views_dir.glob("*.png"):
            stale.unlink()

    bg = torch.zeros(1, 3, 1, 1, device=device)
    labels: list[str] = []
    # 只做"十字"：yaw 扇 + pitch 扇，而不是 yaw×pitch 全网格
    # （全网格会让 22×12 张里大部分是重复的斜向角，SyncFix 成本翻倍）。
    views = [(y, 0.0) for y in yaws]
    views += [(0.0, p) for p in pitches if abs(p) > 1e-9]
    for yaw, pitch in views:
        label = f"yaw{yaw:+.0f}_pitch{pitch:+.0f}_roll+0_frame{base_idx:05d}"
        labels.append(label)
        if args.dry_run:
            print(f"  [dry-run] {label}")
            continue
        if driver is not None:
            turned = driver.turn(base_idx, yaw, pitch)
            cam = cam_fixed
            verts = turned.mesh.unsqueeze(0)
            weight = turned.weight.unsqueeze(0)
        else:
            R_target = rotation_y(yaw)[:3, :3]
            view = orbit_w2c_matrix(w2c, center, R_target @ head_R.T)
            cam = Camera(K=scene.K, w2c=view.unsqueeze(0).to(device),
                         width=scene.width, height=scene.height).to(device)
            verts, weight = base_verts, weight0
        mesh = Mesh(verts=verts, faces=scene.faces.to(device),
                    uvs=scene.uvs.to(device), uv_faces=scene.uv_faces.to(device))
        with torch.no_grad():
            res = render_frame(avatar, rasterizer, mesh, weight, cam, bg)
        img = (res.color[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0
               ).astype(np.uint8)
        Image.fromarray(img).save(views_dir / f"{label}.png")
        print(f"  {label}  (yaw {yaw:+.1f}°, pitch {pitch:+.1f}°)")

    # 参考图：优先用真实的正面/侧面照；没有就退回基帧原图
    if not args.dry_run:
        refs_src = args.refs or (data_dir / "repair" / "refs")
        copied = 0
        if Path(refs_src).is_dir():
            for p in sorted(Path(refs_src).glob("*.png")):
                shutil.copyfile(p, refs_dir / p.name)
                copied += 1
        if copied == 0:
            raw = np.asarray(Image.open(data_dir / "images" / f"{base_idx:05d}.png")
                             .convert("RGBA"))
            rgb = raw[..., :3].copy()
            rgb[raw[..., 3] <= 127] = 0
            Image.fromarray(rgb).save(refs_dir / f"{base_idx:05d}.png")
            copied = 1
        (out / "selected.json").write_text(json.dumps({
            "subject": subject, "model": str(ply), "base_frame": base_idx,
            "yaws": yaws, "views": labels, "refs": copied,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  参考图 : {copied} 张 -> {refs_dir}")
        print(f"  输出   : {views_dir}")

        # ---- SyncFix 修复：按 batch 分组，每组都带参考图 ----
        import subprocess
        fixes = []
        group = max(1, int(args.syncfix_batch))
        labels = sorted(p.stem for p in views_dir.glob("*.png"))
        for gi in range(0, len(labels), group):
            grp = labels[gi:gi + group]
            gdir = out / f"_g{gi // group}"
            (gdir / "views").mkdir(parents=True, exist_ok=True)
            (gdir / "refs").mkdir(parents=True, exist_ok=True)
            for lab in grp:
                shutil.copyfile(views_dir / f"{lab}.png",
                                gdir / "views" / f"{lab}.png")
            for ref in refs_dir.glob("*.png"):
                shutil.copyfile(ref, gdir / "refs" / ref.name)
            subprocess.run([
                sys.executable, str(REPO_ROOT / "scripts" / "syncfix_label.py"),
                "--defects", str(gdir), "--out", str(gdir / "fix"),
                "--repo", str(args.syncfix_repo),
                "--python", str(args.syncfix_python),
                "--model", str(args.syncfix_model),
                "--ref-size", str(args.ref_size),
                "--n-per-pass", str(min(group, 4)),
            ], check=True)
            fixes.extend(sorted((gdir / "fix").glob("*.png")))
        if not fixes:
            raise SystemExit("[error] SyncFix 没有产出图片")

        # ---- 抠图：SyncFix 输出是黑底 RGB，转成 RGBA（alpha=人像）----
        if args.no_matte:
            rgb_dir = out / "syncfix"
            rgb_dir.mkdir(parents=True, exist_ok=True)
            for p in fixes:
                shutil.copyfile(p, rgb_dir / p.name)
            final_src = rgb_dir
            alpha_dir = None
        else:
            rgb_dir = out / "syncfix"
            rgb_dir.mkdir(parents=True, exist_ok=True)
            for p in fixes:
                shutil.copyfile(p, rgb_dir / p.name)
            alpha_dir = out / "matte"
            subprocess.run([
                sys.executable, str(REPO_ROOT / "scripts" / "matte_birefnet.py"),
                "--input", str(rgb_dir), "--out", str(alpha_dir),
                "--force",
            ], check=True)
            final_src = alpha_dir

        # ---- unsharp + 输出到 --out-dir（有 alpha 就保留）----
        final = args.out_dir or (out / "final")
        final.mkdir(parents=True, exist_ok=True)
        from PIL import ImageFilter
        n = 0
        for p in sorted(final_src.glob("*.png")):
            im = Image.open(p)
            alpha = im.getchannel("A") if im.mode == "RGBA" else None
            rgb = im.convert("RGB")
            if args.sharpen > 0:
                arr = np.asarray(rgb, np.float32)
                blur = np.asarray(rgb.filter(ImageFilter.GaussianBlur(radius=1.0)),
                                  np.float32)
                rgb = Image.fromarray(np.clip(arr + args.sharpen * (arr - blur),
                                              0, 255).astype(np.uint8))
            if alpha is not None:
                out_im = rgb.convert("RGBA")
                out_im.putalpha(alpha)
                # alpha 外的 RGB 置 0，与数据集约定一致
                a = np.array(out_im, dtype=np.uint8, copy=True)
                a[a[..., 3] <= 127, :3] = 0
                out_im = Image.fromarray(a)
            else:
                out_im = rgb
            out_im.save(final / p.name)
            n += 1
        print(f"  修复图 : {n} 张（matte={'off' if args.no_matte else 'on'}）-> {final}")

        if args.repair_root is not None:
            root = Path(args.repair_root)
            for sub in ("views", "refs", "masks", "repairs"):
                (root / sub).mkdir(parents=True, exist_ok=True)
            for p in views_dir.glob("*.png"):
                shutil.copyfile(p, root / "views" / p.name)
            for p in refs_dir.glob("*.png"):
                shutil.copyfile(p, root / "refs" / p.name)
            if alpha_dir is not None:
                for p in alpha_dir.glob("*.png"):
                    im = Image.open(p)
                    if im.mode == "RGBA":
                        im.getchannel("A").save(root / "masks" / p.name)
            for p in final.glob("*.png"):
                dst = root / "repairs" / p.name
                if p.resolve() != dst.resolve():   # --out-dir 可能就是 root/repairs
                    shutil.copyfile(p, dst)
            (root / "manifest.json").write_text(json.dumps({
                "subject": subject, "base_frame": base_idx, "yaws": yaws,
                "pitches": pitches,
                "source": "SyncFix (batched, refs per batch) + BiRefNet alpha",
                "files": {"views": "views/", "repairs": "repairs/",
                          "masks": "masks/", "refs": "refs/"},
            }, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  repair/ 结构：{root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
