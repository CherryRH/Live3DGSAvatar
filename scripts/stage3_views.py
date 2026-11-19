#!/usr/bin/env python
"""stage3：[渲染固定 yaw 视图 + 参考图] -> 直接喂给 `syncfix_label.py`。

与 `find_defects.py` 的区别：**不选缺陷**。stage3 的输入模型已经在大角度上
明显好于基础模型，欠的是"没见过的角度"，所以按固定间隔（如 5°）逐个视角
渲染、修复、加入监督即可。

产物（syncfix_label 需要的 defects 目录结构）::

    <out>/views/yaw+45_pitch+0_roll+0_frame02104.png   # 待修复的渲染图
    <out>/refs/*.png                                    # 身份参考图（真实帧）
    <out>/selected.json                                 # 本次视角清单
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
    p = argparse.ArgumentParser(description="stage3 固定 yaw 视图渲染")
    p.add_argument("--subject", default=None)
    p.add_argument("--work-name", default=None,
                   help="要渲染的模型版本名（models/<subject>/<work-name>）")
    p.add_argument("--model", type=Path, default=None,
                   help="直接给 model.ply；给定时忽略 --work-name")
    p.add_argument("--dataset", default=None)
    p.add_argument("--yaws", required=True,
                   help="逗号分隔的绝对 yaw（度），如 45,-45,50,-50")
    p.add_argument("--base-frame", default="auto",
                   help="基帧：auto=训练段里 |yaw| 最小的帧；或给帧号")
    p.add_argument("--refs", type=Path, default=None,
                   help="参考图目录（真实帧）；缺省用 datasets/<obj>/repair/refs")
    p.add_argument("--out", type=Path, required=True)
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
    for yaw in yaws:
        label = f"yaw{yaw:+.0f}_pitch+0_roll+0_frame{base_idx:05d}"
        labels.append(label)
        if args.dry_run:
            print(f"  [dry-run] {label}")
            continue
        R_target = rotation_y(yaw)[:3, :3]
        R_orbit = R_target @ head_R.T
        view = orbit_w2c_matrix(w2c, center, R_orbit)
        cam = Camera(K=scene.K, w2c=view.unsqueeze(0).to(device),
                     width=scene.width, height=scene.height).to(device)
        mesh = Mesh(verts=base_verts, faces=scene.faces.to(device),
                    uvs=scene.uvs.to(device), uv_faces=scene.uv_faces.to(device))
        with torch.no_grad():
            res = render_frame(avatar, rasterizer, mesh, weight0, cam, bg)
        img = (res.color[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0
               ).astype(np.uint8)
        Image.fromarray(img).save(views_dir / f"{label}.png")
        print(f"  {label}  (yaw {yaw:+.1f}°)")

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
