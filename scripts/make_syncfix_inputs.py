#!/usr/bin/env python
"""为 SyncFix 准备一组最小输入（用我们的模型渲染，真实帧作参考）。

## SyncFix 需要什么

`inference.py` 的输入是**目录**，`n_per_pass` 张图一起做联合精修：

```
--input_image <dir>    降质渲染图（按文件名排序，每 n_per_pass 张一起处理）
--ref_image   <dir>    干净参考图（与 input 按同一索引切片！）
--height/--width       统一 resize 尺寸（默认 540x960，横构图）
```

注意两个细节（读 `inference.py` 得到）：

1. 参考图是**并排切片**的：`ref_images[i:i+n]`。`--ref_size 1` 时只有第一批能拿到参考；
   所以小样本实验应当**一次处理完**（`n_per_pass` = 图片数）。
2. 输出会 resize 回输入原尺寸，所以输入尺寸决定输出尺寸。

## 本脚本产出

```
output/syncfix_crh_1/
├── input_pose/     留出集真实姿态的**渲染图**（有 GT，可量化）
├── gt/             对应的真实图像（评估用）
├── input_novel/    同一帧在若干偏航角下的渲染（无 GT，定性看新视角）
├── ref_image/      训练集的真实帧（干净参考）
└── manifest.json   清单：帧号、视角、渲染 PSNR 等
```

用法（需要 GPU）：

    python scripts/make_syncfix_inputs.py --subject crh_1 --work-name run \\
        --dataset crh_1 --frames 6 --frame-offset 2577

    # 然后
    python ~/Projects/SyncFix/inference.py \\
        --input_image output/syncfix_crh_1/input_pose \\
        --ref_image   output/syncfix_crh_1/ref_image \\
        --model_path  <SyncFix 权重> --output_dir output/syncfix_crh_1/out_pose \\
        --n_per_pass 6 --ref_size 1 --height 512 --width 512
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
    p = argparse.ArgumentParser(description="生成 SyncFix 的实验输入")
    p.add_argument("--subject", default=None)
    p.add_argument("--work-name", default=None)
    p.add_argument("--dataset", default=None, help="渲染哪个数据集（默认同 subject）")
    p.add_argument("--frames", type=int, default=6, help="真实姿态用几帧")
    p.add_argument("--frame-offset", type=int, default=None,
                   help="从第几帧开始取（默认取留出段起始）")
    p.add_argument("--novel-frame", type=int, default=None,
                   help="用哪一帧的网格渲染新视角（默认取 --frame-offset）")
    p.add_argument("--yaws", type=float, nargs="+",
                   default=[-60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0],
                   help="新视角的偏航角（度）")
    p.add_argument("--ref-frames", type=int, nargs="+", default=[0, 400, 800],
                   help="参考图取数据集哪几帧（训练段，干净）")
    p.add_argument("--out", type=Path, default=None,
                   help="输出目录（默认 output/syncfix_<对象>）")
    p.add_argument("--dry-run", action="store_true", help="不渲染，只打印计划")
    return p.parse_args(argv)


def rotation_y(deg: float):
    """绕 y 轴旋转（相机绕头转一圈 → 新视角）。"""
    import math

    import torch

    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return torch.tensor([[c, 0.0, s, 0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [-s, 0.0, c, 0.0],
                         [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)


def to_uint8(color):
    """`[1,3,H,W]`（0–1）→ `[H,W,3]` uint8。"""
    import numpy as np

    img = color[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (img * 255.0 + 0.5).astype(np.uint8)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
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

    data_dir = resolve_dataset_dir(cfg, dataset)
    ply = resolve_model_ply(cfg)
    out_root = args.out or (Path(str(cfg.get("paths.output_dir", "output")))
                            / f"syncfix_{subject}")

    total = 2863
    offset = args.frame_offset
    if offset is None:
        # 默认用留出段的起始：那里没参与训练，PSNR 有参考价值
        ratio = float(cfg.get("train.pipeline.test_ratio", 0.1))
        offset = int(round(total * (1.0 - ratio)))
    novel_frame = args.novel_frame if args.novel_frame is not None else offset

    print(f"模型   : {ply}")
    print(f"数据集 : {data_dir}（{dataset}）")
    print(f"输出   : {out_root}")
    print(f"真实姿态帧：{offset} … {offset + args.frames - 1}")
    print(f"新视角     ：第 {novel_frame} 帧，yaw = {args.yaws}")
    print(f"参考帧     ：{args.ref_frames}")

    if args.dry_run:
        print("\n[dry-run] 未渲染")
        return 0

    import numpy as np
    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        print("[error] 渲染需要 GPU", file=sys.stderr)
        return 1

    from live3dgsavatar.core.io import load_ply
    from live3dgsavatar.core.io.ply import read_embedded_config
    from live3dgsavatar.core.metrics import psnr_masked
    from live3dgsavatar.core.render import SimpleRasterizer
    from live3dgsavatar.core.types import Camera, Mesh
    from live3dgsavatar.data.scene import load_scene

    need = max(args.frames, 1)
    scene = load_scene(cfg, frames=need, frame_offset=offset, data_dir=data_dir)
    avatar = load_ply(ply, config=read_embedded_config(ply), device="cuda")
    rasterizer = SimpleRasterizer()

    # 背景与数据集一致（黑底）：参考图是 RGBA 转 RGB，透明处就是黑
    bg = torch.tensor([float(x) for x in cfg.get("render.background", [0, 0, 0])],
                      dtype=torch.float32, device="cuda")
    w2c = torch.eye(4, dtype=torch.float32)
    w2c[:3, :3] = scene.R
    w2c[:3, 3] = scene.T

    def render(mesh, weight, yaw_deg: float = 0.0):
        m = Mesh(verts=mesh, faces=scene.faces, uvs=scene.uvs,
                 uv_faces=scene.uv_faces)
        # 相机绕头旋转 = 新视角；旋转在 CPU 上做再整体送 GPU（w2c 在 CPU 上）
        view = (rotation_y(yaw_deg) @ w2c).unsqueeze(0).to("cuda")
        cam = Camera(K=scene.K, w2c=view,
                     width=scene.width, height=scene.height).to("cuda")
        with torch.no_grad():
            gs = avatar.deform(m, weight)
            return rasterizer.render(gs, cam, bg)

    for sub in ("input_pose", "gt", "input_novel", "ref_image"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    manifest: dict = {"subject": subject, "work_name": work, "dataset": dataset,
                      "model": str(ply), "pose": [], "novel": [], "refs": []}

    # ---- 真实姿态：渲染 + GT ----
    print("\n渲染真实姿态 …")
    for i in range(scene.num_frames):
        idx = offset + i
        mesh = scene.frames[i]["mesh"].unsqueeze(0)
        weight = scene.frames[i]["blend_weight"].unsqueeze(0)
        out = render(mesh, weight, 0.0)
        pred = to_uint8(out.color)
        name = f"{i:05d}.png"
        Image.fromarray(pred).save(out_root / "input_pose" / name)

        gt_path = data_dir / "images" / f"{idx:05d}.png"
        rgba = np.asarray(Image.open(gt_path).convert("RGBA"))
        gt = rgba[..., :3].copy()
        gt[rgba[..., 3] <= 127] = 0          # 透明处填黑，与渲染同底
        Image.fromarray(gt).save(out_root / "gt" / name)
        value = psnr_masked(pred, gt, rgba[..., 3] > 127)
        manifest["pose"].append({"file": name, "frame": idx,
                                 "psnr_masked": None if value is None else round(value, 2)})
        print(f"  {name}  帧 {idx}  PSNR(mask) {value:.2f} dB")

    # ---- 新视角：同一帧的不同偏航 ----
    print("\n渲染新视角 …")
    src = novel_frame - offset
    src = min(max(src, 0), scene.num_frames - 1)
    mesh = scene.frames[src]["mesh"].unsqueeze(0)
    weight = scene.frames[src]["blend_weight"].unsqueeze(0)
    for yaw in args.yaws:
        out = render(mesh, weight, float(yaw))
        name = f"yaw{yaw:+.0f}.png"
        Image.fromarray(to_uint8(out.color)).save(out_root / "input_novel" / name)
        manifest["novel"].append({"file": name, "frame": novel_frame,
                                  "yaw_deg": float(yaw)})
        print(f"  {name}")

    # ---- 参考图：训练段的真实帧（干净） ----
    print("\n复制参考图 …")
    for idx in args.ref_frames:
        src_path = data_dir / "images" / f"{idx:05d}.png"
        if not src_path.is_file():
            print(f"  [warn] 缺少 {src_path}，跳过")
            continue
        rgba = np.asarray(Image.open(src_path).convert("RGBA"))
        rgb = rgba[..., :3].copy()
        rgb[rgba[..., 3] <= 127] = 0
        name = f"{idx:05d}.png"
        Image.fromarray(rgb).save(out_root / "ref_image" / name)
        manifest["refs"].append({"file": name, "frame": idx})
        print(f"  {name}")

    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    n_pose = len(manifest["pose"])
    print(f"\n完成：{out_root}")
    print("下一步（小样本，一次处理完，保证参考图能对上）：")
    print(f"  python ~/Projects/SyncFix/inference.py \\\n"
          f"    --input_image {out_root}/input_pose \\\n"
          f"    --ref_image   {out_root}/ref_image \\\n"
          f"    --model_path  <SyncFix 权重> --output_dir {out_root}/out_pose \\\n"
          f"    --n_per_pass {n_pose} --ref_size 1 --height 512 --width 512")
    print(f"  python ~/Projects/SyncFix/inference.py \\\n"
          f"    --input_image {out_root}/input_novel \\\n"
          f"    --ref_image   {out_root}/ref_image \\\n"
          f"    --model_path  <SyncFix 权重> --output_dir {out_root}/out_novel \\\n"
          f"    --n_per_pass {len(manifest['novel'])} --ref_size 1 "
          f"--height 512 --width 512")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
