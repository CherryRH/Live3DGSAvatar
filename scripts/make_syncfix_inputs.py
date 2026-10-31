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

from live3dgsavatar.core.camera import (as_cpu, orbit_w2c,  # noqa: E402
                                       project_point)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="生成 SyncFix 的实验输入")
    p.add_argument("--subject", default=None)
    p.add_argument("--work-name", default=None)
    p.add_argument("--dataset", default=None, help="渲染哪个数据集（默认同 subject）")
    p.add_argument("--frames", type=int, default=6, help="真实姿态用几帧（随机抽）")
    p.add_argument("--seed", type=int, default=0, help="随机抽样种子")
    p.add_argument("--frame-offset", type=int, default=None,
                   help="从第几帧开始取（默认取留出段起始）")
    p.add_argument("--novel-frame", type=int, default=None,
                   help="用哪一帧的网格渲染新视角（默认取 --frame-offset）")
    p.add_argument("--yaws", type=float, nargs="+",
                   default=[-40.0, -20.0, 0.0, 20.0, 40.0],
                   help="新视角的偏航角（度）；默认 ±40°，更大角度超出可靠范围")
    p.add_argument("--ref-frames", type=int, nargs="+", default=[0, 400, 800],
                   help="参考图取数据集哪几帧（训练段，干净）")
    p.add_argument("--out", type=Path, default=None,
                   help="输出目录（默认 output/syncfix_<对象>）")
    p.add_argument("--dry-run", action="store_true", help="不渲染，只打印计划")
    return p.parse_args(argv)


def pick_holdout_frames(total: int, ratio: float, count: int, seed: int = 0,
                        frame_offset: int | None = None):
    """在**留出段**内随机抽帧。

    连续帧几乎相同（同一段视频相邻两帧差异很小），做多视角/多表情对照没有意义，
    所以在留出段内随机取。

    Args:
        total: 数据集总帧数
        ratio: 验证集比例（与 `train.pipeline.test_ratio` 一致）
        count: 要抽几帧
        seed: 抽样种子
        frame_offset: 显式指定留出段起点（默认 `total - round(total*ratio)`）

    Returns:
        `(offset, held, picks)`，`picks` 升序、互不重复、落在留出段内
    """
    import random

    held = max(1, int(round(total * ratio)))
    offset = total - held if frame_offset is None else int(frame_offset)
    offset = max(0, min(offset, total - 1))
    held = min(held, total - offset)
    k = max(1, min(int(count), held))
    picks = sorted(random.Random(seed).sample(range(offset, offset + held), k))
    return offset, held, picks


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

    # 数据集总帧数：直接读 npz，不写死
    import numpy as np

    with np.load(data_dir / "render_inputs.npz", allow_pickle=True) as d:
        total = int(d["mesh"].shape[0])
    ratio = float(cfg.get("train.pipeline.test_ratio", 0.1))
    offset, held, picks = pick_holdout_frames(
        total, ratio, args.frames, seed=args.seed,
        frame_offset=args.frame_offset)
    default_novel = picks[len(picks) // 2]
    novel_frame = (args.novel_frame if args.novel_frame is not None
                   else default_novel)
    if not (offset <= novel_frame < offset + held):
        print(f"  [warn] --novel-frame {novel_frame} 不在留出段内，改用 "
              f"{default_novel}")
        novel_frame = default_novel

    print(f"模型   : {ply}")
    print(f"数据集 : {data_dir}（{dataset}，共 {total} 帧）")
    print(f"留出段 : [{offset}, {offset + held - 1}]（末尾 {ratio:.0%}，{held} 帧）")
    print(f"输出   : {out_root}")
    print(f"真实姿态帧：{picks}（随机，seed={args.seed}）")
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

    # 载入整段留出段（几百帧，几十 MB），再按 picks 取
    scene = load_scene(cfg, frames=held, frame_offset=offset, data_dir=data_dir)
    avatar = load_ply(ply, config=read_embedded_config(ply), device="cuda")
    rasterizer = SimpleRasterizer()

    # 背景与数据集一致（黑底）：参考图是 RGBA 转 RGB，透明处就是黑
    bg = torch.tensor([float(x) for x in cfg.get("render.background", [0, 0, 0])],
                      dtype=torch.float32, device="cuda")
    # 几何换算全在 CPU 上：新视角的相机由 orbit_w2c 算出后再整体送 GPU
    w2c = torch.eye(4, dtype=torch.float32)
    w2c[:3, :3] = as_cpu(scene.R)
    w2c[:3, 3] = as_cpu(scene.T)
    K_cpu = as_cpu(scene.K)

    def head_center(mesh) -> torch.Tensor:
        """头心 = 当前帧网格的质心（用它做旋转中心，人不会跑出画面）。"""
        return mesh[0].mean(dim=0).detach().cpu()

    def render(mesh, weight, center=None, yaw_deg: float = 0.0):
        m = Mesh(verts=mesh, faces=scene.faces, uvs=scene.uvs,
                 uv_faces=scene.uv_faces)
        view = orbit_w2c(w2c, center if center is not None
                         else head_center(mesh), yaw_deg)
        cam = Camera(K=scene.K, w2c=view.unsqueeze(0).to("cuda"),
                     width=scene.width, height=scene.height).to("cuda")
        with torch.no_grad():
            gs = avatar.deform(m, weight)
            return rasterizer.render(gs, cam, bg)

    for sub in ("input_pose", "gt", "input_novel", "ref_image"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    manifest: dict = {"subject": subject, "work_name": work, "dataset": dataset,
                      "model": str(ply), "pose": [], "novel": [], "refs": []}

    # ---- 真实姿态：渲染 + GT（随机抽自留出段）----
    print("\n渲染真实姿态 …")
    for k, idx in enumerate(picks):
        local = idx - offset
        mesh = scene.frames[local]["mesh"].unsqueeze(0)
        weight = scene.frames[local]["blend_weight"].unsqueeze(0)
        out = render(mesh, weight, yaw_deg=0.0)
        pred = to_uint8(out.color)
        name = f"{k:05d}.png"
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

    # ---- 新视角：同一帧绕头心转 ----
    print("\n渲染新视角 …")
    local = min(max(novel_frame - offset, 0), scene.num_frames - 1)
    mesh = scene.frames[local]["mesh"].unsqueeze(0)
    weight = scene.frames[local]["blend_weight"].unsqueeze(0)
    center = head_center(mesh)
    print(f"  旋转中心（头心）：{center.tolist()}")
    for yaw in args.yaws:
        out = render(mesh, weight, center=center, yaw_deg=float(yaw))
        name = f"yaw{yaw:+.0f}.png"
        img = to_uint8(out.color)
        Image.fromarray(img).save(out_root / "input_novel" / name)
        # 自检：头心应始终落在画面内（绕头心转就不该飞出去）
        view = orbit_w2c(w2c, center, float(yaw))
        u, v = project_point(K_cpu, view, center)
        inside = 0 <= u < scene.width and 0 <= v < scene.height
        fg = float((img.max(axis=-1) > 8).mean())
        manifest["novel"].append({"file": name, "frame": novel_frame,
                                  "yaw_deg": float(yaw),
                                  "center_uv": [round(u, 1), round(v, 1)],
                                  "fg_ratio": round(fg, 4)})
        flag = "" if inside else "  [warn] 头心已在画面外"
        print(f"  {name}  中心 ({u:.0f}, {v:.0f})  前景占比 {fg:.1%}{flag}")

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
