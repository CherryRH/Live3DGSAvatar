#!/usr/bin/env python
"""在线更新（热更新）的实际运行入口。

用数据集里的一小段帧模拟"通话开始时的观测"，在时间预算内微调模型外观，
并报告 PSNR 的前后变化 —— 这就是热更新的验收判据。

两种模式：

1. ``--dry-run``：**不需要 GPU**。只测闭式颜色校正（纯张量运算），
   用数据集原图与模型颜色做点对点比较，验证拟合流程与指标。
2. 默认：完整流程（闭式颜色校正 + 梯度微调）。需要 GPU。

用法::

    python scripts/online_update.py --dry-run
    python scripts/online_update.py --frames 10 --time-budget 10
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ⚠️ 必须在导入参照实现之前施加 numpy 兼容补丁（chumpy）。
import live3dgsavatar  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="在线更新（热更新）")
    p.add_argument("--subject", default=None)
    p.add_argument("--work-name", default=None)
    # 默认值取自 configs/system.yaml 的 online_update 节（None 表示用配置）
    p.add_argument("--frames", type=int, default=None, help="用于更新的帧数")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--time-budget", type=float, default=None, help="墙钟预算（秒）")
    p.add_argument("--lr-color", type=float, default=None)
    p.add_argument("--lr-opacity", type=float, default=None)
    p.add_argument("--no-opacity", action="store_true", help="不更新不透明度")
    p.add_argument("--mask-weight", type=float, default=None,
                   help="掩码外像素权重（0 = 只用前景）")
    p.add_argument("--dry-run", action="store_true",
                   help="只测闭式颜色校正（不需要 GPU）")
    p.add_argument("--dump", type=Path, default=None,
                   help="把更新增量存成 npz")
    return p.parse_args(argv)


def _cfg_get(args, cfg, name, key, default):
    """命令行优先，其次配置，最后兜底默认值。"""
    v = getattr(args, name.replace("-", "_"), None)
    return v if v is not None else cfg.get(f"online_update.{key}", default)


def _setup(args):
    from live3dgsavatar.config import load_config, resolve_data_dir, resolve_model_ply
    cfg = load_config()
    if args.subject:
        cfg.set("subject", args.subject)
    if args.work_name:
        cfg.set("work_name", args.work_name)
    return cfg, resolve_data_dir(cfg), resolve_model_ply(cfg)


def _load_observations(cfg, data_dir: Path, n: int, device: str):
    """从数据集构造观测（与摄像头来源共用同一契约）。"""
    import numpy as np
    import torch
    from PIL import Image

    from live3dgsavatar.data.scene import load_scene
    from live3dgsavatar.training.observations import BatchObservation, FrameObservation
    from live3dgsavatar.core.types import Camera, Mesh

    scene = load_scene(cfg, frames=n, device=device)
    img_dir = data_dir / str(cfg.get("paths.image_subdir", "images"))
    files = sorted(img_dir.iterdir())

    frames = []
    for i in range(scene.num_frames):
        img = np.array(Image.open(files[i]).convert("RGBA")).astype("float32") / 255.0
        rgb = torch.from_numpy(img[..., :3]).permute(2, 0, 1).unsqueeze(0).to(device)
        mask = torch.from_numpy(img[..., 3] > 0.5).unsqueeze(0).unsqueeze(0).to(device)
        frames.append(FrameObservation(
            mesh=Mesh(verts=scene.frames_tensor([i]),
                      faces=scene.faces, uvs=scene.uvs, uv_faces=scene.uv_faces),
            blend_weight=scene.weights_tensor([i]),
            target=rgb, mask=mask, name=f"{i:05d}"))

    # ⚠️ 必须显式 `.to(device)`：`Camera` 的 `w2c` 若留在 CPU 而 `K` 在 GPU，
    #    光栅化内核会把它当作 CUDA 指针 → `illegal memory access`。
    cam = Camera.from_intrinsics_extrinsics(
        K=scene.K, R=scene.R, T=scene.T,
        width=scene.width, height=scene.height).to(device)
    return BatchObservation(frames=frames, camera=cam)


def _dry_run(args) -> int:
    """不需要 GPU：用模型颜色与数据集原图做点对点比较，验证闭式拟合。"""
    import torch

    from live3dgsavatar.core.avatar import AvatarConfig, sh0_to_rgb
    from live3dgsavatar.core.io import load_ply
    from live3dgsavatar.core.render.camera_utils import project_points
    from live3dgsavatar.training.color_fit import fit_color_correction
    import yaml

    cfg, data_dir, ply = _setup(args)
    print(f"[dry-run] model={ply.relative_to(REPO_ROOT)}")
    print(f"[dry-run] data ={data_dir}")

    n_frames = int(_cfg_get(args, cfg, "frames", "frames", 8))
    batch = _load_observations(cfg, data_dir, min(n_frames, 3), "cpu")
    from live3dgsavatar.config import resolve_model_config
    mcfg = yaml.safe_load(open(resolve_model_config(cfg)))
    avatar = load_ply(ply, config=AvatarConfig(
        tex_size=mcfg["model"]["tex_size"],
        num_basis_in=mcfg["model"]["num_basis_in"],
        num_basis_blend=mcfg["model"]["num_basis_blend"]), device="cpu")

    from live3dgsavatar.core.types import Mesh
    obs = batch.frames[0]
    with torch.no_grad():
        gs = avatar.deform(obs.mesh, obs.blend_weight)
    rgb = sh0_to_rgb(gs.color[0].squeeze(1)).clamp(0, 1)
    ndc, _ = project_points(gs.xyz, batch.camera.full_proj, batch.camera.width, batch.camera.height)
    px = ((ndc[0, :, 0] * 0.5 + 0.5) * batch.camera.width).round().long().clamp(0, batch.camera.width - 1)
    py = ((ndc[0, :, 1] * 0.5 + 0.5) * batch.camera.height).round().long().clamp(0, batch.camera.height - 1)
    target = obs.target[0].permute(1, 2, 0)[py, px]              # [N,3]

    corr, metrics = fit_color_correction(rgb, target)
    print("\n[闭式颜色校正]（逐高斯点对点，非真实渲染）")
    print(f"  像素数={metrics['n_pixels']}  参数={metrics['n_params']}")
    print(f"  PSNR {metrics['psnr_before']:.2f} → {metrics['psnr_after']:.2f} dB "
          f"({metrics['psnr_delta']:+.2f})")
    print(f"  矩阵对角={[round(float(x), 3) for x in corr.matrix.diag()]}")
    print("\n  注意：绝对值无意义（不是真实渲染），只看相对提升与矩阵形状。")
    return 0


def main() -> int:
    args = parse_args()
    if args.dry_run:
        return _dry_run(args)

    # ---- 完整流程（需要 GPU）----
    import torch
    from live3dgsavatar.core.avatar import AvatarConfig
    from live3dgsavatar.core.io import load_ply
    from live3dgsavatar.core.render import SimpleRasterizer
    from live3dgsavatar.training import (OnlineUpdater, UpdateConfig, fit_color_correction)
    import yaml

    if not torch.cuda.is_available():
        print("[error] 完整热更新需要 GPU（光栅化无法降级到 CPU）。"
              "可先用 --dry-run 验证闭式颜色校正。", file=sys.stderr)
        return 1
    # 崩溃排查辅助：CUDA 内核错误是异步上报的，建议带 CUDA_LAUNCH_BLOCKING=1 运行
    if not os.environ.get("CUDA_LAUNCH_BLOCKING"):
        print("[提示] 若出现 CUDA 非法访问，请用 "
              "`CUDA_LAUNCH_BLOCKING=1 python scripts/online_update.py ...` "
              "获取准确的出错位置")

    cfg, data_dir, ply = _setup(args)
    print(f"model = {ply.relative_to(REPO_ROOT)}")
    print(f"data  = {data_dir}")

    device = str(cfg.get("runtime.device", "cuda"))
    n_frames = int(_cfg_get(args, cfg, "frames", "frames", 8))
    batch = _load_observations(cfg, data_dir, n_frames, device)
    print(f"观测: {len(batch)} 帧  {batch.camera.width}x{batch.camera.height}")

    from live3dgsavatar.config import resolve_model_config
    mcfg = yaml.safe_load(open(resolve_model_config(cfg)))
    avatar = load_ply(ply, config=AvatarConfig(
        tex_size=mcfg["model"]["tex_size"],
        num_basis_in=mcfg["model"]["num_basis_in"],
        num_basis_blend=mcfg["model"]["num_basis_blend"]), device=device)
    rasterizer = SimpleRasterizer(sh_degree=int(cfg.get("render.sh_degree", 0)))
    bg = torch.tensor([float(x) for x in cfg.get("render.background", [0, 0, 0])],
                      device=device)

    updater = OnlineUpdater(avatar, rasterizer, config=UpdateConfig(
        steps=int(_cfg_get(args, cfg, "steps", "steps", 300)),
        time_budget_s=float(_cfg_get(args, cfg, "time-budget", "time_budget_s", 10.0)),
        lr_color=float(_cfg_get(args, cfg, "lr-color", "lr_color", 2e-3)),
        lr_opacity=float(_cfg_get(args, cfg, "lr-opacity", "lr_opacity", 2e-2)),
        update_opacity=(not args.no_opacity) and bool(
            cfg.get("online_update.update_opacity", True)),
        mask_weight=float(_cfg_get(args, cfg, "mask-weight", "mask_weight", 0.0))))

    report, state = updater.fit(batch, background=bg)
    print(f"\n{report.summary()}")
    if report.history:
        print("  过程:", ", ".join(f"@{h['step']}:{h['psnr']:.2f}" for h in report.history))

    if args.dump:
        import numpy as np
        np.savez(args.dump, **{k: v.numpy() for k, v in state.to_dict().items()})
        print(f"  增量已存: {args.dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
