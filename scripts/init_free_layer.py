#!/usr/bin/env python
"""从硬表面模型的高误差区域初始化自由层。

输入 `find_defects.py` 的 mass.npz；只选**有观测覆盖**的高质量缺陷高斯，
避免把后脑勺/未观测区域复制进自由层。

用法::

    python scripts/init_free_layer.py \
      --model models/bala/layered_hard_init/model.ply \
      --mass tmp/free_defects/mass.npz \
      --out models/bala/layered_free/free_layer.npz --top-k 12000
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import live3dgsavatar  # noqa: F401


def main() -> int:
    p = argparse.ArgumentParser(description="从硬表面高误差区域初始化自由层")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--mass", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=12000)
    p.add_argument("--min-coverage", type=float, default=100.0)
    p.add_argument("--opacity", type=float, default=0.02)
    p.add_argument("--scale-factor", type=float, default=0.85)
    args = p.parse_args()
    from live3dgsavatar.core.avatar import AvatarConfig
    from live3dgsavatar.core.io import load_ply
    cfg = AvatarConfig.from_yaml(args.model.parent / "config.yaml")
    avatar = load_ply(args.model, config=cfg, device="cpu")
    d = np.load(args.mass)
    if "mass" not in d.files or "coverage" not in d.files:
        raise SystemExit("mass.npz 必须含 mass 和 coverage")
    mass = np.asarray(d["mass"], np.float32)
    coverage = np.asarray(d["coverage"], np.float32)
    n = int(avatar.num_gaussians)
    if mass.size != n:
        raise SystemExit(f"mass 长度 {mass.size} != 模型高斯数 {n}")
    observed = coverage >= float(args.min_coverage)
    score = np.where(observed, mass, -1.0)
    if not observed.any():
        raise SystemExit("没有 coverage 达标的高斯；降低 --min-coverage")
    k = min(int(args.top_k), int(observed.sum()))
    idx = np.argpartition(score, -k)[-k:]
    idx = idx[np.argsort(score[idx])[::-1]]
    avatar.init_free_layer(torch.from_numpy(idx.astype("int64")),
                           opacity=args.opacity,
                           scale_factor=args.scale_factor)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    avatar.save_free_layer(args.out)
    print(f"自由层 {k} 个（候选 {int(observed.sum())}），输出 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
