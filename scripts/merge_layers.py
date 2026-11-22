#!/usr/bin/env python
"""把 UV 初始化模型摊到更大的 UV 网格上（给加密留容量）。

UV 初始化模型通常是 tex 256（约 6 万高斯），加密阶段要 split 出更多高斯，
而 256² 只有 65536 个槽位。这里把它复制进一个 tex 更大的空 avatar：
**高斯数不变**，只是把容量放大，后续 split 才有地方放。

用法::

    python scripts/merge_layers.py --hard models/bala/densify_hard/model.ply \
        --out tmp/bala_densify_unified.ply --tex-size 512
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import live3dgsavatar  # noqa: F401,E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hard", type=Path, required=True,
                   help="要扩容的 model.ply（同目录要有 config.yaml）")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tex-size", type=int, default=512)
    a = p.parse_args()

    from live3dgsavatar.core.avatar import AvatarConfig, GaussianAvatar
    from live3dgsavatar.core.deform import Binding
    from live3dgsavatar.core.io import load_ply, save_ply

    cfg_path = a.hard.parent / "config.yaml"
    if not cfg_path.is_file():
        raise SystemExit(f"[error] 缺少结构配置：{cfg_path}")
    cfg = AvatarConfig.from_yaml(cfg_path)
    hard = load_ply(a.hard, config=cfg, device="cpu")
    n = int(hard.num_gaussians)
    if a.tex_size * a.tex_size < n:
        raise SystemExit(
            f"[error] tex_size {a.tex_size} 容量 {a.tex_size ** 2} < 高斯数 {n}")

    new_cfg = replace(cfg, tex_size=int(a.tex_size))
    av = GaussianAvatar(new_cfg, hard.to_binding()).to("cpu")
    with torch.no_grad():
        av.xyz.copy_(hard.xyz)
        av.opacity.copy_(hard.opacity)
        av.scaling.copy_(hard.scaling)
        av.rotation.copy_(hard.rotation)
        av.color.copy_(hard.color)
        av.xyz_b.copy_(hard.xyz_b)
        av.rotation_b.copy_(hard.rotation_b)
        av.color_b.copy_(hard.color_b)
        if new_cfg.use_weight_proj:
            av.weight_module.load_state_dict(hard.weight_module.state_dict())

    a.out.parent.mkdir(parents=True, exist_ok=True)
    save_ply(av, a.out)
    print(f"扩容 {n} 个高斯 -> {a.out}（tex {cfg.tex_size} -> {a.tex_size}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
