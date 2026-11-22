#!/usr/bin/env python
"""生成"未训练"的 UV 初始化模型（跳过 hard_surface 阶段用）。

输出一个和 stage1 产物同构的目录：

    <out-dir>/model.ply     —— 零初始化的 UV 绑定高斯（位置/颜色由绑定和训练填）
    <out-dir>/config.yaml   —— 结构配置（tex_size 等）

之后 `densify_train.sh` 的 merge + stage2 可以照常跑，只是初始化没经过
硬表面训练。用于对照"硬表面阶段到底有没有用"。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="未训练的 UV 初始化模型")
    p.add_argument("--subject", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model-config", default=str(REPO_ROOT / "configs" / "model.yaml"))
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)

    import torch

    from live3dgsavatar.config import load_config
    from live3dgsavatar.core.avatar import AvatarConfig, GaussianAvatar
    from live3dgsavatar.core.deform.binding import build_binding
    from live3dgsavatar.core.io.ply import save_ply
    from live3dgsavatar.data.datasets import resolve_dataset_dir
    from live3dgsavatar.data.scene import load_scene

    cfg = load_config()
    cfg.set("subject", args.subject)
    data = resolve_dataset_dir(cfg, args.subject)
    scene = load_scene(cfg, frames=1, device=args.device, data_dir=data)
    import nvdiffrast.torch as dr

    avatar_cfg = AvatarConfig.from_yaml(Path(args.model_config))
    glctx = dr.RasterizeCudaContext()
    binding = build_binding(scene.uvs.to(args.device), scene.uv_faces.to(args.device),
                            tex_size=avatar_cfg.tex_size, glctx=glctx,
                            device=args.device)
    avatar = GaussianAvatar(avatar_cfg, binding).to(args.device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_ply(avatar, out / "model.ply")
    shutil.copyfile(args.model_config, out / "config.yaml")
    print(f"[init-uv] {out/'model.ply'}  高斯 {avatar.num_gaussians}"
          f"  （tex {avatar_cfg.tex_size}，未训练）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
