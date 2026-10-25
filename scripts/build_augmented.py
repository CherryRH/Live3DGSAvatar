#!/usr/bin/env python
"""把一组"增强图像 + RDF 追踪结果"打包成训练可用的数据集分片。

## 为什么需要

训练要的是 `(mesh 5083, blend_weight 129, image, mask)`。CAP4D 生成的图只有
图像和参数（`rdf_dump.py` 能给出 `x` 与 5023 顶点），还差：

- **补牙齿**（5023 → 5083，`tracking/mesh_build.py`）
- **参数映射**（118 → 129，`tracking/param_map.py`）
- 相机与图像尺寸（与主数据集一致，都是同一套虚拟相机）

本脚本把这些做齐，输出一个与主数据集**同格式**的目录：

    <数据集>/augmented/
    ├── images/00000.png …        增强图（从 --images 复制）
    └── render_inputs.npz         mesh / blend_weight / K / w2v / width / height / names

之后训练加 `--augment` 就会把它按权重混进训练集（验证集仍只用真实数据）。

## 用法

    # ① 先让 RDF 追踪这些增强图（realdenseface 环境）
    conda activate realdenseface
    python scripts/rdf_dump.py --images <增强图目录> --out <增强图目录>/../rdf.npz

    # ② 打包（live3dgs 环境）
    conda activate live3dgs
    python scripts/build_augmented.py --subject crh_1 \\
        --rdf <增强图目录>/../rdf.npz --images <增强图目录>

⚠️ 增强数据只进**训练集**；验证集永远只取真实数据的尾段。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import live3dgsavatar  # noqa: F401


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="打包增强数据为 <数据集>/augmented/")
    p.add_argument("--subject", default=None, help="主数据集名（默认取配置）")
    p.add_argument("--data", type=Path, default=None, help="直接给主数据集目录")
    p.add_argument("--rdf", type=Path, required=True,
                   help="rdf_dump.py 的输出（含 x 与 verts）")
    p.add_argument("--images", type=Path, required=True, help="增强图目录")
    p.add_argument("--out", type=Path, default=None,
                   help="输出目录（默认 <数据集>/augmented）")
    p.add_argument("--limit", type=int, default=0, help="最多用多少帧")
    p.add_argument("--force", action="store_true", help="已存在时覆盖")
    return p.parse_args(argv)


def build(*, data_dir: Path, rdf_path: Path, images_dir: Path, out_dir: Path,
          flame_dir: Path, limit: int = 0) -> Path:
    """生成 `out_dir/{images, render_inputs.npz}`。"""
    import numpy as np
    from PIL import Image

    from live3dgsavatar.data.cap4d import load_rdf_dump, to_frames

    dump = load_rdf_dump(rdf_path)
    if limit > 0:
        for key in ("x", "verts", "names"):
            dump[key] = dump[key][:limit]
    frames = to_frames(dump, flame_dir)

    # 相机/尺寸沿用主数据集：增强图是用同一套虚拟相机生成的
    with np.load(data_dir / "render_inputs.npz", allow_pickle=True) as base:
        K = np.asarray(base["K"], np.float32)
        w2v = np.asarray(base["w2v"], np.float32)
        width, height = int(base["width"]), int(base["height"])

    img_out = out_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    meshes, weights = [], []
    for i, frame in enumerate(frames):
        src = images_dir / dump["names"][i]
        if not src.is_file():
            src = images_dir / f"{i:05d}.png"
        if not src.is_file():
            raise FileNotFoundError(f"缺少增强图：{src}")
        name = f"aug{i:05d}"
        shutil.copyfile(src, img_out / f"{name}.png")
        names.append(name)
        meshes.append(frame["mesh"].numpy())
        weights.append(frame["blend_weight"].numpy())

    if not names:
        raise ValueError("没有可用的增强帧")
    np.savez_compressed(
        out_dir / "render_inputs.npz",
        mesh=np.stack(meshes).astype(np.float32),
        blend_weight=np.stack(weights).astype(np.float32),
        K=K, w2v=w2v, width=np.int64(width), height=np.int64(height),
        names=np.array(names),
    )
    # 顺手校验：图像可读、尺寸一致
    for name in names[:3]:
        arr = np.asarray(Image.open(img_out / f"{name}.png"))
        if arr.shape[0] != height or arr.shape[1] != width:
            raise ValueError(
                f"增强图尺寸 {arr.shape[:2]} 与数据集 {(height, width)} 不一致")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from live3dgsavatar.config import load_config, resolve_data_dir

    cfg = load_config()
    if args.data is not None:
        data_dir = args.data.expanduser().resolve()
    else:
        subject = args.subject or str(cfg.get("subject") or "").strip()
        if not subject:
            raise SystemExit("[error] 需要 --subject 或 --data")
        cfg.set("subject", subject)
        data_dir = resolve_data_dir(cfg)

    out_dir = args.out or (data_dir / "augmented")
    if (out_dir / "render_inputs.npz").exists() and not args.force:
        print(f"[error] {out_dir} 已有增强集；确认覆盖请加 --force")
        return 1

    flame_dir = Path(str(cfg.get("paths.flame_dir")))
    print(f"打包增强集：{args.images} → {out_dir}")
    build(data_dir=data_dir, rdf_path=args.rdf.expanduser().resolve(),
          images_dir=args.images.expanduser().resolve(), out_dir=out_dir,
          flame_dir=flame_dir, limit=args.limit)
    print(f"完成：{out_dir}")
    print("训练时加 --augment 即可启用（验证集仍只用真实数据）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
