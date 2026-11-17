#!/usr/bin/env python
"""重新抠一个已有数据集的 images/（保留人像轮廓，清理边缘白边）。

输入是裁好的 `input/` RGB（512×512），输出同名 `images/` RGBA。
先取 U2Net 的完整人像 alpha，只做 1–2 px 收边；再在轮廓外带去掉接近
背景色的高亮像素，避免上一版 8 px 内缩削掉耳朵、发际和面部边缘。
"""

from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

_WORKER = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="重新抠已有数据集的 images/")
    p.add_argument("--input", type=Path, required=True,
                   help="裁好的 RGB 目录（如 datasets/crh_new/input）")
    p.add_argument("--out", type=Path, required=True,
                   help="输出 RGBA 目录（如 datasets/crh_new/images）")
    p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--erode", type=int, default=1,
                   help="轮廓内缩像素；越小越保留耳朵/发际")
    p.add_argument("--band", type=int, default=8,
                   help="边缘清理带宽（像素）")
    p.add_argument("--bright", type=int, default=185,
                   help="边缘带内 RGB 最小值高于此值视为背景高亮")
    p.add_argument("--upper", type=float, default=0.72,
                   help="只在该高度比例以上清理高亮边缘（避免削白衣）")
    p.add_argument("--model", default="u2net")
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


def _init(model: str) -> None:
    global _WORKER
    from rembg import new_session

    _WORKER = new_session(model)


def _clean(rgb: np.ndarray, *, erode: int, band: int, bright: int,
           upper: float) -> np.ndarray:
    from rembg import remove
    from scipy.ndimage import (binary_erosion, binary_fill_holes,
                               gaussian_filter, label)

    sem = np.asarray(remove(Image.fromarray(rgb), session=_WORKER))[..., 3]
    a = sem.astype(np.float32) / 255.0
    m = binary_fill_holes(a > 0.5)
    lab, n = label(m)
    if n:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        m = lab == int(sizes.argmax())
    if erode > 0:
        m = binary_erosion(m, iterations=int(erode))
    # 只清轮廓外带：去掉头发/背景交界处的高亮背景像素；
    # 内带保留，避免把眼/嘴/耳朵内部的高光当成白边。
    edge = m & ~binary_erosion(m, iterations=max(1, int(band)))
    yy = np.arange(m.shape[0])[:, None]
    edge &= yy < float(upper) * m.shape[0]
    m[edge & (rgb.min(axis=-1) > int(bright))] = False
    af = gaussian_filter(m.astype(np.float32), 0.6)
    af = np.clip(af, 0, 1)
    bg_px = rgb[a < 0.05]
    bg = (np.median(bg_px, axis=0) if len(bg_px) >= 48
          else np.array([232, 232, 230], np.float32))
    f = (rgb.astype(np.float32) - (1 - af[..., None]) * bg) / np.maximum(
        af[..., None], 0.08)
    f = np.clip(f, 0, 255)
    f[af < 0.03] = 0
    return np.concatenate([f, af[..., None] * 255], axis=-1).astype(np.uint8)


def _one(task: tuple) -> str:
    src, dst, erode, band, bright, upper = task
    rgb = np.asarray(Image.open(src).convert("RGB"))
    Image.fromarray(_clean(rgb, erode=erode, band=band, bright=bright,
                          upper=upper)).save(dst)
    return Path(dst).name


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    src = Path(args.input)
    dst = Path(args.out)
    if not src.is_dir():
        raise SystemExit(f"[error] 输入目录不存在：{src}")
    if dst.exists() and any(dst.glob("*.png")) and not args.force:
        raise SystemExit(f"[error] 输出目录已有图：{dst}（确认覆盖加 --force）")
    dst.mkdir(parents=True, exist_ok=True)
    for p in dst.glob("*.png"):
        p.unlink()
    files = sorted(src.glob("*.png"))
    if not files:
        raise SystemExit("[error] 没有输入图")
    tasks = [(str(p), str(dst / p.name), args.erode, args.band,
              args.bright, args.upper) for p in files]
    print(f"重抠：{len(tasks)} 帧，{args.workers} 进程，"
          f"erode={args.erode} band={args.band} bright={args.bright}",
          flush=True)
    with ProcessPoolExecutor(max_workers=max(1, args.workers),
                             initializer=_init, initargs=(args.model,)) as ex:
        for k, _ in enumerate(ex.map(_one, tasks, chunksize=8), 1):
            if k % 100 == 0 or k == len(tasks):
                print(f"  {k}/{len(tasks)}", flush=True)
    print(f"完成：{dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
