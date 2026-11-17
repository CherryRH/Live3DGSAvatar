#!/usr/bin/env python
"""只清理头发/发尾附近的背景色像素，保留人像轮廓。

思路：
1. 在 RGBA 的 fg 内找暗色头发种子（上部 + 两侧，覆盖脖子旁发尾）；
2. 种子小膨胀，得到"头发边缘候选带"；
3. 候选带里只去掉与图像边界连通的背景色像素，内部被头发包围的
   背景块不删，避免误删头发内部；
4. 删除的像素 alpha 置 0、RGB 置黑，防止大角度渲染泛白。
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, label


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="清理头发边缘的背景色像素")
    p.add_argument("--input", type=Path, required=True,
                   help="裁好的 RGB 目录，例如 datasets/crh_new/input")
    p.add_argument("--images", type=Path, required=True,
                   help="当前 RGBA 目录，例如 datasets/crh_new/images")
    p.add_argument("--out", type=Path, required=True, help="输出 RGBA 目录")
    p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--hair-v", type=float, default=0.42,
                   help="头发暗色阈值（HSV V）")
    p.add_argument("--hair-s", type=float, default=0.55,
                   help="头发饱和度上限（排除高饱和皮肤/衣服）")
    p.add_argument("--bg-v", type=float, default=0.60,
                   help="背景亮色阈值（HSV V）")
    p.add_argument("--bg-s", type=float, default=0.28,
                   help="背景低饱和阈值（HSV S）")
    p.add_argument("--dilate", type=int, default=4,
                   help="头发种子膨胀像素，覆盖发丝/发尾边缘")
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


def _clean_one(task: tuple) -> str:
    src, img, dst, hv, hs, bv, bs, dilate = task
    rgb = np.asarray(Image.open(src).convert("RGB"))
    rgba = np.asarray(Image.open(img).convert("RGBA")).copy()
    alpha = rgba[..., 3]
    fg = alpha > 10
    if not fg.any():
        Image.fromarray(rgba).save(dst)
        return Path(dst).name
    ys, xs = np.where(fg)
    top, bottom = int(ys.min()), int(ys.max())
    left, right = int(xs.min()), int(xs.max())
    h = max(1, bottom - top)
    w = max(1, right - left)
    yy, xx = np.mgrid[0:alpha.shape[0], 0:alpha.shape[1]]

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    val = hsv[..., 2].astype(np.float32) / 255.0

    # 上部头发 + 两侧/发尾；避免把中央眼眉/眼镜当成头发种子
    upper = yy < top + 0.55 * h
    sides = ((xx < left + 0.28 * w) | (xx > right - 0.28 * w)) & (yy < top + 0.86 * h)
    hair = fg & (val < hv) & (sat < hs) & (upper | sides)
    # 只要连通到主体头发的暗色区域，避免孤立暗点
    lab, n = label(hair)
    if n:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        hair = lab == int(sizes.argmax())
    band = binary_dilation(hair, iterations=max(1, int(dilate))) & fg

    # 背景色：亮且低饱和；只认与图像边界连通的背景，
    # 被头发包住的白块保留（否则会误删头发内部）。
    bg_like = (val > bv) & (sat < bs) & ~fg
    bg_like |= (val > bv) & (sat < bs) & fg
    lab, n = label(bg_like)
    conn = np.zeros_like(bg_like)
    for k in range(1, n + 1):
        c = lab == k
        if c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any():
            conn |= c
    remove = band & conn & (val > bv) & (sat < bs)
    if remove.any():
        alpha = alpha.copy()
        alpha[remove] = 0
        rgba[..., 3] = alpha
        rgba[..., :3][remove] = 0
    Image.fromarray(rgba).save(dst)
    return Path(dst).name


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    src, imgs, dst = Path(args.input), Path(args.images), Path(args.out)
    for d in (src, imgs):
        if not d.is_dir():
            raise SystemExit(f"[error] 目录不存在：{d}")
    if dst.exists() and any(dst.glob("*.png")) and not args.force:
        raise SystemExit(f"[error] 输出已有文件：{dst}（加 --force）")
    dst.mkdir(parents=True, exist_ok=True)
    for p in dst.glob("*.png"):
        p.unlink()
    files = sorted(imgs.glob("*.png"))
    tasks = []
    for p in files:
        src_p = src / p.name
        if not src_p.is_file():
            raise SystemExit(f"[error] 缺少 input 对应帧：{src_p}")
        tasks.append((str(src_p), str(p), str(dst / p.name),
                      args.hair_v, args.hair_s, args.bg_v, args.bg_s,
                      args.dilate))
    print(f"头发边缘清理：{len(tasks)} 帧，{args.workers} 进程", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for k, _ in enumerate(ex.map(_clean_one, tasks, chunksize=16), 1):
            if k % 200 == 0 or k == len(tasks):
                print(f"  {k}/{len(tasks)}", flush=True)
    print(f"完成：{dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
