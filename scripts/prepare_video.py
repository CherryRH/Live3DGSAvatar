#!/usr/bin/env python
"""把一段人像视频整理成可跟踪的 512 数据集素材。

解决的问题：

- 近景帧裁掉头顶：按每帧人像 bbox 估计头部尺度，用全片稳定的尺度裁一个
  方形窗口；窗口跟随头心缓慢移动，顶部保留固定 margin，不逐帧抖动。
- 抠图白边：U2Net 掩码做最大连通域/填洞/内缩，再用背景色做前景反解，
  边缘的白色背景不再作为 RGB 保留下来。

输出目录（交给后续 RDF/构建脚本）：

```text
<out>/
  input/00000.png ...   裁好的 RGB（给 RDF 跟踪）
  images/00000.png ...  裁好并抠图的 RGBA（训练用）
  crop_stats.json       每帧裁剪参数/统计
```
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from PIL import Image


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="视频抽帧、稳定裁头、去背景")
    p.add_argument("--video", type=Path, required=True,
                   help="视频文件，或只含一个视频的目录")
    p.add_argument("--out", type=Path, required=True, help="输出目录")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--limit", type=int, default=0, help="最多处理多少帧；0 全部")
    p.add_argument("--stride", type=int, default=1, help="抽帧后每 N 帧取一帧")
    p.add_argument("--size", type=int, default=512, help="输出方形边长")
    p.add_argument("--head-factor", type=float, default=2.8,
                   help="方形裁剪边长 / 头宽；2.8 保留头、脖子和肩部，"
                        "超出原图宽度时用背景色补成方形")
    p.add_argument("--top-margin", type=float, default=0.12,
                   help="头顶到裁剪上边的距离占裁剪边长比例")
    p.add_argument("--stat-stride", type=int, default=5,
                   help="每隔多少帧估一次人像 bbox（其余线性插值）")
    p.add_argument("--matte-model", default="u2net")
    p.add_argument("--matte-erode", type=int, default=1,
                   help="抠图掩码内缩像素；默认 1，优先保留耳朵/发际")
    p.add_argument("--matte-band", type=int, default=8,
                   help="只在轮廓外带清理背景高亮")
    p.add_argument("--matte-bright", type=int, default=185,
                   help="边缘带内 RGB 最小值高于此值视为背景高亮")
    p.add_argument("--matte-upper", type=float, default=0.72,
                   help="只在该高度比例以上清理高亮边缘，避免削白衣")
    p.add_argument("--matte-feather", type=float, default=0.6)
    p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                   help="并行抠图进程数；U2Net 是 CPU 推理，默认 4")
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-images", action="store_true",
                   help="只生成裁好的 input/；images/ 交给后续 matting 模型")
    p.add_argument("--keep-raw", action="store_true")
    return p.parse_args(argv)


def resolve_video(path: Path) -> Path:
    path = path.expanduser()
    if path.is_file():
        return path
    if not path.exists():
        for ext in (".mp4", ".mov", ".mkv", ".avi"):
            alt = path.with_suffix(ext)
            if alt.is_file():
                return alt
    if path.is_dir():
        videos = sorted(p for p in path.iterdir()
                        if p.suffix.lower() in (".mp4", ".mov", ".mkv", ".avi"))
        if len(videos) == 1:
            return videos[0]
        if not videos:
            raise SystemExit(f"[error] 目录里没有视频：{path}")
        raise SystemExit("[error] 目录里有多个视频，请直接指定文件：\n  " +
                         "\n  ".join(str(v) for v in videos))
    raise SystemExit(f"[error] 视频不存在：{path}")


def extract_frames(video: Path, raw: Path, fps: float, limit: int) -> list[Path]:
    import imageio_ffmpeg

    raw.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(video),
           "-vf", f"fps={fps:g}", "-start_number", "0"]
    if limit > 0:
        cmd += ["-frames:v", str(limit)]
    cmd.append(str(raw / "%05d.png"))
    subprocess.run(cmd, check=True)
    files = sorted(raw.glob("*.png"))
    if not files:
        raise SystemExit("[error] ffmpeg 没有抽出任何帧")
    return files


def _largest_component(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import label

    lab, n = label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(sizes.argmax())


def person_stats(alpha: np.ndarray) -> dict:
    """从 U2Net alpha 估头顶、头宽和头心。"""
    m = _largest_component(alpha > 0.5)
    ys, xs = np.where(m)
    if xs.size == 0:
        return {"top": 0.0, "center": alpha.shape[1] / 2.0, "head_width": 0.0,
                "person_height": 0.0}
    top, bottom = int(ys.min()), int(ys.max())
    h = max(1, bottom - top)
    # 头顶到人像高度 10% 的横带里，头/发/眼镜最宽；再往下会混入肩。
    band = m[top:top + max(4, int(0.10 * h))]
    widths = [int(band[y].sum()) for y in range(band.shape[0]) if band[y].any()]
    widths = widths or [int(xs.max() - xs.min() + 1)]
    head_width = float(np.percentile(widths, 80))
    # 头心用同一横带的中位数，避免把肩宽算进去
    ys_b, xs_b = np.where(band)
    center = float(np.median(xs_b)) if xs_b.size else float(np.median(xs))
    return {"top": float(top), "center": center,
            "head_width": head_width, "person_height": float(h)}


def smooth_stats(stats: list[dict], step: int, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把稀疏统计线性插值 + 中值平滑到每帧。"""
    from scipy.ndimage import median_filter

    keys = ("top", "center", "head_width")
    xs = np.arange(len(stats)) * step
    xs = np.minimum(xs, n - 1)
    out = []
    for k in keys:
        v = np.asarray([s[k] for s in stats], np.float64)
        xi = np.arange(n)
        arr = np.interp(xi, xs, v)
        arr = median_filter(arr, size=5, mode="nearest")
        out.append(arr)
    return out[0], out[1], out[2]


def clean_alpha(rgba: np.ndarray, *, erode: int, feather: float,
                model: str = "u2net", session=None,
                band: int = 8, bright: int = 185,
                upper: float = 0.72) -> np.ndarray:
    """从 rembg 结果得到干净训练用 RGBA。"""
    from scipy.ndimage import (binary_erosion, binary_fill_holes, gaussian_filter,
                               label)

    from rembg import new_session, remove

    if session is None:
        session = new_session(model)
    rgb = rgba[..., :3]
    out = np.asarray(remove(Image.fromarray(rgb), session=session))
    a = out[..., 3].astype(np.float32) / 255.0
    m = _largest_component(a > 0.5)
    m = binary_fill_holes(m)
    # 去掉边缘白边：内缩后硬掩码；旧数据也几乎是硬 alpha
    if erode > 0:
        m = binary_erosion(m, iterations=erode)
    # 只清轮廓外带：去掉头发/背景交界处的高亮背景像素，
    # 不碰眼/嘴/耳朵内部的高光。
    edge = m & ~binary_erosion(m, iterations=max(1, int(band)))
    yy = np.arange(m.shape[0])[:, None]
    edge &= yy < float(upper) * m.shape[0]
    m[edge & (rgb.min(axis=-1) > int(bright))] = False
    # 背景色估计：只取 alpha≈0 的像素；没有就退回白色
    bg_px = rgb[a < 0.05]
    bg = (np.median(bg_px, axis=0) if bg_px.size >= 48
          else np.array([232, 232, 230], np.float32))
    af = gaussian_filter(m.astype(np.float32), feather)
    af = np.clip(af, 0.0, 1.0)
    # I = aF + (1-a)B  →  F = (I - (1-a)B)/a；边缘不再把白背景带进 RGB
    F = (rgb.astype(np.float32) - (1.0 - af[..., None]) * bg) / np.maximum(
        af[..., None], 0.08)
    F = np.clip(F, 0, 255)
    F[af < 0.03] = 0
    return np.concatenate([F, af[..., None] * 255.0], axis=-1).astype(np.uint8)


_WORKER_SESSION = None


def _init_worker(model: str) -> None:
    global _WORKER_SESSION
    from rembg import new_session

    _WORKER_SESSION = new_session(model)


def _crop_matte_task(task: tuple) -> dict:
    """子进程任务：裁一帧并抠图。"""
    (raw_path, inp_path, img_path, size, side, left, top,
     erode, feather, model, band, bright, upper,
     source_idx, head_top, head_width) = task
    im = Image.open(raw_path).convert("RGB")
    w, h = im.size
    left = max(-side + 1, min(left, w - 1))
    top = max(-side + 1, min(top, h - 1))
    sx0, sy0 = max(0, left), max(0, top)
    sx1, sy1 = min(w, left + side), min(h, top + side)
    crop_im = Image.new("RGB", (side, side), (232, 232, 230))
    crop_im.paste(im.crop((sx0, sy0, sx1, sy1)), (sx0 - left, sy0 - top))
    crop_im = crop_im.resize((size, size), Image.LANCZOS)
    crop_im.save(inp_path)
    rgba = clean_alpha(np.asarray(crop_im), erode=erode, feather=feather,
                       model=model, session=_WORKER_SESSION,
                       band=band, bright=bright, upper=upper)
    Image.fromarray(rgba).save(img_path)
    return {"frame": source_idx, "name": Path(img_path).name,
            "box": [sx0, sy0, sx1, sy1],
            "pad_box": [left, top, left + side, top + side],
            "head_top": float(head_top), "head_width": float(head_width)}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    video = resolve_video(args.video)
    out = args.out
    if out.exists() and not args.force:
        raise SystemExit(f"[error] 输出目录已存在：{out}（确认覆盖加 --force）")
    if out.exists():
        shutil.rmtree(out)
    raw, inp, images = out / "raw", out / "input", out / "images"
    masks = out / "masks"
    for d in (raw, inp, images, masks):
        d.mkdir(parents=True, exist_ok=True)

    print(f"视频：{video}")
    files = extract_frames(video, raw, args.fps, args.limit)
    print(f"抽帧：{len(files)} 帧（{args.fps:g} fps）", flush=True)
    take = list(range(0, len(files), max(1, args.stride)))
    if args.limit > 0:
        take = take[: args.limit]

    # ---- 1) 稀疏统计人像尺度 ----
    from rembg import new_session, remove

    session = new_session(args.matte_model)
    stats = []
    for j, i in enumerate(take[::max(1, args.stat_stride)]):
        im = Image.open(files[i]).convert("RGB")
        scale = 0.5
        small = im.resize((max(1, int(im.width * scale)),
                           max(1, int(im.height * scale))), Image.BILINEAR)
        a = np.asarray(remove(small, session=session))[..., 3]
        a = np.asarray(Image.fromarray(a).resize(im.size, Image.BILINEAR))
        st = person_stats(a.astype(np.float32) / 255.0)
        stats.append(st)
        print(f"  统计 {j + 1}/{len(range(0, len(take), args.stat_stride))}"
              f"  帧 {i:05d}  head={st['head_width']:.0f}px", flush=True)
    if not stats:
        raise SystemExit("[error] 没有可统计的帧")
    top_s, center_s, width_s = smooth_stats(
        stats, max(1, args.stat_stride), len(take))

    # 全片稳定裁剪尺度：用较宽的头（90 分位）保证近景不切头。
    # 方形边长可以大于原图宽度；超出的左右用背景色补齐，保持 1:1 像素，
    # 不做非等比缩放（否则 FLAME 脸会被拉伸）。
    head_ref = float(np.percentile(width_s, 90))
    side = int(round(args.head_factor * head_ref))
    side = max(512, side)
    print(f"裁剪：{side}px 方形（原图宽 {im.width}），"
          f"头宽参考 {head_ref:.0f}px，顶部 margin {args.top_margin:.0%}",
          flush=True)

    tasks = []
    for j, i in enumerate(take):
        cx = int(round(center_s[j]))
        top = int(round(top_s[j] - args.top_margin * side))
        top = max(-side + 1, min(top, 4096))  # 源高度这里只用于 pad 坐标
        left = int(round(cx - side / 2))
        tasks.append((str(files[i]), str(inp / f"{j:05d}.png"),
                      str(images / f"{j:05d}.png"), args.size, side, left, top,
                      args.matte_erode, args.matte_feather, args.matte_model,
                      args.matte_band, args.matte_bright, args.matte_upper,
                      i, float(top_s[j]), float(width_s[j])))
    records = []
    workers = max(1, int(args.workers))
    print(f"并行抠图：{workers} 进程", flush=True)
    if workers == 1:
        _init_worker(args.matte_model)
        for k, task in enumerate(tasks, 1):
            records.append(_crop_matte_task(task))
            if k % 25 == 0 or k == len(tasks):
                print(f"  裁图/抠图 {k}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers,
                                 initializer=_init_worker,
                                 initargs=(args.matte_model,)) as ex:
            for k, rec in enumerate(ex.map(_crop_matte_task, tasks, chunksize=8), 1):
                records.append(rec)
                if k % 100 == 0 or k == len(tasks):
                    print(f"  裁图/抠图 {k}/{len(tasks)}", flush=True)

    (out / "crop_stats.json").write_text(
        json.dumps({"video": str(video), "fps": args.fps, "size": args.size,
                    "crop": side, "head_factor": args.head_factor,
                    "top_margin": args.top_margin, "frames": records},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    if not args.keep_raw:
        shutil.rmtree(raw)
    print(f"\n产物：{out}")
    print(f"  input/   {len(take)} 张 RGB（RDF 跟踪用）")
    if args.skip_images:
        print("  images/  已跳过（后续 matting 生成）")
    else:
        print(f"  images/  {len(take)} 张 RGBA（训练用）")
    print(f"  crop_stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
