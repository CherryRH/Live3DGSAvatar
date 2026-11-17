#!/usr/bin/env python
"""用 MODNet 从裁好的 RGB 帧生成干净的人像 RGBA。

MODNet 是面向人像 matting 的模型，对发丝、耳饰和眼镜边缘比通用
U2Net 更稳；模型为 ONNX（25 MB，Apache-2.0），CPU 单帧约 0.05 s。

用法：
    python scripts/matte_modnet.py --input datasets/crh_new/input \
        --out datasets/crh_new/images --model models/matting/modnet.onnx --force
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MODNet 批量抠图")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", type=Path, default=Path("models/matting/modnet.onnx"))
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input.is_dir():
        raise SystemExit(f"[error] 输入目录不存在：{args.input}")
    if not args.model.is_file():
        raise SystemExit(
            f"[error] 找不到 MODNet 模型：{args.model}\n"
            "        下载：\n"
            "        mkdir -p models/matting && curl -L \\\n"
            "          https://hf-mirror.com/Xenova/modnet/resolve/main/onnx/model.onnx \\\n"
            "          -o models/matting/modnet.onnx")
    if args.out.exists() and any(args.out.glob("*.png")) and not args.force:
        raise SystemExit(f"[error] 输出已有图：{args.out}（加 --force）")
    args.out.mkdir(parents=True, exist_ok=True)
    for p in args.out.glob("*.png"):
        p.unlink()

    import onnxruntime as ort

    sess = ort.InferenceSession(str(args.model),
                                providers=["CPUExecutionProvider"])
    files = sorted(args.input.glob("*.png"))
    if not files:
        raise SystemExit("[error] 没有输入图")
    print(f"MODNet：{len(files)} 帧，模型 {args.model}", flush=True)
    for k, p in enumerate(files, 1):
        rgb = np.asarray(Image.open(p).convert("RGB"), np.float32)
        x = (rgb / 255.0 - 0.5) / 0.5
        x = x.transpose(2, 0, 1)[None].astype(np.float32)
        alpha = sess.run(None, {"input": x})[0][0, 0]
        alpha = np.clip(alpha, 0, 1)
        rgba = np.dstack([rgb.astype(np.uint8),
                          np.round(alpha * 255).astype(np.uint8)])
        Image.fromarray(rgba).save(args.out / p.name)
        if k % 200 == 0 or k == len(files):
            print(f"  {k}/{len(files)}", flush=True)
    print(f"完成：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
