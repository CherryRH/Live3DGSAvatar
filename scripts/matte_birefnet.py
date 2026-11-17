#!/usr/bin/env python
"""用 PyTorch 版 BiRefNet 从裁好的 RGB 帧生成人像 RGBA。

BiRefNet-portrait 的权重和自定义模型代码放在
`models/matting/birefnet_portrait/`，在 GPU 上单帧约 0.08 s。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BiRefNet-portrait 批量抠图")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", type=Path,
                   default=Path("models/matting/birefnet_portrait"))
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input.is_dir():
        raise SystemExit(f"[error] 输入目录不存在：{args.input}")
    if not (args.model / "model.safetensors").is_file():
        raise SystemExit(
            f"[error] 找不到 BiRefNet 权重：{args.model}/model.safetensors\n"
            "        下载：\n"
            "        python -c \"from huggingface_hub import snapshot_download; "
            "snapshot_download('ZhengPeng7/BiRefNet-portrait', "
            "local_dir='models/matting/birefnet_portrait')\"")
    if args.out.exists() and any(args.out.glob("*.png")) and not args.force:
        raise SystemExit(f"[error] 输出已有图：{args.out}（加 --force）")
    args.out.mkdir(parents=True, exist_ok=True)
    for p in args.out.glob("*.png"):
        p.unlink()

    import torch
    from transformers import AutoModelForImageSegmentation

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[error] CUDA 不可用；需要 GPU 跑 BiRefNet")
    model = AutoModelForImageSegmentation.from_pretrained(
        str(args.model), trust_remote_code=True, local_files_only=True)
    model.eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406],
                        dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225],
                       dtype=torch.float32, device=device).view(1, 3, 1, 1)

    files = sorted(args.input.glob("*.png"))
    if not files:
        raise SystemExit("[error] 没有输入图")
    print(f"BiRefNet：{len(files)} 帧，模型 {args.model}，device={device}",
          flush=True)
    for k, p in enumerate(files, 1):
        rgb = Image.open(p).convert("RGB")
        small = rgb.resize((args.size, args.size), Image.LANCZOS)
        x = torch.from_numpy(np.asarray(small, np.float32) / 255.0)
        x = x.permute(2, 0, 1)[None].to(device)
        x = (x - mean) / std
        with torch.no_grad():
            out = model(x)[-1].sigmoid()[0, 0]
        alpha = out.detach().cpu().numpy()
        alpha = np.asarray(Image.fromarray((alpha * 255).astype(np.uint8))
                           .resize(rgb.size, Image.BILINEAR), np.float32) / 255.0
        rgba = np.dstack([np.asarray(rgb), np.round(alpha * 255).astype(np.uint8)])
        Image.fromarray(rgba).save(args.out / p.name)
        if k % 200 == 0 or k == len(files):
            print(f"  {k}/{len(files)}", flush=True)
    print(f"完成：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
