#!/usr/bin/env python
"""用 PyTorch 版 BiRefNet 从裁好的 RGB 帧生成人像 RGBA。

BiRefNet-portrait 的权重和自定义模型代码放在
`data/matting/birefnet_portrait/`，在 GPU 上单帧约 0.08 s。
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
                   default=Path("data/matting/birefnet_portrait"))
    p.add_argument("--clothes-model", type=Path,
                   default=Path("data/matting/clothes_seg"))
    p.add_argument("--keep-clothes", action="store_true",
                   help="保留衣服（默认去掉 T 恤/外套等衣物，保留脸/发/露出的皮肤）")
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
            "local_dir='data/matting/birefnet_portrait')\"")
    if args.out.exists() and any(args.out.glob("*.png")) and not args.force:
        raise SystemExit(f"[error] 输出已有图：{args.out}（加 --force）")
    args.out.mkdir(parents=True, exist_ok=True)
    for p in args.out.glob("*.png"):
        p.unlink()

    import cv2
    import torch
    import torch.nn.functional as F
    from transformers import (AutoImageProcessor,
                              AutoModelForImageSegmentation,
                              AutoModelForSemanticSegmentation)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[error] CUDA 不可用；需要 GPU 跑 BiRefNet")
    model = AutoModelForImageSegmentation.from_pretrained(
        str(args.model), trust_remote_code=True, local_files_only=True)
    model.eval().to(device)
    clothes_proc = clothes_model = None
    if not args.keep_clothes:
        if not (args.clothes_model / "model.safetensors").is_file():
            raise SystemExit(
                f"[error] 找不到衣服分割模型：{args.clothes_model}")
        clothes_proc = AutoImageProcessor.from_pretrained(
            str(args.clothes_model), local_files_only=True)
        clothes_model = AutoModelForSemanticSegmentation.from_pretrained(
            str(args.clothes_model), local_files_only=True).eval().to(device)
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
        if clothes_model is not None:
            ins = clothes_proc(images=rgb, return_tensors="pt")
            ins = {k: v.to(device) for k, v in ins.items()}
            with torch.no_grad():
                logits = clothes_model(**ins).logits
            seg = F.interpolate(logits, size=alpha.shape,
                                mode="bilinear", align_corners=False)
            seg = seg.argmax(dim=1)[0].cpu().numpy()
            # 4=Upper-clothes, 7=Dress；11=Face，2=Hair。
            hsv = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2HSV)
            sat = hsv[..., 1].astype(np.float32) / 255.0
            val = hsv[..., 2].astype(np.float32) / 255.0
            hair = (val < 0.45) & (sat < 0.6)
            m = alpha > 0.02
            ys = np.where(m)[0]
            lower = np.arange(alpha.shape[0])[:, None] > (
                int(ys.min()) + 0.45 * max(1, int(ys.max() - ys.min())))
            remove = ((seg == 4) | (seg == 7)) | (lower & (seg == 0))
            remove &= ~hair & (seg != 11) & m
            alpha[remove] = 0
        rgb_arr = np.asarray(rgb).copy()
        # 数据集约定：alpha 外的 RGB 置黑，避免任何读取方看到采集白墙。
        rgb_arr[alpha < 0.02] = 0
        rgba = np.dstack([rgb_arr, np.round(alpha * 255).astype(np.uint8)])
        Image.fromarray(rgba).save(args.out / p.name)
        if k % 500 == 0 or k == len(files):
            print(f"  {k}/{len(files)}", flush=True)
    print(f"完成：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
