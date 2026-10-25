#!/usr/bin/env python
"""把数据集的图像打包成一个 `images.npy`，供训练时 **memmap 直读**。

## 为什么值得

训练时一个 batch 要解 10 张 512² PNG，实测约 **86 ms**（CPU 单线程），
而一步计算只有十几毫秒 —— GPU 大部分时间在等数据。打包成数组后，
读取退化成一次内存拷贝，实测单帧从 ~8 ms 降到 ~1 ms 量级。

## 用法

    python scripts/cache_images.py --subject crh_1
    python scripts/cache_images.py --data datasets/crh_2

产物：`<数据集>/images.npy`（uint8，`[N,H,W,3或4]`），行序与
`render_inputs.npz` 的 `names` 一致。`ManifestSource` 会自动检测并优先用它；
想强制走 PNG，构造时传 `use_cache=False`。

⚠️ 图像更新后要重新生成（缓存不会自动失效）。文件名固定，方便直接覆盖。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import live3dgsavatar  # noqa: F401


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="把数据集图像打包成 images.npy")
    p.add_argument("--subject", default=None, help="数据集名（默认取配置）")
    p.add_argument("--data", type=Path, default=None, help="直接给数据集目录")
    p.add_argument("--out", type=Path, default=None,
                   help="输出路径（默认 <数据集>/images.npy）")
    p.add_argument("--force", action="store_true", help="已存在时覆盖")
    return p.parse_args(argv)


def build_cache(data_dir: Path, out: Path) -> Path:
    """读 `<data_dir>/images/` 下的图，按 `names` 顺序写成 uint8 数组。"""
    import numpy as np
    from PIL import Image

    manifest = data_dir / "render_inputs.npz"
    if not manifest.is_file():
        raise FileNotFoundError(f"找不到 {manifest}")
    with np.load(manifest, allow_pickle=True) as d:
        if "names" not in d.files:
            raise ValueError(f"{manifest} 缺少 names 字段（无法对齐图像顺序）")
        names = [str(n) for n in d["names"]]

    arrays = []
    for name in names:
        path = data_dir / "images" / f"{name}.png"
        if not path.is_file():
            path = data_dir / "images" / name
        if not path.is_file():
            raise FileNotFoundError(f"缺少图像：{path}")
        arr = np.array(Image.open(path))
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        arrays.append(arr)

    shapes = {a.shape for a in arrays}
    if len(shapes) != 1:
        raise ValueError(f"图像尺寸不一致，无法打包：{sorted(shapes)}")
    stack = np.stack(arrays).astype(np.uint8)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, stack)
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from live3dgsavatar.config import load_config, resolve_data_dir

    cfg = load_config()
    if args.data is not None:
        data_dir = Path(args.data).expanduser().resolve()
    else:
        subject = args.subject or str(cfg.get("subject") or "").strip()
        if not subject:
            raise SystemExit("[error] 需要 --subject 或 --data")
        cfg.set("subject", subject)
        data_dir = resolve_data_dir(cfg)

    out = args.out or (data_dir / "images.npy")
    if out.exists() and not args.force:
        print(f"[error] {out} 已存在；确认覆盖请加 --force")
        return 1

    print(f"打包 {data_dir} → {out}")
    path = build_cache(data_dir, out)
    import numpy as np

    arr = np.load(path, mmap_mode="r")
    size_mib = path.stat().st_size / 2 ** 20
    print(f"完成：{arr.shape} {arr.dtype}  {size_mib:.0f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
