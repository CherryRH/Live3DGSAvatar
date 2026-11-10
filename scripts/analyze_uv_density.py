#!/usr/bin/env python
"""量化 UV 采样密度：每个三角面实际分到几个 texel。

## 为什么要看这个

模型里"一个有效 texel = 一个高斯"，所以**UV 图对网格的采样密度**决定了每个三角面
有几个采样点。采样不足的三角形只能靠邻接面的斑点覆盖；在斜视角下叠加层数不够，
alpha 就掉到 0.5 以下，表现为空洞 —— 这是大角度崩坏的一个来源，且与数据采集无关。

面积法可以离线预测（`count_f ≈ area_uv(f) · tex_size²`），与真实绑定一致时即可用来
评估"提高分辨率能改善多少"。

## 用法

```bash
python scripts/analyze_uv_density.py --tex-size 256 320 384
python scripts/analyze_uv_density.py --ply models/<对象>/<版本>/model.ply   # 对照真实绑定
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import live3dgsavatar  # noqa: F401


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="统计每个三角面的 texel 采样数")
    p.add_argument("--tex-size", type=int, nargs="+", default=[256],
                   help="要评估的分辨率（可多个）")
    p.add_argument("--ply", type=Path, default=None,
                   help="可选：与真实模型的绑定对照")
    p.add_argument("--out", type=Path, default=None, help="写出 JSON")
    return p.parse_args(argv)


def per_face_texels(uvs, uv_faces, tex_size: int):
    """按 UV 面积预测每个面的 texel 数（`[F]` float）。"""
    import numpy as np

    tri = np.asarray(uvs, dtype=np.float64)[np.asarray(uv_faces, dtype=np.int64)]
    area = 0.5 * np.abs(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]))
    return area * (tex_size ** 2)


def summarise(counts, tex_size: int) -> dict:
    """把逐面 texel 数压成可读指标。"""
    import numpy as np

    counts = np.asarray(counts, dtype=np.float64)
    return {
        "tex_size": tex_size,
        "predicted_texels": int(round(float(counts.sum()))),
        "per_face_median": round(float(np.median(counts)), 2),
        "per_face_p10": round(float(np.percentile(counts, 10)), 2),
        "per_face_p90": round(float(np.percentile(counts, 90)), 2),
        "zero_texel_faces": round(float((counts < 0.5).mean()), 4),
        "below_one_texel": round(float((counts < 1.0).mean()), 4),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import numpy as np

    from live3dgsavatar.config import load_config
    from live3dgsavatar.data.flame_topology import load as load_topo

    cfg = load_config()
    topo = load_topo(REPO_ROOT / "data" / "FLAME2020",
                     Path(str(cfg.get("paths.output_dir", "output"))) / "cache",
                     "cpu")
    uvs = topo["uvs"].numpy()
    uv_faces = topo["uv_faces"].numpy()
    print(f"  面数 {len(uv_faces)}   UV 顶点 {len(uvs)}")

    rows = []
    for tex_size in args.tex_size:
        counts = per_face_texels(uvs, uv_faces, tex_size)
        row = summarise(counts, tex_size)
        rows.append(row)
        print(f"  tex_size={tex_size:4d}: 有效 texel {row['predicted_texels']:7d}  "
              f"每面中位 {row['per_face_median']:5.2f}  "
              f"零 texel 面 {row['zero_texel_faces']:.1%}  "
              f"<1 texel {row['below_one_texel']:.1%}")

    if args.ply is not None:
        from live3dgsavatar.core.io import load_ply
        from live3dgsavatar.core.io.ply import read_embedded_config

        avatar = load_ply(args.ply, config=read_embedded_config(args.ply),
                          device="cpu")
        face_id = avatar.binding_face_id.numpy()
        n_faces = int(uv_faces.max()) + 1
        counts = np.bincount(face_id, minlength=n_faces).astype(np.float64)
        row = summarise(counts, int(avatar.config.tex_size))
        row["source"] = "binding"
        rows.append(row)
        print(f"  真实绑定（{args.ply.name}）：有效 texel {int(counts.sum())}  "
              f"每面中位 {row['per_face_median']:.2f}  "
              f"零 texel 面 {row['zero_texel_faces']:.1%}")

    if args.out is not None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps({"rows": rows}, indent=2, ensure_ascii=False),
            encoding="utf-8")
        print(f"  写出 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
