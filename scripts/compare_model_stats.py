#!/usr/bin/env python
"""比较两个模型的参数统计：定位"变糊"到底是哪一参数造成的。

渲染图上的差别往往能追到参数分布的差别上。最典型的两种：

- **尺度膨胀**：`scaling` 整体变大 → 表面被拉平、锐度下降、空洞被"糊住"；
- **不透明度抬高**：`opacity` 整体变大 → 表面更实，但同样会丢掉细节。

本脚本只读 PLY，不需要 GPU。

## 用法

```bash
python scripts/compare_model_stats.py --subject crh_1 --base-work run --work-name run2
```
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import live3dgsavatar  # noqa: F401


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="比较两个模型的参数统计")
    p.add_argument("--subject", default=None)
    p.add_argument("--base-work", default="run")
    p.add_argument("--work-name", default=None, help="要比较的版本")
    p.add_argument("--dataset", default=None, help="仅为兼容习惯，未使用")
    return p.parse_args(argv)


def load_stats(ply: Path) -> dict:
    """读一个模型，返回尺度/不透明度的分位数（都是**激活后**的物理量）。"""
    import numpy as np
    import torch

    from live3dgsavatar.core.io import load_ply
    from live3dgsavatar.core.io.ply import read_embedded_config

    avatar = load_ply(ply, config=read_embedded_config(ply), device="cpu")
    with torch.no_grad():
        opacity = torch.sigmoid(avatar.opacity).reshape(-1).numpy()
        scale = torch.exp(avatar.scaling).numpy().mean(axis=1)   # 三个轴的平均
    return {
        "n": int(opacity.size),
        "opacity_median": float(np.median(opacity)),
        "opacity_p90": float(np.percentile(opacity, 90)),
        "opacity_gt_0.5": float((opacity > 0.5).mean()),
        "scale_mm_median": float(np.median(scale)) * 1e3,
        "scale_mm_p90": float(np.percentile(scale, 90)) * 1e3,
        "scale_mm_max": float(scale.max()) * 1e3,
        "scale_gt_3mm": float((scale > 3e-3).mean()),
        "scale_gt_10mm": float((scale > 1e-2).mean()),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from live3dgsavatar.config import load_config
    from live3dgsavatar.data.models import models_root

    cfg = load_config()
    subject = args.subject or str(cfg.get("subject") or "")
    work = args.work_name or str(cfg.get("work_name") or "")
    if not (subject and work):
        raise SystemExit("[error] 需要 --subject 与 --work-name")
    root = models_root(cfg) / subject
    base_ply = root / args.base_work / "model.ply"
    new_ply = root / work / "model.ply"
    for ply in (base_ply, new_ply):
        if not ply.is_file():
            raise SystemExit(f"[error] 找不到模型：{ply}")

    a = load_stats(base_ply)
    b = load_stats(new_ply)
    print(f"对照 : {base_ply}")
    print(f"新模型: {new_ply}")
    print(f"\n  {'指标':22s} {'对照':>10s} {'新模型':>10s} {'倍数':>8s}")
    for key in ("opacity_median", "opacity_p90", "opacity_gt_0.5",
                "scale_mm_median", "scale_mm_p90", "scale_mm_max",
                "scale_gt_3mm", "scale_gt_10mm"):
        va, vb = a[key], b[key]
        ratio = (vb / va) if va not in (0, None) else float("nan")
        print(f"  {key:22s} {va:10.4f} {vb:10.4f} {ratio:8.2f}")
    print("\n  判读：`scale_mm_median` 明显变大 → 尺度膨胀（表面被拉平）；"
          "`opacity_*` 明显变大 → 不透明度抬高。两者都可能导致锐度下降。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
