#!/usr/bin/env python
"""消融：限制/归零每个高斯的切空间偏移（`xyz`），检验"位置"是不是缺陷的来源。

## 为什么做这个

容量翻倍的对照实验给出的是负结果：采样点从 6 万加到 12.6 万，缺陷视角的结构缺口
没有变化（0.170 → 0.173），说明**缺的不是采样点**。

那么剩下的嫌疑就是**位置**：每个高斯的 `xyz` 是自由参数，实测可见高斯的中位偏移
约 27 mm，而它只在被观测的视角下受约束。如果把这个偏移压小/归零后缺陷明显变化，
就说明"点云被训练拉离了网格"是主因，后续该做的是位置约束，而不是继续加采样点。

本脚本**不训练**，只改参数再存一份，用现有的 `compare_defects` 对比即可。

## 用法

```bash
# 偏移归零：所有点回到绑定面上
python scripts/ablate_offsets.py --subject crh_1 --work-name run --mode zero

# 偏移裁剪到 2 mm
python scripts/ablate_offsets.py --subject crh_1 --work-name run --mode clip --clip-mm 2

# 偏移缩小到 50%
python scripts/ablate_offsets.py --subject crh_1 --work-name run --mode scale --scale 0.5
```

产物装进 `models/<对象>/<版本>_<后缀>/`，之后：

```bash
python scripts/compare_defects.py --subject crh_1 --work-name run_zero --base-work run
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
    p = argparse.ArgumentParser(description="消融：限制高斯偏移")
    p.add_argument("--subject", default=None)
    p.add_argument("--work-name", default=None)
    p.add_argument("--mode", choices=("zero", "clip", "scale"), required=True)
    p.add_argument("--clip-mm", type=float, default=2.0,
                   help="clip 模式：偏移上限（毫米）")
    p.add_argument("--scale", type=float, default=0.5,
                   help="scale 模式：偏移缩放系数")
    p.add_argument("--suffix", default=None,
                   help="产物目录后缀（默认 <mode>）")
    return p.parse_args(argv)


def limit_offsets(xyz, mode: str, *, clip_mm: float = 2.0,
                  scale: float = 0.5):
    """按模式改写偏移（原地返回新张量）。

    Args:
        xyz: `[N, 3]` 切空间偏移（米）
        mode: ``zero`` 归零 / ``clip`` 按模长裁剪 / ``scale`` 整体缩放
        clip_mm: clip 上限（毫米）
        scale: scale 系数

    Returns:
        新的 `[N, 3]` 张量（不修改输入）
    """
    import torch

    if mode == "zero":
        return torch.zeros_like(xyz)
    if mode == "scale":
        return xyz * float(scale)
    if mode == "clip":
        limit = float(clip_mm) / 1000.0
        if limit <= 0:
            raise ValueError(f"clip_mm 必须为正，实际 {clip_mm}")
        norm = xyz.norm(dim=1, keepdim=True).clamp(min=1e-9)
        factor = (limit / norm).clamp(max=1.0)
        return xyz * factor
    raise ValueError(f"未知模式 {mode!r}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import shutil

    import torch

    from live3dgsavatar.config import load_config
    from live3dgsavatar.core.io import load_ply
    from live3dgsavatar.core.io.ply import read_embedded_config, save_ply
    from live3dgsavatar.data.models import models_root

    cfg = load_config()
    subject = args.subject or str(cfg.get("subject") or "")
    work = args.work_name or str(cfg.get("work_name") or "")
    if not (subject and work):
        raise SystemExit("[error] 需要 --subject 与 --work-name")
    root = models_root(cfg) / subject
    src = root / work / "model.ply"
    if not src.is_file():
        raise SystemExit(f"[error] 找不到模型：{src}")

    name = f"{work}_{args.suffix or args.mode}"
    avatar = load_ply(src, config=read_embedded_config(src), device="cpu")
    with torch.no_grad():
        before = avatar.xyz.norm(dim=1)
        avatar.xyz.copy_(limit_offsets(avatar.xyz, args.mode,
                                       clip_mm=args.clip_mm,
                                       scale=args.scale))
        after = avatar.xyz.norm(dim=1)

    target = root / name
    target.mkdir(parents=True, exist_ok=True)
    save_ply(avatar, target / "model.ply")
    cfg_src = src.parent / "config.yaml"
    if cfg_src.is_file():
        shutil.copyfile(cfg_src, target / "config.yaml")
    info = {
        "source": str(src), "mode": args.mode,
        "clip_mm": args.clip_mm, "scale": args.scale,
        "offset_median_mm_before": round(float(before.median()) * 1e3, 2),
        "offset_median_mm_after": round(float(after.median()) * 1e3, 2),
        "offset_p90_mm_after": round(float(after.quantile(0.9)) * 1e3, 2),
    }
    (target / "ablation.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"源模型 : {src}")
    print(f"模式   : {args.mode}")
    print(f"偏移中位: {info['offset_median_mm_before']} → "
          f"{info['offset_median_mm_after']} mm（P90 {info['offset_p90_mm_after']}）")
    print(f"产物   : {target}")
    print(f"\n对比：python scripts/compare_defects.py --subject {subject} "
          f"--work-name {name} --base-work {work} --dataset {subject}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
