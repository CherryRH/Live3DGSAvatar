#!/usr/bin/env python3
"""打印**解析后**的最终配置，便于排查「为什么路径不对」。

配置来源与优先级见 `docs/CONFIG.md`。本脚本不访问 GPU，也不需要数据集存在。

用法::

    python scripts/show_config.py              # 全部
    python scripts/show_config.py --section paths
    python scripts/show_config.py --sources    # 显示各文件是否被读到
    python scripts/show_config.py --json       # 机器可读
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import live3dgsavatar  # noqa: E402, F401  （导入即施加 numpy 兼容补丁）
from live3dgsavatar.config import CONFIG_DIR, load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="查看解析后的配置")
    p.add_argument("--section", default=None,
                   help="只看某一节（如 paths / render / model / runtime）")
    p.add_argument("--json", action="store_true", help="以 JSON 输出")
    p.add_argument("--sources", action="store_true",
                   help="显示配置文件来源与是否存在")
    p.add_argument("--config-dir", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(config_dir=args.config_dir)

    if args.sources:
        print(f"配置目录: {CONFIG_DIR if args.config_dir is None else args.config_dir}")
        for name in ("system.yaml", "render.yaml", "local.yaml"):
            path = (args.config_dir or CONFIG_DIR) / name
            mark = "✓ 已加载" if path.exists() else "· 不存在（跳过）"
            print(f"  {mark}  {path}")
        import os

        print("\n环境变量覆盖:")
        for var in ("LIVE3DGS_DATA_ROOT", "LIVE3DGS_OUTPUT_DIR",
                    "LIVE3DGS_REFERENCE_ROOT", "LIVE3DGS_SUBJECT",
                    "LIVE3DGS_DEVICE"):
            value = os.environ.get(var)
            print(f"  {var} = {value!r}" if value is not None
                  else f"  {var} 未设置")
        print()

    data = cfg.resolved()
    if args.section:
        if args.section not in data:
            print(f"[error] 没有名为 {args.section!r} 的配置节；"
                  f"可用：{sorted(data.keys())}")
            return 1
        data = {args.section: data[args.section]}

    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return 0

    _print_tree(data)
    _warn_missing(data)
    return 0


def _print_tree(data: dict, prefix: str = "") -> None:
    for key, value in data.items():
        label = f"{prefix}{key}"
        if isinstance(value, dict):
            print(f"{label}:")
            _print_tree(value, prefix + "  ")
        else:
            print(f"{label} = {value!r}")


def _warn_missing(data: dict) -> None:
    """对关键路径做存在性提醒 —— 只是提醒，不当成错误。"""
    checks = [
        ("paths.data_root", "数据集根目录"),
        ("paths.model_ply", "预训练模型"),
    ]
    missing = []
    for key, label in checks:
        value = data
        for part in key.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        if value is None:
            continue
        if not Path(value).exists():
            missing.append(f"  {label:<12} {value}")
    if missing:
        print("\n[提示] 以下路径当前不存在（换机器/未下载时正常）：")
        print("\n".join(missing))


if __name__ == "__main__":
    raise SystemExit(main())
