#!/usr/bin/env python
"""把 `configs/train_<recipe>.yaml` 的配方参数导出成 shell 变量。

用法：
    eval "$(python scripts/show_recipe.py --name avatar --shell)"
"""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="导出训练配方参数")
    p.add_argument("--name", required=True,
                   choices=("rgb", "base", "avatar"))
    p.add_argument("--shell", action="store_true")
    args = p.parse_args(argv)
    import yaml

    path = REPO_ROOT / "configs" / f"train_{args.name}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    flat: dict[str, object] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            for sub, sub_value in value.items():
                flat[f"DENSIFY_{sub.upper()}"] = sub_value
        else:
            flat[key.upper()] = value
    for key, value in flat.items():
        print(f"RECIPE_{key}={shlex.quote('' if value is None else str(value))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
