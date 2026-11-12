#!/usr/bin/env python
"""用 SyncFix 对整幅渲染图做联合修复（全图，不走掩码）。

这是后训练的第二种标签来源：
    find_defects 的 views/  -> SyncFix 多视图 -> <out>/*.png -> refine

⚠️ SyncFix 是独立仓库/环境（默认 /home/test/Projects/SyncFix），当前只做
接口与守卫，不在环境未就绪时调用。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SyncFix 全图联合修复标签")
    p.add_argument("--defects", type=Path, required=True,
                   help="find_defects 产物目录（含 views/、refs/）")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repo", default="/home/test/Projects/SyncFix")
    p.add_argument("--python", default="syncfix",
                   help="SyncFix 的 conda 环境名或 python 可执行文件")
    p.add_argument("--model", default="/home/test/Projects/SyncFix/models/paper")
    p.add_argument("--n-per-pass", type=int, default=2)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--ready", action="store_true",
                   help="确认 SyncFix 环境已配置；否则拒绝调用")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.ready:
        raise SystemExit(
            "[error] SyncFix 环境未配置（syncfix_ready=false），本次不调用。\n"
            "        配好环境后在 configs/train.yaml 里设 post_train.syncfix_ready: true")
    repo = Path(args.repo).expanduser()
    infer = repo / "inference.py"
    model = Path(args.model).expanduser()
    views = Path(args.defects) / "views"
    refs = Path(args.defects) / "refs"
    if not infer.is_file():
        raise SystemExit(f"[error] 找不到 SyncFix inference.py：{infer}")
    if not model.exists():
        raise SystemExit(f"[error] 找不到 SyncFix 权重：{model}")
    if not views.is_dir():
        raise SystemExit(f"[error] 没有待修复的 views：{views}")
    args.out.mkdir(parents=True, exist_ok=True)

    py = Path(args.python).expanduser()
    if py.is_file():
        pycmd = [str(py)]
    else:
        pycmd = ["conda", "run", "-n", args.python,
                 "--no-capture-output", "python"]
    cmd = [*pycmd, str(infer),
           "--input_image", str(views),
           "--ref_image", str(refs),
           "--model_path", str(model),
           "--output_dir", str(args.out),
           "--n_per_pass", str(args.n_per_pass),
           "--ref_size", "1",
           "--height", str(args.height),
           "--width", str(args.width)]
    print("$ " + " ".join(cmd), flush=True)
    if args.dry_run:
        return 0
    p = subprocess.run(cmd, cwd=str(repo), check=False)
    if p.returncode != 0:
        print("[error] SyncFix 推理失败；检查 syncfix conda 环境与权重",
              file=sys.stderr)
    return p.returncode


if __name__ == "__main__":
    raise SystemExit(main())
