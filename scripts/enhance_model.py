#!/usr/bin/env python
"""模型增强流程：找缺陷 → 外部精修 → 微调 → 落地 → 评测，一条命令跑完。

```
find_defects.py        选出问题视角，渲染 views/ 与 refs/
        ↓              （refs/ 是各视角基帧的真实图，作为精修的参考）
<外部精修器>           处理 views/，输出到 refined/（SyncFix 等）
        ↓
refine_from_images.py  用 refined/ 微调，按"PSNR − 锐度惩罚"选检查点
        ↓
promote                装进 models/<对象>/<工作名>_refined/
        ↓
render_test.py         留出集质量（基线 + 新模型）
compare_defects.py     同一批视角的结构缺口对比
```

## 为什么用子进程而不是 import

每一步都是独立可复用的脚本，也都能单独跑。这里只是把它们串起来，**不复制逻辑**；
任何一步都可以单独重跑（`--skip-find` / `--skip-syncfix` / `--skip-train`）。

## 断点续跑

状态写在 `<缺陷目录>/enhance_pipeline.json`：每步完成后记录状态与产物。重跑时若
产物仍在且没加 `--force`，该步自动跳过。

## 用法

```bash
python scripts/enhance_model.py --subject crh_1 --work-name run \\
    --syncfix-dir ~/Projects/SyncFix --syncfix-model <权重路径> \\
    --promote-name run_refined
```

`--dry-run` 只打印将要执行的命令；`--skip-syncfix` 表示 `refined/` 已经有了。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import live3dgsavatar  # noqa: F401

STATE_NAME = "enhance_pipeline.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="模型增强流程：缺陷 → 精修 → 微调 → 评测")
    p.add_argument("--subject", default=None)
    p.add_argument("--work-name", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--budget", type=int, default=16, help="缺陷视角数量")
    p.add_argument("--defects", type=Path, default=None,
                   help="缺陷目录（默认 output/defects/<对象>/<版本>/<数据集>）")

    # 精修
    p.add_argument("--syncfix-dir", type=Path, default=None,
                   help="SyncFix 仓库目录（含 inference.py）")
    p.add_argument("--syncfix-model", default=None, help="SyncFix 权重路径")
    p.add_argument("--syncfix-python", default=None,
                   help="跑 SyncFix 的解释器路径（优先级最高）")
    p.add_argument("--syncfix-env", default=None,
                   help="用 conda 环境跑 SyncFix，如 `syncfix`；不指定时若存在同名"
                        "环境则自动使用，否则用当前 python")
    p.add_argument("--syncfix-out", type=Path, default=None,
                   help="SyncFix 的原始输出目录（默认 <缺陷目录>/syncfix_raw）")
    p.add_argument("--syncfix-arg", action="append", default=[],
                   help="额外透传给推理脚本的参数（可重复，如 --syncfix-arg=--n_per_pass=1）")

    # 微调
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--ssim-weight", type=float, default=0.2)
    p.add_argument("--refined-per-batch", type=int, default=4)
    p.add_argument("--real-per-batch", type=int, default=4)
    p.add_argument("--promote-name", default=None,
                   help="落地目录名（默认 <版本>_refined）")

    # 评测
    p.add_argument("--eval-frames", type=int, default=-1,
                   help="留出集评测帧数；-1 表示全部")
    p.add_argument("--no-render-test", action="store_true",
                   help="跳过留出集渲染测试")

    # 控制
    p.add_argument("--skip-find", action="store_true")
    p.add_argument("--skip-syncfix", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--force", action="store_true", help="忽略已完成状态，重跑各步")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


# ----------------------------------------------------------------- 工具 --

def run(cmd: list[str], *, cwd: Path | None = None, dry_run: bool = False) -> None:
    """执行一条命令；失败时抛出带上下文的异常。"""
    text = " ".join(str(c) for c in cmd)
    print(f"\n$ {text}" + (f"    (cwd={cwd})" if cwd else ""), flush=True)
    if dry_run:
        return
    proc = subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None)
    if proc.returncode != 0:
        raise RuntimeError(f"命令失败（退出码 {proc.returncode}）：{text}")


def load_state(path: Path) -> dict:
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def save_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False),
                    encoding="utf-8")


def collect_refined(raw_dir: Path, target_dir: Path, labels: list[str]) -> list[str]:
    """把精修器的输出整理成 `refined/<label>.png`。

    优先按文件名匹配（stem）；匹配不上再按排序顺序对应，并在返回值里标注。
    Returns: 整理好的 label 列表
    """
    if not raw_dir.is_dir():
        raise RuntimeError(f"精修输出目录不存在：{raw_dir}")
    found = {p.stem: p for p in raw_dir.rglob("*.png")}
    target_dir.mkdir(parents=True, exist_ok=True)
    done: list[str] = []
    matched = [lab for lab in labels if lab in found]
    if matched:
        for lab in matched:
            shutil.copyfile(found[lab], target_dir / f"{lab}.png")
            done.append(lab)
        return done
    # 文件名对不上：按排序顺序一一对应（打印警告，提示用户复核）
    ordered = sorted(found.values())
    if len(ordered) < len(labels):
        raise RuntimeError(
            f"精修输出只有 {len(ordered)} 张，少于 {len(labels)} 个视角；"
            "请确认精修器是否完整跑完")
    print(f"  [warn] 精修输出文件名与视角标签不一致，按排序顺序对应 {len(labels)} 张")
    for lab, src in zip(labels, ordered):
        shutil.copyfile(src, target_dir / f"{lab}.png")
        done.append(lab)
    return done


def syncfix_prefix(args) -> list[str]:
    """SyncFix 的解释器前缀。

    优先级：`--syncfix-python` > `--syncfix-env` > 自动检测名为 `syncfix` 的
    conda 环境 > 当前 `python`。

    为什么需要：SyncFix 通常有自己的环境（依赖 imageio 等），而本流程跑在
    训练环境里。用 `conda run --no-capture-output` 保证输出能实时透传。
    """
    if args.syncfix_python:
        return [str(args.syncfix_python)]
    env = args.syncfix_env or _detect_conda_env("syncfix")
    if env:
        return ["conda", "run", "--no-capture-output", "-n", str(env), "python"]
    return ["python"]


def _detect_conda_env(name: str) -> str | None:
    """若存在名为 `name` 的 conda 环境则返回它，否则 None。"""
    import os

    conda = os.environ.get("CONDA_EXE")
    if not conda:
        return None
    try:
        out = subprocess.run([conda, "env", "list"], capture_output=True,
                             text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        parts = line.split()
        if parts and parts[0] == name:
            return name
    return None


def syncfix_command(args, *, views: int) -> list[str]:
    """拼出 SyncFix 的推理命令（含解释器前缀）。"""
    if args.syncfix_dir is None:
        raise RuntimeError("需要 --syncfix-dir（或加 --skip-syncfix 复用已有精修图）")
    script = Path(args.syncfix_dir) / "inference.py"
    cmd = syncfix_prefix(args) + [str(script),
           "--input_image", "{views}", "--ref_image", "{refs}",
           "--output_dir", "{raw}",
           "--n_per_pass", str(max(1, views)), "--ref_size", "1",
           "--height", "512", "--width", "512"]
    if args.syncfix_model:
        cmd += ["--model_path", str(args.syncfix_model)]
    cmd += [str(a) for a in args.syncfix_arg]
    return cmd


def read_report(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def render_metrics(report: dict | None) -> dict:
    if not report:
        return {}
    q = report.get("quality", {})
    out = {}
    for key, name in (("psnr_masked", "psnr"), ("psnr_full", "psnr_full")):
        if key in q:
            out[name] = round(float(q[key]["median"]), 2)
    if "sharpness" in q:
        out["sharp_ratio"] = q["sharpness"]["ratio"]
    return out


def defect_metrics(report: dict | None) -> dict:
    if not report:
        return {}
    s = report.get("summary", {})
    out = {}
    if "hole_base" in s:
        out["hole_before"] = s["hole_base"]["mean"]
        out["hole_after"] = s["hole_new"]["mean"]
    if "hole_rel" in s:
        out["hole_rel"] = s["hole_rel"]
    if "outside_base" in s:
        out["outside_before"] = s["outside_base"]["mean"]
        out["outside_after"] = s["outside_new"]["mean"]
    if "sharp_rel" in s:
        out["sharp_rel"] = s["sharp_rel"]
    return out


# ----------------------------------------------------------------- 主流程 --

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from live3dgsavatar.config import load_config
    from live3dgsavatar.data.datasets import resolve_dataset_dir

    cfg = load_config()
    subject = args.subject or str(cfg.get("subject") or "")
    work = args.work_name or str(cfg.get("work_name") or "")
    if not (subject and work):
        raise SystemExit("[error] 需要 --subject 与 --work-name")
    dataset = args.dataset or subject
    try:
        data_dir = resolve_dataset_dir(cfg, dataset)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"[error] {e}") from None

    defects_dir = Path(args.defects or (
        Path(str(cfg.get("paths.output_dir", "output"))) / "defects"
        / subject / work / dataset))
    refined_dir = defects_dir / "refined"
    syncfix_raw = Path(args.syncfix_out or (defects_dir / "syncfix_raw"))
    new_work = args.promote_name or f"{work}_refined"
    state_path = defects_dir / STATE_NAME
    state = load_state(state_path)

    def done(stage: str) -> bool:
        return (not args.force and state.get(stage, {}).get("status") == "done")

    def mark(stage: str, **info) -> None:
        state[stage] = {"status": "done", **info}
        save_state(state_path, state)

    py = sys.executable
    scripts = REPO_ROOT / "scripts"
    base_cmd = [py, str(scripts)]

    print("=" * 70)
    print(f"模型增强流程：{subject}/{work} → {new_work}")
    print(f"缺陷目录：{defects_dir}")
    print("=" * 70)

    # ---- 1. 找缺陷 ----
    if args.skip_find or done("find_defects"):
        print("\n[1/4] 找缺陷：跳过（" +
              ("--skip-find" if args.skip_find else "已完成") + "）")
    else:
        print("\n[1/4] 找缺陷")
        run([py, str(scripts / "find_defects.py"),
             "--subject", subject, "--work-name", work,
             "--dataset", dataset, "--budget", str(args.budget),
             "--out", str(defects_dir)], dry_run=args.dry_run)
        mark("find_defects", defects_dir=str(defects_dir))

    selected = defects_dir / "selected.json"
    labels: list[str] = []
    if selected.is_file():
        labels = json.loads(selected.read_text(encoding="utf-8"))["views"]["chosen"]
    elif not args.dry_run:
        raise SystemExit(f"[error] 找不到 {selected}（先跑 find_defects.py）")

    # ---- 2. 外部精修 ----
    if args.skip_syncfix or done("syncfix"):
        print("\n[2/4] 外部精修：跳过（" +
              ("--skip-syncfix" if args.skip_syncfix else "已完成") + "）")
    else:
        print(f"\n[2/4] 外部精修（{len(labels)} 个视角）")
        prefix = syncfix_prefix(args)
        print(f"  解释器：{' '.join(prefix)}", flush=True)
        if not args.dry_run:
            probe = prefix + ["-c", "import imageio"]
            if subprocess.run(probe, capture_output=True).returncode != 0:
                raise SystemExit(
                    "[error] SyncFix 的 Python 无法 import imageio。\n"
                    "        请用 --syncfix-env <环境名> 或 --syncfix-python <解释器> "
                    "指定它的运行环境")
        cmd = syncfix_command(args, views=len(labels))
        cmd = [c.format(views=defects_dir / "views", refs=defects_dir / "refs",
                        raw=syncfix_raw) for c in cmd]
        run(cmd, cwd=Path(args.syncfix_dir) if args.syncfix_dir else None,
            dry_run=args.dry_run)
        if not args.dry_run:
            got = collect_refined(syncfix_raw, refined_dir, labels)
            print(f"  整理出 {len(got)} 张精修图 → {refined_dir}")
            mark("syncfix", raw=str(syncfix_raw), refined=str(refined_dir),
                 count=len(got))

    # ---- 3. 微调 + 落地 ----
    if args.skip_train or done("refine"):
        print("\n[3/4] 微调：跳过（" +
              ("--skip-train" if args.skip_train else "已完成") + "）")
    else:
        print("\n[3/4] 微调（颜色 + 不透明度 + 尺度，按留出真值与锐度选点）")
        run([py, str(scripts / "refine_from_images.py"),
             "--subject", subject, "--work-name", work, "--dataset", dataset,
             "--defects", str(defects_dir),
             "--steps", str(args.steps), "--ssim-weight", str(args.ssim_weight),
             "--refined-per-batch", str(args.refined_per_batch),
             "--real-per-batch", str(args.real_per_batch),
             "--promote", "--promote-name", new_work,
             "--force"], dry_run=args.dry_run)
        mark("refine", work=new_work)

    # ---- 4. 评测 ----
    print("\n[4/4] 评测")
    results: dict = {"subject": subject, "base_work": work, "new_work": new_work}
    if args.no_render_test:
        print("  留出集渲染测试：跳过（--no-render-test）")
    else:
        offset = None
        try:
            import numpy as np

            with np.load(Path(data_dir) / "render_inputs.npz",
                         allow_pickle=True) as d:
                total = int(d["mesh"].shape[0])
            ratio = float(cfg.get("train.pipeline.test_ratio", 0.1))
            offset = total - max(1, int(round(total * ratio)))
        except Exception:                                  # noqa: BLE001
            print("  [warn] 无法推算留出段起点，渲染全部帧")
        for name in (work, new_work):
            cmd = [py, str(scripts / "render_test.py"),
                   "--subject", subject, "--work-name", name, "--dataset", dataset,
                   "--frames", str(args.eval_frames)]
            if offset is not None:
                cmd += ["--frame-offset", str(offset)]
            try:
                run(cmd, dry_run=args.dry_run)
            except RuntimeError as e:
                print(f"  [warn] {name} 渲染测试失败：{e}")
            report_path = (Path(str(cfg.get("paths.output_dir", "output")))
                           / "render_test" / subject / name / dataset
                           / "report.json")
            results.setdefault("render_test", {})[name] = render_metrics(
                read_report(report_path))
    if not args.dry_run:
        run([py, str(scripts / "compare_defects.py"),
             "--subject", subject, "--work-name", new_work,
             "--base-work", work, "--dataset", dataset,
             "--out", str(defects_dir / f"compare_{new_work}")],
            dry_run=args.dry_run)
        results["defects"] = defect_metrics(
            read_report(defects_dir / f"compare_{new_work}"
                        / "compare_defects.json"))

    # ---- 汇总 ----
    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    base_rt = results.get("render_test", {}).get(work, {})
    new_rt = results.get("render_test", {}).get(new_work, {})
    if base_rt or new_rt:
        print(f"  留出集 PSNR  : {base_rt.get('psnr', '—')} → "
              f"{new_rt.get('psnr', '—')} dB")
        print(f"  锐度比       : {base_rt.get('sharp_ratio', '—')} → "
              f"{new_rt.get('sharp_ratio', '—')}")
    d = results.get("defects", {})
    if d:
        print(f"  结构缺口     : {d.get('hole_before', '—')} → "
              f"{d.get('hole_after', '—')}")
        print(f"  云外溢       : {d.get('outside_before', '—')} → "
              f"{d.get('outside_after', '—')}")

    state["results"] = results
    save_state(state_path, state)
    print(f"\n状态与结果：{state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
