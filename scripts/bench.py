#!/usr/bin/env python3
"""综合性能测试：纯渲染 / RDF 跟踪 / RDF 驱动渲染（**不需要前端与后端服务**）。

三段各自独立，可以只跑其中一段：

===============  ================================================  ==========
段               测什么                                            需要
===============  ================================================  ==========
``render``       从已有参数到出图（deform + 光栅化）                GPU
``rdf``          一帧画面送追踪服务、拿回参数与网格的往返耗时         GPU + RDF 环境
``full``         完整链路：取帧 → 跟踪 → 转换 → 渲染                 GPU + RDF 环境
===============  ================================================  ==========

## 用法

::

    conda activate live3dgs

    # 三段全跑：用 <数据集> 驱动 <对象>/<版本> 的模型，各 100 帧
    python scripts/bench.py \\
        --subject <对象> --work-name <版本> --data-subject <数据集> \\
        --frames 100

    # 只测纯渲染（最常用：验证机器性能有没有退化）
    python scripts/bench.py --subject <对象> --work-name <版本> --parts render

    # 追踪服务已经在别处跑着，复用而不重复启动（省 30–60 s 的模型加载）
    python scripts/bench.py --data-subject <数据集> --parts rdf,full \\
        --no-launch-rdf --rdf-port 8980

## 为什么能一次跑完三段

渲染要 CUDA 扩展（``live3dgs`` 环境），追踪要 ``flame_solver``
（``realdenseface`` 环境，cpython-311）。两者**不能同进程**，
所以本脚本把追踪服务作为**子进程**拉起，本进程只当客户端 ——
这也正是产品里的部署形态，测出来的数字与真实链路一致。

追踪服务用哪个解释器：``--rdf-python`` > ``LIVE3DGS_RDF_PYTHON`` >
自动探测 ``realdenseface`` 环境。找不到时会明确报错并给出设置方法。

## 指标口径

- **纯渲染**：每帧 ``deform + 光栅化`` 的墙钟时间（`render()` 内部已同步 CUDA）。
- **RDF 跟踪**：发一帧图画 + 读回结果的往返时间，含 IPC（回环，可忽略）。
- **RDF 驱动渲染**：端到端每帧时间；再按 ``跟踪`` 与 ``渲染`` 拆开，
  便于判断瓶颈在哪一段。
- 所有段都先跑 ``--warmup`` 帧预热（首帧含 CUDA 初始化、显存分配），
  预热帧**不计入**统计。
- 帧号循环覆盖整个数据集，避免只测到某一小段。

## 产物

终端打印一张表；``--json <路径>`` 会把同样的数字写成 JSON（便于对比多次结果）。
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import live3dgsavatar  # noqa: E402, F401  （导入即施加 numpy 兼容补丁）

PARTS = ("render", "rdf", "full")
DEFAULT_PORT = 8980


# ------------------------------------------------------------------ 统计 --


def summarize(samples_ms: list[float]) -> dict:
    """把一组耗时（毫秒）汇总成一组指标。

    用**中位数**作为主指标：单帧偶发的调度抖动会把均值拉高，
    而中位数更能代表"用户实际感受到的速度"。p95 用来观察长尾。
    """
    if not samples_ms:
        return {}
    xs = sorted(float(v) for v in samples_ms)
    mean = sum(xs) / len(xs)
    return {
        "n": len(xs),
        "mean_ms": mean,
        "median_ms": statistics.median(xs),
        "p95_ms": xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))],
        "min_ms": xs[0],
        "max_ms": xs[-1],
        # 帧率按中位数折算：与"实际观感"一致
        "fps": 1000.0 / statistics.median(xs) if statistics.median(xs) > 0 else 0.0,
    }


# -------------------------------------------------------------- 环境探测 --


def default_rdf_python() -> Path | None:
    """探测 `realdenseface` 环境的解释器。

    顺序：显式环境变量 → 当前环境的兄弟目录 → 常见 conda 安装位置。
    找不到返回 ``None``（由调用方给出可执行的提示）。
    """
    explicit = os.environ.get("LIVE3DGS_RDF_PYTHON")
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.is_file() else None

    candidates: list[Path] = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        # <base>/envs/live3dgs → <base>/envs/realdenseface
        candidates.append(Path(conda_prefix).parent / "realdenseface" / "bin" / "python")
    for base in ("~/miniconda3", "~/anaconda3", "~/miniforge3"):
        candidates.append(Path(base).expanduser() / "envs" / "realdenseface"
                          / "bin" / "python")
    for p in candidates:
        if p.is_file():
            return p
    return None


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) == 0


def wait_for_service(host: str, port: int, proc: subprocess.Popen | None,
                     timeout_s: float = 300.0) -> None:
    """等到追踪服务可以连接。

    超时或子进程提前退出都**立刻报错**（带上它的退出码），
    否则会在这里干等 5 分钟，看不出是环境问题还是模型在加载。
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _port_open(host, port):
            return
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"追踪服务提前退出（退出码 {proc.returncode}）。"
                "请检查上面的日志；常见原因是 RDF 资产缺失（weights/ 未部署）、"
                "环境不对（缺 flame_solver）或端口被占用。\n"
                "        可单独验证：realdenseface 环境下运行 "
                "python scripts/rdf_bridge.py --check-deps")
        time.sleep(0.3)
    raise RuntimeError(
        f"等待追踪服务超时（{timeout_s:.0f}s，{host}:{port}）。"
        "首次启动含 torch.compile，可能需要更久；也可先手动起好服务再 "
        "加 --no-launch-rdf 复用。")


def check_tracker_ready(python: Path, rdf_root: Path | None = None) -> None:
    """先用 `--check-deps` 做一次预检，**在启动之前**把可诊断的问题说清楚。

    不做这一步的后果：追踪服务在构建模型时崩在第三层调用栈里（例如
    `torch.load` 找不到权重），而我们只看到"子进程退出码 1" ——
    部署到新机器时几乎必然踩到（权重目录没一起拷过去）。
    """
    cmd = [str(python), str(REPO_ROOT / "scripts" / "rdf_bridge.py"),
           "--check-deps"]
    if rdf_root is not None:
        cmd += ["--rdf-root", str(rdf_root)]
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit(
            "[error] 追踪服务自检未通过：\n"
            + (res.stdout + res.stderr).rstrip()
            + "\n\n        RDF 仓库需要 weights/ 与 configs/ 一起部署；"
              "只测渲染可用 --parts render。")


def launch_tracker(args, host: str, port: int) -> subprocess.Popen:
    """把追踪服务拉起来（子进程，`realdenseface` 环境）。"""
    python = Path(args.rdf_python) if args.rdf_python else default_rdf_python()
    if python is None:
        raise SystemExit(
            "[error] 找不到 realdenseface 环境。请指定解释器：\n"
            "        --rdf-python /path/to/envs/realdenseface/bin/python\n"
            "        或设置环境变量 LIVE3DGS_RDF_PYTHON\n"
            "        （只测纯渲染则加 --parts render，不需要它）")

    check_tracker_ready(python, args.rdf_root)

    cmd = [str(python), str(REPO_ROOT / "scripts" / "rdf_bridge.py"),
           "--host", host, "--port", str(port)]
    if args.rdf_root is not None:
        cmd += ["--rdf-root", str(args.rdf_root)]
    if args.rdf_no_compile:
        cmd.append("--no-compile")
    print(f"  启动追踪服务：{python}", flush=True)
    # 继承 stdout/stderr：让用户看到它自己的进度（模型加载约 30–60 s）
    return subprocess.Popen(cmd, cwd=str(REPO_ROOT))


# ------------------------------------------------------------------ 装配 --


def _require_gpu() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "[error] 本脚本需要 GPU（渲染无法降级到 CPU）。\n"
            "        请在能访问 GPU 的终端运行；仅检查配置可用 --dry-run。")


def _session(cfg, subject: str, work_name: str):
    """构造并加载一个与 GUI 同路径的 Session。"""
    from live3dgsavatar.app.session import Session

    session = Session(cfg)
    session.load_model(subject, work_name)
    return session


def _frames(cfg, data_subject: str):
    """取帧器：优先 `input/`，无则 `images/`（与实时驱动一致）。"""
    from live3dgsavatar.data.datasets import resolve_dataset_dir
    from live3dgsavatar.tracking.frame_source import DatasetFrames

    data_dir = resolve_dataset_dir(cfg, data_subject)
    frames = DatasetFrames(data_dir, fps=0,
                           subdir=str(cfg.get("tracking.frame_subdir", "auto")))
    return frames, data_dir


def _pick_indices(total: int, n: int) -> list[int]:
    """要测的帧号：不足就循环，避免 `n > total` 时只测一遍。"""
    return [i % total for i in range(n)] if total else []


# -------------------------------------------------------------- 三段实现 --


def bench_render(session, n: int, warmup: int) -> dict:
    """纯渲染：给已有参数、只测出图。"""
    total = int(session.scene.total_frames)
    idx = _pick_indices(total, n)

    for i in _pick_indices(total, warmup):
        session.jump_frame(i)
        session.render()

    frame_ms, deform_ms, raster_ms = [], [], []
    for i in idx:
        session.jump_frame(i)
        _rgb, _target, timings = session.render()
        frame_ms.append(timings["frame"])
        deform_ms.append(timings["deform"])
        raster_ms.append(timings["rasterize"])

    out = summarize(frame_ms)
    out.update({
        "part": "render",
        "frames": n,
        "deform_ms": summarize(deform_ms)["median_ms"],
        "rasterize_ms": summarize(raster_ms)["median_ms"],
        "peak_memory_mib": session.peak_memory_mib,
        "output_size": list(session.output_size()),
    })
    return out


def bench_rdf(client, frames, n: int) -> dict:
    """RDF 跟踪：一帧画面 → 参数与顶点，测往返。"""
    total = frames.num_frames
    idx = _pick_indices(total, n)

    lat = []
    for i in idx:
        rgb = frames.read_at(i)
        if rgb is None:
            continue
        t0 = time.perf_counter()
        client.request(rgb, seq=i)
        lat.append((time.perf_counter() - t0) * 1000.0)

    out = summarize(lat)
    out.update({"part": "rdf", "frames": len(lat), "frame_source": frames.subdir})
    return out


def bench_full(session, client, frames, n: int, warmup: int) -> dict:
    """完整链路：取帧 → 跟踪 → 转换（补牙齿/参数映射）→ 渲染。"""
    import time as _time

    from live3dgsavatar.tracking.live_source import LiveSource

    total = frames.num_frames
    idx = _pick_indices(total, n)

    live = LiveSource(session.cfg, session.device, client=client, frames=frames)
    live.set_enabled(True)
    session.live = live
    session.params.source_kind = "live"

    def one(i: int) -> tuple[float, float, float]:
        rgb = frames.read_at(i)
        t0 = _time.perf_counter()
        obs = client.request(rgb, seq=i)
        track_ms = (_time.perf_counter() - t0) * 1000.0
        live.ingest(obs)
        t1 = _time.perf_counter()
        session.render()
        render_ms = (_time.perf_counter() - t1) * 1000.0
        return track_ms, render_ms, (_time.perf_counter() - t0) * 1000.0

    for i in _pick_indices(total, warmup):
        one(i)

    track_ms, render_ms, total_ms = [], [], []
    for i in idx:
        a, b, c = one(i)
        track_ms.append(a)
        render_ms.append(b)
        total_ms.append(c)

    out = summarize(total_ms)
    out.update({
        "part": "full",
        "frames": len(total_ms),
        "track_ms": summarize(track_ms)["median_ms"],
        "render_ms": summarize(render_ms)["median_ms"],
        "frame_source": frames.subdir,
    })
    return out


# ------------------------------------------------------------------ 输出 --


def print_row(tag: str, r: dict) -> None:
    if not r:
        print(f"  {tag}：无数据")
        return
    head = (f"  {tag:<12} {r['frames']:>4} 帧  "
            f"中位 {r['median_ms']:7.2f} ms   {r['fps']:6.1f} FPS   "
            f"p95 {r['p95_ms']:7.2f} ms")
    detail = ""
    if r["part"] == "render":
        detail = (f"   (deform {r['deform_ms']:.2f} + 光栅 {r['rasterize_ms']:.2f}"
                  f" · 峰值 {r['peak_memory_mib']:.0f} MiB)")
    elif r["part"] == "full":
        detail = (f"   (跟踪 {r['track_ms']:.2f} + 渲染 {r['render_ms']:.2f})")
    elif r["part"] == "rdf":
        detail = f"   (画面来源 {r['frame_source']}/)"
    print(head + detail)


def plan(args, cfg) -> dict:
    """`--dry-run`：只解析配置与路径，把要做的事打印出来（不需要 GPU）。"""
    from live3dgsavatar.data.datasets import resolve_dataset_dir
    from live3dgsavatar.data.models import list_models

    subject = args.subject or str(cfg.get("subject") or "")
    work = args.work_name or str(cfg.get("work_name") or "")
    if not subject and not work:
        infos = list_models(cfg)
        subject, work = (infos[0].subject, infos[0].work_name) if infos else ("", "")

    data_subject = args.data_subject or subject
    out = {"parts": args.parts, "frames": args.frames,
           "subject": subject, "work_name": work,
           "data_subject": data_subject}
    if subject and work:
        from live3dgsavatar.config import resolve_model_ply

        cfg.set("subject", subject)
        cfg.set("work_name", work)
        out["model"] = str(resolve_model_ply(cfg))
    if data_subject:
        try:
            out["dataset"] = str(resolve_dataset_dir(cfg, data_subject))
        except (FileNotFoundError, ValueError) as e:
            # 只测纯渲染时不需要数据集：别让它挡住 --dry-run
            out["dataset_error"] = str(e).splitlines()[0]
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="综合性能测试：纯渲染 / RDF 跟踪 / RDF 驱动渲染",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="各段可独立运行；只测纯渲染时不需要 realdenseface 环境。")
    p.add_argument("--subject", default=None,
                   help="模型的对象名（默认取配置，其次列表第一个）")
    p.add_argument("--work-name", default=None, help="模型的版本名（同上）")
    p.add_argument("--data-subject", default=None,
                   help="用哪个数据集驱动（默认与对象同名）")
    p.add_argument("--frames", type=int, default=50,
                   help="每段测多少帧（默认 50）")
    p.add_argument("--warmup", type=int, default=5,
                   help="预热帧数，不计入统计（默认 5）")
    p.add_argument("--parts", default=",".join(PARTS),
                   help=f"要测的段，逗号分隔（默认 {','.join(PARTS)}）")
    p.add_argument("--rdf-host", default=None, help="追踪服务地址（默认取配置）")
    p.add_argument("--rdf-port", type=int, default=None,
                   help=f"追踪服务端口（默认取配置，兜底 {DEFAULT_PORT}）")
    p.add_argument("--rdf-python", default=None,
                   help="realdenseface 环境的解释器（默认自动探测）")
    p.add_argument("--rdf-root", default=None,
                   help="RealDenseFace 仓库路径（默认取 rdf_bridge.py 的默认值）")
    p.add_argument("--no-launch-rdf", action="store_true",
                   help="不启动追踪服务，直接连已有的（需先在别处起好）")
    p.add_argument("--rdf-no-compile", action="store_true",
                   help="启动追踪服务时加 --no-compile（更快起，但每帧更慢）")
    p.add_argument("--json", type=Path, default=None,
                   help="把结果写成 JSON（默认只打印）")
    p.add_argument("--dry-run", action="store_true",
                   help="只解析配置与路径并打印计划，不跑测试（无需 GPU）")
    p.add_argument("--config-dir", type=Path, default=None)
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    args.parts = [s.strip() for s in str(args.parts).split(",") if s.strip()]
    bad = [s for s in args.parts if s not in PARTS]
    if bad:
        p.error(f"--parts 里有未知的段：{bad}（可选 {list(PARTS)}）")
    if args.frames < 1:
        p.error("--frames 至少为 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from live3dgsavatar.config import load_config

    cfg = load_config(args.config_dir)
    host = args.rdf_host or str(cfg.get("tracking.host", "127.0.0.1"))
    port = args.rdf_port or int(cfg.get("tracking.port", DEFAULT_PORT))

    info = plan(args, cfg)
    print("=" * 68)
    print("综合性能测试")
    print(f"  模型：{info.get('model') or '（未指定）'}")
    print(f"  数据集：{info.get('dataset') or '（未指定）'}"
          f"  驱动对象：{info.get('data_subject') or '—'}")
    print(f"  分段：{','.join(info['parts'])}   每段 {info['frames']} 帧"
          f"（预热 {args.warmup}）")
    print("=" * 68)

    if args.dry_run:
        for key in ("model", "dataset"):
            print(f"  [dry-run] {key}: {info.get(key) or '未解析'}")
        if info.get("dataset_error"):
            print(f"  [dry-run] 数据集不可用：{info['dataset_error']}")
        print("  [dry-run] 未运行任何测试")
        return 0

    _require_gpu()
    subject, work = info["subject"], info["work_name"]
    if not subject or not work:
        raise SystemExit("[error] 没有可用模型：请用 --subject/--work-name 指定，"
                         "或把模型放到 models/<对象>/<版本>/ 下。")
    data_subject = info["data_subject"]

    results: dict[str, dict] = {}
    frames = None
    if "rdf" in args.parts or "full" in args.parts:
        frames, data_dir = _frames(cfg, data_subject)
        print(f"  画面来源：{frames.subdir}/（{frames.num_frames} 帧）"
              f"  {data_dir}")

    # ---- 纯渲染 ----
    session = None
    if "render" in args.parts:
        print("\n[1/3] 纯渲染", flush=True)
        session = _session(cfg, subject, work)
        results["render"] = bench_render(session, args.frames, args.warmup)
        print_row("纯渲染", results["render"])

    # ---- 需要追踪服务的两段 ----
    if "rdf" in args.parts or "full" in args.parts:
        from live3dgsavatar.tracking.rdf_client import RdfClient

        proc = None
        if not args.no_launch_rdf:
            if _port_open(host, port):
                raise SystemExit(
                    f"[error] {host}:{port} 已被占用。"
                    "若那是你的追踪服务，请加 --no-launch-rdf 复用它。")
            print("\n  正在加载追踪服务（首次约 30–60 s）…", flush=True)
            proc = launch_tracker(args, host, port)
        try:
            wait_for_service(host, port, proc)
            client = RdfClient(host, port, request_timeout_s=60.0)
            try:
                if "rdf" in args.parts:
                    print("\n[2/3] RDF 跟踪", flush=True)
                    results["rdf"] = bench_rdf(client, frames, args.frames)
                    print_row("RDF 跟踪", results["rdf"])

                if "full" in args.parts:
                    print("\n[3/3] RDF 驱动渲染", flush=True)
                    if session is None:
                        session = _session(cfg, subject, work)
                    results["full"] = bench_full(
                        session, client, frames, args.frames, args.warmup)
                    print_row("RDF 驱动渲染", results["full"])
            finally:
                client.close()
        finally:
            if proc is not None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:      # pragma: no cover
                    proc.kill()

    # ---- 汇总 ----
    print("\n" + "-" * 68)
    for tag, key in (("纯渲染", "render"), ("RDF 跟踪", "rdf"),
                     ("RDF 驱动渲染", "full")):
        if key in results:
            print_row(tag, results[key])
    if "full" in results and "render" in results:
        r = results["full"]
        share = r["track_ms"] / r["mean_ms"] * 100 if r["mean_ms"] else 0.0
        print(f"\n  端到端里跟踪占 {share:.0f}%"
              f"（跟踪 {r['track_ms']:.2f} ms vs 渲染 {r['render_ms']:.2f} ms）")

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"args": {k: str(v) for k, v in vars(args).items()},
                   "plan": info, "results": results,
                   "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        print(f"\n  结果已写入：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
