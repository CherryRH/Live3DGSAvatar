#!/usr/bin/env python
"""渲染测试：本项目 `core/` 出图，与数据集原图对比，并测性能。

判据：

- **重建质量**：PSNR（全图 / 前景掩码内）、SSIM、LPIPS、锐度比
- **性能**：单帧耗时、FPS、峰值显存
- **产物**：`report.json` + 对比视频（原图 | 渲染，逐帧叠加指标）

用法::

    python scripts/render_test.py --frames 20
    python scripts/render_test.py --frames -1 --batch-size 4
    python scripts/render_test.py --dry-run          # 无需 GPU 的形状/约定自检

产物：`report.json`（PSNR / SSIM / LPIPS / 锐度 / 性能）+ `compare.mp4`
（原图 | 渲染并排，每帧叠加指标；缺 imageio 时退化为 `compare.webp`）。
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

# ⚠️ 必须在导入任何可能触发 pickle 的模块之前施加 numpy 兼容补丁。
#    （本项目已不再加载 FLAME 封装，但保留这一步以防间接依赖。）
import live3dgsavatar  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from live3dgsavatar.config import (Config, load_config,  # noqa: E402
                                   resolve_model_config, resolve_model_ply)


# ------------------------------------------------------------------ CLI --


def _normalize_argv(argv: list[str]) -> list[str]:
    """把 `--work_name` 这类**选项名**规整成 `--work-name`（两种写法都接受）。

    只改以 `--` 开头的 token。之前对**所有**参数做替换，会把取值也改掉：
    `--subject my_name` 变成 `my-name`、路径里的下划线被改掉 —— 后者只在测试里
    暴露（真实运行时 `main()` 走的是未规整的分支），很难查。
    """
    return [a.replace("_", "-") if a.startswith("--") else a
            for a in argv if a.strip()]


def clean_previous_outputs(out_root: Path, share_dir: Path | None) -> int:
    """清掉上一次运行留下的旧产物（`images/`、旧视频）。

    产物目录按"对象/版本/数据集"复用，但**帧数会变**（换 `--frames`、换数据集）。
    实测踩过一次：先跑 2863 帧中途中断，再跑 212 帧，`images/` 里留下 1025 张
    新旧混杂的图，统计与肉眼检查都会被污染。视频同理（旧文件会看起来像新结果）。

    Returns:
        删除的文件数
    """
    removed = 0
    dirs = [d for d in (share_dir, Path(out_root) / "images") if d is not None]
    for sub in dirs:
        if not Path(sub).is_dir():
            continue
        for path in Path(sub).glob("*.png"):
            path.unlink()
            removed += 1
    for pattern in ("compare.mp4", "compare.webp"):
        path = Path(out_root) / pattern
        if path.is_file():
            path.unlink()
            removed += 1
    return removed


def _images_have_alpha(directory: Path) -> bool:
    """抽一张图看是否有 alpha 通道（决定对比时的背景色）。"""
    from PIL import Image

    for path in sorted(Path(directory).glob("*.png"))[:1]:
        with Image.open(path) as im:
            return im.mode in ("RGBA", "LA") or "transparency" in im.info
    return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="渲染测试（本项目 core vs 数据集原图）")
    p.add_argument("--subject", default=None, help="模型所属对象（models/<对象>/...）")
    p.add_argument("--work-name", default=None, help="模型版本名（models/<对象>/<版本>/）")
    p.add_argument("--dataset", default=None,
                   help="用哪个数据集评测（datasets/<名字>，必须是标准数据集；"
                        "默认取 --subject 同名数据集）")
    p.add_argument("--frames", type=int, default=None,
                   help="评测多少帧；-1 表示全部")
    p.add_argument("--frame-mode", choices=("head", "tail", "random"),
                   default="head",
                   help="取帧方式：head=从开头（默认）、tail=从结尾、random=随机")
    p.add_argument("--seed", type=int, default=None,
                   help="random 模式的随机种子（默认取配置）")
    p.add_argument("--video-fps", type=float, default=None,
                   help="对比视频帧率（默认取配置，30 = 实时指标口径）")
    # `default=None`：脚本不硬编码业务默认量，未给时在 `_finalize` 里取 0
    p.add_argument("--frame-offset", type=int, default=None,
                   help="起始帧（仅 head 模式生效）")
    p.add_argument("--batch-size", type=int, default=None, help="批大小")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="产物根目录（默认 output/render_test/<对象>/<版本>/<数据集>）")
    p.add_argument("--save-images", action="store_true",
                   help="额外落盘渲染 PNG（默认只出对比视频）")
    p.add_argument("--no-lpips", action="store_true",
                   help="跳过 LPIPS（首次使用需下载 AlexNet 权重）")
    p.add_argument("--profile", action="store_true",
                   help="额外输出 deform / 光栅化分阶段耗时（诊断用）")
    p.add_argument("--background", type=float, nargs=3, default=None,
                   metavar=("R", "G", "B"), help="背景色")
    p.add_argument("--tex-size", type=int, default=None)
    p.add_argument("--num-basis-in", type=int, default=None)
    p.add_argument("--num-basis-blend", type=int, default=None)
    p.add_argument("--mlp-hidden", type=int, nargs="+", default=None)
    p.add_argument("--no-weight-proj", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="无需 GPU：用真实模型 + 合成帧验证形状与约定")
    import sys as _sys

    raw = list(_sys.argv[1:] if argv is None else argv)
    return p.parse_args(_normalize_argv(raw))


def _validate(args: argparse.Namespace) -> None:
    for name in ("frames", "batch_size"):
        v = getattr(args, name)
        if v is not None and v == 0:
            raise SystemExit(f"[error] --{name.replace('_', '-')} 不能为 0")
    if args.frames is not None and args.frames < -1:
        raise SystemExit("[error] --frames 应为 -1 或正整数")


def _finalize(args: argparse.Namespace, cfg: Config) -> argparse.Namespace:
    """把配置与 CLI 合并成最终参数。"""
    # 先记下哪些结构参数是**用户显式给的**：只有那时才用它覆盖 PLY 的自描述，
    # 否则会让"模型结构与配置默认值不一致"的文件被错误加载。
    args.frame_offset = 0 if args.frame_offset is None else int(args.frame_offset)
    args.seed = int(cfg.get("render.test_seed", 0)
                   if args.seed is None else args.seed)
    args.video_fps = float(cfg.get("render.video_fps", 30.0)
                           if args.video_fps is None else args.video_fps)
    args._explicit_tex_size = args.tex_size is not None
    args._explicit_basis_in = args.num_basis_in is not None
    args._explicit_basis_blend = args.num_basis_blend is not None
    args._explicit_mlp_hidden = args.mlp_hidden is not None
    if args.subject is not None:
        cfg.set("subject", args.subject)
    if args.work_name is not None:
        cfg.set("work_name", args.work_name)
    if not (str(cfg.get("subject") or "").strip()
            and str(cfg.get("work_name") or "").strip()):
        raise SystemExit(
            "[error] 需要指定模型：--subject 与 --work-name\n"
            "        例如 --subject <对象> --work-name <版本>")
    args.subject = str(cfg.get("subject"))
    args.work_name = str(cfg.get("work_name"))

    # 数据集：必须是标准数据集（datasets/<名字>，含 images/ + render_inputs.npz）。
    # 不允许直接指任意目录 —— 数据集的构建走 build_dataset.sh，渠道唯一。
    from live3dgsavatar.data.datasets import resolve_dataset_dir

    args.dataset = str(args.dataset or args.subject)
    # ⚠️ 不能改 cfg 的 subject：模型路径是按 subject 拼的，
    # 改了就会去找 `models/<数据集>/<版本>/model.ply`（渲染别的数据集时必错）
    try:
        args.data = resolve_dataset_dir(cfg, args.dataset)
        args.ply = resolve_model_ply(cfg)
        args.model_config = resolve_model_config(cfg)
    except (FileNotFoundError, ValueError) as e:
        # 模型/数据集缺一不可，错误要能直接指出该补哪个
        raise SystemExit(f"[error] {e}") from None
    args.gt_dir = args.data / str(cfg.get("paths.image_subdir", "images"))

    # 三层目录：<对象>/<版本>/<数据集>；内部统一 report.json + 对比视频
    out_root = Path(args.output_dir) if args.output_dir else (
        Path(str(cfg.get("paths.output_dir", "output"))) / "render_test"
        / args.subject / args.work_name / args.dataset)
    args.output_dir = out_root

    args.frames = int(cfg.get("render.frames", 20) if args.frames is None else args.frames)
    args.batch_size = int(cfg.get("render.batch_size", 1)
                          if args.batch_size is None else args.batch_size)
    if args.tex_size is None:
        args.tex_size = int(cfg.get("model.network.tex_size", 256))
    if args.num_basis_in is None:
        args.num_basis_in = int(cfg.get("model.network.num_basis_in", 129))
    if args.num_basis_blend is None:
        args.num_basis_blend = int(cfg.get("model.network.num_basis_blend", 20))
    if args.mlp_hidden is None:
        args.mlp_hidden = list(cfg.get("model.network.mlp_hidden") or [])
    args.use_weight_proj = (not args.no_weight_proj) and bool(
        cfg.get("model.network.use_weight_proj", True))
    args._explicit_background = args.background is not None
    if args.background is None:
        args.background = [float(x) for x in cfg.get("render.background", [0, 0, 0])]
    return args


# --------------------------------------------------------------- 指标 --
# 口径与 GUI/在线更新一致：图像度量来自 `core.metrics`，
# 聚合与报告结构来自 `evaluation.protocol`（脚本不再自己算一遍）。

from live3dgsavatar.core.metrics import psnr, psnr_masked  # noqa: E402
from live3dgsavatar.evaluation import (  # noqa: E402
    EvaluationReport, summarize_no_reference, summarize_quality)
from live3dgsavatar.evaluation.protocol import (  # noqa: E402
    frame_metrics, lpips_score)


# ------------------------------------------------------------- 渲染 --


def render_core(args, scene, frames: list[int], out_dir: Path,
                share_dir: Path | None = None):
    """用本项目 `core/` 渲染。

    Args:
        share_dir: 非 ``None`` 时额外落盘 PNG。图像总是留在内存里（拼视频用）。
    """
    import torch

    from live3dgsavatar.core.avatar import AvatarConfig
    from live3dgsavatar.core.io import load_ply
    from live3dgsavatar.core.render import SimpleRasterizer
    from live3dgsavatar.core.types import Camera, Mesh

    # 结构优先级：命令行显式 > PLY 内嵌的自描述 > 配置/默认值。
    # 训练产出的 PLY 自带 `gaussian_config`；参照实现写出的文件没有，
    # 这时回退到 `--tex-size` 等参数（默认取 configs/）。
    from live3dgsavatar.core.io.ply import read_embedded_config

    explicit = (args._explicit_tex_size or args._explicit_basis_in
                or args._explicit_basis_blend or args._explicit_mlp_hidden
                or args.no_weight_proj)
    embedded = None if explicit else read_embedded_config(args.ply)
    if embedded is not None:
        print(f"  结构  : 来自 PLY 内嵌（tex {embedded.tex_size}, "
              f"D {embedded.num_basis_in}, K {embedded.num_basis_blend}）")
        avatar = load_ply(args.ply, config=embedded, device="cuda")
    else:
        if not explicit:
            print("  结构  : 该 PLY 未内嵌结构，回退到配置/默认值")
        cfg = AvatarConfig(tex_size=args.tex_size, num_basis_in=args.num_basis_in,
                           num_basis_blend=args.num_basis_blend,
                           mlp_hidden=tuple(args.mlp_hidden),
                           use_weight_proj=args.use_weight_proj)
        avatar = load_ply(args.ply, config=cfg, device="cuda")
    print(f"  结构  : tex {avatar.config.tex_size}  D {avatar.config.num_basis_in}  "
          f"K {avatar.config.num_basis_blend}  "
          f"hidden {avatar.config.mlp_hidden}  "
          f"proj {avatar.config.use_weight_proj}")
    w2c = torch.eye(4, dtype=torch.float32)
    w2c[:3, :3] = scene.R
    w2c[:3, 3] = scene.T
    w2c = w2c.unsqueeze(0)
    bg = torch.tensor(args.background, dtype=torch.float32, device="cuda")
    rasterizer = SimpleRasterizer()

    stages: dict[str, list[float]] = {"deform": [], "rasterize": []}

    def run_batch(idx: list[int]):
        mesh = Mesh(verts=scene.frames_tensor(idx, device="cuda"),
                    faces=scene.faces, uvs=scene.uvs, uv_faces=scene.uv_faces)
        # 逐帧内参（每帧裁剪框可能不同）；数据集的各帧共用一份，走广播
        K = scene.intrinsics(idx, device="cuda")
        camera = Camera(K=K, w2c=w2c.expand(len(idx), 4, 4).contiguous(),
                        width=scene.width, height=scene.height).to("cuda")
        weight = scene.weights_tensor(idx, device="cuda")
        with torch.no_grad():
            if args.profile:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                gs = avatar.deform(mesh, weight)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                out = rasterizer.render(gs, camera, bg)
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                stages["deform"].append((t1 - t0) * 1000.0 / len(idx))
                stages["rasterize"].append((t2 - t1) * 1000.0 / len(idx))
                return out
            gs = avatar.deform(mesh, weight)
            return rasterizer.render(gs, camera, bg)

    run_batch(frames[:1])                       # 预热
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    images: dict[int, np.ndarray] = {}
    per_frame_ms: list[float] = []
    total_frames = len(frames)
    # 进度：不打印的话，几百帧的渲染在终端里看起来就是"卡住"。
    # 每 10% 打一次，且随总帧数自适应（少帧时每帧都打太吵）。
    step = max(1, total_frames // 10)
    for start in range(0, len(frames), args.batch_size):
        if start % step == 0:
            print(f"  [render] {start}/{total_frames}", flush=True)
        chunk = frames[start:start + args.batch_size]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = run_batch(chunk)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        per_frame_ms.extend([dt / len(chunk)] * len(chunk))

        from PIL import Image

        arr = (out.color.clamp(0, 1).permute(0, 2, 3, 1) * 255).byte().cpu().numpy()
        for j, i in enumerate(chunk):
            images[i] = arr[j]
            if share_dir is not None:
                Image.fromarray(arr[j]).save(share_dir / f"{i:05d}.png")

    print(f"  [render] {total_frames}/{total_frames} 完成", flush=True)
    perf = {
        "mean_ms_per_frame": float(np.mean(per_frame_ms)),
        "median_ms_per_frame": float(np.median(per_frame_ms)),
        "fps": 1000.0 / float(np.mean(per_frame_ms)),
        "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
        "batch_size": args.batch_size,
        "num_gaussian": int(avatar.num_gaussians),
    }
    if args.profile:
        perf["stages_ms"] = {k: float(np.mean(v)) for k, v in stages.items() if v}
    return images, perf


def select_indices(total: int, frames: int | None, mode: str, *,
                   seed: int = 0, offset: int = 0) -> list[int]:
    """按取帧方式返回**全局帧号**列表。

    - `head`  ：从 `offset` 开始连续取（`--frame-offset` 只在这里生效）
    - `tail`  ：取最后 n 帧
    - `random`：全片随机取 n 帧（`seed` 固定则可复现）

    `frames` 为 ``None`` 或 `-1` 时表示全部。
    """
    import numpy as np

    total = int(total)
    n = total if frames is None or int(frames) < 0 else min(int(frames), total)
    if mode == "tail":
        return list(range(total - n, total))
    if mode == "random":
        rng = np.random.default_rng(int(seed))
        return sorted(int(i) for i in rng.choice(total, size=n, replace=False))
    start = max(0, min(int(offset), total))
    return list(range(start, min(total, start + n)))


def load_gt(args, frames: list[int], offset: int = 0):
    """读取数据集原图（RGBA）。返回 `{i: (rgb HxWx3, mask HxW bool)}`。

    `frames` 是**场景内的下标**（0..n-1），`offset` 是它们在数据集里的起始
    帧号。GT 文件必须按 `offset + i` 取 —— 之前漏了这一步，`--frame-offset`
    时会把"渲染第 2513 帧"和"原图第 0 帧"相比，PSNR 直接掉到 13 dB 以下。
    """
    from PIL import Image

    if not args.gt_dir.is_dir():
        return None
    files = sorted(args.gt_dir.iterdir())
    out = {}
    for i in frames:
        gid = int(offset) + int(i)
        if gid < 0 or gid >= len(files):
            continue
        arr = np.array(Image.open(files[gid]))
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[2] == 4:
            out[i] = (arr[..., :3], arr[..., 3] > 250)   # alpha 即 mask
        else:
            # 白底图没有 alpha：把"接近纯白"的像素当背景。
            # 这不是完美的抠图，但足以把 PSNR 限制在前景区域。
            fg = arr[..., :3].min(axis=-1) < 250
            out[i] = (arr[..., :3], fg)
    return out or None


# ----------------------------------------------------------------- 主流程 --


def main() -> int:
    args = parse_args()
    _validate(args)
    cfg = load_config()
    args = _finalize(args, cfg)

    print(f"渲染测试 —— {args.subject}/{args.work_name}")
    print(f"  model = {args.ply}")
    print(f"  data  = {args.data}")
    print(f"  数据集: {args.dataset}")
    frames_text = "全部" if args.frames < 0 else f"{args.frames} 帧"
    print(f"  配置  : {frames_text}（起始 {args.frame_offset}）, "
          f"batch={args.batch_size}, {args.tex_size} tex, "
          f"D={args.num_basis_in}, K={args.num_basis_blend}")
    print(f"  产物  : {args.output_dir}")

    if args.dry_run:
        return _dry_run(args)

    import torch

    if not torch.cuda.is_available():
        print("[error] 渲染需要 GPU。可用 --dry-run 做无 GPU 自检。", file=sys.stderr)
        return 1

    from live3dgsavatar.data.scene import load_scene

    # ⚠️ 必须显式传 `data_dir`：`--dataset` 选的数据集与模型所属对象的
    # subject 不一定同名，而 `load_scene(cfg)` 只按 cfg 的 subject 找数据集 ——
    # 漏了这个参数就会"显示的是 A 数据集，实际渲染 B 数据集"。
    # 统一整段载入：三种取帧方式用同一套**全局下标**，GT 与渲染不会错位。
    # 帧数据很小（2863 帧约 175 MB），不值得为 head 模式单独优化。
    scene = load_scene(cfg, frames=-1, data_dir=args.data)
    total = scene.total_frames
    indices = select_indices(total, args.frames, args.frame_mode,
                             seed=args.seed, offset=args.frame_offset)
    n = len(indices)
    print(f"  数据集: 共 {total} 帧，取 {n} 帧（{args.frame_mode}）", flush=True)

    # 背景按图像本身判定：带 alpha 的是实拍（用配置的黑底），
    # 纯 RGB 的多为白底合成图（用白底，否则 PSNR 全被背景吃掉）。
    if not args._explicit_background and not _images_have_alpha(args.gt_dir):
        args.background = [1.0, 1.0, 1.0]

    out_root = args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)
    share_dir = out_root / "images" if args.save_images else None
    if share_dir is not None:
        share_dir.mkdir(parents=True, exist_ok=True)
    stale = clean_previous_outputs(out_root, share_dir)
    if stale:
        print(f"  清理旧产物：{stale} 张", flush=True)

    images, perf = render_core(args, scene, indices, out_root, share_dir)

    result = EvaluationReport(
        subject=args.subject, work_name=args.work_name, dataset=args.dataset,
        num_frames=n, dataset_frames=total,
        width=scene.width, height=scene.height,
        perf=perf, extra={"model_path": str(args.ply)})
    report: dict = result.to_dict()

    print(f"  读取原图：{n} 张（解码 512² PNG，约 {n * 14 / 1000:.0f} s）…",
          flush=True)
    gt = load_gt(args, indices)
    video_path = None
    if gt is None:
        print("      [warn] 找不到数据集原图，改做无真值统计", flush=True)
        report["no_reference"] = summarize_no_reference(images)
    else:
        print(f"  原图就绪：{len(gt)} 张", flush=True)
        report["quality"] = summarize_quality(images, gt)
        per_frame, lpips_acc = [], []
        for i in sorted(gt):
            ref_rgb, ref_mask = gt[i]
            img = images.get(i)
            if img is None or img.shape != ref_rgb.shape:
                continue
            m = frame_metrics(img, ref_rgb, ref_mask)
            if m.get("lpips") is not None:
                lpips_acc.append(float(m["lpips"]))
            per_frame.append({"frame": int(i), **m})
        if per_frame:
            report["quality"]["per_frame"] = per_frame
        if lpips_acc:
            arr = np.asarray(lpips_acc, dtype=np.float64)
            report["quality"]["lpips"] = {
                "median": round(float(np.median(arr)), 4),
                "min": round(float(arr.min()), 4),
                "max": round(float(arr.max()), 4),
                "mean": round(float(arr.mean()), 4),
                "n": int(arr.size)}
        elif not args.no_lpips:
            print("      [warn] LPIPS 不可用（缺 lpips 或权重），已跳过",
                  flush=True)

    # 对比视频：原图 | 渲染，每帧叠加 PSNR / SSIM / LPIPS；帧率固定 30
    frames_out = []
    for item in per_frame:
        i = item["frame"]
        ref_rgb, _ = gt[i]
        img = images.get(i)
        if img is None:
            continue
        left = f"target {i:05d}"
        parts = [f"PSNR {item['psnr']:.2f}"]
        if item["ssim"] is not None:
            parts.append(f"SSIM {item['ssim']:.4f}")
        if item["lpips"] is not None:
            parts.append(f"LPIPS {item['lpips']:.4f}")
        right = f"render {i:05d} | " + " | ".join(parts)
        frames_out.append(compose_pair(ref_rgb, img, left, right))
    if frames_out:
        video_path = write_video(out_root / "compare", frames_out,
                                 args.video_fps)
        print(f"\n  对比视频：{video_path}（{len(frames_out)} 帧，"
              f"{args.video_fps:.0f} FPS；渲染均值 {perf['fps']:.0f} FPS）")

    report_path = out_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    _print_summary(report)
    print(f"\n  产物：{out_root}")
    print(f"    report.json  指标")
    if video_path is not None:
        print(f"    {video_path.name}  对比视频（原图 | 渲染，带逐帧指标）")
    if share_dir is not None:
        print(f"    images/      渲染图（{n} 张）")
    return 0


def _print_summary(report: dict) -> None:
    sep = "=" * 78
    print(f"\n{sep}")
    head = f"{report['subject']}/{report['work_name']}"
    print(f"渲染测试报告 —— {head}，"
          f"{report['num_frames']} 帧，{report['width']}x{report['height']}")
    print(sep)

    q = report.get("quality")
    if q:
        print("\n[重建质量]  vs 数据集原图")
        for label, key in (("PSNR（全图）", "psnr_full"),
                           ("PSNR（mask 内）", "psnr_masked")):
            s = q.get(key)
            if s:
                print(f"  {label:24s} 中位 {s['median']:.2f} dB  "
                      f"最小 {s['min']:.2f} dB  (n={s['n']})")
        s = q.get("ssim")
        if s:
            print(f"  {'SSIM（mask 内）':24s} 中位 {s['median']:.4f}  "
                  f"最小 {s['min']:.4f}  (n={s['n']})")
        s = q.get("lpips")
        if s:
            print(f"  {'LPIPS（越低越好）':24s} 中位 {s['median']:.4f}  "
                  f"最大 {s['max']:.4f}  (n={s['n']})")
        s = q.get("max_abs_diff")
        if s:
            print(f"  {'max|Δ| (0-255)':24s} 中位 {s['median']:.0f}  "
                  f"最大 {s['max']:.0f}")
        sharp = q.get("sharpness")
        if sharp:
            r, g = sharp["render"], sharp["gt"]
            print(f"  {'锐度（梯度, 掩码内）':24s} 渲染 {r['median']:.2f} / "
                  f"原图 {g['median']:.2f}（比值 {sharp['ratio']:.2f}）")
    else:
        nr = report.get("no_reference") or {}
        print("\n[无真值统计]  缺少数据集原图，只给出图像自身指标")
        for label, key, unit in (("前景占比", "foreground", "%.3f"),
                                 ("锐度（梯度）", "sharpness", "%.2f"),
                                 ("饱和度", "saturation", "%.1f"),
                                 ("截断比例", "clip_ratio", "%.3f")):
            s = nr.get(key)
            if s:
                print(f"  {label:24s} 中位 {unit % s['median']}  (n={s['n']})")

    p = report["perf"]
    print("\n[性能]")
    print(f"  {'平均耗时 (ms/帧)':28s} {p['mean_ms_per_frame']:.2f}")
    print(f"  {'中位耗时 (ms/帧)':28s} {p['median_ms_per_frame']:.2f}")
    print(f"  {'吞吐 (FPS)':28s} {p['fps']:.1f}")
    print(f"  {'峰值显存 (MiB)':28s} {p['peak_memory_mib']:.0f}")
    print(f"  {'批大小':28s} {p['batch_size']}")
    print(f"  {'高斯数':28s} {p['num_gaussian']}")
    stages = p.get("stages_ms")
    if stages:
        print("  [分阶段] ms/帧")
        for key, value in stages.items():
            print(f"    {key:26s} {value:.2f}")
    print()


def _label_strip(width: int, text: str, height: int = 20):
    """面板上方的深色标签条（ASCII，避免字体依赖）。"""
    from PIL import Image, ImageDraw

    strip = Image.new("RGB", (width, height), (24, 26, 32))
    ImageDraw.Draw(strip).text((6, 4), text, fill=(230, 232, 238))
    return strip


def compose_pair(left: np.ndarray, right: np.ndarray,
                 left_label: str, right_label: str,
                 strip_h: int = 22, gap: int = 4) -> np.ndarray:
    """左=原图、右=渲染，一格一条标签，返回 `HxWx3` uint8（拼视频用）。"""
    from PIL import Image, ImageDraw

    tiles = [Image.fromarray(np.ascontiguousarray(x)) for x in (left, right)]
    w, h = tiles[0].size
    canvas = Image.new("RGB", (w * 2 + gap, h + strip_h), (10, 11, 14))
    draw = ImageDraw.Draw(canvas)
    for i, (tile, text) in enumerate(zip(tiles, (left_label, right_label))):
        x = i * (w + gap)
        draw.text((x + 4, 4), text, fill=(232, 234, 240))
        canvas.paste(tile, (x, strip_h))
    return np.asarray(canvas)


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> Path:
    """把已拼好的帧写成视频。

    优先用 `imageio`（mp4/H.264）；没装时退化为**动图 WebP**（PIL 直接写，
    浏览器可播放）。两条路都能用，缺依赖不会让整个测试失败。
    """
    if not frames:
        return path
    fps = max(1.0, float(fps))
    try:
        import imageio.v2 as iio

        real = path.with_suffix(".mp4")
        real.parent.mkdir(parents=True, exist_ok=True)
        writer = iio.get_writer(str(real), fps=fps, codec="libx264",
                                quality=8, macro_block_size=1)
        try:
            for frame in frames:
                writer.append_data(frame)
        finally:
            writer.close()
        return real
    except Exception:                                   # noqa: BLE001
        from PIL import Image

        real = path.with_suffix(".webp")
        real.parent.mkdir(parents=True, exist_ok=True)
        images = [Image.fromarray(np.ascontiguousarray(f)) for f in frames]
        images[0].save(real, save_all=True, append_images=images[1:],
                       duration=int(1000 / fps), loop=0, quality=90)
        return real


def write_comparison(path: Path, panels) -> None:
    """一排多图：`[(标签, HxWx3 uint8)]`，每格上方带标签。

    用于"原图 | 模型1 | 模型2 | …"，标签里写模型名与它的 PSNR，
    一眼能看出哪个模型在哪一帧更好。
    """
    from PIL import Image

    if not panels:
        return
    tiles = [Image.fromarray(np.ascontiguousarray(img)) for _, img in panels]
    w, h = tiles[0].size
    gap, strip_h = 4, 20
    total_w = w * len(tiles) + gap * (len(tiles) - 1)
    canvas = Image.new("RGB", (total_w, h + strip_h), (10, 11, 14))
    x = 0
    for tile, (label, _) in zip(tiles, panels):
        canvas.paste(_label_strip(w, label, strip_h), (x, 0))
        canvas.paste(tile, (x, strip_h))
        x += w + gap
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _dry_run(args) -> int:
    """CPU 自检：用真实模型 + 合成帧验证 core 链路的形状与约定。"""
    import torch

    from live3dgsavatar.core.avatar import AvatarConfig
    from live3dgsavatar.core.io import load_ply
    from live3dgsavatar.core.render import SimpleRasterizer, to_kernel_matrix
    from live3dgsavatar.core.types import Camera, Mesh

    print("\n[dry-run] 只验证 core/ 链路与指标函数，不需要 GPU。")

    if not args.ply.exists():
        print(f"[error] 模型不存在：{args.ply}")
        return 1

    cfg = AvatarConfig(tex_size=args.tex_size, num_basis_in=args.num_basis_in,
                       num_basis_blend=args.num_basis_blend,
                       mlp_hidden=tuple(args.mlp_hidden),
                       use_weight_proj=args.use_weight_proj)
    avatar = load_ply(args.ply, config=cfg, device="cpu")
    print(f"  avatar      N={avatar.num_gaussians}, K={avatar.num_basis}")

    n_faces = int(avatar.binding_face_id.max()) + 1
    n_verts = int(avatar.binding_face_id.numel())
    torch.manual_seed(0)
    faces = torch.stack(
        [torch.arange(0, n_faces * 3, 3, dtype=torch.int32) % n_verts,
         torch.arange(1, n_faces * 3 + 1, 3, dtype=torch.int32) % n_verts,
         torch.arange(2, n_faces * 3 + 2, 3, dtype=torch.int32) % n_verts], dim=1)
    mesh = Mesh(verts=torch.randn(1, n_verts, 3) * 0.1, faces=faces,
                uvs=torch.rand(n_verts, 2), uv_faces=faces)
    gs = avatar.deform(mesh, torch.randn(1, cfg.num_basis_in) * 0.1)
    print(f"  deform      xyz={tuple(gs.xyz.shape)} 有限={bool(torch.isfinite(gs.xyz).all())}")

    K = torch.tensor([[400.0, 0, 256], [0, 400.0, 256], [0, 0, 1]])
    cam = Camera.from_intrinsics_extrinsics(
        K=K, R=torch.eye(3), T=torch.tensor([0.0, 0.0, -0.5]), width=512, height=512)
    km = to_kernel_matrix(cam.full_proj[0])
    print(f"  camera      kernel 矩阵连续={km.is_contiguous()}")

    r = SimpleRasterizer()
    print(f"  指标函数    psnr 相同={psnr(np.zeros((4,4,3),np.uint8), np.zeros((4,4,3),np.uint8))}")
    print("\n  [ok] 链路自检通过（渲染本身需要 GPU）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
