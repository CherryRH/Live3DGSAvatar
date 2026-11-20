#!/usr/bin/env python
"""渲染测试：本项目 `core/` 出图，与数据集原图对比，并测性能。

判据：

- **重建质量**：PSNR（全图 / 前景掩码内）、SSIM、LPIPS、锐度比
- **性能**：单帧耗时、FPS、峰值显存
- **产物**：`report.json` + 对比视频（原图 | 渲染，逐帧叠加指标）

用法::

    python scripts/render_test.py --frames 20
    python scripts/render_test.py --frames -1 --batch-size 4
    python scripts/render_test.py --subject crh_1 --work-names base base_post_r2 wk1116_mask_r2 wk1116_none_none
    python scripts/render_test.py --dry-run          # 无需 GPU 的形状/约定自检

单模型时：`report.json`（PSNR / SSIM / LPIPS / 锐度 / 性能）+ `compare.mp4`
（原图 | 渲染并排，每帧叠加指标）+ `showcase.png`（固定 8 视角 2x4 拼图：
GT、-15/-30/-45、0、+15/+30/+45；只渲染 7 张）。
多模型时：`--work-names` 给多个版本，`compare.mp4` 为“原图 + N 个渲染”横排，
`showcase.png` 为固定 2x4 拼图；报告为每个模型单独记录指标。
"""

from __future__ import annotations

import argparse
import gc
import json
import shlex
import sys
import time
from pathlib import Path

# ⚠️ 必须在导入任何可能触发 pickle 的模块之前施加 numpy 兼容补丁。
#    （保留这一步以防间接依赖触发 pickle。）
import live3dgsavatar  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from live3dgsavatar.config import (Config, load_config,  # noqa: E402
                                   output_dir_for, resolve_model_config,
                                   resolve_model_ply)


# ------------------------------------------------------------------ CLI --


def _normalize_argv(argv: list[str]) -> list[str]:
    """把 `--work_name` 这类**选项名**规整成 `--work-name`（两种写法都接受）。

    只改以 `--` 开头的 token，不能碰取值：`--subject my_name` 必须保持
    `my_name`，路径里的下划线也不能被改写。
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
        for path in Path(sub).rglob("*.png"):
            path.unlink()
            removed += 1
    for pattern in ("report.json", "compare.mp4", "compare.webp",
                    "showcase.mp4", "showcase.webp", "showcase.png"):
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
    p.add_argument("--work-name", default=None,
                   help="主模型版本名（models/<对象>/<版本>/）")
    p.add_argument("--work-names", nargs="+", default=None,
                   help="多版本对比：依次给出多个版本名；第一个为主版本。"
                        "省略则只渲染 --work-name")
    p.add_argument("--dataset", default=None,
                   help="用哪个数据集评测（datasets/<名字>，必须是标准数据集；"
                        "默认取 --subject 同名数据集）")
    p.add_argument("--frames", type=int, default=None,
                   help="评测多少帧；-1 表示全部")
    p.add_argument("--frame-mode", choices=("head", "tail", "random", "heldout"),
                   default="head",
                   help="取帧方式：head=从开头（默认）、tail=从结尾、"
                        "random=随机、heldout=末尾留出集（比例取 test_ratio）")
    p.add_argument("--test-ratio", type=float, default=None,
                   help="heldout 模式下留出集比例（默认取 train.pipeline.test_ratio）")
    p.add_argument("--seed", type=int, default=None,
                   help="random 模式的随机种子（默认取配置）")
    p.add_argument("--video-fps", type=float, default=None,
                   help="对比视频帧率（默认取配置，30 = 实时指标口径）")
    # `default=None`：脚本不硬编码业务默认量，未给时在 `_finalize` 里取 0
    p.add_argument("--frame-offset", type=int, default=None,
                   help="起始帧（仅 head 模式生效）")
    p.add_argument("--batch-size", type=int, default=1,
                   help="批大小（默认 1；FPS 按单帧口径统计）")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="产物根目录（默认 output/render_test/<对象>/<版本>/；数据集不同时追加 /<数据集>）")
    p.add_argument("--save-images", action="store_true",
                   help="额外落盘渲染 PNG（默认只出对比视频；多模型时按版本分子目录）")
    p.add_argument("--showcase", dest="showcase", action="store_true",
                   default=True,
                   help="额外输出大角度展示视频（默认开）：yaw 0→40→-40→0，"
                        "再 pitch 20→-20，共 30 帧")
    p.add_argument("--no-showcase", dest="showcase", action="store_false",
                   help="不输出展示视频")
    p.add_argument("--showcase-frames", type=int, default=8,
                   help="兼容参数；展示已改为固定 8 张 2x4 拼图")
    p.add_argument("--showcase-fps", type=float, default=None,
                   help="展示视频帧率（默认与 --video-fps 一致）")
    p.add_argument("--showcase-frame", type=int, default=None,
                   help="展示用的基础帧下标（默认取本次评测中头姿最正的一帧）")
    p.add_argument("--frontface", action="store_true",
                   help="只渲染朝向相机的高斯（硬表面/背面剔除）")
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


def _work_name_list(primary: str | None, raw: list[str] | None) -> list[str]:
    """合并 `--work-name` 与 `--work-names`，去重并保持顺序。"""
    names: list[str] = []
    if primary:
        names.append(str(primary))
    for item in raw or []:
        for name in str(item).split(","):
            name = name.strip()
            if name and name not in names:
                names.append(name)
    return names


def _validate(args: argparse.Namespace) -> None:
    for name in ("frames", "batch_size"):
        v = getattr(args, name)
        if v is not None and v == 0:
            raise SystemExit(f"[error] --{name.replace('_', '-')} 不能为 0")
    if args.frames is not None and args.frames < -1:
        raise SystemExit("[error] --frames 应为 -1 或正整数")
    if args.showcase_frames <= 0:
        raise SystemExit("[error] --showcase-frames 必须为正")
    if args.showcase_fps is not None and args.showcase_fps <= 0:
        raise SystemExit("[error] --showcase-fps 必须为正")
    if args.showcase_frame is not None and args.showcase_frame < 0:
        raise SystemExit("[error] --showcase-frame 不能为负")


def _finalize(args: argparse.Namespace, cfg: Config) -> argparse.Namespace:
    """把配置与 CLI 合并成最终参数。"""
    # 先记下哪些结构参数是**用户显式给的**：只有那时才用它覆盖 PLY 的自描述，
    # 否则会让"模型结构与配置默认值不一致"的文件被错误加载。
    args.frame_offset = 0 if args.frame_offset is None else int(args.frame_offset)
    args.test_ratio = float(cfg.get("train.pipeline.test_ratio", 0.3)
                            if args.test_ratio is None else args.test_ratio)
    if not 0.0 <= args.test_ratio < 1.0:
        raise SystemExit(
            f"[error] --test-ratio 应在 [0, 1)，实际 {args.test_ratio}")
    args.seed = int(cfg.get("render.test_seed", 0)
                   if args.seed is None else args.seed)
    args.video_fps = float(cfg.get("render.video_fps", 30.0)
                           if args.video_fps is None else args.video_fps)
    args.showcase_fps = float(args.video_fps if args.showcase_fps is None
                              else args.showcase_fps)
    args._explicit_tex_size = args.tex_size is not None
    args._explicit_basis_in = args.num_basis_in is not None
    args._explicit_basis_blend = args.num_basis_blend is not None
    args._explicit_mlp_hidden = args.mlp_hidden is not None
    if args.subject is not None:
        cfg.set("subject", args.subject)
    names = _work_name_list(args.work_name, args.work_names)
    if not (str(cfg.get("subject") or "").strip() and names):
        raise SystemExit(
            "[error] 需要指定模型：--subject 与 --work-name/--work-names\n"
            "        例如 --subject <对象> --work-name <版本>\n"
            "        或 --subject <对象> --work-names base base_post_r2")
    args.subject = str(cfg.get("subject"))
    args.work_names = names
    args.work_name = names[0]
    args._multi = len(names) > 1
    cfg.set("work_name", args.work_name)

    # 数据集：必须是标准数据集（datasets/<名字>，含 images/ + render_inputs.npz）。
    # 不允许直接指任意目录 —— 数据集的构建走 build_dataset.sh，渠道唯一。
    from live3dgsavatar.data.datasets import resolve_dataset_dir

    args.dataset = str(args.dataset or args.subject)
    # ⚠️ 不能改 cfg 的 subject：模型路径是按 subject 拼的，
    # 改了就会去找 `models/<数据集>/<版本>/model.ply`（渲染别的数据集时必错）
    try:
        args.data = resolve_dataset_dir(cfg, args.dataset)
        args.plies = []
        args.model_configs = []
        for name in names:
            cfg.set("work_name", name)
            args.plies.append(resolve_model_ply(cfg))
            args.model_configs.append(resolve_model_config(cfg))
    except (FileNotFoundError, ValueError) as e:
        # 模型/数据集缺一不可，错误要能直接指出该补哪个
        raise SystemExit(f"[error] {e}") from None
    finally:
        cfg.set("work_name", args.work_name)
    args.ply = args.plies[0]
    args.model_config = args.model_configs[0]
    args.gt_dir = args.data / str(cfg.get("paths.image_subdir", "images"))

    # 多模型时不要覆盖单模型目录：默认用 `<主版本>_multi` 作为对比实验名。
    out_work = args.work_name if not args._multi else f"{args.work_name}_multi"
    out_root = (Path(args.output_dir) if args.output_dir
                else output_dir_for(cfg, "render_test", args.subject,
                                    out_work, args.dataset))
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
#: 指标口径版本：口径一变就必须改这个串（报告里可追溯）。
METRIC_VERSION = "2026-11-20.pose-headturn"


def _provenance(models: list[dict]) -> dict:
    """报告溯源：commit / 模型哈希 / 口径版本 / mask 与阈值定义。"""
    import hashlib
    import subprocess

    def _git(*args: str):
        try:
            out = subprocess.run(["git", *args], capture_output=True,
                                 text=True, timeout=5)
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None

    files: dict[str, dict] = {}
    for m in models:
        path = Path(m["model_path"])
        digest = None
        try:
            h = hashlib.sha256()
            with path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            digest = h.hexdigest()[:16]
        except OSError:
            pass
        files[m["work_name"]] = {"path": str(path), "sha256": digest}
    try:
        import lpips as _lpips
        lpips_version = getattr(_lpips, "__version__", None)
    except Exception:
        lpips_version = None
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "model_files": files,
        "metric_version": METRIC_VERSION,
        "mask": "数据集人像 alpha>127（masked PSNR/SSIM）",
        "hole_threshold": 0.5,
        "alpha_deficit": "mean(1-alpha) over FLAME surface (robust score)",
        "lpips": {"backbone": "alex", "version": lpips_version},
        "sharpness": "mask 内梯度均值（渲染 / GT）",
    }


# 聚合与报告结构来自 `evaluation.protocol`。

from live3dgsavatar.core.metrics import psnr, psnr_masked  # noqa: E402
from live3dgsavatar.evaluation import (  # noqa: E402
    EvaluationReport, summarize_no_reference, summarize_quality)
from live3dgsavatar.evaluation.protocol import (  # noqa: E402
    frame_metrics, lpips_score)


# ------------------------------------------------------------- 渲染 --


def _frontal_frame(scene, frames: list[int]) -> int:
    """在本次评测帧里选头姿最接近正脸的一帧做 showcase 基帧。"""
    import numpy as np

    from live3dgsavatar.data.pose import head_pitch, head_yaw

    w = scene.weights_tensor(frames, device="cpu").detach().cpu().numpy()
    yaw = np.abs(head_yaw(w))
    pitch = np.abs(head_pitch(w))
    return int(frames[int(np.argmin(yaw + pitch))])


def render_core(args, scene, frames: list[int], out_dir: Path,
                share_dir: Path | None = None, *, ply: Path | None = None,
                draw_angle: bool = True):
    """用本项目 `core/` 渲染一个模型。

    Args:
        ply: 要加载的 PLY；``None`` 时用 ``args.ply``（单模型兼容路径）。
        share_dir: 非 ``None`` 时额外落盘 PNG。图像总是留在内存里（拼视频用）。
        draw_angle: 单模型 showcase 时把角度写在图上；多模型由拼图标签统一写。
    """
    import torch

    from live3dgsavatar.core.avatar import AvatarConfig
    from live3dgsavatar.core.io import load_ply
    from live3dgsavatar.core.render import (SimpleRasterizer, render_frame,
                                            render_gaussians)
    from live3dgsavatar.core.types import Camera, Mesh

    # 结构优先级：命令行显式 > PLY 内嵌的自描述 > 配置/默认值。
    # 训练产出的 PLY 自带 `gaussian_config`；参照实现写出的文件没有，
    # 这时回退到 `--tex-size` 等参数（默认取 configs/）。
    from live3dgsavatar.core.io.ply import read_embedded_config

    ply = Path(ply) if ply is not None else Path(args.ply)
    explicit = (args._explicit_tex_size or args._explicit_basis_in
                or args._explicit_basis_blend or args._explicit_mlp_hidden
                or args.no_weight_proj)
    embedded = None if explicit else read_embedded_config(ply)
    if embedded is not None:
        print(f"  结构  : 来自 PLY 内嵌（tex {embedded.tex_size}, "
              f"D {embedded.num_basis_in}, K {embedded.num_basis_blend}）")
        avatar = load_ply(ply, config=embedded, device="cuda")
    else:
        if not explicit:
            print("  结构  : 该 PLY 未内嵌结构，回退到配置/默认值")
        cfg = AvatarConfig(tex_size=args.tex_size, num_basis_in=args.num_basis_in,
                           num_basis_blend=args.num_basis_blend,
                           mlp_hidden=tuple(args.mlp_hidden),
                           use_weight_proj=args.use_weight_proj)
        avatar = load_ply(ply, config=cfg, device="cuda")
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
        timings: dict[str, float] = {}
        if args.frontface:
            from live3dgsavatar.core.render.frontface import frontface_gaussians
            from live3dgsavatar.core.render import render_gaussians
            with torch.no_grad():
                gs = avatar.deform(mesh, weight)
                gs = frontface_gaussians(gs, camera, mesh,
                                         avatar.binding_face_id)
                out = render_gaussians(gs, rasterizer, camera, bg)
        else:
            out = render_frame(avatar, rasterizer, mesh, weight, camera, bg,
                               timings=timings if args.profile else None,
                               sync=bool(args.profile))
        if args.profile:
            stages["deform"].append(timings["deform"] / len(idx))
            stages["rasterize"].append(timings["rasterize"] / len(idx))
        return out

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
        # 报告里写**实际加载**的结构，而不是 CLI 默认值（评审 A4）
        "structure": {
            "tex_size": int(avatar.config.tex_size),
            "num_basis_in": int(avatar.config.num_basis_in),
            "num_basis_blend": int(avatar.config.num_basis_blend),
            "mlp_hidden": list(avatar.config.mlp_hidden or ()),
            "use_weight_proj": bool(avatar.config.use_weight_proj),
        },
    }
    if args.profile:
        perf["stages_ms"] = {k: float(np.mean(v)) for k, v in stages.items() if v}
    showcase = None
    if args.showcase:
        base = (int(args.showcase_frame) if args.showcase_frame is not None
                else _frontal_frame(scene, frames))
        args._showcase_frame_used = base
        showcase = render_showcase(args, scene, avatar, rasterizer, bg, base,
                                   draw_angle=draw_angle)
    return images, perf, showcase


def showcase_poses(n: int = 30) -> list[tuple[float, float]]:
    """展示轨迹：0 yaw → +40 → -40 → 0 → pitch +20 → -20。

    关键帧精确落在采样点上；段内用 smoothstep 缓动（每段起步/收尾都变慢）。
    返回 `(yaw, pitch)`。
    """
    keys = [(0.0, 0.0), (40.0, 0.0), (-40.0, 0.0), (0.0, 0.0),
            (0.0, 20.0), (0.0, -20.0), (0.0, 0.0)]
    n = max(len(keys), int(n))
    nseg = len(keys) - 1
    # 关键帧的帧号，例如 n=30 → [0, 6, 12, 18, 24, 29]
    idx = [round(j * (n - 1) / nseg) for j in range(nseg + 1)]
    out = []
    j = 0
    for i in range(n):
        while j < nseg - 1 and i > idx[j + 1]:
            j += 1
        span = max(1, idx[j + 1] - idx[j])
        f = (i - idx[j]) / span
        f = max(0.0, min(1.0, f))
        f = f * f * (3.0 - 2.0 * f)      # smoothstep：每段缓入缓出
        y0, p0 = keys[j]
        y1, p1 = keys[j + 1]
        out.append((y0 + (y1 - y0) * f, p0 + (p1 - p0) * f))
    return out

def render_showcase(args, scene, avatar, rasterizer, bg,
                    frame_idx: int, *, draw_angle: bool = True) -> np.ndarray:
    """固定 8 个视角的 2x4 展示拼图（GT + 7 个 yaw）。

    第一排：GT、-15、-30、-45
    第二排：0、+15、+30、+45
    只渲染 7 张，不再生成 120 帧视频。
    """
    import torch
    from PIL import Image, ImageDraw

    from live3dgsavatar.core.camera import (orbit_w2c_matrix, rotation_x,
                                            rotation_y)
    from live3dgsavatar.core.render import render_gaussians
    from live3dgsavatar.core.types import Camera, Mesh
    from live3dgsavatar.data.pose import rot6d_to_matrix

    mesh = Mesh(verts=scene.frames_tensor([frame_idx], device="cuda"),
                faces=scene.faces, uvs=scene.uvs, uv_faces=scene.uv_faces)
    weight = scene.weights_tensor([frame_idx], device="cuda")
    K = scene.intrinsics([frame_idx], device="cuda")
    w2c_data = torch.eye(4, dtype=torch.float32)
    w2c_data[:3, :3] = scene.R
    w2c_data[:3, 3] = scene.T
    center = mesh.verts[0].mean(dim=0).detach().cpu()
    head_R = torch.from_numpy(
        rot6d_to_matrix(weight.detach().cpu().numpy()[:, 100:106])[0]).float()
    with torch.no_grad():
        gs = avatar.deform(mesh, weight)

    # 有 RDF 跟踪结果就按"头转"渲染（与 GT 语义一致），否则退回相机绕转
    driver = None
    data_dir = getattr(args, "data", None)
    if data_dir is not None:
        from live3dgsavatar.tracking.pose_drive import PoseDriver
        try:
            driver = PoseDriver(Path(data_dir))
        except Exception:
            driver = None

    yaws = (-15.0, -30.0, -45.0, 0.0, 15.0, 30.0, 45.0)
    images = []
    for yaw in yaws:
        draw_gs = gs
        if driver is not None:
            turned = driver.turn(frame_idx, yaw)
            mesh_t = Mesh(verts=turned.mesh.unsqueeze(0).cuda(),
                          faces=scene.faces, uvs=scene.uvs,
                          uv_faces=scene.uv_faces)
            with torch.no_grad():
                draw_gs = avatar.deform(mesh_t, turned.weight.unsqueeze(0).cuda())
            cam = Camera(K=K, w2c=w2c_data.unsqueeze(0),
                         width=scene.width, height=scene.height).to("cuda")
        else:
            R_target = (rotation_x(0.0) @ rotation_y(yaw))[:3, :3]
            view = orbit_w2c_matrix(w2c_data, center, R_target @ head_R.T)
            cam = Camera(K=K, w2c=view.unsqueeze(0),
                         width=scene.width, height=scene.height).to("cuda")
        if getattr(args, "frontface", False):
            from live3dgsavatar.core.render.frontface import frontface_gaussians
            draw_gs = frontface_gaussians(draw_gs, cam, mesh,
                                          avatar.binding_face_id)
        out = render_gaussians(draw_gs, rasterizer, cam, bg)
        arr = (out.color.clamp(0, 1).permute(0, 2, 3, 1) * 255).byte()
        images.append(np.asarray(Image.fromarray(arr[0].cpu().numpy())))

    # GT 使用与评测相同的 alpha 合成背景，避免把采集白墙带进拼图。
    gt = load_gt(args, [frame_idx])
    if gt is not None and frame_idx in gt:
        gt_rgb, _ = gt[frame_idx]
        gt_img = gt_rgb.astype(np.uint8)
    else:
        gt_img = np.zeros((scene.height, scene.width, 3), np.uint8)
    tiles = [(f"GT {frame_idx:05d}", gt_img)] + [
        (f"yaw {yaw:+.0f}", img) for yaw, img in zip(yaws, images)]

    # 4x2 拼图，第一排 GT/-15/-30/-45，第二排 0/+15/+30/+45。
    order = [0, 1, 2, 3, 4, 5, 6, 7]
    from PIL import Image as _Image, ImageDraw as _ImageDraw
    tile_w, tile_h = scene.width, scene.height
    gap = 4
    label_h = 24
    canvas = _Image.new("RGB", (tile_w * 4 + gap * 3,
                                (tile_h + label_h) * 2 + gap),
                        (10, 11, 14))
    draw = _ImageDraw.Draw(canvas)
    for n, idx in enumerate(order):
        row, col = divmod(n, 4)
        x = col * (tile_w + gap)
        y = row * (tile_h + label_h + gap)
        label, img = tiles[idx]
        draw.text((x + 4, y + 4), label, fill=(232, 234, 240))
        canvas.paste(_Image.fromarray(img), (x, y + label_h))
    return np.asarray(canvas)



def select_indices(total: int, frames: int | None, mode: str, *,
                   seed: int = 0, offset: int = 0,
                   test_ratio: float = 0.3) -> list[int]:
    """按取帧方式返回**全局帧号**列表。

    - `head`   ：从 `offset` 开始连续取（`--frame-offset` 只在这里生效）
    - `tail`   ：取最后 n 帧
    - `random` ：全片随机取 n 帧（`seed` 固定则可复现）
    - `heldout`：取末尾 `test_ratio` 的留出集；`frames` 限定时取该段前 n 帧

    `frames` 为 ``None`` 或 `-1` 时表示全部。留出集比例与训练切分共用
    `train.pipeline.test_ratio`，避免评测和训练对"留出"的定义分叉。
    """
    import numpy as np

    total = int(total)
    n = total if frames is None or int(frames) < 0 else min(int(frames), total)
    if mode == "heldout":
        ratio = float(test_ratio)
        if not 0.0 <= ratio < 1.0:
            raise ValueError(f"test_ratio 应在 [0,1)，实际 {ratio}")
        k = int(round(total * ratio))
        if ratio > 0 and k == 0 and total > 1:
            k = 1
        k = min(k, total - 1) if total > 1 else 0
        start = total - k
        return list(range(start, start + min(n, k)))
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
    帧号。GT 文件必须按 `offset + i` 取，否则 `--frame-offset` 会把
    "渲染第 2513 帧"和"原图第 0 帧"相比。
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
            alpha = arr[..., 3]
            mask = alpha > 250
            # 新数据集用软 alpha 保留发丝；对比视频必须把 alpha 外
            # 的背景合成到统一背景，否则会直接显示采集时的白墙。
            rgb = arr[..., :3].copy()
            bg = np.asarray(getattr(args, "background", [0, 0, 0]),
                            dtype=np.float32) * 255.0
            rgb[~mask] = bg.astype(rgb.dtype)
            out[i] = (rgb, mask)
        else:
            # 白底图没有 alpha：把"接近纯白"的像素当背景。
            # 这不是完美的抠图，但足以把 PSNR 限制在前景区域。
            fg = arr[..., :3].min(axis=-1) < 250
            out[i] = (arr[..., :3], fg)
    return out or None


def _evaluate_images(images: dict[int, np.ndarray], gt, args):
    """单个模型：汇总质量 + 逐帧指标（视频标签用）。"""
    if gt is None:
        return None, None
    quality = summarize_quality(images, gt)
    per_frame: dict[int, dict] = {}
    lpips_acc: list[float] = []
    for i in sorted(gt):
        ref_rgb, ref_mask = gt[i]
        img = images.get(i)
        if img is None or img.shape != ref_rgb.shape:
            continue
        m = frame_metrics(img, ref_rgb, ref_mask,
                          with_lpips=not args.no_lpips)
        if m.get("lpips") is not None:
            lpips_acc.append(float(m["lpips"]))
        per_frame[i] = {"frame": int(i), **m}
    if per_frame:
        quality["per_frame"] = [per_frame[i] for i in sorted(per_frame)]
    if lpips_acc:
        arr = np.asarray(lpips_acc, dtype=np.float64)
        quality["lpips"] = {
            "median": round(float(np.median(arr)), 4),
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
            "mean": round(float(arr.mean()), 4),
            "n": int(arr.size)}
    elif not args.no_lpips:
        print("      [warn] LPIPS 不可用（缺 lpips 或权重），已跳过",
              flush=True)
    return quality, per_frame


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
    offset_note = (f"起始 {args.frame_offset}" if args.frame_mode == "head"
                   else f"frame-mode={args.frame_mode}（frame-offset 仅 head 生效）")
    print(f"  配置  : {frames_text}（{offset_note}）, "
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
    if (args.showcase_frame is not None
            and not 0 <= int(args.showcase_frame) < total):
        print(f"[error] --showcase-frame 越界：{args.showcase_frame}"
              f"（数据集共 {total} 帧）", file=sys.stderr)
        return 1
    indices = select_indices(total, args.frames, args.frame_mode,
                             seed=args.seed, offset=args.frame_offset,
                             test_ratio=args.test_ratio)
    n = len(indices)
    rng = (f"，范围 {indices[0]}–{indices[-1]}" if indices else "")
    print(f"  数据集: 共 {total} 帧，取 {n} 帧（{args.frame_mode}{rng}）",
          flush=True)

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

    # 逐个模型渲染。一次只把一套模型放上 GPU，渲染完释放显存；
    # 图像留在内存里，后面统一做指标、拼视频。
    models: list[dict] = []
    images_root = out_root / "images" if share_dir is not None else None
    for k, name in enumerate(args.work_names):
        print(f"\n  [model {k + 1}/{len(args.work_names)}] {name}", flush=True)
        model_share = None
        if images_root is not None:
            model_share = (images_root / name) if args._multi else images_root
            model_share.mkdir(parents=True, exist_ok=True)
        draw_angle = not args._multi
        model_images, model_perf, model_showcase = render_core(
            args, scene, indices, out_root, model_share,
            ply=args.plies[k], draw_angle=draw_angle)
        models.append({
            "work_name": name,
            "model_path": str(args.plies[k]),
            "model_config": (str(args.model_configs[k])
                             if args.model_configs[k] else None),
            "images": model_images,
            "perf": model_perf,
            "showcase": model_showcase,
        })
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    primary = models[0]
    perf = primary["perf"]
    env = {"python": sys.version.split()[0], "torch": torch.__version__,
           "cuda": torch.version.cuda}
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
    model_params = [{
        "work_name": m["work_name"],
        "model_path": m["model_path"],
        "model_config": m["model_config"],
    } for m in models]
    params = {
        "command": shlex.join(sys.argv),
        "model_path": primary["model_path"],
        "model_config": primary["model_config"],
        "models": model_params,
        # 实际加载的结构（每个模型一份），CLI 默认值不再冒充
        "model_structure": primary["perf"].get("structure", {}),
        "model_structure_per_model": [
            {"work_name": m["work_name"],
             "num_gaussian": m["perf"].get("num_gaussian"),
             **m["perf"].get("structure", {})} for m in models],
        # 溯源：报告必须能落回具体 commit / 文件 / 口径版本（评审 A3/A4/C）
        "provenance": _provenance(models),
        # 评测口径：留出集同时用于选模型和报数，必须写明（评审 A1）
        "evaluation": {
            "test_split": f"last {args.test_ratio:.0%} (heldout)",
            "selection_split": (f"last {args.test_ratio:.0%} (same frames; "
                                "early-stop / model_best)"),
            "selection_metric": "robust = PSNR - 10*alpha_deficit "
                                "(±20/30/40deg head-turn views)",
            "note": "选模型与报数共用同一留出集：绝对值偏乐观，"
                    "配置间比较请用同一选择规则与同一指标版本",
        },
        "dataset_dir": str(args.data),
        "gt_dir": str(args.gt_dir),
        "frames": {
            "mode": args.frame_mode,
            "count": n,
            "indices": indices,
            "range": ([indices[0], indices[-1]] if indices else []),
            "test_ratio": args.test_ratio,
            "seed": args.seed,
            "offset": args.frame_offset,
        },
        "render": {
            "background": args.background,
            "batch_size": args.batch_size,
            "video_fps": args.video_fps,
            "lpips": not args.no_lpips,
            "multi_model": bool(args._multi),
            "frontface": bool(args.frontface),
            "showcase": bool(args.showcase),
            "showcase_frame": getattr(args, "_showcase_frame_used", None),
            "showcase_frames": args.showcase_frames,
            "showcase_fps": args.showcase_fps,
        },
        "config": cfg.resolved(),
        "environment": env,
    }
    result = EvaluationReport(
        subject=args.subject, work_name=args.work_name, dataset=args.dataset,
        num_frames=n, dataset_frames=total,
        width=scene.width, height=scene.height,
        perf=perf, params=params,
        extra={"model_path": primary["model_path"],
               "model_count": len(models),
               "models": [m["work_name"] for m in models],
               "frame_mode": args.frame_mode,
               "frame_range": ([indices[0], indices[-1]]
                               if indices else []),
               "test_ratio": args.test_ratio,
               "showcase_frames": 8 if primary["showcase"] is not None else 0})
    report: dict = result.to_dict()

    print(f"  读取原图：{n} 张（解码 512² PNG，约 {n * 14 / 1000:.0f} s）…",
          flush=True)
    gt = load_gt(args, indices)
    if gt is None:
        print("      [warn] 找不到数据集原图，改做无真值统计", flush=True)
    else:
        print(f"  原图就绪：{len(gt)} 张", flush=True)

    for m in models:
        quality, per_frame = _evaluate_images(m["images"], gt, args)
        m["quality"] = quality
        m["per_frame"] = per_frame
        if gt is None:
            m["no_reference"] = summarize_no_reference(m["images"])
    report["models"] = [{
        "work_name": m["work_name"],
        "model_path": m["model_path"],
        "quality": m.get("quality"),
        "no_reference": m.get("no_reference"),
        "perf": m["perf"],
    } for m in models]
    if models[0].get("quality") is not None:
        report["quality"] = models[0]["quality"]
    else:
        report["no_reference"] = models[0].get("no_reference")

    video_path = None
    if gt is not None:
        # 对比视频：原图 | 模型1 | 模型2 | …；单模型保持原来的两格布局。
        frames_out = []
        for i in sorted(gt):
            ref_rgb, _ = gt[i]
            if args._multi:
                panels = [(f"target {i:05d}", ref_rgb)]
                for m in models:
                    item = (m.get("per_frame") or {}).get(i)
                    if item is None:
                        continue
                    parts = []
                    if item.get("psnr") is not None:
                        parts.append(f"PSNR {item['psnr']:.2f}")
                    if item.get("ssim") is not None:
                        parts.append(f"SSIM {item['ssim']:.4f}")
                    if item.get("lpips") is not None:
                        parts.append(f"LPIPS {item['lpips']:.4f}")
                    panels.append((f"{m['work_name']} | " + " | ".join(parts),
                                   m["images"][i]))
                if len(panels) > 1:
                    frames_out.append(compose_panels(panels))
            else:
                item = (models[0].get("per_frame") or {}).get(i)
                img = models[0]["images"].get(i)
                if item is None or img is None:
                    continue
                parts = [f"PSNR {item['psnr']:.2f}"]
                if item["ssim"] is not None:
                    parts.append(f"SSIM {item['ssim']:.4f}")
                if item["lpips"] is not None:
                    parts.append(f"LPIPS {item['lpips']:.4f}")
                frames_out.append(compose_pair(
                    ref_rgb, img, f"target {i:05d}",
                    f"render {i:05d} | " + " | ".join(parts)))
        if frames_out:
            video_path = write_video(out_root / "compare", frames_out,
                                     args.video_fps)
            print(f"\n  对比视频：{video_path}（{len(frames_out)} 帧，"
                  f"{args.video_fps:.0f} FPS；主模型均值 {perf['fps']:.0f} FPS）")

    showcase_path = None
    showcase_grids = [(m["work_name"], m["showcase"]) for m in models
                      if m.get("showcase") is not None]
    if showcase_grids:
        from PIL import Image
        paths = []
        if args._multi:
            for name, grid in showcase_grids:
                path = out_root / f"showcase_{name}.png"
                Image.fromarray(grid).save(path)
                paths.append(path)
            showcase_path = paths[0]
        else:
            showcase_path = out_root / "showcase.png"
            Image.fromarray(showcase_grids[0][1]).save(showcase_path)
        print(f"  展示拼图：{showcase_path}（2x4；GT/-15/-30/-45 + 0/+15/+30/+45）")

    report_path = out_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    _print_summary(report)
    _print_multi_summary(report)
    print(f"\n  产物：{out_root}")
    print(f"    report.json  指标")
    if video_path is not None:
        print(f"    {video_path.name}  对比视频（原图 | 渲染，带逐帧指标）")
    if showcase_path is not None:
        print(f"    {showcase_path.name}  大角度展示拼图")
    if share_dir is not None:
        count = n * len(models)
        extra = "，按版本分子目录" if args._multi else ""
        print(f"    images/      渲染图（{count} 张{extra}）")
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


def _print_multi_summary(report: dict) -> None:
    models = report.get("models") or []
    if len(models) < 2:
        return
    print("[多模型对比]  mask 内中位指标")
    print(f"  {'版本':28s} {'PSNR':>7s} {'SSIM':>8s} {'LPIPS':>8s} {'锐度比':>7s}")
    for m in models:
        q = m.get("quality") or {}
        psnr = q.get("psnr_masked") or q.get("psnr_full") or {}
        ssim = q.get("ssim") or {}
        lpips = q.get("lpips") or {}
        sharp = q.get("sharpness") or {}
        print(f"  {m['work_name']:28s} "
              f"{psnr.get('median', float('nan')):7.2f} "
              f"{ssim.get('median', float('nan')):8.4f} "
              f"{lpips.get('median', float('nan')):8.4f} "
              f"{sharp.get('ratio', float('nan')):7.3f}")
    print()


def _label_strip(width: int, text: str, height: int = 20):
    """面板上方的深色标签条（ASCII，避免字体依赖）。"""
    from PIL import Image, ImageDraw

    strip = Image.new("RGB", (width, height), (24, 26, 32))
    ImageDraw.Draw(strip).text((6, 4), text, fill=(230, 232, 238))
    return strip


def compose_panels(panels: list[tuple[str, np.ndarray]],
                   strip_h: int = 22, gap: int = 4) -> np.ndarray:
    """任意多图横排：`[(标签, HxWx3 uint8)]`，每格上方带标签。"""
    from PIL import Image, ImageDraw

    if not panels:
        raise ValueError("compose_panels 需要至少一个 panel")
    tiles = [Image.fromarray(np.ascontiguousarray(img)) for _, img in panels]
    w, h = tiles[0].size
    canvas = Image.new("RGB", (w * len(tiles) + gap * (len(tiles) - 1),
                               h + strip_h), (10, 11, 14))
    draw = ImageDraw.Draw(canvas)
    for i, (tile, (label, _)) in enumerate(zip(tiles, panels)):
        x = i * (w + gap)
        draw.text((x + 4, 4), label, fill=(232, 234, 240))
        canvas.paste(tile, (x, strip_h))
    return np.asarray(canvas)


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
