#!/usr/bin/env python
"""渲染测试：本项目 `core/` 出图，与数据集原图对比，并测性能。

判据只有两项：

- **重建质量**：与数据集原图的 PSNR（全图 / 前景掩码内）
- **性能**：单帧耗时、FPS、峰值显存

用法::

    python scripts/render_test.py --frames 20
    python scripts/render_test.py --frames -1 --batch-size 4 --dump-diff
    python scripts/render_test.py --dry-run          # 无需 GPU 的形状/约定自检
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

from live3dgsavatar.config import (Config, load_config, resolve_data_dir,  # noqa: E402
                                   resolve_model_config, resolve_model_ply)


# ------------------------------------------------------------------ CLI --


def _normalize_argv(argv: list[str]) -> list[str]:
    """把 `--work_name` 规整成 `--work-name`（两种写法都接受）。"""
    return [a.replace("_", "-") for a in argv if a.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="渲染测试（本项目 core vs 数据集原图）")
    p.add_argument("--subject", default=None, help="覆盖配置里的 subject")
    p.add_argument("--work-name", default=None, help="覆盖配置里的 work_name")
    p.add_argument("--frames", type=int, default=None, help="帧数；-1 表示全部")
    # `default=None`：脚本不硬编码业务默认量，未给时在 `_finalize` 里取 0
    p.add_argument("--frame-offset", type=int, default=None,
                   help="从第几帧开始取（配合 --frames 可只评估留出的尾段）")
    p.add_argument("--batch-size", type=int, default=None, help="批大小")
    p.add_argument("--ply", type=Path, default=None,
                   help="直接指定模型文件（默认取模型目录下的 model.ply）")
    p.add_argument("--data", type=Path, default=None, help="数据集目录（覆盖配置）")
    p.add_argument("--cap4d", type=Path, default=None,
                   help="CAP4D 的输出目录（含 generated_images/）；"
                        "给了它就渲染这些生成图并与它们对比")
    p.add_argument("--output-dir", type=Path, default=None, help="产物目录")
    p.add_argument("--no-save", action="store_true",
                   help="不落盘、不拷贝到主机（干净测量渲染耗时）")
    p.add_argument("--dump-diff", action="store_true",
                   help="输出对比图（默认开）")
    p.add_argument("--no-diff", action="store_true",
                   help="不输出对比图（只出 images 与 report）")
    p.add_argument("--background", type=float, nargs=3, default=None,
                   metavar=("R", "G", "B"), help="背景色")
    p.add_argument("--tex-size", type=int, default=None)
    p.add_argument("--num-basis-in", type=int, default=None)
    p.add_argument("--num-basis-blend", type=int, default=None)
    p.add_argument("--mlp-hidden", type=int, nargs="+", default=None)
    p.add_argument("--no-weight-proj", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="无需 GPU：用真实模型 + 合成帧验证形状与约定")
    return p.parse_args(_normalize_argv(argv) if argv is not None else None)


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
    args._explicit_tex_size = args.tex_size is not None
    args._explicit_basis_in = args.num_basis_in is not None
    args._explicit_basis_blend = args.num_basis_blend is not None
    args._explicit_mlp_hidden = args.mlp_hidden is not None
    if args.subject is not None:
        cfg.set("subject", args.subject)
    if args.work_name is not None:
        cfg.set("work_name", args.work_name)
    if args.data is not None:
        cfg.set("paths.data_root", args.data.parent)
        cfg.set("subject", args.data.name)
    # `--ply` 是"用户直接指的文件"，按当前工作目录解析；不能塞进
    # `paths.model_ply`（那里是**相对模型目录的文件名**），否则会被拼成
    # `models/.../models/...` 这种路径
    explicit_ply = None
    if args.ply is not None:
        explicit_ply = Path(args.ply).expanduser()
        if not explicit_ply.is_file():
            raise SystemExit(f"[error] --ply 不存在：{explicit_ply}")

    if not (str(cfg.get("subject") or "").strip()
            and str(cfg.get("work_name") or "").strip()):
        raise SystemExit(
            "[error] 需要指定模型：--subject 与 --work-name\n"
            "        默认配置不再写死对象名，所以必须显式给出。")
    args.subject = str(cfg.get("subject"))
    args.work_name = str(cfg.get("work_name"))
    args.data = resolve_data_dir(cfg)
    args.ply = (explicit_ply.resolve() if explicit_ply is not None
                else resolve_model_ply(cfg))
    args.model_config = resolve_model_config(cfg)
    args.gt_dir = args.data / str(cfg.get("paths.image_subdir", "images"))
    # 对比图默认输出；配置里可整体关掉，命令行 --no-diff 优先
    args.dump_diff = not args.no_diff and (
        args.dump_diff or bool(cfg.get("test.dump_diff", True)))

    # 默认按 "<对象>/<版本>" 分目录：否则不同模型的对比图会互相覆盖。
    out_root = Path(args.output_dir) if args.output_dir else (
        Path(str(cfg.get("paths.output_dir", "output"))) / "render_test"
        / str(cfg.get("subject")) / str(cfg.get("work_name") or "default"))
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
    if args.background is None:
        args.background = [float(x) for x in cfg.get("render.background", [0, 0, 0])]
    return args


# --------------------------------------------------------------- 指标 --
# 口径与 GUI/在线更新一致，统一由 `core.metrics` 提供。

from live3dgsavatar.core.metrics import (Accum, max_abs_diff,  # noqa: E402
                                         psnr, psnr_masked)


# ------------------------------------------------------------- 渲染 --


def render_core(args, scene, frames: list[int], out_dir: Path,
                share_dir: Path | None = None):
    """用本项目 `core/` 渲染。

    Args:
        share_dir: 非 ``None`` 时落盘 PNG 并返回图像数组；``None``
            （`--no-save`）时**跳过 D2H 拷贝与编码**，只做渲染与计时。
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

    def run_batch(idx: list[int]):
        mesh = Mesh(verts=scene.frames_tensor(idx, device="cuda"),
                    faces=scene.faces, uvs=scene.uvs, uv_faces=scene.uv_faces)
        # 逐帧内参（CAP4D 每帧裁剪框不同）；数据集走广播
        K = scene.intrinsics(idx, device="cuda")
        camera = Camera(K=K, w2c=w2c.expand(len(idx), 4, 4).contiguous(),
                        width=scene.width, height=scene.height).to("cuda")
        with torch.no_grad():
            gs = avatar.deform(mesh, scene.weights_tensor(idx, device="cuda"))
            return rasterizer.render(gs, camera, bg)

    run_batch(frames[:1])                       # 预热
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    images: dict[int, np.ndarray] = {}
    per_frame_ms: list[float] = []
    for start in range(0, len(frames), args.batch_size):
        chunk = frames[start:start + args.batch_size]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = run_batch(chunk)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        per_frame_ms.extend([dt / len(chunk)] * len(chunk))

        if share_dir is None:
            # 干净测量：连 D2H 拷贝都不做（窗口前的 `.cpu()` 会同步等待内核）
            continue

        from PIL import Image

        arr = (out.color.clamp(0, 1).permute(0, 2, 3, 1) * 255).byte().cpu().numpy()
        for j, i in enumerate(chunk):
            images[i] = arr[j]
            Image.fromarray(arr[j]).save(share_dir / f"{i:05d}.png")

    return images, {
        "mean_ms_per_frame": float(np.mean(per_frame_ms)),
        "median_ms_per_frame": float(np.median(per_frame_ms)),
        "fps": 1000.0 / float(np.mean(per_frame_ms)),
        "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
        "batch_size": args.batch_size,
        "num_gaussian": int(avatar.num_gaussians),
    }


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
            # 白底图（CAP4D）没有 alpha：把"接近纯白"的像素当背景。
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
    print(f"  配置  : {args.frames} 帧（起始 {args.frame_offset}）, "
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

    scene = load_scene(cfg, frames=args.frames, frame_offset=args.frame_offset)
    total = scene.total_frames
    n = scene.num_frames

    if args.cap4d is not None:
        # 用 **RDF 拟合生成图** 的结果：网格与相机天然自洽，不需要任何换算。
        # RDF 的虚拟相机（fov 30°、(0,0,1)、diag(1,-1,-1)）与本数据集相机一致，
        # 所以这里直接沿用 scene 的 K/R/T。
        from live3dgsavatar.data.cap4d import images_dir, load_rdf_dump, to_frames

        assets = Path(args.cap4d)
        dump_path = assets / "cap4d_rdf.npz"
        if not dump_path.is_file():
            raise SystemExit(
                f"[error] 找不到 {dump_path}\n"
                "        先在 realdenseface 环境里让 RDF 拟合这些图：\n"
                f"        python scripts/rdf_dump.py --images {images_dir(assets)} "
                f"--out {dump_path}")
        dump = load_rdf_dump(dump_path)
        scene.frames = to_frames(dump, Path(str(cfg.get("paths.flame_dir"))))
        scene.dataset_frames = len(scene.frames)
        n = len(scene.frames)
        # 生成图是白底，对比时背景也用白（否则 PSNR 全被背景吃掉）
        if args.background is None:
            args.background = [1.0, 1.0, 1.0]
        args.gt_dir = images_dir(assets)
        print(f"  CAP4D: {n} 帧（{dump_path}）")
    print(f"  数据集: 载入 {n} / 共 {total} 帧")

    out_root = args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)
    share_dir = None if args.no_save else out_root / "images"
    if share_dir is not None:
        share_dir.mkdir(parents=True, exist_ok=True)

    images, perf = render_core(args, scene, list(range(n)), out_root, share_dir)

    report: dict = {"subject": args.subject, "work_name": args.work_name,
                    "model_path": str(args.ply),
                    "num_frames": n, "dataset_frames": total,
                    "width": scene.width, "height": scene.height, "perf": perf}

    gt = load_gt(args, list(range(n)), offset=args.frame_offset)
    if gt is None:
        print("      [warn] 找不到数据集原图，跳过质量对比")
    else:
        full, masked, mad = Accum(), Accum(), Accum()
        for i, img in images.items():
            if i not in gt:
                continue
            ref_rgb, ref_mask = gt[i]
            full.add(psnr(img, ref_rgb))
            masked.add(psnr_masked(img, ref_rgb, ref_mask))
            mad.add(max_abs_diff(img, ref_rgb))
        report["quality"] = {"psnr_full": full.summary(),
                             "psnr_masked": masked.summary(),
                             "max_abs_diff": mad.summary()}

    # 原图 | 渲染结果（每格带 PSNR 标签）—— 一眼能看出差在哪一帧
    if args.dump_diff and not args.no_save and gt and images:
        diff_dir = out_root / "diff"
        written = 0
        for i in sorted(gt):
            ref_rgb, ref_mask = gt[i]
            img = images.get(i)
            if img is None or img.shape != ref_rgb.shape:
                continue
            value = psnr_masked(img, ref_rgb, ref_mask)
            if value is None:
                value = psnr(img, ref_rgb)
            label = f"{value:.2f} dB"
            write_comparison(diff_dir / f"{i:05d}.png",
                             [(f"target {i:05d}", ref_rgb),
                              (f"render  {label}", img)])
            written += 1
        print(f"\n  对比图：{diff_dir}（{written} 张，原图 | 渲染）")

    report_path = out_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    _print_summary(report)
    print(f"\n  报告：{report_path}")
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
        s = q.get("max_abs_diff")
        if s:
            print(f"  {'max|Δ| (0-255)':24s} 中位 {s['median']:.0f}  "
                  f"最大 {s['max']:.0f}")
    else:
        print("\n[重建质量]  未做（缺少数据集原图）")

    p = report["perf"]
    print("\n[性能]")
    print(f"  {'平均耗时 (ms/帧)':28s} {p['mean_ms_per_frame']:.2f}")
    print(f"  {'中位耗时 (ms/帧)':28s} {p['median_ms_per_frame']:.2f}")
    print(f"  {'吞吐 (FPS)':28s} {p['fps']:.1f}")
    print(f"  {'峰值显存 (MiB)':28s} {p['peak_memory_mib']:.0f}")
    print(f"  {'批大小':28s} {p['batch_size']}")
    print(f"  {'高斯数':28s} {p['num_gaussian']}")
    print()


def _label_strip(width: int, text: str, height: int = 20):
    """面板上方的深色标签条（ASCII，避免字体依赖）。"""
    from PIL import Image, ImageDraw

    strip = Image.new("RGB", (width, height), (24, 26, 32))
    ImageDraw.Draw(strip).text((6, 4), text, fill=(230, 232, 238))
    return strip


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
