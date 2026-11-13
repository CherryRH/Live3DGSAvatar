"""迭代修复：数据驱动选角 + 累计伪标签 + 保守真实锚定 + 时间记录。

流程（每轮）：
1. `find_defects` 在当前角度圈里选 top-N 最差缺陷视角；
2. `pseudo_label` 为这些视角生成 epiC 伪标签（只接受通过门控的）；
3. 把新标签并入累计 defects 目录（旧标签继续监督）；
4. `refine_from_images` 从当前模型短训，真实帧权重更高（保守）；
5. 记录 before/after、耗时；只有严重崩坏才停，否则继续下一圈。

用法：
    python scripts/iterative_repair.py --subject biden \
        --base-work train1_repair_r1 --prefix top2 --top-n 2 \
        --rings auto --round-offset 1 \
        --seed-defects output/post/biden/top2/r1/defects

产物（每次运行一个目录）：
    output/post/<subject>/<prefix>/
        report.json
        cum/            （累计伪标签 + finetune 产物）
        r1/defects/     （find_defects：views/refs/selected.json）
        r1/pseudo/      （pseudo_label：accepted/debug/report.json）
        r2/ ...
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from live3dgsavatar.config import load_config  # noqa: E402


def parse_rings(text: str) -> list[tuple[float, float]]:
    if str(text).strip().lower() == "auto":
        return []          # 由 main 里按数据集覆盖范围计算
    out = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        yaw, pitch = part.split(",")
        out.append((float(yaw), float(pitch)))
    return out


def run(cmd: list[str], *, dry: bool = False) -> tuple[int, float]:
    print("$ " + " ".join(str(x) for x in cmd), flush=True)
    if dry:
        return 0, 0.0
    t = time.perf_counter()
    p = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    dt = time.perf_counter() - t
    tail = (p.stdout or "").strip().splitlines()[-6:]
    if tail:
        print("\n".join("  " + x for x in tail), flush=True)
    if p.returncode != 0:
        print((p.stderr or "")[-1500:], flush=True)
    return p.returncode, dt


def install_in_place(models_root: Path, subject: str, name: str,
                     src_ply: Path, report_src: Path | None = None) -> Path:
    """把改进后的模型写回同名目录（如 `best`），config 保持不动。"""
    target = models_root / subject / name
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_ply, target / "model.ply")
    if report_src is not None and report_src.is_file():
        shutil.copy2(report_src, target / "report.json")
    return target


def _merge_selected(dst: Path, src: Path, *, keep: set[str],
                    refined_src: Path | None = None) -> None:
    """把 src 的 selected.json 合并到 dst，只保留累计 chosen。

    `refined_src` 指定这一轮通过门控的标签目录。新流程用
    `pseudo/accepted/`，不往 `defects/refined/` 复制一份；
    （`src/refined`）仍然兼容。
    """
    sd = json.loads((src / "selected.json").read_text(encoding="utf-8"))
    sd["views"]["chosen"] = [v for v in sd["views"]["chosen"] if v in keep]
    for sub in ("views", "refs"):
        if (src / sub).is_dir():
            shutil.copytree(src / sub, dst / sub, dirs_exist_ok=True)
    if (src / "mass.npz").is_file():
        shutil.copy2(src / "mass.npz", dst / "mass.npz")
    labels = refined_src if refined_src is not None else src / "refined"
    if not labels.is_dir():
        # 新布局：<run>/rN/defects 的兄弟目录 pseudo/accepted
        alt = src.parent / "pseudo" / "accepted"
        if alt.is_dir():
            labels = alt
    if labels.is_dir():
        out = dst / "refined"
        out.mkdir(parents=True, exist_ok=True)
        for png in labels.glob("*.png"):
            # 只保留 selected.json 里累计 chosen 的标签，避免 accepted 里
            # 残留的旧图混进监督集
            if png.stem in keep:
                shutil.copy2(png, out / png.name)
    (dst / "selected.json").write_text(
        json.dumps(sd, ensure_ascii=False, indent=1), encoding="utf-8")


def label_stems(defects: Path) -> set[str]:
    p = defects / "selected.json"
    if not p.is_file():
        return set()
    d = json.loads(p.read_text(encoding="utf-8"))
    return set(d.get("views", {}).get("chosen", []))


def _drop_round_raw(defects: Path, labels: Path) -> None:
    """cum 已持有合并后的视图/参考/标签，删掉本轮原始副本。"""
    for sub in ("views", "refs"):
        path = defects / sub
        if path.is_dir():
            shutil.rmtree(path)
    mass = defects / "mass.npz"
    if mass.is_file():
        mass.unlink()
    if labels.is_dir():
        shutil.rmtree(labels)


def _drop_finetune_models(finetune: Path) -> None:
    """模型已装进 models/，输出目录只保留报告。"""
    for name in ("model.ply", "model_best.ply"):
        path = finetune / name
        if path.is_file():
            path.unlink()


def _drop_rejected_model(models_dir: Path, subject: str, name: str) -> None:
    """本轮未通过容差而被拒绝时，别在 models/ 留下更差的版本。"""
    target = models_dir / subject / name
    if target.is_dir():
        shutil.rmtree(target)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--base-work", required=True,
                    help="当前模型名（models/<subject>/<name>）")
    ap.add_argument("--prefix", required=True, help="输出前缀，如 top2 / top1")
    ap.add_argument("--in-place", action="store_true",
                    help="每轮只把改进后的模型写回 --base-work（如 best），"
                         "不产生 <prefix>_rN；基座 checkpoint 仍在 output/train/")
    ap.add_argument("--force", action="store_true",
                    help="覆盖已存在的运行目录/模型版本（默认拒绝）")
    # 未给出的数值由 configs/train.yaml 的 post_train 兜底；
    # post_train.sh 会显式传全部用户参数。
    ap.add_argument("--top-n", type=int, default=None)
    ap.add_argument("--rings", default=None,
                    help="每轮 (yaw,pitch) 上限，如 '30,20;40,30;50,30'；"
                         "填 auto 则按数据集覆盖边缘自动生成")
    ap.add_argument("--ring-quantile", type=float, default=0.90,
                    help="auto 模式：用 |yaw|/|pitch| 的该分位数作为覆盖边缘")
    ap.add_argument("--ring-step", type=float, default=10.0)
    ap.add_argument("--ring-max-yaw", type=float, default=50.0)
    ap.add_argument("--ring-max-pitch", type=float, default=30.0)
    ap.add_argument("--round-offset", type=int, default=0,
                    help="第一轮编号偏移（top2 从 r2 开始时填 1）")
    ap.add_argument("--seed-defects", type=Path, default=None,
                    help="已有累计 defects（top2 从 r1 续跑时传 r1）")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--refined-per-batch", type=int, default=None)
    ap.add_argument("--real-per-batch", type=int, default=None)
    ap.add_argument("--lr-xyz", type=float, default=None)
    ap.add_argument("--repair-mode",
                    choices=("syncfix", "mask_diffusion", "none"),
                    default=None,
                    help="syncfix=全图+SyncFix（默认）；"
                         "mask_diffusion=掩码+扩散；none=纯真实帧消融")
    ap.add_argument("--syncfix-repo", default=None)
    ap.add_argument("--syncfix-python", default=None)
    ap.add_argument("--syncfix-model", default=None)
    ap.add_argument("--syncfix-n-per-pass", type=int, default=None)
    ap.add_argument("--syncfix-ref-size", type=int, default=None)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--ip-adapter", choices=("none", "faceid", "faceid_plus"),
                    default=None)
    ap.add_argument("--ip-adapter-weight", default="ip-adapter-faceid_sd15.bin")
    ap.add_argument("--ip-adapter-image-encoder",
                    default="h94/IP-Adapter/sdxl_models/image_encoder")
    ap.add_argument("--ip-adapter-scale", type=float, default=0.6)
    ap.add_argument("--trusted-tol", type=float, default=None)
    ap.add_argument("--mask-dilate", type=int, default=None,
                    help="伪标签掩码向周围扩多少像素（默认 12；扩散上下文）")
    ap.add_argument("--core-dilate", type=int, default=None,
                    help="真正替换回渲染图的 core 外扩（默认 12 = 整掩码替换）")
    ap.add_argument("--strength", type=float, default=None,
                    help="重绘强度（SD1.5 人脸加强版默认 0.85）")
    ap.add_argument("--identity-threshold", type=float, default=0.30,
                    help="ArcFace 身份相似度下限（默认 0.30）")
    ap.add_argument("--color-match", dest="color_match", action="store_true",
                    default=None,
                    help="把生成图颜色分布对齐到周围渲染（默认关：实测压锐度）")
    ap.add_argument("--no-color-match", dest="color_match", action="store_false",
                    default=None, help="关闭颜色对齐（覆盖配置）")
    ap.add_argument("--color-mean-only", action="store_true",
                    help="只对齐颜色均值，保留对比度")
    ap.add_argument("--color-ring", type=int, default=None,
                    help="取渲染图周围多少像素的分布做对齐")
    ap.add_argument("--feather", type=int, default=None,
                    help="core 边界羽化半径（0=硬替换）")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--negative", default=None)
    ap.add_argument("--head-cut", dest="head_cut", action="store_true",
                    default=None,
                    help="unsupported 掩码只修到最低观测像素附近，"
                         "避免补出脖子/胸口/衣服")
    ap.add_argument("--no-head-cut", dest="head_cut", action="store_false",
                    default=None, help="关闭 head_cut（覆盖配置）")
    ap.add_argument("--head-cut-margin", type=int, default=None)
    ap.add_argument("--min-fill", type=float, default=None,
                    help="大掩码下填充率门限（默认 0.45）")
    ap.add_argument("--sharpen", type=float, default=None,
                    help="SD 往返后的 unsharp 增益")
    ap.add_argument("--min-sharp", type=float, default=None,
                    help="整幅前景锐度比下限")
    ap.add_argument("--min-mask-sharp", type=float, default=None,
                    help="掩码内锐度比下限（大掩码可适当放宽）")
    ap.add_argument("--sharp-tolerance", type=float, default=0.12)
    ap.add_argument("--ssim-tolerance", type=float, default=0.03)
    ap.add_argument("--lpips-tolerance", type=float, default=0.08)
    ap.add_argument("--gross-psnr", type=float, default=30.0)
    ap.add_argument("--gross-ssim", type=float, default=0.93)
    ap.add_argument("--gross-lpips", type=float, default=0.10)
    ap.add_argument("--quality-tol", type=float, default=None,
                    help="质量分(SSIM-LPIPS+psnr_weight*PSNR)相对最佳下降超过该值即早停")
    ap.add_argument("--psnr-weight", type=float, default=None,
                    help="PSNR 在质量分中的小权重（默认 0.002/dB）")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="运行根目录；默认 output/post。"
                         "每次运行落在 <out-root>/<subject>/<prefix>/")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    # 命令行未给出的参数由 configs/train.yaml 的 post_train 兜底。
    cfg = load_config()
    def _post(key, default):
        return cfg.get(f"train.pipeline.post_train.{key}", default)
    args.top_n = int(_post("top_n", 2)) if args.top_n is None else args.top_n
    args.rings = str(_post("rings", "auto")) if args.rings is None else args.rings
    args.steps = int(_post("steps", 100)) if args.steps is None else args.steps
    args.real_per_batch = (int(_post("real_per_batch", 6))
                           if args.real_per_batch is None else args.real_per_batch)
    args.refined_per_batch = (int(_post("refined_per_batch", 2))
                              if args.refined_per_batch is None
                              else args.refined_per_batch)
    args.lr_xyz = float(_post("lr_xyz", 1e-5)) if args.lr_xyz is None else args.lr_xyz
    args.repair_mode = (str(_post("repair_mode", "syncfix"))
                        if args.repair_mode is None else args.repair_mode)
    args.syncfix_repo = (str(_post("syncfix_repo", ""))
                         if args.syncfix_repo is None else args.syncfix_repo)
    args.syncfix_python = (str(_post("syncfix_python", "syncfix"))
                           if args.syncfix_python is None else args.syncfix_python)
    args.syncfix_model = (str(_post("syncfix_model", ""))
                          if args.syncfix_model is None else args.syncfix_model)
    args.syncfix_n_per_pass = (int(_post("syncfix_n_per_pass", 4))
                               if args.syncfix_n_per_pass is None
                               else args.syncfix_n_per_pass)
    args.syncfix_ref_size = (int(_post("syncfix_ref_size", 4))
                             if args.syncfix_ref_size is None
                             else args.syncfix_ref_size)
    args.base_model = str(_post("base_model", "emilianJR/epiCRealism")) \
        if args.base_model is None else args.base_model
    args.ip_adapter = str(_post("ip_adapter", "none")) \
        if args.ip_adapter is None else args.ip_adapter
    args.trusted_tol = float(_post("trusted_tol", 3.0)) \
        if args.trusted_tol is None else args.trusted_tol
    args.mask_dilate = int(_post("mask_dilate", 12)) \
        if args.mask_dilate is None else args.mask_dilate
    args.core_dilate = int(_post("core_dilate", 12)) \
        if args.core_dilate is None else args.core_dilate
    args.strength = float(_post("strength", 0.6)) \
        if args.strength is None else args.strength
    args.color_match = bool(_post("color_match", False)) \
        if args.color_match is None else args.color_match
    args.color_ring = int(_post("color_ring", 8)) \
        if args.color_ring is None else args.color_ring
    args.feather = int(_post("feather", 4)) \
        if args.feather is None else args.feather
    args.head_cut = bool(_post("head_cut", True)) \
        if args.head_cut is None else args.head_cut
    args.head_cut_margin = int(_post("head_cut_margin", 0)) \
        if args.head_cut_margin is None else args.head_cut_margin
    args.min_fill = float(_post("min_fill", 0.4)) \
        if args.min_fill is None else args.min_fill
    args.sharpen = float(_post("sharpen", 0.35)) \
        if args.sharpen is None else args.sharpen
    args.min_sharp = float(_post("min_sharp", 0.85)) \
        if args.min_sharp is None else args.min_sharp
    args.min_mask_sharp = float(_post("min_mask_sharp", 0.85)) \
        if args.min_mask_sharp is None else args.min_mask_sharp
    args.quality_tol = float(_post("quality_tol", 0.01)) \
        if args.quality_tol is None else args.quality_tol
    args.psnr_weight = float(_post("psnr_weight", 0.002)) \
        if args.psnr_weight is None else args.psnr_weight
    args.prompt = str(_post("prompt", "")) if args.prompt is None else args.prompt
    args.negative = str(_post("negative", "")) if args.negative is None else args.negative

    if args.repair_mode == "syncfix":
        if not args.syncfix_repo or not Path(args.syncfix_repo).is_dir():
            raise SystemExit(
                f"[error] 找不到 SyncFix 仓库：{args.syncfix_repo or '<空>'}；"
                "请用 --syncfix-repo 或 configs/local.yaml 配置")
        if not args.syncfix_model or not Path(args.syncfix_model).is_dir():
            raise SystemExit(
                f"[error] 找不到 SyncFix 权重：{args.syncfix_model or '<空>'}；"
                "请用 --syncfix-model 或 configs/local.yaml 配置")

    dataset = args.dataset or args.subject
    rings = parse_rings(args.rings)
    if not rings:
        # 从数据集训练段的头姿分布估计"已经覆盖到的边缘"：p90 作为起点，
        # 再按 ring_step 向外扩到配置上限。避免在已有真实数据的角度上
        # 继续让扩散模型重绘（那只会引入噪声）。
        import numpy as np

        from live3dgsavatar.data.datasets import resolve_dataset_dir
        from live3dgsavatar.data.pose import head_pitch, head_yaw

        data_dir = resolve_dataset_dir(cfg, dataset)
        with np.load(data_dir / "render_inputs.npz", allow_pickle=True) as d:
            w = np.asarray(d["blend_weight"], np.float32)
        ratio = float(cfg.get("train.pipeline.test_ratio", 0.3))
        n = len(w)
        train_end = max(1, n - max(1, int(round(n * ratio))))
        yaw = np.abs(head_yaw(w[:train_end]))
        pitch = np.abs(head_pitch(w[:train_end]))
        q = float(np.clip(args.ring_quantile, 0.5, 1.0)) * 100.0
        step = max(1.0, float(args.ring_step))
        edge_yaw = float(np.ceil(np.percentile(yaw, q) / step) * step)
        edge_pitch = float(np.ceil(np.percentile(pitch, q) / step) * step)
        if edge_yaw > args.ring_max_yaw and edge_pitch > args.ring_max_pitch:
            raise SystemExit(
                f"[error] 数据集覆盖边缘 ({edge_yaw:g},{edge_pitch:g}) 已超过上限 "
                f"({args.ring_max_yaw:g},{args.ring_max_pitch:g})")
        y, pi = edge_yaw, edge_pitch
        while y <= args.ring_max_yaw or pi <= args.ring_max_pitch:
            rings.append((min(y, args.ring_max_yaw),
                          min(pi, args.ring_max_pitch)))
            y += step
            pi += step
        print(f"自动选角：数据集 p{args.ring_quantile * 100:.0f} "
              f"|yaw|={np.percentile(yaw, q):.1f} / "
              f"|pitch|={np.percentile(pitch, q):.1f} → "
              f"ring 起点 ({edge_yaw:g},{edge_pitch:g})，共 {len(rings)} 轮",
              flush=True)
    if not rings:
        raise SystemExit("[error] --rings 为空")
    out_root = (args.out_root if args.out_root is not None
                else Path(str(cfg.get("paths.output_dir", "output"))) / "post")
    run_dir = out_root / args.subject / args.prefix
    resume = args.round_offset > 0 or args.seed_defects is not None
    if args.in_place and not args.force:
        raise SystemExit("[error] --in-place 会覆盖 models/<subject>/<base-work>；"
                         "确认请加 --force")
    if run_dir.exists() and not args.dry_run and not resume:
        if not args.force:
            raise SystemExit(
                f"[error] 运行目录已存在：{run_dir}\n"
                "         覆盖会删除旧的轮次与报告；确认请加 --force，"
                "或换个 --prefix")
        shutil.rmtree(run_dir)
    models_dir = Path(str(cfg.get("paths.models_dir", "models")))
    if not args.dry_run and not args.in_place and not resume:
        pattern = (f"{args.prefix}_none" if args.repair_mode == "none"
                   else f"{args.prefix}_r*")
        stale_models = sorted((models_dir / args.subject).glob(pattern))
        if stale_models:
            if not args.force:
                raise SystemExit(
                    f"[error] 已有同名模型版本：{stale_models[0].parent}/"
                    f"{args.prefix}_r*；确认覆盖请加 --force，或换个 --prefix")
            for path in stale_models:
                shutil.rmtree(path)
    run_dir.mkdir(parents=True, exist_ok=True)
    cum = run_dir / "cum"
    if cum.exists() and not args.dry_run:
        shutil.rmtree(cum)
    cum.mkdir(parents=True, exist_ok=True)

    in_place = bool(args.in_place)
    models_root = models_dir

    if args.repair_mode == "none":
        # 消融：只用真实帧微调，不做任何选角/伪标签/SyncFix。
        # 训练总步数与修复路线对齐：steps × 圈数。
        total_steps = args.steps * max(1, len(rings))
        promote = args.base_work if in_place else f"{args.prefix}_none"
        defects_dir = run_dir / "real_only"
        t0 = time.perf_counter()
        cmd = [PY, "scripts/refine_from_images.py",
               "--subject", args.subject, "--work-name", args.base_work,
               "--dataset", dataset, "--defects", str(defects_dir),
               "--real-only", "--steps", str(total_steps),
               "--eval-every", "50", "--eval-frames", "64",
               "--real-per-batch", str(args.real_per_batch),
               "--refined-per-batch", "0",
               "--select-by", "last",
               "--quality-tol", f"{args.quality_tol:g}",
               "--psnr-weight", f"{args.psnr_weight:g}",
               "--promote", "--promote-name", promote, "--force",
               "--lr-xyz", f"{args.lr_xyz:g}",
               "--ssim-weight", "0.2", "--lpips-weight", "0.1",
               "--sharp-tolerance", f"{args.sharp_tolerance:g}",
               "--ssim-tolerance", f"{args.ssim_tolerance:g}",
               "--lpips-tolerance", f"{args.lpips_tolerance:g}"]
        rc, dt = run(cmd, dry=args.dry_run)
        ref_path = defects_dir / "finetune" / "report.json"
        rep = (json.loads(ref_path.read_text(encoding="utf-8"))
               if ref_path.is_file() else {})
        after = rep.get("after")
        out = {
            "prefix": args.prefix, "base": args.base_work,
            "repair_mode": "none", "rings": rings,
            "total_seconds": round(time.perf_counter() - t0, 1),
            "accepted": rc == 0 and after is not None,
            "improved": None,
            "best_model": promote,
            "best_round": {
                "round": "none", "selected": rep.get("selected"),
                "best": after,
                "score": rep.get("selected_score"),
                "best_seconds": rep.get("best_seconds"),
            } if after is not None else None,
            "rounds": [{
                "round": "none", "labels": 0, "selected": rep.get("selected"),
                "best": after, "accepted": rc == 0 and after is not None,
                "seconds": round(dt, 1),
            }],
            "params": {"repair_mode": "none", "steps": total_steps,
                       "real_per_batch": args.real_per_batch,
                       "refined_per_batch": 0},
        }
        (run_dir / "report.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        if rc == 0:
            _drop_finetune_models(defects_dir / "finetune")
        print(f"完成 {args.prefix}（none）：{total_steps} 步，"
              f"{out['total_seconds']}s", flush=True)
        return 0 if out["accepted"] else 1

    keep: set[str] = set()
    if args.seed_defects is not None:
        if not (args.seed_defects / "selected.json").is_file():
            raise SystemExit(f"[error] seed defects 不完整：{args.seed_defects}")
        keep = label_stems(args.seed_defects)
        _merge_selected(cum, args.seed_defects, keep=keep)
        print(f"累计种子：{len(keep)} 个视角", flush=True)

    start = time.perf_counter()
    base = args.base_work
    rounds = []
    best_quality = None
    best_round_tag = None
    best_model_name = args.base_work
    last_round = None
    for i, (yaw, pitch) in enumerate(rings):
        k = args.round_offset + i + 1
        rtag = f"r{k}"
        rdir = run_dir / rtag / "defects"
        gendir = run_dir / rtag / (
            "pseudo" if args.repair_mode == "mask_diffusion" else "syncfix")
        if rdir.exists() and not args.dry_run:
            shutil.rmtree(rdir)
        if gendir.exists() and not args.dry_run:
            shutil.rmtree(gendir)
        t_round = time.perf_counter()
        print(f"\n=== {args.prefix} {rtag}: yaw ±{yaw:g} / pitch ±{pitch:g} "
              f"top-{args.top_n}（base={base}）===", flush=True)

        # SyncFix 联合修复一次要 4 张：视图和参考图都按 n_per_pass 扩；
        # 每帧最多取 1 个视角，保证有同样数量的不同参考帧。
        sel_n = (max(args.top_n, args.syncfix_n_per_pass)
                 if args.repair_mode == "syncfix" else args.top_n)
        sel_extra = (["--per-base-max", "1"]
                     if args.repair_mode == "syncfix" else [])
        rc, dt_sel = run([PY, "scripts/find_defects.py",
                          "--subject", args.subject, "--work-name", base,
                          "--dataset", dataset,
                          "--yaw-limit", f"{yaw:g}",
                          "--pitch-limit", f"{pitch:g}",
                          "--step", "10", "--dense-from", "100",
                          "--budget", str(sel_n),
                          *sel_extra,
                          "--out", str(rdir)], dry=args.dry_run)
        if rc != 0:
            print("[stop] find_defects 失败", flush=True)
            break

        if args.repair_mode == "mask_diffusion":
            gen_cmd = [PY, "scripts/pseudo_label.py",
                       "--subject", args.subject, "--work-name", base,
                       "--dataset", dataset,
                       "--defects", str(rdir), "--out", str(gendir),
                       "--limit", "0", "--mask-mode", "unsupported",
                       "--mask-dilate", str(args.mask_dilate),
                       "--strength", f"{args.strength:g}",
                       "--core-dilate", str(args.core_dilate),
                       "--identity-threshold", f"{args.identity_threshold:g}",
                       "--color-ring", str(args.color_ring),
                       *(["--color-mean-only"] if args.color_mean_only else []),
                       "--feather", str(args.feather),
                       *(["--color-match"] if args.color_match
                         else ["--no-color-match"]),
                       "--prompt", args.prompt,
                       "--negative", args.negative,
                       *(["--head-cut", "--head-cut-margin",
                          str(args.head_cut_margin)]
                         if args.head_cut else []),
                       "--steps", "40", "--consistency", "2",
                       "--photometric-frames", "80",
                       "--photometric-min-px", "16",
                       "--sharpen", f"{args.sharpen:g}",
                       "--min-sharp", f"{args.min_sharp:g}",
                       "--min-mask-sharp", f"{args.min_mask_sharp:g}",
                       "--min-fill", f"{args.min_fill:g}",
                       "--trusted-tol", f"{args.trusted_tol:g}",
                       "--base-model", args.base_model,
                       "--ip-adapter", args.ip_adapter,
                       "--ip-adapter-weight", args.ip_adapter_weight,
                       "--ip-adapter-image-encoder",
                       args.ip_adapter_image_encoder,
                       "--ip-adapter-scale", f"{args.ip_adapter_scale:g}"]
        else:
            gen_cmd = [PY, "scripts/syncfix_label.py",
                       "--defects", str(rdir), "--out", str(gendir),
                       "--repo", args.syncfix_repo,
                       "--python", args.syncfix_python,
                       "--model", args.syncfix_model,
                       "--n-per-pass", str(args.syncfix_n_per_pass),
                       "--ref-size", str(args.syncfix_ref_size),
                       "--height", "512", "--width", "512"]
        rc, dt_gen = run(gen_cmd, dry=args.dry_run)
        if rc != 0:
            print(f"[stop] 标签生成失败（{args.repair_mode}）", flush=True)
            break
        if args.dry_run:
            print("  [dry-run] 跳过累计合并与微调", flush=True)
            rounds.append({"round": rtag, "yaw": yaw, "pitch": pitch,
                           "views": [], "labels": 0, "before": None,
                           "best": None, "seconds": 0.0, "dry_run": True})
            continue

        new = label_stems(rdir)
        if not new:
            print("[stop] 没有选中视角", flush=True)
            break
        keep |= new
        refined_src = (gendir / "accepted"
                       if args.repair_mode == "mask_diffusion" else gendir)
        _merge_selected(cum, rdir, keep=keep, refined_src=refined_src)
        _drop_round_raw(rdir, gendir)
        labels = len(list((cum / "refined").glob("*.png")))
        if labels == 0:
            print("[stop] 没有可用伪标签", flush=True)
            break

        promote = f"{args.prefix}_{rtag}"
        promote_args = ([] if in_place else
                        ["--promote", "--promote-name", promote, "--force"])
        rc, dt_ft = run([PY, "scripts/refine_from_images.py",
                         "--subject", args.subject, "--work-name", base,
                         "--dataset", dataset, "--defects", str(cum),
                         "--steps", str(args.steps), "--eval-every", "50",
                         "--eval-frames", "64",
                         "--select-by", "last",
                         "--quality-tol", f"{args.quality_tol:g}",
                         "--psnr-weight", f"{args.psnr_weight:g}",
                         *promote_args,
                         "--lr-xyz", f"{args.lr_xyz:g}",
                         "--ssim-weight", "0.2", "--lpips-weight", "0.1",
                         "--refined-per-batch", str(args.refined_per_batch),
                         "--real-per-batch", str(args.real_per_batch),
                         "--sharp-tolerance", f"{args.sharp_tolerance:g}",
                         "--ssim-tolerance", f"{args.ssim_tolerance:g}",
                         "--lpips-tolerance", f"{args.lpips_tolerance:g}"],
                        dry=args.dry_run)

        report_path = cum / "finetune" / "report.json"
        best = None
        rep = None
        if report_path.is_file():
            rep = json.loads(report_path.read_text(encoding="utf-8"))
            hist = [h for h in rep.get("score_history", []) if h.get("passed")]
            # refine 的最终选择：last（容差内）或 best；after 就是实际落盘的模型
            best = rep.get("after")
            if best is None and hist:
                best = max(hist, key=lambda h: (
                    h["score"] if h.get("score") is not None else
                    h.get("ssim", 0.0) - h.get("lpips", 1.0)
                    + args.psnr_weight * h.get("psnr", 0.0)))
        dt_round = time.perf_counter() - t_round
        best_secs = rep.get("best_seconds") if rep is not None else None
        q = (rep.get("selected_score") if rep is not None else None)
        if q is None and best is not None:
            q = (best.get("ssim", 0.0) - best.get("lpips", 1.0)
                 + args.psnr_weight * best.get("psnr", 0.0))
        rounds.append({
            "round": rtag, "yaw": yaw, "pitch": pitch,
            "views": sorted(new), "labels": labels,
            "model": args.base_work if in_place else promote,
            "promoted": False,
            "improved": False,
            "accepted": False,
            "selected": (rep.get("selected") if rep is not None else None),
            "score": (round(q, 4) if q is not None else None),
            "best_seconds": best_secs,
            "before": (rep.get("before") if rep is not None else None),
            "best": best,
            "seconds": round(dt_round, 1),
            "select_seconds": round(dt_sel, 1),
            "gen_seconds": round(dt_gen, 1),
            "ft_seconds": round(dt_ft, 1),
        })
        print(f"  {rtag} 耗时 {dt_round:.1f}s（选角 {dt_sel:.1f}s / "
              f"生成 {dt_gen:.1f}s / 微调 {dt_ft:.1f}s），标签 {labels} 张"
              + (f"，选中 {rep.get('selected')}" if rep is not None else ""),
              flush=True)
        if rc != 0:
            print("[stop] refine 失败", flush=True)
            break
        if best is None or q is None:
            _drop_finetune_models(cum / "finetune")
            print("[stop] 没有通过的检查点", flush=True)
            break
        if not in_place:
            # refine 内部已经把本轮模型装进 models/<subject>/<prefix>_rN
            rounds[-1]["promoted"] = True
        before = rep.get("before") if rep is not None else None
        degraded = bool(before) and (
            (before.get("psnr") is not None
             and best.get("psnr", 0.0) < before["psnr"] - 1.0)
            or (before.get("ssim") is not None
                and best.get("ssim", 0.0) < before["ssim"] - 0.01)
            or (before.get("lpips") is not None
                and best.get("lpips", 1.0) > before["lpips"] + 0.01))
        if (best["psnr"] < args.gross_psnr
                or best["ssim"] < args.gross_ssim
                or best["lpips"] > args.gross_lpips) and degraded:
            if not in_place and rounds[-1].get("promoted"):
                _drop_rejected_model(models_dir, args.subject, promote)
                rounds[-1]["promoted"] = False
            _drop_finetune_models(cum / "finetune")
            print(f"[stop] 严重崩坏（低于基座 {before}）：{best}", flush=True)
            break

        improved = best_quality is None or q > best_quality
        # 用户策略：小角度留出集测不出大角度变化；只要没超出下降容差，
        # 就接受"最后一个"模型，而不是挑分数最高的那一轮。
        accepted = best_quality is None or q >= best_quality - args.quality_tol
        rounds[-1]["improved"] = bool(improved)
        rounds[-1]["accepted"] = bool(accepted)
        if not accepted:
            if not in_place and rounds[-1].get("promoted"):
                _drop_rejected_model(models_dir, args.subject, promote)
                rounds[-1]["promoted"] = False
            _drop_finetune_models(cum / "finetune")
            print(f"[stop] 质量分 {q:.4f} 低于最佳 {best_quality:.4f} "
                  f"超过 {args.quality_tol:g}，早停（保留上一版）", flush=True)
            break
        if improved:
            best_quality = q
        if in_place:
            install_in_place(models_root, args.subject, args.base_work,
                             cum / "finetune" / "model.ply",
                             cum / "finetune" / "report.json")
            rounds[-1]["promoted"] = True
            print(f"  写回 models/{args.subject}/{args.base_work}"
                  f"（{'本轮更优' if improved else '容差内，接受最后一版'}）",
                  flush=True)
            best_model_name = args.base_work
        else:
            base = promote
            best_model_name = promote
        last_round = rounds[-1]
        if rounds[-1].get("promoted"):
            _drop_finetune_models(cum / "finetune")
        print(f"  下一轮 base={base}", flush=True)

    total = time.perf_counter() - start
    _drop_finetune_models(cum / "finetune")
    # 选最后一个被接受的轮次
    best_round = last_round
    out = {
        "prefix": args.prefix, "base": args.base_work,
        "top_n": args.top_n, "rings": rings,
        # 记录生成这条结果的确切参数：换生成器/掩码后仍能追溯来源
        "params": {
            "repair_mode": args.repair_mode,
            "syncfix_model": args.syncfix_model,
            "syncfix_n_per_pass": args.syncfix_n_per_pass,
            "syncfix_ref_size": args.syncfix_ref_size,
            "base_model": args.base_model,
            "ip_adapter": args.ip_adapter,
            "mask_dilate": args.mask_dilate,
            "core_dilate": args.core_dilate,
            "strength": args.strength,
            "min_fill": args.min_fill,
            "trusted_tol": args.trusted_tol,
            "sharpen": args.sharpen,
            "min_sharp": args.min_sharp,
            "min_mask_sharp": args.min_mask_sharp,
            "steps": args.steps,
            "lr_xyz": args.lr_xyz,
            "real_per_batch": args.real_per_batch,
            "refined_per_batch": args.refined_per_batch,
            "quality_tol": args.quality_tol,
            "psnr_weight": args.psnr_weight,
            "color_match": args.color_match,
            "feather": args.feather,
            "head_cut": args.head_cut,
            "head_cut_margin": args.head_cut_margin,
        },
        "rounds": rounds, "total_seconds": round(total, 1),
        "in_place": in_place,
        "improved": any(r.get("improved") for r in rounds),
        "accepted": any(r.get("accepted") for r in rounds),
        "selection": "last-within-tol",
        "best_round": best_round,
        "best_model": best_model_name,
    }
    report_path = run_dir / "report.json"
    report_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n完成 {args.prefix}：{len(rounds)} 轮，总 {total:.1f}s", flush=True)
    if in_place:
        updated = any(r.get("promoted") for r in rounds)
        print(f"  模型：models/{args.subject}/{args.base_work}"
              f"（{'已更新' if updated else '未接受，保持基座'}）", flush=True)
    if best_round:
        best_secs = best_round.get("best_seconds")
        print(f"  最终 {best_round['round']}（{best_round.get('selected')}）："
              f"{best_round['best']}"
              + (f"，累计 {best_secs:.1f}s" if best_secs is not None else ""),
              flush=True)
    elif not in_place:
        print(f"  未接受，保持 models/{args.subject}/{args.base_work}", flush=True)
    print(f"  报告：{report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
