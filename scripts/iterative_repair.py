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
        --rings "30,20;40,30;50,30" --round-offset 1 \
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


def parse_rings(text: str) -> list[tuple[float, float]]:
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
    `pseudo/accepted/`，不再往 `defects/refined/` 复制一份；旧目录
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="biden")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--base-work", required=True,
                    help="当前模型名（models/<subject>/<name>）")
    ap.add_argument("--prefix", required=True, help="输出前缀，如 top2 / top1")
    ap.add_argument("--in-place", action="store_true",
                    help="每轮只把改进后的模型写回 --base-work（如 best），"
                         "不产生 <prefix>_rN；基座 checkpoint 仍在 output/train/")
    ap.add_argument("--top-n", type=int, default=2)
    ap.add_argument("--rings", required=True,
                    help="每轮 (yaw,pitch) 上限，如 '30,20;40,30;50,30'")
    ap.add_argument("--round-offset", type=int, default=0,
                    help="第一轮编号偏移（top2 从 r2 开始时填 1）")
    ap.add_argument("--seed-defects", type=Path, default=None,
                    help="已有累计 defects（top2 从 r1 续跑时传 r1）")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--refined-per-batch", type=int, default=2)
    ap.add_argument("--real-per-batch", type=int, default=8)
    ap.add_argument("--lr-xyz", type=float, default=1e-5)
    ap.add_argument("--base-model", default="emilianJR/epiCRealism")
    ap.add_argument("--ip-adapter", choices=("none", "faceid", "faceid_plus"),
                    default="none")
    ap.add_argument("--ip-adapter-weight", default="ip-adapter-faceid_sd15.bin")
    ap.add_argument("--ip-adapter-image-encoder",
                    default="h94/IP-Adapter/sdxl_models/image_encoder")
    ap.add_argument("--ip-adapter-scale", type=float, default=0.6)
    ap.add_argument("--trusted-tol", type=float, default=3.0)
    ap.add_argument("--mask-dilate", type=int, default=12,
                    help="伪标签掩码向周围扩多少像素（默认 12；扩散上下文）")
    ap.add_argument("--core-dilate", type=int, default=12,
                    help="真正替换回渲染图的 core 外扩（默认 12 = 整掩码替换）")
    ap.add_argument("--strength", type=float, default=0.85,
                    help="重绘强度（SD1.5 人脸加强版默认 0.85）")
    ap.add_argument("--identity-threshold", type=float, default=0.30,
                    help="ArcFace 身份相似度下限（默认 0.30）")
    ap.add_argument("--color-match", dest="color_match", action="store_true",
                    default=False,
                    help="把生成图颜色分布对齐到周围渲染（默认关：实测压锐度）")
    ap.add_argument("--no-color-match", dest="color_match", action="store_false",
                    help="关闭颜色对齐（默认）")
    ap.add_argument("--color-mean-only", action="store_true",
                    help="只对齐颜色均值，保留对比度")
    ap.add_argument("--color-ring", type=int, default=8,
                    help="取渲染图周围多少像素的分布做对齐")
    ap.add_argument("--feather", type=int, default=4,
                    help="core 边界羽化半径（0=硬替换）")
    ap.add_argument("--prompt",
                    default="close-up of a bare face and neck, natural bare "
                            "skin, visible pores, eyes looking straight ahead, "
                            "soft light, plain background, sharp focus")
    ap.add_argument("--negative",
                    default="looking at camera, looking at viewer, eye contact, "
                            "side glance, cross-eyed, clothing, shirt, collar, "
                            "jacket, scarf, necklace, earrings, jewelry, hat, "
                            "glasses, makeup, lipstick, eyeshadow, blush, "
                            "painted face, beauty filter, blurry, smooth plastic, "
                            "deformed, extra eyes, low resolution, watermark")
    ap.add_argument("--head-cut", action="store_true",
                    help="unsupported 掩码只修到最低观测像素附近，"
                         "避免补出脖子/胸口/衣服")
    ap.add_argument("--head-cut-margin", type=int, default=0)
    ap.add_argument("--min-fill", type=float, default=0.45,
                    help="大掩码下填充率门限（默认 0.45）")
    ap.add_argument("--sharpen", type=float, default=0.35,
                    help="SD 往返后的 unsharp 增益")
    ap.add_argument("--min-sharp", type=float, default=0.95,
                    help="整幅前景锐度比下限")
    ap.add_argument("--min-mask-sharp", type=float, default=0.90,
                    help="掩码内锐度比下限（大掩码可适当放宽）")
    ap.add_argument("--sharp-tolerance", type=float, default=0.12)
    ap.add_argument("--ssim-tolerance", type=float, default=0.03)
    ap.add_argument("--lpips-tolerance", type=float, default=0.08)
    ap.add_argument("--gross-psnr", type=float, default=30.0)
    ap.add_argument("--gross-ssim", type=float, default=0.93)
    ap.add_argument("--gross-lpips", type=float, default=0.10)
    ap.add_argument("--quality-tol", type=float, default=0.01,
                    help="质量分(SSIM-LPIPS+psnr_weight*PSNR)相对最佳下降超过该值即早停")
    ap.add_argument("--psnr-weight", type=float, default=0.002,
                    help="PSNR 在质量分中的小权重（默认 0.002/dB）")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="运行根目录；默认 output/post。"
                         "每次运行落在 <out-root>/<subject>/<prefix>/")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    dataset = args.dataset or args.subject
    rings = parse_rings(args.rings)
    if not rings:
        raise SystemExit("[error] --rings 为空")
    out_root = (args.out_root if args.out_root is not None
                else REPO_ROOT / "output" / "post")
    run_dir = out_root / args.subject / args.prefix
    # 全新运行先清掉同名目录，避免上一次多出来的 r3/r4 残留；
    # 续跑（--round-offset / --seed-defects）则只重建 cum。
    if (run_dir.exists() and not args.dry_run
            and args.round_offset == 0 and args.seed_defects is None):
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cum = run_dir / "cum"
    if cum.exists() and not args.dry_run:
        shutil.rmtree(cum)
    cum.mkdir(parents=True, exist_ok=True)

    in_place = bool(args.in_place)
    models_root = REPO_ROOT / "models"

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
    for i, (yaw, pitch) in enumerate(rings):
        k = args.round_offset + i + 1
        rtag = f"r{k}"
        rdir = run_dir / rtag / "defects"
        gendir = run_dir / rtag / "pseudo"
        if rdir.exists() and not args.dry_run:
            shutil.rmtree(rdir)
        if gendir.exists() and not args.dry_run:
            shutil.rmtree(gendir)
        t_round = time.perf_counter()
        print(f"\n=== {args.prefix} {rtag}: yaw ±{yaw:g} / pitch ±{pitch:g} "
              f"top-{args.top_n}（base={base}）===", flush=True)

        rc, dt_sel = run([PY, "scripts/find_defects.py",
                          "--subject", args.subject, "--work-name", base,
                          "--dataset", dataset,
                          "--yaw-limit", f"{yaw:g}",
                          "--pitch-limit", f"{pitch:g}",
                          "--step", "10", "--dense-from", "100",
                          "--budget", str(args.top_n),
                          "--out", str(rdir)], dry=args.dry_run)
        if rc != 0:
            print("[stop] find_defects 失败", flush=True)
            break

        rc, dt_gen = run([PY, "scripts/pseudo_label.py",
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
                          "--ip-adapter-scale", f"{args.ip_adapter_scale:g}",
                          ], dry=args.dry_run)
        if rc != 0:
            print("[stop] pseudo_label 失败", flush=True)
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
        _merge_selected(cum, rdir, keep=keep,
                        refined_src=gendir / "accepted")
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
                         "--select-by", "quality",
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
            if hist:
                # 与 model_best 的选择一致：quality = SSIM − LPIPS + w·PSNR
                best = max(hist, key=lambda h: (
                    h.get("ssim", 0.0) - h.get("lpips", 1.0)
                    + args.psnr_weight * h.get("psnr", 0.0)))
        dt_round = time.perf_counter() - t_round
        best_secs = rep.get("best_seconds") if rep is not None else None
        rounds.append({
            "round": rtag, "yaw": yaw, "pitch": pitch,
            "views": sorted(new), "labels": labels,
            "model": args.base_work if in_place else promote,
            "promoted": False,
            "improved": False,
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
              + (f"，最佳 {best_secs:.1f}s" if best_secs is not None else ""),
              flush=True)
        if rc != 0:
            print("[stop] refine 失败", flush=True)
            break
        if best is None:
            print("[stop] 没有通过的检查点", flush=True)
            break
        if not in_place:
            # refine 内部已经把本轮模型装进 models/<subject>/<prefix>_rN
            rounds[-1]["promoted"] = True
        before = rep.get("before") if rep is not None else None
        # 绝对阈值只是兜底：不同对象/基座的 SSIM 本来就不一样（biden 0.95，
        # 别的对象可能 0.92），只按绝对值会把"正常但偏低"当成崩坏。
        # 真正的崩坏 = 既低于绝对阈值，又比本轮基座明显退步。
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
            print(f"[stop] 严重崩坏（低于基座 {before}）：{best}", flush=True)
            break
        q = (best.get("ssim", 0.0) - best.get("lpips", 1.0)
             + args.psnr_weight * best.get("psnr", 0.0))
        improved = best_quality is None or q > best_quality
        rounds[-1]["improved"] = bool(improved)
        if improved:
            best_quality = q
            best_round_tag = rtag
            if not in_place:
                best_model_name = promote
        if in_place:
            # 只有更好才覆盖，避免后一轮退化把 best 写坏
            if improved:
                install_in_place(models_root, args.subject, args.base_work,
                                 cum / "finetune" / "model.ply",
                                 cum / "finetune" / "report.json")
                rounds[-1]["promoted"] = True
                print(f"  写回 models/{args.subject}/{args.base_work}"
                      "（本轮质量更好）", flush=True)
            elif q < best_quality - args.quality_tol:
                print(f"[stop] 质量分 {q:.4f} 低于最佳 {best_quality:.4f} "
                      f"超过 {args.quality_tol:g}，早停（不动 best）", flush=True)
                break
        else:
            if not improved and q < best_quality - args.quality_tol:
                print(f"[stop] 质量分 {q:.4f} 低于最佳 {best_quality:.4f} "
                      f"超过 {args.quality_tol:g}，早停", flush=True)
                break
            # 只有更好的轮次才作为下一轮起点；更差但仍可接受的轮次
            # 不接进链里，避免 LPIPS/感知质量逐轮累积退化
            base = promote if improved else best_model_name
        print(f"  下一轮 base={base}", flush=True)

    total = time.perf_counter() - start
    passed = [r for r in rounds if r.get("best")]
    best_round = max(
        passed,
        key=lambda r: (r["best"].get("ssim", 0.0)
                       - r["best"].get("lpips", 1.0)
                       + args.psnr_weight * r["best"].get("psnr", 0.0)),
    ) if passed else None
    out = {
        "prefix": args.prefix, "base": args.base_work,
        "top_n": args.top_n, "rings": rings,
        # 记录生成这条结果的确切参数：换生成器/掩码后仍能追溯来源
        "params": {
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
              f"（{'已更新' if updated else '未改进，保持基座'}）", flush=True)
    if best_round:
        best_secs = best_round.get("best_seconds")
        print(f"  最佳 {best_round['round']}：{best_round['best']}"
              + (f"，最佳累计 {best_secs:.1f}s" if best_secs is not None else ""),
              flush=True)
    print(f"  报告：{report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
