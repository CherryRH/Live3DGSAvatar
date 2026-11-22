#!/usr/bin/env python
"""训练入口：在**我们自己的管线**上训一个约简高斯 blendshape 头像。

流程：

1. 从 `datasets/<subject>/` 读 `render_inputs.npz` + 图像，末尾 N 帧作验证
2. 从模型目录的 `config.yaml`（没有就用 `configs/model.yaml`）读网络结构
3. 用 UV 绑定初始化高斯，按参数计划优化，期间定期验证/存档
4. 产出：`model_XXXXXX.ply` + `report.json`（含验证指标与耗时）

用法::

    conda activate live3dgs
    python scripts/train.py --subject <对象> --work-name <版本>

默认超参与 模型目录的 `config.yaml` 一致：batch 10、50000 步、
L1 + 新视角 spill；SSIM/LPIPS 可选（默认关，见 config 的 ssim_weight /
lpips_weight）。验证集取末尾 30%。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import live3dgsavatar  # noqa: F401


def resolve_structure_config(models_dir: Path, subject: str, work: str,
                             *, explicit: Path | None = None,
                             repo_default: Path | None = None) -> Path:
    """决定训练用哪个"模型结构"配置。

    训练**新模型**时 `<models_dir>/<subject>/<work>/config.yaml` 通常还不存在，
    所以这里按优先级回退，而不是直接报"没有选择模型"：

    1. `--model-config` 显式给出
    2. `<models_dir>/<subject>/<work>/config.yaml`（续训/复现已有模型）
    3. `<models_dir>/<subject>/*/config.yaml`（同对象下任一版本的结构）
    4. 仓库模板 `configs/model.yaml`

    Raises:
        FileNotFoundError: 以上都没有 —— 此时应当明确告诉用户给一个配置，
            而不是让人以为"数据或模型没选对"
    """
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"--model-config 不存在：{path}")
        return path

    direct = Path(models_dir) / subject / work / "config.yaml"
    if direct.is_file():
        return direct

    for candidate in sorted((Path(models_dir) / subject).glob("*/config.yaml")):
        return candidate

    if repo_default is not None and Path(repo_default).is_file():
        return Path(repo_default)

    raise FileNotFoundError(
        f"找不到可用的模型结构配置：试过 {direct}、"
        f"{(Path(models_dir) / subject)}/*/config.yaml"
        f"{(', ' + str(repo_default)) if repo_default else ''}。\n"
        "训练新模型时请给一个结构配置（可用 --model-config 指定）。")


def snapshot_structure_config(model_cfg: Path, out_dir: Path) -> Path:
    """把结构配置复制进产物目录，返回快照路径。

    训练期间 `models/<subject>/<version>/` 可能被清理（重训前删旧模型），
    结束时再回头 copy 源文件就会 FileNotFoundError。开训前先留一份。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "config.yaml"
    shutil.copyfile(model_cfg, target)
    return target


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="训练一个约简高斯 blendshape 头像")
    p.add_argument("--subject", default=None, help="对象名（默认取配置）")
    p.add_argument("--work-name", default=None, help="版本名（默认取配置）")
    p.add_argument("--iterations", type=int, default=None, help="覆盖训练步数")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--blend-start", type=int, default=None,
                   help="第几步开始启用形变基（默认 3000）")
    p.add_argument("--mask-weight", type=float, default=None,
                   help="背景在损失里的权重；1 表示与前景同权（默认）")
    p.add_argument("--ssim-weight", type=float, default=None,
                   help="SSIM 项权重（默认 0；基础训练可试 0.02–0.05）")
    p.add_argument("--lpips-weight", type=float, default=None,
                   help="LPIPS 项权重（默认 0；建议 >=20000 步再开 0.01–0.03）")
    p.add_argument("--lpips-from", type=int, default=None,
                   help="第几步开始启用 LPIPS（默认取配置）")
    p.add_argument("--random-bg", action="store_true",
                   help="每批随机背景色（RGBAvatar 的 random_bg_color）")
    p.add_argument("--train-split", choices=("train", "all"), default=None,
                   help="训练帧来源：train=前 70%%（默认）；all=全数据集"
                        "（仅用于对齐后训练口径的诊断，会让留出集指标虚高）")
    p.add_argument("--test-ratio", type=float, default=None,
                   help="验证集比例（0.3 表示末尾 30%%，默认 0.3）")
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--save-every", type=int, default=None)
    p.add_argument("--patience", type=int, default=None,
                   help="验证连续多少次无提升就停（次数 × eval-every 步）")
    p.add_argument("--min-delta", type=float, default=None,
                   help="认为'有提升'的最小 dB 幅度")
    p.add_argument("--prefetch-workers", type=int, default=None,
                   help="后台预取线程数；0 表示不预取")
    p.add_argument("--early-metric",
                   choices=("psnr", "psnr_surface", "robust", "score"),
                   default=None,
                   help="早停指标；psnr=mask 内 PSNR，psnr_surface=FLAME 表面内 PSNR"
                        "（硬表面阶段默认），robust=大角度综合分")
    p.add_argument("--eval-frames", type=int, default=None,
                   help="验证只用其中 N 帧（等间隔抽样）；缺省用全部留出集")
    p.add_argument("--early-stop-after", type=int, default=None,
                   help="第几步之后允许早停；硬表面阶段默认 200，其余默认 blend_start+1000")
    p.add_argument("--robust-eval", dest="robust_eval", action="store_true",
                   default=None,
                   help="验证时加入大角度视角，用 正脸PSNR − λ·缺口 早停")
    p.add_argument("--no-robust-eval", dest="robust_eval", action="store_false",
                   help="关闭大角度验证（只按正脸 PSNR 早停）")
    p.add_argument("--robust-eval-frames", type=int, default=None,
                   help="参与大角度统计的验证帧数（默认 24）")
    p.add_argument("--robust-eval-weight", type=float, default=None,
                   help="λ：缺口比例折算成 dB 的系数（默认 10，即 10%% 缺口 ≈ 1 dB）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-config", type=Path, default=None,
                   help="模型结构配置（默认：模型目录里的 config.yaml，"
                        "没有就用仓库模板 configs/model.yaml）")
    p.add_argument("--stage", choices=("base", "hard_surface"), default=None,
                   help="训练阶段标签：base 保持旧流程；hard_surface 从 UV/FLAME "
                        "初始化，先拟合硬表面（不读取 base）")
    p.add_argument("--surface-spill-weight", type=float, default=None,
                   help="硬表面阶段：FLAME 表面外半透明溢出的权重（默认取配置）")
    p.add_argument("--surface-margin", type=int, default=None,
                   help="硬表面阶段 surface_spill 的边界余量像素")
    p.add_argument("--surface-every", type=int, default=None,
                   help="硬表面阶段每多少步计算一次表面 alpha 约束；默认取配置")
    p.add_argument("--surface-point-weight", type=float, default=None,
                   help="硬表面阶段：高斯中心到 FLAME 表面的距离权重")
    p.add_argument("--surface-normal-weight", type=float, default=None,
                   help="硬表面阶段：最短轴与面法线对齐权重")
    p.add_argument("--surface-scale-weight", type=float, default=None,
                   help="硬表面阶段：法向尺度压短权重")
    p.add_argument("--surface-size-weight", type=float, default=None,
                   help="硬表面阶段：切向尺度上限权重")
    p.add_argument("--surface-iso-weight", type=float, default=None,
                   help="硬表面阶段：切向两轴等长（小圆片）权重")
    p.add_argument("--surface-max-scale", type=float, default=None,
                   help="硬表面阶段：切向尺度上限（米）")
    p.add_argument("--surface-outside-weight", type=float, default=None,
                   help="硬表面阶段：FLAME 表面外 alpha 归零权重")
    p.add_argument("--alpha-weight", type=float, default=None,
                   help="alpha 直接监督权重：向 GT 前景 mask 回归（0=关）")
    p.add_argument("--surface-bg-weight", type=float, default=None,
                   help="硬表面阶段：FLAME 表面内、原图无数据区域的 alpha 归零权重")
    p.add_argument("--repair-dir", type=Path, default=None,
                   help="硬表面初始化：额外混入的 SyncFix 修复图目录")
    p.add_argument("--repair-schedule", default=None,
                   choices=("none", "linear", "cosine", "ratio"),
                   help="修复图退火：linear/cosine 衰减权重；ratio 后期减半（2/8→1/8）")
    p.add_argument("--repair-final-weight", type=float, default=None,
                   help="退火结束时的修复图样本权重（默认见配置）")
    p.add_argument("--repair-per-batch", type=int, default=None,
                   help="每个 batch 的修复图数量；缺省为 batch/2")
    p.add_argument("--angle-weight", type=float, default=0.0,
                   help="训练帧按 |yaw| 加权的尺度：20 = 每 20° 权重翻倍；0=均匀")
    p.add_argument("--freeze-xyz", action="store_true",
                   help="冻结 xyz / xyz_b（只训练外观与形状参数）")
    p.add_argument("--densify", action="store_true",
                   help="统一集合加密：全部高斯漂移，高残差处统一 split/prune")
    p.add_argument("--densify-every", type=int, default=None)
    p.add_argument("--densify-count", type=int, default=None)
    p.add_argument("--densify-max", type=int, default=None)
    p.add_argument("--densify-prune-opacity", type=float, default=None)
    p.add_argument("--densify-split-mode", choices=("copy", "subdivide"),
                   default=None,
                   help="copy=父不动+追加副本（偏软）；subdivide=父换成两个子高斯")
    p.add_argument("--densify-split-jitter", type=float, default=None,
                   help="分裂副本沿自身尺度的随机位移倍数（1σ；0=原地复制）")
    p.add_argument("--hard-xyz-lr", type=float, default=None,
                   help="分层全量阶段硬表面 xyz 的小学习率；缺省沿用默认")
    p.add_argument("--hard-offset-cap", type=float, default=None,
                   help="硬表面阶段切空间偏移上限（米）；每步后投影，防止长训练缓慢外漂")
    p.add_argument("--resume-hard", action="store_true",
                   help="hard_surface 阶段允许 --init-ply 热启动（带几何约束的 refine）")
    p.add_argument("--init-ply", type=Path, default=None,
                   help="从已有 PLY warm start（默认从 UV 绑定新建）")
    p.add_argument("--lr-scale", type=float, default=1.0,
                   help="所有参数组学习率乘该系数（warm start 微调用，默认 1）")
    p.add_argument("--force", action="store_true",
                   help="覆盖已存在的产物目录/模型版本（默认拒绝）")
    p.add_argument("--nv-every", type=int, default=None,
                   help="每 N 步做一次新视角几何监督；0 关闭（默认取配置）")
    p.add_argument("--nv-hole-weight", type=float, default=None,
                   help="新视角 supported 面上的 alpha 孔洞权重（默认 0）")
    p.add_argument("--nv-spill-weight", type=float, default=None,
                   help="新视角表面外半透明溢出权重（默认 0）")
    p.add_argument("--nv-spill-margin", type=int, default=None,
                   help="FLAME 表面外扩多少像素后仍允许（给头发留余量）")
    p.add_argument("--nv-alpha-min", type=float, default=None,
                   help="溢出惩罚的 alpha 下限（默认 0.05）")
    p.add_argument("--nv-alpha-max", type=float, default=None,
                   help="溢出惩罚的 alpha 上限（默认 0.6；实色头发不罚）")
    p.add_argument("--nv-min-px", type=int, default=None,
                   help="面至少多少真实前景像素才算 supported")
    p.add_argument("--nv-frames", type=int, default=None,
                   help="统计逐面支持时抽多少帧真实图（默认 80）")
    p.add_argument("--nv-yaws", type=float, nargs="+", default=None,
                   help="新视角的 yaw 候选（默认 ±10/20/30/40）")
    p.add_argument("--robust-spill-weight", type=float, default=None,
                   help="早停指标里 面外溢出 的权重（默认 0 = 不计）")
    p.add_argument("--no-spill", action="store_true",
                   help="关闭新视角 spill（原版无几何监督基线）")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--promote-name", default=None,
                   help="装进 models/ 时用的名字（默认与 --work-name 相同）")
    p.add_argument("--dry-run", action="store_true",
                   help="不需要 GPU：检查数据与配置，打印将要做的事")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from live3dgsavatar.config import load_config, resolve_data_dir
    from live3dgsavatar.data.models import promote_model

    cfg = load_config()
    if not bool(cfg.get("train.pipeline.enabled", True)):
        raise SystemExit(
            "[error] train.pipeline.enabled=false：训练入口已关闭（configs/train.yaml）")
    subject = args.subject or str(cfg.get("subject") or "").strip()
    work = args.work_name or str(cfg.get("work_name") or "").strip()
    if not subject or not work:
        raise SystemExit("[error] 需要 --subject 与 --work-name")

    # `model_dir()` 需要 subject+work_name 都在配置里；训练新模型时
    # `work` 只是产物目录名，不一定有对应模型 —— 所以下面**不走**
    # `resolve_model_config`，改用带回退的解析器。
    cfg.set("subject", subject)
    cfg.set("work_name", work)
    data_dir = resolve_data_dir(cfg)
    out_dir = Path(args.output_dir) if args.output_dir else (
        Path(str(cfg.get("paths.output_dir", "output"))) / "train" / subject / work)
    if not args.dry_run and out_dir.exists() and not args.force:
        raise SystemExit(
            f"[error] 产物目录已存在：{out_dir}\n"
            "         覆盖会删除旧 checkpoint；确认请加 --force，或换个 --work-name")

    iterations = int(args.iterations if args.iterations is not None
                     else cfg.get("train.pipeline.iterations", 50_000))
    batch_size = int(args.batch_size if args.batch_size is not None
                     else cfg.get("train.pipeline.batch_size", 10))
    blend_start = int(args.blend_start if args.blend_start is not None
                      else cfg.get("train.pipeline.blend_start", 3_000))
    test_ratio = float(args.test_ratio if args.test_ratio is not None
                       else cfg.get("train.pipeline.test_ratio", 0.3))
    eval_every = int(args.eval_every if args.eval_every is not None
                     else cfg.get("train.pipeline.eval_every", 500))
    eval_frames = (int(args.eval_frames) if args.eval_frames is not None
                   else cfg.get("train.pipeline.eval_frames", None))
    save_every = int(args.save_every if args.save_every is not None
                     else cfg.get("train.pipeline.save_every", 5000))
    patience = int(args.patience if args.patience is not None
                   else cfg.get("train.pipeline.patience", 2))
    min_delta = float(args.min_delta if args.min_delta is not None
                      else cfg.get("train.pipeline.min_delta", 0.05))
    workers = int(args.prefetch_workers if args.prefetch_workers is not None
                  else cfg.get("train.pipeline.prefetch_workers", 2))
    robust_enabled = (args.robust_eval if args.robust_eval is not None
                      else bool(cfg.get("train.pipeline.robust_eval", True)))
    robust_frames = int(args.robust_eval_frames if args.robust_eval_frames is not None
                        else cfg.get("train.pipeline.robust_eval_frames", 24))
    robust_weight = float(args.robust_eval_weight
                          if args.robust_eval_weight is not None
                          else cfg.get("train.pipeline.robust_eval_weight", 10.0))

    lr_scale = float(args.lr_scale)
    if lr_scale <= 0:
        raise SystemExit(f"[error] --lr-scale 必须为正，实际 {args.lr_scale}")

    sel_cfg = cfg.get("train.pipeline.selection", {}) or {}
    sel_ssim_weight = float(sel_cfg.get("ssim", 50.0))
    sel_lpips_weight = float(sel_cfg.get("lpips", 50.0))
    sel_percept_frames = int(sel_cfg.get("perceptual_frames", 240))

    def _nv(key, default):
        return cfg.get(f"train.pipeline.novel_view.{key}", default)

    nv_every = int(args.nv_every if args.nv_every is not None
                   else _nv("every", 0))
    nv_hole = float(args.nv_hole_weight if args.nv_hole_weight is not None
                    else _nv("hole_weight", 0.0))
    nv_spill = float(args.nv_spill_weight if args.nv_spill_weight is not None
                     else _nv("spill_weight", 0.0))
    nv_margin = int(args.nv_spill_margin if args.nv_spill_margin is not None
                    else _nv("spill_margin", 12))
    nv_alpha_min = float(args.nv_alpha_min if args.nv_alpha_min is not None
                         else _nv("alpha_min", 0.05))
    nv_alpha_max = float(args.nv_alpha_max if args.nv_alpha_max is not None
                         else _nv("alpha_max", 0.6))
    nv_min_px = int(args.nv_min_px if args.nv_min_px is not None
                    else _nv("min_px", 16))
    nv_frames = int(args.nv_frames if args.nv_frames is not None
                    else _nv("frames", 80))
    nv_yaws = list(args.nv_yaws if args.nv_yaws is not None
                   else _nv("yaws", [-50, -40, -30, -20, -10, 10, 20, 30, 40, 50]))
    robust_spill = float(args.robust_spill_weight
                         if args.robust_spill_weight is not None
                         else _nv("robust_spill_weight", 0.0))
    if args.no_spill:
        nv_spill = 0.0
    stage = str(args.stage if args.stage is not None
                else cfg.get("train.pipeline.stage", "base"))
    if stage not in ("base", "hard_surface"):
        raise SystemExit(f"[error] 未知训练阶段：{stage!r}")

    if args.early_stop_after is not None:
        early_stop_after = max(0, int(args.early_stop_after))
    elif stage == "hard_surface":
        # 硬表面只做初始化：形变基一直是关的，早停没必要等 blend_start。
        early_stop_after = int(
            cfg.get("train.pipeline.hard_surface.early_stop_after", 200))
    else:
        # 形变基在 blend_start 才打开；早停至少等它生效 1000 步
        early_stop_after = max(0, blend_start + 1000)
    surface_spill = float(
        args.surface_spill_weight if args.surface_spill_weight is not None
        else cfg.get("train.pipeline.hard_surface.spill_weight", 0.05))
    surface_margin = int(
        args.surface_margin if args.surface_margin is not None
        else cfg.get("train.pipeline.hard_surface.margin", 4))
    surface_every = int(
        args.surface_every if args.surface_every is not None
        else cfg.get("train.pipeline.hard_surface.every", 10))
    surface_point = float(
        args.surface_point_weight if args.surface_point_weight is not None
        else cfg.get("train.pipeline.hard_surface.point_weight", 2000.0))
    surface_normal = float(
        args.surface_normal_weight if args.surface_normal_weight is not None
        else cfg.get("train.pipeline.hard_surface.normal_weight", 0.10))
    surface_scale = float(
        args.surface_scale_weight if args.surface_scale_weight is not None
        else cfg.get("train.pipeline.hard_surface.scale_weight", 0.05))
    surface_size = float(
        args.surface_size_weight if args.surface_size_weight is not None
        else cfg.get("train.pipeline.hard_surface.size_weight", 1.0))
    surface_iso = float(
        args.surface_iso_weight if args.surface_iso_weight is not None
        else cfg.get("train.pipeline.hard_surface.iso_weight", 0.1))
    surface_max_scale = float(
        args.surface_max_scale if args.surface_max_scale is not None
        else cfg.get("train.pipeline.hard_surface.max_scale", 0.0020))
    surface_outside = float(
        args.surface_outside_weight if args.surface_outside_weight is not None
        else cfg.get("train.pipeline.hard_surface.outside_weight", 1.0))
    surface_bg = float(
        args.surface_bg_weight if args.surface_bg_weight is not None
        else cfg.get("train.pipeline.hard_surface.bg_weight", 1.0))
    avatar_section = cfg.get("train.pipeline.avatar", {}) or {}
    repair_schedule = (args.repair_schedule
                       or avatar_section.get("repair_schedule", "none"))
    repair_final_weight = float(
        args.repair_final_weight if args.repair_final_weight is not None
        else avatar_section.get("repair_final_weight", 0.2))
    alpha_weight = float(
        args.alpha_weight if args.alpha_weight is not None
        else cfg.get("train.pipeline.alpha_weight", 0.0))
    if surface_every <= 0:
        raise SystemExit("[error] --surface-every 必须为正")
    if stage != "hard_surface":
        surface_spill = 0.0
        surface_point = surface_normal = surface_scale = 0.0
        surface_size = surface_iso = surface_outside = surface_bg = 0.0
    else:
        # 硬表面阶段使用真实视角的表面约束；新视角项留到后续阶段。
        nv_hole = nv_spill = 0.0
    if (stage == "hard_surface" and args.init_ply is not None
            and not args.resume_hard):
        raise SystemExit(
            "[error] hard_surface 阶段默认从 UV/FLAME 初始化；"
            "若要做带几何约束的 refine，请加 --resume-hard")
    nv_enabled = (stage != "hard_surface") and nv_every > 0 and (
        nv_hole > 0 or nv_spill > 0)
    if nv_enabled and not nv_yaws:
        raise SystemExit("[error] 新视角监督需要至少一个 --nv-yaws")
    need_support = nv_enabled and nv_hole > 0

    from live3dgsavatar.data.sources import Batch, ManifestSource

    class MixedHardSource:
        """真实帧 + 修复帧 1:1；修复帧使用各自的相机和 FLAME surface mask。"""

        def __init__(self, real, repair_batches, surface_masks, K_data,
                     w2c_data, repair_per_batch=None, *,
                     schedule="none", total_steps=0, final_weight=0.2):
            self.real = real
            self.repair = repair_batches
            self.surface_masks = surface_masks
            self.K_data = K_data
            self.w2c_data = w2c_data
            self.repair_per_batch = repair_per_batch
            self.batch_size = 10
            # 修复图退火：前期当"锚定"权重高，后期让真实帧主导
            self.schedule = str(schedule)
            self.total_steps = max(1, int(total_steps))
            self.final_weight = float(final_weight)
            self._step = 0

        def _repair_weight(self) -> float:
            """当前步修复图的样本权重（1 = 与真实帧同权）。"""
            if self.schedule == "none":
                return 1.0
            t = min(1.0, self._step / self.total_steps)
            if self.schedule == "linear":
                return 1.0 + (self.final_weight - 1.0) * t
            if self.schedule == "cosine":
                return self.final_weight + (1.0 - self.final_weight) * 0.5 * (
                    1.0 + __import__("math").cos(__import__("math").pi * t))
            return 1.0

        def _repair_count(self, n: int) -> int:
            if self.schedule != "ratio":
                return (int(self.repair_per_batch) if self.repair_per_batch
                        is not None else max(1, int(n) // 2))
            full = int(self.repair_per_batch or 0)
            t = min(1.0, self._step / self.total_steps)
            # 温和退火：后段只减半，不清零（清零会让大角度轮廓后期失守）
            if t < 0.5:
                return full
            return max(1, full // 2)

        def random_batch(self, n: int):
            import torch as _torch
            n_rep = self._repair_count(n)
            rep_w = self._repair_weight()
            self._step += 1
            n_rep = min(n_rep, int(n) - 1, len(self.repair))
            n_real = int(n) - n_rep
            rb = self.real.random_batch(n_real)
            rb.K = self.K_data.unsqueeze(0).repeat(n_real, 1, 1)
            rb.w2c = self.w2c_data.unsqueeze(0).repeat(n_real, 1, 1)
            rb.surface = _torch.stack(
                [self.surface_masks[int(i)] for i in rb.indices]).unsqueeze(1)
            picks = [_torch.randint(0, len(self.repair), (1,)).item()
                     for _ in range(n_rep)]
            reps = [self.repair[i] for i in picks]
            def cat(name):
                vals = [getattr(rb, name)] + [getattr(r, name) for r in reps]
                return _torch.cat(vals, 0)
            weight = None
            if n_rep > 0 and rep_w != 1.0:
                weight = _torch.cat([
                    _torch.ones(n_real),
                    _torch.full((n_rep,), float(rep_w)),
                ])
            return Batch(
                mesh=cat("mesh"), blend_weight=cat("blend_weight"),
                image=cat("image"), mask=cat("mask"),
                indices=cat("indices"), K=cat("K"), w2c=cat("w2c"),
                surface=cat("surface"), weight=weight)

        def __len__(self):
            return len(self.real)

        def close(self):
            close = getattr(self.real, "close", None)
            if callable(close):
                close()

        def random_indices(self, n):
            return self.real.random_indices(n)

        def iter_batches(self, batch_size):
            """透传真实帧批次（逐面统计只用真实帧，修复帧没有对应 GT）。"""
            return self.real.iter_batches(batch_size)

        def set_sample_weights(self, weights):
            """透传给真实帧源：修复帧按固定配额混入，不受权重影响。"""
            self.real.set_sample_weights(weights)

    max_frames = cfg.get("train.pipeline.max_frames", None)
    if max_frames is not None and int(max_frames) <= 0:
        max_frames = None
    train_split = str(args.train_split
                      if args.train_split is not None
                      else cfg.get("train.pipeline.train_split", "train"))
    train_src = ManifestSource(data_dir, split=train_split, test_ratio=test_ratio,
                               device="cpu", seed=args.seed,
                               max_frames=(None if max_frames is None
                                           else int(max_frames)),
                               random_bg=bool(args.random_bg))
    val_src = ManifestSource(data_dir, split="test", test_ratio=test_ratio,
                             device="cpu")

    print(f"训练 {subject}/{work}")
    print(f"  数据   : {data_dir}")
    print(f"  阶段   : {stage}")
    print(f"  训练集 : {len(train_src)} 帧    验证集 : {len(val_src)} 帧"
          f"（末尾 {test_ratio:.0%}）")
    models_dir = Path(str(cfg.get("paths.models_dir")))
    init_path = (Path(args.init_ply).expanduser()
                 if args.init_ply is not None else None)
    if init_path is not None and not init_path.is_file():
        raise SystemExit(f"[error] --init-ply 不存在：{init_path}")
    target_name = args.promote_name or work
    target_dir = models_dir / subject / target_name
    if not args.dry_run and target_dir.exists() and not args.force:
        raise SystemExit(
            f"[error] 模型版本已存在：{target_dir}\n"
            "         覆盖会替换旧模型；确认请加 --force，或换个 --promote-name")
    if stage == "hard_surface" and args.model_config is None:
        # 新阶段从 UV/FLAME 初始化，不从旧 base 目录借结构配置，避免隐式耦合。
        model_cfg = REPO_ROOT / "configs" / "model.yaml"
    else:
        model_cfg = resolve_structure_config(
            models_dir, subject, work, explicit=args.model_config,
            repo_default=REPO_ROOT / "configs" / "model.yaml")
    if init_path is not None and args.model_config is None:
        sibling = init_path.parent / "config.yaml"
        if sibling.is_file():
            model_cfg = sibling
    if not args.dry_run and out_dir.exists() and args.force:
        if out_dir.resolve() in Path(model_cfg).resolve().parents:
            raise SystemExit(
                f"[error] --force 会删除结构配置所在目录：{model_cfg}\n"
                "         请先把 config 移出产物目录")
        shutil.rmtree(out_dir)

    from live3dgsavatar.core.avatar import AvatarConfig

    avatar_cfg = AvatarConfig.from_yaml(model_cfg)

    print(f"  结构   : {model_cfg}")
    print(f"           tex {avatar_cfg.tex_size}  D {avatar_cfg.num_basis_in}  "
          f"K {avatar_cfg.num_basis_blend}")
    print(f"  步数   : {iterations}   batch {batch_size}   "
          f"blend_start {blend_start}")
    print(f"  早停   : 每 {eval_every} 步验证，连续 {patience} 次 "
          f"< {min_delta:g} dB；第 {early_stop_after} 步后生效")
    if blend_start >= iterations:
        print(f"  [hint] blend_start({blend_start}) ≥ 步数({iterations})："
              "整段都不会启用形变基，只在拟合静态外观")

    if args.dry_run:
        b = train_src.random_batch(min(batch_size, 2))
        print(f"  [dry-run] batch: mesh {tuple(b.mesh.shape)}  "
              f"weight {tuple(b.blend_weight.shape)}  image {tuple(b.image.shape)}")
        if init_path is not None:
            print(f"  [dry-run] warm start: {init_path}")
        if nv_enabled:
            print(f"  [dry-run] 新视角监督: 每 {nv_every} 步，yaw {nv_yaws}，"
                  f"hole {nv_hole:g} + spill {nv_spill:g}"
                  f"（margin {nv_margin}px, alpha "
                  f"{nv_alpha_min:g}–{nv_alpha_max:g}）")
        if stage == "hard_surface":
            print(f"  [dry-run] 硬表面: spill "
                  f"{surface_spill:g}（每 {surface_every} 步，"
                  f"margin {surface_margin}px）")
        print(f"  [dry-run] 产物目录 {out_dir}")
        print("  [dry-run] 未执行训练（需要 GPU）")
        return 0

    # 结构配置可能来自 models/<subject>/ 下的某个兄弟版本；训练期间那个目录
    # 可能被清理（例如重训时先删旧 best），所以立刻在产物目录留一份快照，
    # 后面训练/装入都用快照，避免结束时 copyfile 找不到源。
    try:
        model_cfg = snapshot_structure_config(model_cfg, out_dir)
    except FileNotFoundError:
        raise SystemExit(
            f"[error] 结构配置在训练开始前已被删除；"
            "可用 --model-config configs/model.yaml 指定一份")

    import random

    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("[error] 训练需要 GPU。可用 --dry-run 检查数据。")

    import nvdiffrast.torch as dr

    from live3dgsavatar.core.avatar import GaussianAvatar
    from live3dgsavatar.core.deform.binding import build_binding
    from live3dgsavatar.core.io import load_ply
    from live3dgsavatar.core.camera import orbit_w2c
    from live3dgsavatar.core.render import (SimpleRasterizer, render_frame,
                                            render_gaussians)
    from live3dgsavatar.core.render.faceid import detect_flip, face_id_map
    from live3dgsavatar.evaluation import (HOLE_THRESHOLD, RobustConfig,
                                           alpha_deficit,
                                           robust_score, robust_views,
                                           subsample, summarise_coverage,
                                           summarise_holes)
    from live3dgsavatar.core.types import Camera, Mesh
    from live3dgsavatar.data.scene import load_scene
    from live3dgsavatar.training.geometry import hard_surface_terms
    from live3dgsavatar.training.objectives import Objective
    from live3dgsavatar.training.plans import ParameterPlan
    from live3dgsavatar.training.schedule import EarlyStopping
    from live3dgsavatar.training.trainer import (CheckpointHook, LogHook, Trainer,
                                                 ValidationHook)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(str(cfg.get("runtime.device", "cuda")))
    # ---- 几何与相机 ----
    scene = load_scene(cfg, frames=1, device=str(device))
    glctx = dr.RasterizeCudaContext()
    if init_path is not None:
        avatar = load_ply(init_path, config=avatar_cfg, device=str(device))
        print(f"  初始化 : {init_path}（warm start）")
    else:
        binding = build_binding(scene.uvs.to(device), scene.uv_faces.to(device),
                                tex_size=avatar_cfg.tex_size, glctx=glctx,
                                device=device)
        avatar = GaussianAvatar(avatar_cfg, binding).to(device)
    rasterizer = SimpleRasterizer()

    # 头转（pose 驱动）需要 RDF 的 identity/pose；没有就退回相机绕转
    pose_driver = None
    try:
        from live3dgsavatar.tracking.pose_drive import PoseDriver
        pose_driver = PoseDriver(data_dir, device="cpu")
    except Exception as exc:  # 资产/拓扑不匹配等：不阻断训练
        print(f"  提示   : pose 驱动不可用（{exc}），退回相机绕转")

    camera = Camera.from_intrinsics_extrinsics(
        K=scene.K, R=scene.R, T=scene.T,
        width=scene.width, height=scene.height).to(str(device))
    print(f"  高斯数 : {avatar.num_gaussians}   基 K={avatar.num_basis}")

    # 硬表面从零初始化时，可把 SyncFix 修复图作为等比例额外观测。
    # 修复图没有相机/网格，用基准帧的 mesh + 目标 yaw 构造 camera。
    if args.repair_dir is not None:
        if stage not in ("hard_surface", "base"):
            raise SystemExit("--repair-dir 只用于 --stage hard_surface/base")
        from PIL import Image
        from live3dgsavatar.core.camera import (orbit_w2c_matrix, rotation_x,
                                                rotation_y)
        from live3dgsavatar.core.render.faceid import face_id_map
        from live3dgsavatar.core.types import Camera
        from live3dgsavatar.data.pose import rot6d_to_matrix, head_yaw
        from live3dgsavatar.evaluation.views import parse_view_label

        import numpy as _np
        with _np.load(data_dir / "render_inputs.npz", allow_pickle=True) as d:
            meshes_np = d["mesh"]; weights_np = d["blend_weight"]
        labels = sorted(Path(args.repair_dir).glob("*.png"))
        if not labels:
            raise SystemExit(f"repair-dir 没有 PNG：{args.repair_dir}")
        # 修复图使用基准帧（第一个标签的 frame 表达式）
        base_idx = int(parse_view_label(labels[0].stem)["expr"][5:])
        base_mesh = torch.from_numpy(meshes_np[base_idx].copy()).float()
        base_weight = torch.from_numpy(weights_np[base_idx].copy()).float()
        head_R = torch.from_numpy(rot6d_to_matrix(
            base_weight.numpy()[None, 100:106])[0]).float()
        center = base_mesh.mean(dim=0)
        w2c_data = torch.eye(4, dtype=torch.float32)
        w2c_data[:3, :3] = scene.R.detach().cpu()
        w2c_data[:3, 3] = scene.T.detach().cpu()
        K_data = scene.K.detach().cpu()
        # 修复图是"头转、胸不动"：网格必须用同一套 pose 驱动重建，
        # 相机保持数据集相机；没有 RDF 跟踪结果才退回相机绕转。
        driver = pose_driver
        repair_batches = []
        for pth in labels:
            info = parse_view_label(pth.stem)
            yaw = float(info["yaw"])
            pitch = float(info["pitch"]) if info.get("pitch") is not None else 0.0
            if driver is not None:
                turned = driver.turn(base_idx, yaw, pitch)
                mesh_t = turned.mesh.unsqueeze(0)
                weight_t = turned.weight.unsqueeze(0)
                view = w2c_data
                cam = Camera(K=K_data, w2c=w2c_data.unsqueeze(0),
                             width=scene.width, height=scene.height).to("cuda")
            else:
                R_target = (rotation_x(0.0) @ rotation_y(yaw))[:3, :3]
                view = orbit_w2c_matrix(w2c_data, center, R_target @ head_R.T)
                cam = Camera(K=K_data, w2c=view.unsqueeze(0),
                             width=scene.width, height=scene.height).to("cuda")
                mesh_t = base_mesh.unsqueeze(0)
                weight_t = base_weight.unsqueeze(0)
            with torch.no_grad():
                ids = face_id_map(mesh_t.cuda(),
                                  scene.faces.cuda(), cam, glctx, flip_y=False)
                surf = (ids >= 0).unsqueeze(0).unsqueeze(0).cpu()
            raw = Image.open(pth)
            if raw.mode in ("RGBA", "LA") or "transparency" in raw.info:
                arr = _np.asarray(raw.convert("RGBA"), _np.float32)
                rgb = arr[..., :3]; fg = arr[..., 3] > 127
            else:
                arr = _np.asarray(raw.convert("RGB"), _np.float32)
                rgb = arr; fg = arr.max(axis=-1) > 8
            image = torch.from_numpy(rgb.transpose(2, 0, 1) / 255.0).float()
            mask = torch.from_numpy(fg[None, None]).bool()
            repair_batches.append(Batch(
                mesh=mesh_t, blend_weight=weight_t,
                image=image.unsqueeze(0), mask=mask,
                indices=torch.tensor([base_idx], dtype=torch.long),
                K=K_data.unsqueeze(0), w2c=view.unsqueeze(0),
                surface=surf))
        with _np.load(data_dir / "surface_masks.npz") as d:
            surface_masks = {int(k): torch.from_numpy(d[k].copy()).bool()
                              for k in d.files}
        train_src = MixedHardSource(
            train_src, repair_batches, surface_masks, K_data, w2c_data,
            repair_per_batch=args.repair_per_batch,
            schedule=repair_schedule,
            total_steps=int(iterations),
            final_weight=repair_final_weight)
        ratio = (f"{int(args.repair_per_batch)}:{int(batch_size) - int(args.repair_per_batch)}"
                 if args.repair_per_batch is not None else "1:1")
        print(f"  初始化混合: {len(repair_batches)} 张修复图，batch 修复:真实={ratio}")

    # 数据相机（world→camera）：大角度验证要绕它旋转
    w2c_data = torch.eye(4, dtype=torch.float32)
    w2c_data[:3, :3] = scene.R.detach().cpu()
    w2c_data[:3, 3] = scene.T.detach().cpu()
    robust_cfg = (RobustConfig(frames=robust_frames, weight=robust_weight)
                  if robust_enabled else None)
    if robust_cfg is not None:
        print(f"  验证   : {'头转' if pose_driver is not None else '相机绕转'} "
              f"±{', ±'.join(f'{y:g}' for y in robust_cfg.yaws)}°"
              f"（{robust_cfg.frames} 帧，λ={robust_cfg.weight:g}）")

    plan = ParameterPlan.rgba_avatar_defaults(
        scene_extent=float(cfg.get("train.pipeline.scene_extent", 1.0)))
    if args.freeze_xyz:
        plan = ParameterPlan(
            groups={k: v for k, v in plan.groups.items()
                    if k not in ("xyz", "xyz_b")},
            scene_extent=plan.scene_extent)
        print("  位移   : 冻结硬表面 xyz / xyz_b")
    hard_xyz_lr = (args.hard_xyz_lr if args.hard_xyz_lr is not None
                   else (cfg.get("train.pipeline.hard_surface.xyz_lr")
                         if stage == "hard_surface" else None))
    if hard_xyz_lr is not None:
        from live3dgsavatar.training.plans import GroupPlan
        plan.groups["xyz"] = GroupPlan(
            lr=float(hard_xyz_lr),
            final_lr=float(hard_xyz_lr) * 0.1,
            max_steps=30000)
        print(f"  硬位移 : xyz lr {hard_xyz_lr:g}（轻度）")
    if lr_scale != 1.0:
        plan = plan.scaled(lr_scale)
        print(f"  lr 缩放 : {lr_scale:g}")

    # 硬表面允许微小形变（补偿 FLAME/RDF 误差），但每步把切空间偏移投影到
    # 上限之内：否则长期训练时 1% 的高斯会慢慢漂到表面外 8mm+，既不再是
    # "硬表面"，也浪费步数。
    offset_guard = None
    if stage == "hard_surface":
        cap = float(args.hard_offset_cap if args.hard_offset_cap is not None
                    else cfg.get("train.pipeline.hard_surface.max_offset", 0.003))
        if cap > 0:
            def offset_guard() -> None:
                with torch.no_grad():
                    n = avatar.xyz.data.norm(dim=-1, keepdim=True)
                    f = (cap / n.clamp_min(1e-8)).clamp(max=1.0)
                    avatar.xyz.data.mul_(f)
            print(f"  硬偏移 : 切空间偏移上限 {cap * 1000:.1f} mm（每步投影）")
    mask_weight = float(args.mask_weight if args.mask_weight is not None
                        else cfg.get("train.pipeline.mask_weight", 1.0))
    ssim_weight = float(args.ssim_weight if args.ssim_weight is not None
                        else cfg.get("train.pipeline.ssim_weight", 0.0))
    lpips_weight = float(args.lpips_weight if args.lpips_weight is not None
                         else cfg.get("train.pipeline.lpips_weight", 0.0))
    lpips_from = int(args.lpips_from if args.lpips_from is not None
                     else cfg.get("train.pipeline.lpips_from", 20_000))
    objective = Objective.rgba_avatar_defaults()
    # 硬表面阶段只监督 FLAME mask 内部，背景/头发/衣服不进入 L1。
    objective.mask_weight = 0.0 if stage == "hard_surface" else mask_weight
    objective.nv_hole = nv_hole
    objective.nv_spill = nv_spill
    objective.surface_spill = surface_spill
    objective.surface_point = surface_point
    objective.surface_normal = surface_normal
    objective.surface_scale = surface_scale
    objective.surface_size = surface_size
    objective.surface_iso = surface_iso
    objective.surface_outside = surface_outside
    objective.surface_bg = surface_bg
    objective.alpha = alpha_weight
    objective.ssim = ssim_weight
    objective.lpips = lpips_weight
    objective.lpips_from = lpips_from
    supported_face = None
    effective_mask_weight = 0.0 if stage == "hard_surface" else mask_weight
    loss_terms = [f"L1（背景权重 {effective_mask_weight:.2f}）"]
    if nv_spill > 0:
        loss_terms.append(f"nv_spill {nv_spill:g}")
    if nv_hole > 0:
        loss_terms.append(f"nv_hole {nv_hole:g}")
    if surface_spill > 0:
        loss_terms.append(f"surface_spill {surface_spill:g}")
    if surface_point > 0:
        loss_terms.append(f"surface_point {surface_point:g}")
    if surface_normal > 0:
        loss_terms.append(f"surface_normal {surface_normal:g}")
    if surface_scale > 0:
        loss_terms.append(f"surface_scale {surface_scale:g}")
    if surface_size > 0:
        loss_terms.append(f"surface_size {surface_size:g}")
    if surface_iso > 0:
        loss_terms.append(f"surface_iso {surface_iso:g}")
    if surface_outside > 0:
        loss_terms.append(f"surface_outside {surface_outside:g}")
    if surface_bg > 0:
        loss_terms.append(f"surface_bg {surface_bg:g}")
    if repair_schedule != "none":
        loss_terms.append(f"修复图退火 {repair_schedule}"
                          f"（末权重 {repair_final_weight:g}）")
    if alpha_weight > 0:
        loss_terms.append(f"alpha {alpha_weight:g}(对 GT mask)")
    if ssim_weight > 0:
        loss_terms.append(f"ssim {ssim_weight:g}")
    if lpips_weight > 0:
        loss_terms.append(f"lpips {lpips_weight:g}")
    print(f"  损失   : {' + '.join(loss_terms)}")
    if ssim_weight > 0 or lpips_weight > 0:
        print(f"  感知项 : SSIM {ssim_weight:g} + LPIPS {lpips_weight:g}"
              f"（LPIPS 第 {lpips_from} 步后启用）")
    else:
        print("  感知项 : 关闭（SSIM/LPIPS 只在后训练 refine 启用）")
    if nv_enabled:
        print(f"  新视角 : 每 {nv_every} 步，yaw {nv_yaws}，"
              f"hole {nv_hole:g} + spill {nv_spill:g}"
              f"（margin {nv_margin}px, alpha {nv_alpha_min:g}–{nv_alpha_max:g}）")
    else:
        print("  新视角 : 关闭")
    if stage == "hard_surface":
        print(f"  硬表面 : spill {surface_spill:g}"
              f"（每 {surface_every} 步，margin {surface_margin}px），"
              f"point {surface_point:g} + normal {surface_normal:g} + "
              f"scale {surface_scale:g} + size {surface_size:g} + "
              f"iso {surface_iso:g} + outside {surface_outside:g} + "
              f"bg {surface_bg:g}（cap {surface_max_scale:g}m）")
    # 硬表面 mask 用半分辨率面 id 图再上采样；512² 每步算代价太高。
    k_half = scene.K.clone()
    k_half[:2] *= 0.5
    cam_half = Camera(K=k_half, w2c=camera.w2c[:1],
                      width=scene.width // 2, height=scene.height // 2).to(device)
    surface_cache: dict[int, torch.Tensor] = {}
    surface_cache_path = data_dir / "surface_masks.npz"
    if stage == "hard_surface" and surface_cache_path.is_file():
        with np.load(surface_cache_path) as d:
            surface_cache = {
                int(k): torch.from_numpy(d[k].copy()).bool()[None, None]
                for k in d.files}
        print(f"  表面mask: 缓存 {len(surface_cache)} 帧（{surface_cache_path.name}）")
    elif stage == "hard_surface":
        print("  表面mask: 未找到缓存，训练中按需光栅化（可能较慢）", flush=True)

    def _surface_mask(batch, i: int) -> torch.Tensor:
        """按全局帧号缓存半分辨率 FLAME mask，避免每步重复光栅化。

        返回 CPU bool `[1,1,H,W]`；缓存缺失时才做一次面 id 光栅化。
        """
        explicit = getattr(batch, "surface", None)
        if explicit is not None:
            return explicit[i:i + 1].to(device)
        key = (int(batch.indices[i]) if getattr(batch, "indices", None)
               is not None else -1)
        cached = surface_cache.get(key)
        if cached is not None:
            return cached
        with torch.no_grad():
            ids = face_id_map(
                batch.mesh[i:i + 1].to(device), scene.faces.to(device),
                cam_half, glctx, flip_y=False)
            s = (ids >= 0).float()[None, None]
            s = torch.nn.functional.interpolate(
                s, size=(scene.height, scene.width), mode="nearest")
            s = (s > 0.5).cpu()
        if key >= 0:
            surface_cache[key] = s
        return s

    from live3dgsavatar.training.forward import ForwardOptions, forward_avatar
    hard_forward = ForwardOptions(build_graph=True, frontface=True,
                                  clamp_max_scale=surface_max_scale)
    base_forward = ForwardOptions(build_graph=True)
    hard_eval = ForwardOptions(frontface=True,
                               clamp_max_scale=surface_max_scale)
    base_eval = ForwardOptions()

    def render_fn(mesh_verts, weight, batch, iteration):
        mesh = Mesh(verts=mesh_verts.to(device), faces=scene.faces.to(device),
                    uvs=scene.uvs.to(device), uv_faces=scene.uv_faces.to(device))
        if getattr(batch, "K", None) is not None:
            cam = Camera(K=batch.K.to(device), w2c=batch.w2c.to(device),
                         width=scene.width, height=scene.height).to(device)
        else:
            cam = camera
        if getattr(batch, "bg", None) is not None:
            bg = batch.bg.to(device)
        else:
            bg = torch.zeros(batch.image.shape[0], 3, 1, 1, device=device)
        if stage == "hard_surface":
            res = forward_avatar(
                avatar, rasterizer, mesh,
                None if weight is None else weight.to(device), cam, bg,
                options=hard_forward, target_image=batch.image.to(device))
            out = res.out
            gs = res.gaussians
            terms = hard_surface_terms(
                gs, mesh, avatar.binding_face_id, avatar.binding_face_bary,
                max_scale=surface_max_scale)
            if surface_point > 0:
                out.surface_point_loss = terms["surface_point"]
            if surface_normal > 0:
                out.surface_normal_loss = terms["surface_normal"]
            if surface_scale > 0:
                out.surface_scale_loss = terms["surface_scale"]
            if surface_size > 0:
                out.surface_size_loss = terms["surface_size"]
            if surface_iso > 0:
                out.surface_iso_loss = terms["surface_iso"]

            # FLAME 支撑面作为硬表面 loss mask；只拟合表面内部，不把
            # 头发/背景/衣服一并纳入 L1。半分辨率面 id 上采样足够稳定。
            surface = torch.cat(
                [_surface_mask(batch, i).to(device)
                 for i in range(int(mesh_verts.shape[0]))], dim=0)
            out.loss_mask = surface & batch.mask.to(device).bool()
        else:
            out = forward_avatar(
                avatar, rasterizer, mesh,
                None if weight is None else weight.to(device), cam, bg,
                options=base_forward, target_image=batch.image.to(device)).out
        if nv_enabled and (int(iteration) + 1) % int(nv_every) == 0:
            from live3dgsavatar.training.losses import (outside_alpha_loss,
                                                       surface_bg_alpha_loss,
                                                       surface_spill_loss)
            i = random.randrange(int(mesh_verts.shape[0]))
            yaw = float(random.choice(nv_yaws))
            if pose_driver is not None:
                # 与 GT 同语义：相机不动，头转
                turned = pose_driver.turn(int(batch.indices[i]), yaw, 0.0)
                verts_nv = turned.mesh.unsqueeze(0).to(device)
                # ⚠️ warm-up 期间主路径不传 weight，这里也必须一致，
                #    否则形变基会在 blend_start 之前被偷偷训练（评审 B2）。
                weight_nv = (None if weight is None
                             else turned.weight.unsqueeze(0).to(device))
                cam_nv = Camera(K=scene.K,
                                w2c=w2c_data.unsqueeze(0).to(device),
                                width=scene.width,
                                height=scene.height).to(device)
            else:
                verts_nv = mesh_verts[i:i + 1].to(device)
                weight_nv = None if weight is None else weight[i:i + 1].to(device)
                center_nv = verts_nv[0].mean(dim=0).detach().cpu()
                view_nv = orbit_w2c(w2c_data, center_nv, yaw, 0.0)
                cam_nv = Camera(K=scene.K, w2c=view_nv.unsqueeze(0).to(device),
                                width=scene.width,
                                height=scene.height).to(device)
            mesh_nv = Mesh(verts=verts_nv, faces=scene.faces.to(device),
                           uvs=scene.uvs.to(device),
                           uv_faces=scene.uv_faces.to(device))
            with torch.no_grad():
                ids_nv = face_id_map(verts_nv, scene.faces.to(device),
                                     cam_nv, glctx, flip_y=False)
                surface_nv = ids_nv >= 0
                supported_nv = None
                if supported_face is not None:
                    supported_nv = (surface_nv
                                    & supported_face[ids_nv.clamp(min=0)]
                                    & surface_nv)
            out_nv = forward_avatar(
                avatar, rasterizer, mesh_nv, weight_nv, cam_nv,
                torch.zeros(1, 3, 1, 1, device=device),
                options=base_forward).out
            if nv_hole > 0 and supported_nv is not None:
                out.nv_hole_loss = alpha_hole_loss(
                    out_nv.alpha, supported_nv.unsqueeze(0).unsqueeze(0))
            if nv_spill > 0:
                out.nv_spill_loss = surface_spill_loss(
                    out_nv.alpha, surface_nv.unsqueeze(0).unsqueeze(0),
                    margin=nv_margin, alpha_min=nv_alpha_min,
                    alpha_max=nv_alpha_max)

        if (stage == "hard_surface" and surface_spill > 0
                and (int(iteration) + 1) % surface_every == 0):
            from live3dgsavatar.training.losses import (alpha_hole_loss,
                                                       outside_alpha_loss,
                                                       surface_bg_alpha_loss,
                                                       surface_spill_loss)
            if surface_spill > 0:
                out.surface_spill_loss = surface_spill_loss(
                    out.alpha, surface, margin=surface_margin)
            if surface_outside > 0:
                out.surface_outside_loss = outside_alpha_loss(out.alpha, surface)
            if surface_bg > 0:
                out.surface_bg_loss = surface_bg_alpha_loss(
                    out.alpha, surface, batch.mask.to(device).bool())

        return out

    last_eval: dict[str, float] = {}
    geom_ref: dict[str, tuple] = {}

    @torch.no_grad()
    def evaluate(source) -> dict[str, float]:
        """验证集 PSNR（前景掩码内）。

        这里**不走** `render_fn`：训练态会把 `target_image` 传给光栅化器，
        由它额外算 `est_color`/`est_weight`（供颜色回写用）。评测只需要
        颜色与 alpha，带上 `target_image` 会走一条在 `no_grad` 下容易出错的
        内核路径（实测在第 2000 步验证时触发过 CUDA illegal memory access）。
        所以评测单独走一条纯推理路径。
        """
        from live3dgsavatar.core.metrics import ssim as _ssim
        from live3dgsavatar.evaluation.protocol import lpips_score as _lpips

        total, n = 0.0, 0
        surf_psnr, surf_hole = 0.0, 0.0
        # 感知项只在抽样子集上算（全量太慢），PSNR 仍是全量。
        perc_budget = int(sel_percept_frames)
        perc_seen = 0
        ssim_sum, lpips_sum, perc_n = 0.0, 0.0, 0
        try:
            batches = list(source.iter_batches(max(1, batch_size)))
            if eval_frames is not None and eval_frames > 0 and batches:
                # 等间隔抽批：硬表面阶段每 100 步就要验证一次，跑全量太贵
                per_batch = max(1, int(batches[0].mesh.shape[0]))
                want = max(1, int(eval_frames) // per_batch)
                if want < len(batches):
                    step = max(1, len(batches) // want)
                    batches = batches[::step][:want]
        except Exception as e:                                  # noqa: BLE001
            print(f"[warn] 验证集遍历失败：{type(e).__name__}: {e}", flush=True)
            raise
        for batch in batches:
            mesh = Mesh(verts=batch.mesh.to(device), faces=scene.faces.to(device),
                        uvs=scene.uvs.to(device), uv_faces=scene.uv_faces.to(device))
            bg = torch.zeros(batch.image.shape[0], 3, 1, 1, device=device)
            out = forward_avatar(
                avatar, rasterizer, mesh, batch.blend_weight.to(device),
                camera, bg,
                options=(hard_eval if stage == "hard_surface"
                         else base_eval)).out
            pred, tgt, mask = out.color, batch.image.to(device), batch.mask.to(device)
            mse = (((pred - tgt) ** 2) * mask).sum() / (mask.sum() * 3).clamp(min=1)
            total += float(10.0 * torch.log10(1.0 / mse.clamp(min=1e-10))) * len(batch)
            n += len(batch)
            if perc_seen < perc_budget:
                img_np = (pred.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
                          * 255.0).astype(np.uint8)
                tgt_np = (tgt.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
                          * 255.0).astype(np.uint8)
                msk_np = mask[:, 0].cpu().numpy().astype(bool)
                for j in range(int(img_np.shape[0])):
                    if perc_seen >= perc_budget:
                        break
                    v = _ssim(img_np[j], tgt_np[j], msk_np[j])
                    l = _lpips(img_np[j], tgt_np[j])
                    if v is not None and l is not None:
                        ssim_sum += float(v)
                        lpips_sum += float(l)
                        perc_n += 1
                    perc_seen += 1
            if stage == "hard_surface":
                # 硬表面只负责 FLAME 表面那一片：mask 内 PSNR 会被头发/衣服
                # 拖住，不能反映"壳做得好不好"。这里按表面 mask 单独统计，
                # 顺带量一下表面被覆盖的比例（孔洞率）。
                surf = torch.cat(
                    [_surface_mask(batch, i).to(device)
                     for i in range(int(batch.mesh.shape[0]))], dim=0) & mask
                mse_s = (((pred - tgt) ** 2) * surf).sum() \
                    / (surf.sum() * 3).clamp(min=1)
                surf_psnr += float(
                    10.0 * torch.log10(1.0 / mse_s.clamp(min=1e-10))) * len(batch)
                surf_hole += float(
                    (out.alpha < HOLE_THRESHOLD)[surf].float().mean()) * len(batch)
        metrics = {"psnr": total / max(1, n)}
        if perc_n:
            metrics["ssim"] = ssim_sum / perc_n
            metrics["lpips"] = lpips_sum / perc_n
            # 选模型/早停用：PSNR + w_s·SSIM + w_l·(1−LPIPS)
            metrics["score"] = (metrics["psnr"]
                                + float(sel_ssim_weight) * metrics["ssim"]
                                + float(sel_lpips_weight) * (1.0 - metrics["lpips"]))
        if stage == "hard_surface":
            metrics["psnr_surface"] = surf_psnr / max(1, n)
            metrics["surface_hole"] = surf_hole / max(1, n)
            # "初始化是否定型"：几何参数每次验证之间还在挪多少（mm）。
            # 覆盖与贴面会先收敛，颜色/PSNR 会后收敛 —— 后者 stage2 会重练，
            # 所以硬表面段的早停应该看几何，而不是看 PSNR 平台。
            with torch.no_grad():
                geom = (avatar.xyz.detach().clone(),
                        avatar.scaling.detach().clone())
                # 离绑定面的切空间偏移（mm）：初始化时是 0，训练中被允许
                # "微小形变"补偿 FLAME/RDF 误差，但不应长期缓慢外漂。
                off_mm = avatar.xyz.detach().norm(dim=-1) * 1000.0
                metrics["offset_mean_mm"] = float(off_mm.mean())
                metrics["offset_p99_mm"] = float(off_mm.quantile(0.99))
            prev = geom_ref.get("prev")
            if prev is not None:
                d_xyz = float((geom[0] - prev[0]).norm(dim=-1).mean()) * 1000.0
                d_sc = float((geom[1] - prev[1]).abs().mean()) * 1000.0
                metrics["geom_drift_mm"] = d_xyz
                metrics["scale_drift_mm"] = d_sc
                # EarlyStopping 越大越好：取负号
                metrics["geom_settled"] = -d_xyz
            geom_ref["prev"] = geom

        if robust_cfg is not None:
            # 只在部分验证帧上加大角度视角：全量会拖慢训练
            picked = set(subsample(n, robust_cfg.frames))
            holes: list[float] = []
            deficits: list[float] = []
            spills: list[float] = []
            frame = 0
            for batch in batches:
                for i in range(batch.mesh.shape[0]):
                    if frame not in picked:
                        frame += 1
                        continue
                    verts = batch.mesh[i:i + 1].to(device)
                    weight_i = batch.blend_weight[i:i + 1].to(device)
                    mesh = Mesh(verts=verts, faces=scene.faces.to(device),
                                uvs=scene.uvs.to(device),
                                uv_faces=scene.uv_faces.to(device))
                    with torch.no_grad():
                        gs = avatar.deform(mesh, weight_i)
                        center = verts[0].mean(dim=0).detach().cpu()
                        for view in robust_views(robust_cfg):
                            verts_v, gs_v = verts, gs
                            if pose_driver is not None:
                                # 与 GT 同语义：相机不动，头转
                                turned = pose_driver.turn(
                                    int(batch.indices[i]), view.yaw, view.pitch)
                                verts_v = turned.mesh.unsqueeze(0).to(device)
                                mesh_v = Mesh(verts=verts_v,
                                              faces=scene.faces.to(device),
                                              uvs=scene.uvs.to(device),
                                              uv_faces=scene.uv_faces.to(device))
                                gs_v = avatar.deform(
                                    mesh_v, turned.weight.unsqueeze(0).to(device))
                                cam = Camera(
                                    K=scene.K,
                                    w2c=w2c_data.unsqueeze(0).to(device),
                                    width=scene.width,
                                    height=scene.height).to(device)
                            else:
                                view_w2c = orbit_w2c(w2c_data, center, view.yaw,
                                                     view.pitch)
                                cam = Camera(K=scene.K,
                                             w2c=view_w2c.unsqueeze(0).to(device),
                                             width=scene.width,
                                             height=scene.height).to(device)
                            out = render_gaussians(
                                gs_v, rasterizer, cam,
                                torch.zeros(1, 3, 1, 1, device=device))
                            ids = face_id_map(verts_v, scene.faces, cam, glctx)
                            surface = ids >= 0
                            if bool(surface.any()):
                                alpha = out.alpha[0, 0]
                                holes.append(float(
                                    (alpha[surface] < HOLE_THRESHOLD).float()
                                    .mean()))
                                deficits.append(alpha_deficit(alpha, surface))
                                filled = alpha > 0.5
                                spills.append(float((filled & ~surface).sum())
                                              / max(1, int(filled.sum())))
                    frame += 1
            summary = summarise_holes(holes)
            metrics.update(summary)
            metrics.update(summarise_coverage(deficits))
            # 名字写实：统计的是"FLAME 轮廓外的**实心**像素"（含头发/肩），
            # 与训练里同名 loss（alpha∈[0.05,0.6] 的半透明溢出）不是一回事。
            outside = float(np.mean(spills)) if spills else 0.0
            metrics["opaque_outside_flame"] = outside
            metrics["outside_mean"] = outside          # 兼容旧字段名
            # 早停用阈值孔洞：与离线评估（compare_defects）同一口径。
            # 连续缺口只作为诊断量记录 —— 实测它与阈值口径几乎同值。
            # 面外溢出按 robust_spill 权重计入（默认 0，保持旧行为）。
            metrics["robust"] = (robust_score(metrics["psnr"],
                                              metrics["hole_mean"],
                                              weight=robust_cfg.weight)
                                 - float(robust_spill)
                                 * metrics["outside_mean"])
        last_eval.clear()
        last_eval.update(metrics)
        return metrics

    out_dir.mkdir(parents=True, exist_ok=True)

    from live3dgsavatar.core.io.ply import save_ply

    # 训练循环内的墙钟：每次存盘打印累计耗时，report 里也留一份。
    # t0 在 trainer.run() 前才置位，不把数据加载/预处理算进训练时间。
    timing = {"t0": None, "best": None, "saves": []}

    def _elapsed() -> float:
        return 0.0 if timing["t0"] is None else time.perf_counter() - timing["t0"]

    def save(iteration: int) -> None:
        path = out_dir / f"model_{iteration:06d}.ply"
        save_ply(avatar, path)
        secs = _elapsed()
        timing["saves"].append({"iteration": iteration, "file": path.name,
                                "seconds": round(secs, 1)})
        print(f"  [save] 第 {iteration} 步 {path.name}（累计 {secs:.1f}s）",
              flush=True)

    early_metric_name = (args.early_metric or
                         ("psnr_surface" if stage == "hard_surface" else "score"))
    # 指标必须真的会被算出来，否则 ValidationHook 永远不 update，
    # 跑满预算却没有 best 模型（评审 B6）。
    if early_metric_name == "robust" and robust_cfg is None:
        raise SystemExit(
            "[error] --early-metric robust 需要新视角验证（别加 --no-robust-eval）")
    if early_metric_name == "psnr_surface" and stage != "hard_surface":
        raise SystemExit(
            "[error] --early-metric psnr_surface 只适用于 --stage hard_surface")
    if early_metric_name == "score" and stage == "hard_surface":
        raise SystemExit(
            "[error] --early-metric score 只适用于 base 阶段")

    def save_best(iteration: int, metric: float) -> None:
        """验证指标刷新最好成绩时存一份 —— 后训变差也能拿回好的那份。"""
        save_ply(avatar, out_dir / "model_best.ply")
        secs = _elapsed()
        timing["best"] = secs
        shown_psnr = (last_eval.get("psnr_surface", last_eval.get("psnr", float("nan")))
                      if stage == "hard_surface"
                      else last_eval.get("psnr", float("nan")))
        if early_metric_name == "score":
            print(f"  [best] 第 {iteration} 步 score {metric:.2f}"
                  f"（PSNR {last_eval.get('psnr', float('nan')):.2f}，"
                  f"SSIM {last_eval.get('ssim', float('nan')):.4f}，"
                  f"LPIPS {last_eval.get('lpips', float('nan')):.4f}）"
                  f" → model_best.ply（累计 {secs:.1f}s）", flush=True)
        elif robust_cfg is not None:
            print(f"  [best] 第 {iteration} 步 robust {metric:.2f}"
                  f"（PSNR {shown_psnr:.2f}，"
                  f"缺口 {last_eval.get('deficit_mean', float('nan')):.3f}，"
                  f"阈值孔洞 {last_eval.get('hole_mean', float('nan')):.1%}，"
                  f"面外溢出 {last_eval.get('outside_mean', float('nan')):.1%}）"
                  f" → model_best.ply（累计 {secs:.1f}s）", flush=True)
        else:
            label = {"psnr_surface": "表面 PSNR", "psnr": "PSNR（mask 内）"}.get(
                early_metric_name, early_metric_name)
            print(f"  [best] 第 {iteration} 步 {label} {metric:.2f} dB"
                  f" → model_best.ply（累计 {secs:.1f}s）", flush=True)

    val_hook = ValidationHook(
        source=val_src, evaluate=evaluate, every=eval_every,
        early_stop=EarlyStopping(patience=patience, min_delta=min_delta),
        metric=early_metric_name,
        on_improve=save_best, start_after=early_stop_after)
    hooks = [
        LogHook(every=100),
        val_hook,
        CheckpointHook(save=save, every=save_every),
    ]
    if need_support:
        def sample_face_stats(n_frames: int):
            """逐面真实前景像素数（只统计真实数据，给新视角 hole 用）。

            脖子/胸口在图像里是背景，mask 会排除，所以不会被当成"看到过"。
            """
            n_faces = int(scene.faces.shape[0])
            counts = np.zeros(n_faces, np.int64)
            flip = None
            seen = 0
            bg_stats = torch.zeros(1, 3, 1, 1, device=device)
            for batch in train_src.iter_batches(batch_size):
                for i in range(batch.mesh.shape[0]):
                    if seen >= n_frames:
                        break
                    verts = batch.mesh[i:i + 1].to(device)
                    weight_i = batch.blend_weight[i:i + 1].to(device)
                    if flip is None:
                        mesh = Mesh(verts=verts, faces=scene.faces.to(device),
                                    uvs=scene.uvs.to(device),
                                    uv_faces=scene.uv_faces.to(device))
                        with torch.no_grad():
                            out = render_frame(avatar, rasterizer, mesh,
                                               weight_i, camera, bg_stats)
                        flip, _ = detect_flip(
                            verts, scene.faces, camera, glctx,
                            out.alpha[0, 0].detach().cpu().numpy())
                    ids = face_id_map(verts, scene.faces, camera, glctx,
                                      flip_y=flip).detach().cpu().numpy()
                    fg = batch.mask[i, 0].numpy() > 0.5
                    sel = fg & (ids >= 0)
                    if bool(sel.any()):
                        np.add.at(counts, ids[sel], 1)
                    seen += 1
                if seen >= n_frames:
                    break
            print(f"  逐面前景统计: {seen} 帧，"
                  f"{int((counts > 0).sum())} 个面被看到", flush=True)
            return counts

        counts = sample_face_stats(nv_frames)
        supported_face = torch.as_tensor(counts >= int(nv_min_px),
                                         device=device)
        print(f"  新视角支持面: {int(supported_face.sum())}/"
              f"{supported_face.numel()}（≥{nv_min_px} px）", flush=True)

    from live3dgsavatar.data.sources import PrefetchSource

    if float(args.angle_weight) > 0:
        from live3dgsavatar.data.pose import angle_weights, head_yaw
        with np.load(data_dir / "render_inputs.npz", allow_pickle=True) as _d:
            _bw = np.asarray(_d["blend_weight"], dtype=np.float32)
        _yaw = np.abs(head_yaw(_bw))
        train_src.set_sample_weights(
            angle_weights(_yaw, scale=float(args.angle_weight), floor=1.0))
        print(f"  采样   : 按 |yaw| 加权（每 {float(args.angle_weight):g}° 翻倍）")

    train_source = train_src
    if workers > 0:
        train_source = PrefetchSource(train_source, batch_size, workers=workers,
                                      depth=max(2, workers * 2))
    print(f"  取数   : 预取 {workers} 线程（批 {batch_size}）")
    if getattr(train_src, "has_image_cache", False):
        print("  取数   : 使用 images.npy 缓存（跳过 PNG 解码）")
    else:
        print("  取数   : 逐帧解 PNG —— 可跑 scripts/cache_images.py 加速")

    densify_hook = None
    if args.densify:
        avatar_cfg = cfg.get("train.pipeline.avatar", {}) or {}
        dcfg = avatar_cfg.get("densify", {}) or {}
        d_every = int(args.densify_every if args.densify_every is not None
                      else dcfg.get("every", 300))
        d_count = int(args.densify_count if args.densify_count is not None
                      else dcfg.get("count", 500))
        d_max = int(args.densify_max if args.densify_max is not None
                    else dcfg.get("max_gaussians", avatar.num_gaussians * 2))
        d_op = float(args.densify_prune_opacity
                     if args.densify_prune_opacity is not None
                     else dcfg.get("prune_opacity", 0.01))
        d_jitter = float(args.densify_split_jitter
                         if args.densify_split_jitter is not None
                         else dcfg.get("split_jitter", 0.8))
        d_state = {"step": 0}

        d_hole = float(dcfg.get("split_hole_weight", 0.0))
        d_pattern = str(dcfg.get("split_pattern", "random"))
        d_mode = str(args.densify_split_mode
                     if args.densify_split_mode is not None
                     else dcfg.get("split_mode", "copy"))
        d_scale = float(dcfg.get("split_scale", 0.7))
        d_state = {"step": 0, "splits": 0}

        def densify_post():
            d_state["step"] += 1
            if d_state["step"] % d_every != 0:
                return
            if int(avatar.num_gaussians) >= d_max:
                return
            idx = None
            try:
                lb, lo = trainer.last_batch, trainer.last_out
                if lb is None or lo is None:
                    return
                nf = int(scene.faces.shape[0])
                face_err = torch.zeros(nf, device=device)
                face_cnt = torch.zeros(nf, device=device)
                face_hole = torch.zeros(nf, device=device)
                for i in range(int(lb.mesh.shape[0])):
                    if getattr(lb, "K", None) is not None:
                        cam_i = Camera(K=lb.K[i:i+1].to(device),
                                       w2c=lb.w2c[i:i+1].to(device),
                                       width=scene.width,
                                       height=scene.height).to(device)
                    else:
                        cam_i = camera
                    v = lb.mesh[i:i+1].to(device)
                    ids = face_id_map(v, scene.faces.to(device), cam_i,
                                      glctx, flip_y=False)
                    if ids.ndim == 3:
                        ids = ids[0]
                    err = (lo.color[i:i+1] - lb.image[i:i+1].to(device)).abs().mean(1).reshape(-1)
                    ids = ids.reshape(-1)
                    m = lb.mask[i:i+1, 0].to(device).reshape(-1).bool()
                    valid = (ids >= 0) & m
                    if bool(valid.any()):
                        f = ids[valid].long()
                        face_err.index_add_(0, f, err[valid])
                        face_cnt.index_add_(0, f, torch.ones_like(
                            err[valid], dtype=face_cnt.dtype))

                # 大角度覆盖缺口：渲染一个随机侧视角，统计表面内 alpha<0.5
                # 的像素。颜色残差会把"有高斯但纹理糊"和"根本没盖住"混在一起，
                # 这一项专门找后者，让分裂长到真正缺覆盖的面上。
                if d_hole > 0:
                    i = random.randrange(int(lb.mesh.shape[0]))
                    yaw_nv = float(random.choice((-40.0, -30.0, 30.0, 40.0)))
                    if pose_driver is not None:
                        turned = pose_driver.turn(int(lb.indices[i]), yaw_nv, 0.0)
                        verts_nv = turned.mesh.unsqueeze(0).to(device)
                        weight_nv = turned.weight.unsqueeze(0).to(device)
                        cam_nv = Camera(K=scene.K,
                                        w2c=w2c_data.unsqueeze(0).to(device),
                                        width=scene.width,
                                        height=scene.height).to(device)
                    else:
                        verts_nv = lb.mesh[i:i + 1].to(device)
                        weight_nv = lb.blend_weight[i:i + 1].to(device)
                        center_nv = verts_nv[0].mean(dim=0).detach().cpu()
                        view_nv = orbit_w2c(w2c_data, center_nv, yaw_nv, 0.0)
                        cam_nv = Camera(K=scene.K,
                                        w2c=view_nv.unsqueeze(0).to(device),
                                        width=scene.width,
                                        height=scene.height).to(device)
                    mesh_nv = Mesh(verts=verts_nv, faces=scene.faces.to(device),
                                   uvs=scene.uvs.to(device),
                                   uv_faces=scene.uv_faces.to(device))
                    with torch.no_grad():
                        ids_nv = face_id_map(verts_nv, scene.faces.to(device),
                                             cam_nv, glctx, flip_y=False)
                        gs_nv = avatar.deform(mesh_nv, weight_nv)
                        out_nv = render_gaussians(
                            gs_nv, rasterizer, cam_nv,
                            torch.zeros(1, 3, 1, 1, device=device))
                        surf_nv = ids_nv >= 0
                        hole_nv = (out_nv.alpha[0, 0] < 0.5) & surf_nv
                        if bool(surf_nv.any()):
                            f_nv = ids_nv[surf_nv].long()
                            face_hole.index_add_(0, f_nv, hole_nv[surf_nv].float())

                cnt = face_cnt.clamp_min(1.0)
                err_mean = face_err / cnt
                err_mean = err_mean / err_mean.max().clamp_min(1e-8)
                hole_rate = face_hole / cnt
                score = d_hole * hole_rate + (1.0 - d_hole) * err_mean
                if float(score.max()) > 0:
                    idx = avatar.residual_indices(
                        score,
                        min(d_count, d_max - int(avatar.num_gaussians)),
                        min_opacity=d_op,
                        max_offset=float(dcfg.get("max_offset", 0.08)))
            except Exception as e:
                print(f"  [densify] 残差选择失败：{e}", flush=True)
            if idx is None or idx.numel() == 0:
                return
            n_before = int(avatar.num_gaussians)
            added = avatar.split_unified(
                idx, scale_factor=(d_scale if d_mode == "subdivide" else 0.5),
                jitter=d_jitter, pattern=d_pattern,
                phase=d_state["splits"] % 8, mode=d_mode)
            d_state["splits"] += 1
            if added:
                with torch.no_grad():
                    # 初始硬表面/已有高斯全部保留；只剪本次新增的低 opacity 副本。
                    keep = torch.ones(int(avatar.num_gaussians), dtype=torch.bool,
                                      device=avatar.opacity.device)
                    keep[n_before:] = (
                        torch.sigmoid(avatar.opacity).squeeze(-1)[n_before:] >= d_op)
                avatar.prune_unified(keep)
                trainer.rebuild_optimizer()
                print(f"  [densify] split +{added} → {int(avatar.num_gaussians)}", flush=True)
        densify_hook = densify_post

    trainer = Trainer(source=train_source, plan=plan, objective=objective,
                      render_fn=render_fn, params=avatar.parameter_groups,
                      iterations=iterations, batch_size=batch_size,
                      blend_start=blend_start, hooks=hooks, device=device,
                      post_step=(densify_hook if densify_hook is not None
                                 else offset_guard))
    timing["t0"] = time.perf_counter()
    report = trainer.run()

    # 最终模型（训练结束时的那一份）—— 定期存档已在 CheckpointHook 里做过
    from live3dgsavatar.core.io.ply import save_ply

    final_ply = out_dir / "model.ply"
    save_ply(avatar, final_ply)

    print(f"\n最终模型：{final_ply}（累计 {_elapsed():.1f}s）")

    print(f"\n训练结束：{report.iterations} 步，{report.seconds:.1f} s，"
          f"{report.iters_per_s:.2f} it/s")
    if report.best_metric is not None:
        best_secs = timing["best"]
        suffix = f"，累计 {best_secs:.1f}s" if best_secs is not None else ""
        print(f"  最佳验证 PSNR {report.best_metric:.2f} dB"
              f"（第 {report.best_iteration} 步{suffix}）")
    if report.stopped_early:
        print(f"  早停   : 连续 {patience} 次验证提升 < {min_delta:g} dB"
              f"（每 {eval_every} 步验证）")
    eval_history = [
        {"iteration": int(it),
         **{k: float(v) for k, v in metrics.items()
            if isinstance(v, (int, float, np.integer, np.floating))}}
        for it, metrics in val_hook.history
    ]
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps({
        "subject": subject, "work_name": work, "stage": stage,
        "iterations": report.iterations, "seconds": report.seconds,
        "best_psnr": report.best_metric, "best_iteration": report.best_iteration,
        "best_seconds": (round(timing["best"], 1)
                         if timing["best"] is not None else None),
        "checkpoints": timing["saves"],
        "eval_every": eval_every, "patience": patience,
        "min_delta": min_delta, "early_stop_after": early_stop_after,
        "ssim_weight": ssim_weight, "lpips_weight": lpips_weight,
        "lpips_from": lpips_from,
        "eval_history": eval_history,
        "stopped_early": report.stopped_early,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    best_ply = out_dir / "model_best.ply"
    src = best_ply if best_ply.is_file() else final_ply
    print(f"  装入来源：{src.name}")
    name = args.promote_name or work
    target = models_dir / subject / name
    try:
        promote_model(source_ply=src, structure_config=model_cfg,
                      report=report_path, target=target, force=args.force)
    except (FileExistsError, FileNotFoundError) as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1
    print(f"已装入：{target}")
    # 最终模型只保留在 models/；output/train 只留 checkpoint 与报告。
    for duplicate in (final_ply, best_ply):
        if duplicate.is_file():
            duplicate.unlink()
    print(f"  评测：bash eval.sh --subject {subject} --work-name {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
