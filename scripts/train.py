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
只用 L1；验证集取末尾 350 帧。
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
    p.add_argument("--init-ply", type=Path, default=None,
                   help="从已有 PLY warm start（默认从 UV 绑定新建）")
    p.add_argument("--lr-scale", type=float, default=1.0,
                   help="所有参数组学习率乘该系数（warm start 微调用，默认 1）")
    p.add_argument("--no-overwrite", action="store_true",
                   help="目标模型已存在则直接退出（保护旧模型做基线）")
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
    p.add_argument("--post-train", dest="post_train", action="store_true",
                   default=None,
                   help="训练后接迭代修复（默认取配置 post_train.enabled）")
    p.add_argument("--no-post-train", dest="post_train", action="store_false",
                   help="强制关闭后训练")
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

    iterations = int(args.iterations if args.iterations is not None
                     else cfg.get("train.pipeline.iterations", 50_000))
    batch_size = int(args.batch_size if args.batch_size is not None
                     else cfg.get("train.pipeline.batch_size", 10))
    blend_start = int(args.blend_start if args.blend_start is not None
                      else cfg.get("train.pipeline.blend_start", 3_000))
    # 形变基在 blend_start 才打开；早停至少等它生效 1000 步
    early_stop_after = max(0, blend_start + 1000)
    test_ratio = float(args.test_ratio if args.test_ratio is not None
                       else cfg.get("train.pipeline.test_ratio", 0.3))
    eval_every = int(args.eval_every if args.eval_every is not None
                     else cfg.get("train.pipeline.eval_every", 500))
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
    nv_enabled = nv_every > 0 and (nv_hole > 0 or nv_spill > 0)
    if nv_enabled and not nv_yaws:
        raise SystemExit("[error] 新视角监督需要至少一个 --nv-yaws")
    need_support = nv_enabled and nv_hole > 0

    def _pt(key, default):
        return cfg.get(f"train.pipeline.post_train.{key}", default)

    post_enabled = bool(args.post_train if args.post_train is not None
                        else _pt("enabled", False))
    post_top_n = int(_pt("top_n", 2))
    post_rings = str(_pt("rings", "20,10;30,20;40,30;50,30"))
    post_steps = int(_pt("steps", 200))
    post_refined = int(_pt("refined_per_batch", 2))
    post_real = int(_pt("real_per_batch", 8))
    post_quality_tol = float(_pt("quality_tol", 0.01))
    post_psnr_weight = float(_pt("psnr_weight", 0.002))
    post_mask_dilate = int(_pt("mask_dilate", 12))
    post_min_fill = float(_pt("min_fill", 0.45))
    post_base_model = str(_pt("base_model", "emilianJR/epiCRealism"))
    post_ip_adapter = str(_pt("ip_adapter", "none"))
    post_ip_adapter_weight = str(_pt("ip_adapter_weight",
                                     "ip-adapter-faceid_sd15.bin"))
    post_ip_adapter_image_encoder = str(_pt(
        "ip_adapter_image_encoder", "h94/IP-Adapter/sdxl_models/image_encoder"))
    post_ip_adapter_scale = float(_pt("ip_adapter_scale", 0.6))
    post_strength = float(_pt("strength", 0.85))
    post_core_dilate = int(_pt("core_dilate", 12))
    post_identity_threshold = float(_pt("identity_threshold", 0.30))
    post_trusted_tol = float(_pt("trusted_tol", 3.0))
    post_lr_xyz = float(_pt("lr_xyz", 1e-5))
    post_color_match = bool(_pt("color_match", False))
    post_color_mean_only = bool(_pt("color_mean_only", False))
    post_color_ring = int(_pt("color_ring", 8))
    post_feather = int(_pt("feather", 4))
    post_prompt = str(_pt("prompt", "professional photo of a human head, "
                                   "natural skin texture, fine detail, sharp focus"))
    post_negative = str(_pt("negative", "blurry, smooth plastic, deformed, "
                                       "cartoon, extra eyes, low resolution, "
                                       "watermark"))

    from live3dgsavatar.data.sources import ManifestSource

    max_frames = cfg.get("train.pipeline.max_frames", None)
    if max_frames is not None and int(max_frames) <= 0:
        max_frames = None
    train_src = ManifestSource(data_dir, split="train", test_ratio=test_ratio,
                               device="cpu", seed=args.seed,
                               max_frames=(None if max_frames is None
                                           else int(max_frames)))
    val_src = ManifestSource(data_dir, split="test", test_ratio=test_ratio,
                             device="cpu")

    print(f"训练 {subject}/{work}")
    print(f"  数据   : {data_dir}")
    print(f"  训练集 : {len(train_src)} 帧    验证集 : {len(val_src)} 帧"
          f"（末尾 {test_ratio:.0%}）")
    models_dir = Path(str(cfg.get("paths.models_dir")))
    init_path = (Path(args.init_ply).expanduser()
                 if args.init_ply is not None else None)
    if init_path is not None and not init_path.is_file():
        raise SystemExit(f"[error] --init-ply 不存在：{init_path}")
    target_name = args.promote_name or work
    target_dir = models_dir / subject / target_name
    if args.no_overwrite and target_dir.exists():
        raise SystemExit(
            f"[error] 目标已存在：{target_dir}\n"
            "         --no-overwrite：换个 --promote-name，或去掉该开关")
    model_cfg = resolve_structure_config(
        models_dir, subject, work, explicit=args.model_config,
        repo_default=REPO_ROOT / "configs" / "model.yaml")
    if init_path is not None and args.model_config is None:
        sibling = init_path.parent / "config.yaml"
        if sibling.is_file():
            model_cfg = sibling
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
        if post_enabled:
            print(f"  [dry-run] 后训练: top{post_top_n} rings {post_rings}，"
                  f"{post_steps} 步/轮（默认关，需 epiC 模型）")
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
    from live3dgsavatar.core.render import SimpleRasterizer
    from live3dgsavatar.core.render.faceid import detect_flip, face_id_map
    from live3dgsavatar.evaluation import (RobustConfig, alpha_deficit,
                                           robust_score, robust_views,
                                           subsample, summarise_coverage,
                                           summarise_holes)
    from live3dgsavatar.core.types import Camera, Mesh
    from live3dgsavatar.data.scene import load_scene
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

    camera = Camera.from_intrinsics_extrinsics(
        K=scene.K, R=scene.R, T=scene.T,
        width=scene.width, height=scene.height).to(str(device))
    print(f"  高斯数 : {avatar.num_gaussians}   基 K={avatar.num_basis}")

    # 数据相机（world→camera）：大角度验证要绕它旋转
    w2c_data = torch.eye(4, dtype=torch.float32)
    w2c_data[:3, :3] = scene.R.detach().cpu()
    w2c_data[:3, 3] = scene.T.detach().cpu()
    robust_cfg = (RobustConfig(frames=robust_frames, weight=robust_weight)
                  if robust_enabled else None)
    if robust_cfg is not None:
        print(f"  验证   : 额外视角 ±{', ±'.join(f'{y:g}' for y in robust_cfg.yaws)}°"
              f"（{robust_cfg.frames} 帧，λ={robust_cfg.weight:g}）")

    plan = ParameterPlan.rgba_avatar_defaults(
        scene_extent=float(cfg.get("render.scene_extent", 1.0)))
    if lr_scale != 1.0:
        plan = plan.scaled(lr_scale)
        print(f"  lr 缩放 : {lr_scale:g}")
    mask_weight = float(args.mask_weight if args.mask_weight is not None
                        else cfg.get("train.pipeline.mask_weight", 1.0))
    ssim_weight = float(args.ssim_weight if args.ssim_weight is not None
                        else cfg.get("train.pipeline.ssim_weight", 0.0))
    lpips_weight = float(args.lpips_weight if args.lpips_weight is not None
                         else cfg.get("train.pipeline.lpips_weight", 0.0))
    lpips_from = int(args.lpips_from if args.lpips_from is not None
                     else cfg.get("train.pipeline.lpips_from", 20_000))
    objective = Objective.rgba_avatar_defaults()
    objective.mask_weight = mask_weight
    objective.nv_hole = nv_hole
    objective.nv_spill = nv_spill
    objective.ssim = ssim_weight
    objective.lpips = lpips_weight
    objective.lpips_from = lpips_from
    supported_face = None
    loss_terms = [f"L1（背景权重 {mask_weight:.2f}）"]
    if nv_spill > 0:
        loss_terms.append(f"nv_spill {nv_spill:g}")
    if nv_hole > 0:
        loss_terms.append(f"nv_hole {nv_hole:g}")
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
    def render_fn(mesh_verts, weight, batch, iteration):
        mesh = Mesh(verts=mesh_verts.to(device), faces=scene.faces.to(device),
                    uvs=scene.uvs.to(device), uv_faces=scene.uv_faces.to(device))
        gs = avatar.deform(mesh, None if weight is None else weight.to(device))
        bg = torch.zeros(batch.image.shape[0], 3, 1, 1, device=device)
        out = rasterizer.render(gs, camera, bg, target_image=batch.image.to(device))

        if nv_enabled and (int(iteration) + 1) % int(nv_every) == 0:
            from live3dgsavatar.training.losses import (alpha_hole_loss,
                                                       surface_spill_loss)
            i = random.randrange(int(mesh_verts.shape[0]))
            yaw = float(random.choice(nv_yaws))
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
            out_nv = rasterizer.render(
                avatar.deform(mesh_nv, weight_nv), cam_nv,
                torch.zeros(1, 3, 1, 1, device=device))
            if nv_hole > 0 and supported_nv is not None:
                out.nv_hole_loss = alpha_hole_loss(
                    out_nv.alpha, supported_nv.unsqueeze(0).unsqueeze(0))
            if nv_spill > 0:
                out.nv_spill_loss = surface_spill_loss(
                    out_nv.alpha, surface_nv.unsqueeze(0).unsqueeze(0),
                    margin=nv_margin, alpha_min=nv_alpha_min,
                    alpha_max=nv_alpha_max)

        return out

    last_eval: dict[str, float] = {}

    @torch.no_grad()
    def evaluate(source) -> dict[str, float]:
        """验证集 PSNR（前景掩码内）。

        这里**不走** `render_fn`：训练态会把 `target_image` 传给光栅化器，
        由它额外算 `est_color`/`est_weight`（供颜色回写用）。评测只需要
        颜色与 alpha，带上 `target_image` 会走一条在 `no_grad` 下容易出错的
        内核路径（实测在第 2000 步验证时触发过 CUDA illegal memory access）。
        所以评测单独走一条纯推理路径。
        """
        total, n = 0.0, 0
        try:
            batches = list(source.iter_batches(max(1, batch_size)))
        except Exception as e:                                  # noqa: BLE001
            print(f"[warn] 验证集遍历失败：{type(e).__name__}: {e}", flush=True)
            raise
        for batch in batches:
            mesh = Mesh(verts=batch.mesh.to(device), faces=scene.faces.to(device),
                        uvs=scene.uvs.to(device), uv_faces=scene.uv_faces.to(device))
            with torch.no_grad():
                gs = avatar.deform(mesh, batch.blend_weight.to(device))
                bg = torch.zeros(batch.image.shape[0], 3, 1, 1, device=device)
                out = rasterizer.render(gs, camera, bg)
            pred, tgt, mask = out.color, batch.image.to(device), batch.mask.to(device)
            mse = (((pred - tgt) ** 2) * mask).sum() / (mask.sum() * 3).clamp(min=1)
            total += float(10.0 * torch.log10(1.0 / mse.clamp(min=1e-10))) * len(batch)
            n += len(batch)
        metrics = {"psnr": total / max(1, n)}

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
                            view_w2c = orbit_w2c(w2c_data, center, view.yaw,
                                                 view.pitch)
                            cam = Camera(K=scene.K,
                                         w2c=view_w2c.unsqueeze(0).to(device),
                                         width=scene.width,
                                         height=scene.height).to(device)
                            out = rasterizer.render(
                                gs, cam, torch.zeros(1, 3, 1, 1, device=device))
                            ids = face_id_map(verts, scene.faces, cam, glctx)
                            surface = ids >= 0
                            if bool(surface.any()):
                                alpha = out.alpha[0, 0]
                                holes.append(float((alpha[surface] < 0.5).float()
                                                   .mean()))
                                deficits.append(alpha_deficit(alpha, surface))
                                filled = alpha > 0.5
                                spills.append(float((filled & ~surface).sum())
                                              / max(1, int(filled.sum())))
                    frame += 1
            summary = summarise_holes(holes)
            metrics.update(summary)
            metrics.update(summarise_coverage(deficits))
            metrics["outside_mean"] = (float(np.mean(spills)) if spills else 0.0)
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

    def save_best(iteration: int, metric: float) -> None:
        """验证指标刷新最好成绩时存一份 —— 后训变差也能拿回好的那份。"""
        save_ply(avatar, out_dir / "model_best.ply")
        secs = _elapsed()
        timing["best"] = secs
        if robust_cfg is not None:
            print(f"  [best] 第 {iteration} 步 robust {metric:.2f}"
                  f"（PSNR {last_eval.get('psnr', float('nan')):.2f}，"
                  f"缺口 {last_eval.get('deficit_mean', float('nan')):.3f}，"
                  f"阈值孔洞 {last_eval.get('hole_mean', float('nan')):.1%}，"
                  f"面外溢出 {last_eval.get('outside_mean', float('nan')):.1%}）"
                  f" → model_best.ply（累计 {secs:.1f}s）", flush=True)
        else:
            print(f"  [best] 第 {iteration} 步 验证 {metric:.2f} dB"
                  f" → model_best.ply（累计 {secs:.1f}s）", flush=True)

    val_hook = ValidationHook(
        source=val_src, evaluate=evaluate, every=eval_every,
        early_stop=EarlyStopping(patience=patience, min_delta=min_delta),
        metric="robust" if robust_cfg is not None else "psnr",
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
                            gs = avatar.deform(mesh, weight_i)
                            out = rasterizer.render(gs, camera, bg_stats)
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

    train_source = train_src
    if workers > 0:
        train_source = PrefetchSource(train_source, batch_size, workers=workers,
                                      depth=max(2, workers * 2))
        print(f"  取数   : 预取 {workers} 线程（批 {batch_size}）")
    if train_src.has_image_cache:
        print("  取数   : 使用 images.npy 缓存（跳过 PNG 解码）")
    else:
        print("  取数   : 逐帧解 PNG —— 可跑 scripts/cache_images.py 加速")

    trainer = Trainer(source=train_source, plan=plan, objective=objective,
                      render_fn=render_fn, params=avatar.parameter_groups(),
                      iterations=iterations, batch_size=batch_size,
                      blend_start=blend_start, hooks=hooks, device=device)
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
        "subject": subject, "work_name": work,
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

    # 训练结束一律落盘：结果不装进 models/ 就没有可用的模型，
    # 所以不提供"不保存"的开关。目标已存在时直接覆盖（同名即视为重训）。
    best_ply = out_dir / "model_best.ply"
    src = best_ply if best_ply.is_file() else final_ply
    print(f"  装入来源：{src.name}")
    name = args.promote_name or work
    target = models_dir / subject / name
    try:
        promote_model(source_ply=src, structure_config=model_cfg,
                      report=report_path, target=target, force=True)
    except (FileExistsError, FileNotFoundError) as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1
    print(f"已装入：{target}")
    print(f"  评测：bash eval.sh --subject {subject} --work-name {name}")
    if post_enabled:
        import subprocess
        prefix = f"{work}_post"
        cmd = [sys.executable, "scripts/iterative_repair.py",
               "--subject", subject, "--base-work", name,
               "--prefix", prefix, "--top-n", str(post_top_n),
               "--rings", post_rings, "--steps", str(post_steps),
               "--refined-per-batch", str(post_refined),
               "--real-per-batch", str(post_real),
               "--quality-tol", str(post_quality_tol),
               "--psnr-weight", str(post_psnr_weight),
               "--mask-dilate", str(post_mask_dilate),
               "--min-fill", str(post_min_fill),
               "--base-model", post_base_model,
               "--ip-adapter", post_ip_adapter,
               "--ip-adapter-weight", post_ip_adapter_weight,
               "--ip-adapter-image-encoder", post_ip_adapter_image_encoder,
               "--ip-adapter-scale", str(post_ip_adapter_scale),
               "--strength", str(post_strength),
               "--core-dilate", str(post_core_dilate),
               "--lr-xyz", str(post_lr_xyz),
               "--identity-threshold", str(post_identity_threshold),
               "--trusted-tol", str(post_trusted_tol),
               "--color-ring", str(post_color_ring),
               *(["--color-match"] if post_color_match
                 else ["--no-color-match"]),
               *(["--color-mean-only"] if post_color_mean_only else []),
               "--feather", str(post_feather),
               "--prompt", post_prompt, "--negative", post_negative]
        print(f"\n后训练：{' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[1]),
                            check=False).returncode
        if rc != 0:
            print("[warn] 后训练失败；基础模型已保留", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
