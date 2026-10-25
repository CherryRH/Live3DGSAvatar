#!/usr/bin/env python
"""训练入口：在**我们自己的管线**上训一个约简高斯 blendshape 头像。

流程：

1. 从 `datasets/<subject>/` 读 `render_inputs.npz` + 图像，末尾 N 帧作验证
2. 从模型目录的 `config.yaml`（没有就用 `configs/model.yaml`）读网络结构
3. 用 UV 绑定初始化高斯，按参数计划优化，期间定期验证/存档
4. 产出：`model_XXXXXX.ply` + `report.json`（含验证指标与耗时）

用法::

    conda activate live3dgs
    python scripts/train.py --subject crh_1 --work-name test

默认超参与 `models/crh_1/test/config.yaml` 一致：batch 10、50000 步、
只用 L1；验证集取末尾 350 帧。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import live3dgsavatar  # noqa: F401


def promote_model(*, source_ply: Path, structure_config: Path, report: Path | None,
                  target: Path, force: bool = False,
                  filename: str = "model.ply") -> Path:
    """把一个训练好的模型装进 `models/<对象>/<版本>/`，供渲染直接使用。

    拷贝三样东西：

    - `model.ply`（或 `filename`）：高斯参数（PLY 头内嵌 `gaussian_config`）
    - `config.yaml`：结构配置的可读副本（换机器/人工查看用）
    - `report.json`：训练指标（可选，便于回溯训练来源）

    Raises:
        FileExistsError: 目标已存在且没给 `force` —— 覆盖已有模型是破坏性的，
            必须显式确认
    """
    import shutil

    if not Path(source_ply).is_file():
        raise FileNotFoundError(f"找不到要装的模型：{source_ply}")
    target = Path(target)
    dest = target / filename
    # 只拦"要装的这个文件已存在"：同一目录下 `model.ply` 与 `model_aug.ply`
    # 应该能并存（这正是对比增强训练效果所需要的）
    if dest.exists() and not force:
        raise FileExistsError(
            f"{dest} 已存在；确认覆盖请加 --force")
    target.mkdir(parents=True, exist_ok=True)
    def _copy(src: Path, dst: Path) -> None:
        # 源与目标可能是同一个文件：结构配置常常就是目标目录里的那份
        # （续训时 `resolve_structure_config` 会选中它），直接拷会抛 SameFileError
        if Path(src).resolve() == Path(dst).resolve():
            return
        shutil.copyfile(src, dst)

    _copy(source_ply, target / filename)
    _copy(structure_config, target / "config.yaml")
    if report is not None and Path(report).is_file():
        _copy(report, target / "report.json")
    return target


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
    p.add_argument("--augment", action="store_true",
                   help="把 <数据集>/augmented/ 混进训练集（验证集仍只用真实数据）")
    p.add_argument("--augment-weight", type=float, default=None,
                   help="增强样本的损失权重（0 = 完全不考虑）")
    p.add_argument("--test-ratio", type=float, default=None,
                   help="验证集比例（0.1 表示末尾 10%%）；默认 0.1")
    p.add_argument("--test-frames", type=int, default=None,
                   help="按帧数指定验证集（与 --test-ratio 二选一）")
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--save-every", type=int, default=None)
    p.add_argument("--patience", type=int, default=None,
                   help="验证连续多少次无提升就停（次数 × eval-every 步）")
    p.add_argument("--min-delta", type=float, default=None,
                   help="认为'有提升'的最小 dB 幅度")
    p.add_argument("--prefetch-workers", type=int, default=None,
                   help="后台预取线程数；0 表示不预取")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-config", type=Path, default=None,
                   help="模型结构配置（默认：模型目录里的 config.yaml，"
                        "没有就用仓库模板 configs/model.yaml）")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--promote", action="store_true",
                   help="训练结束后把模型装进 models/<对象>/<名字>/，可直接用于渲染")
    p.add_argument("--promote-name", default=None,
                   help="装进 models/ 时用的名字（默认与 --work-name 相同）")
    p.add_argument("--promote-from", type=Path, default=None,
                   help="装哪个文件（默认列表末尾的 model.ply）")
    p.add_argument("--force", action="store_true",
                   help="目标目录已存在时覆盖")
    p.add_argument("--dry-run", action="store_true",
                   help="不需要 GPU：检查数据与配置，打印将要做的事")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from live3dgsavatar.config import load_config, resolve_data_dir

    cfg = load_config()
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
    test_ratio = (None if args.test_frames is not None else
                  float(args.test_ratio if args.test_ratio is not None
                        else cfg.get("train.pipeline.test_ratio", 0.1)))
    test_frames = int(args.test_frames if args.test_frames is not None else 0)
    augment_weight = float(args.augment_weight if args.augment_weight is not None
                           else cfg.get("train.pipeline.augment_weight", 1.0))
    eval_every = int(args.eval_every if args.eval_every is not None
                     else cfg.get("train.pipeline.eval_every", 1000))
    save_every = int(args.save_every if args.save_every is not None
                     else cfg.get("train.pipeline.save_every", 5000))
    patience = int(args.patience if args.patience is not None
                   else cfg.get("train.pipeline.patience", 3))
    min_delta = float(args.min_delta if args.min_delta is not None
                      else cfg.get("train.pipeline.min_delta", 0.05))
    workers = int(args.prefetch_workers if args.prefetch_workers is not None
                  else cfg.get("train.pipeline.prefetch_workers", 2))

    from live3dgsavatar.data.sources import ManifestSource

    train_src = ManifestSource(data_dir, split="train", test_frames=test_frames,
                               test_ratio=test_ratio, device="cpu", seed=args.seed)
    val_src = ManifestSource(data_dir, split="test", test_frames=test_frames,
                             test_ratio=test_ratio, device="cpu")

    print(f"训练 {subject}/{work}")
    print(f"  数据   : {data_dir}")
    split_text = (f"比例 {test_ratio:.0%}" if test_ratio is not None
                  else f"{test_frames} 帧")
    print(f"  训练集 : {len(train_src)} 帧    验证集 : {len(val_src)} 帧"
          f"（{split_text}）")
    models_dir = Path(str(cfg.get("paths.models_dir")))
    model_cfg = resolve_structure_config(
        models_dir, subject, work, explicit=args.model_config,
        repo_default=REPO_ROOT / "configs" / "model.yaml")
    from live3dgsavatar.core.avatar import AvatarConfig

    avatar_cfg = AvatarConfig.from_yaml(model_cfg)

    print(f"  结构   : {model_cfg}")
    print(f"           tex {avatar_cfg.tex_size}  D {avatar_cfg.num_basis_in}  "
          f"K {avatar_cfg.num_basis_blend}")
    print(f"  步数   : {iterations}   batch {batch_size}   "
          f"blend_start {blend_start}")
    if blend_start >= iterations:
        print(f"  [hint] blend_start({blend_start}) ≥ 步数({iterations})："
              "整段都不会启用形变基，只在拟合静态外观")

    if args.dry_run:
        b = train_src.random_batch(min(batch_size, 2))
        print(f"  [dry-run] batch: mesh {tuple(b.mesh.shape)}  "
              f"weight {tuple(b.blend_weight.shape)}  image {tuple(b.image.shape)}")
        print(f"  [dry-run] 产物目录 {out_dir}")
        print("  [dry-run] 未执行训练（需要 GPU）")
        return 0

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("[error] 训练需要 GPU。可用 --dry-run 检查数据。")

    import nvdiffrast.torch as dr

    from live3dgsavatar.core.avatar import GaussianAvatar
    from live3dgsavatar.core.deform.binding import build_binding
    from live3dgsavatar.core.render import SimpleRasterizer
    from live3dgsavatar.core.types import Camera, Mesh
    from live3dgsavatar.data.scene import load_scene
    from live3dgsavatar.training.objectives import Objective
    from live3dgsavatar.training.plans import ParameterPlan
    from live3dgsavatar.training.schedule import EarlyStopping
    from live3dgsavatar.training.trainer import (CheckpointHook, LogHook, Trainer,
                                                 ValidationHook)

    torch.manual_seed(args.seed)
    device = torch.device(str(cfg.get("runtime.device", "cuda")))
    # ---- 几何与相机 ----
    scene = load_scene(cfg, frames=1, device=str(device))
    glctx = dr.RasterizeCudaContext()
    binding = build_binding(scene.uvs.to(device), scene.uv_faces.to(device),
                            tex_size=avatar_cfg.tex_size, glctx=glctx,
                            device=device)
    avatar = GaussianAvatar(avatar_cfg, binding).to(device)
    rasterizer = SimpleRasterizer()
    camera = Camera.from_intrinsics_extrinsics(
        K=scene.K, R=scene.R, T=scene.T,
        width=scene.width, height=scene.height).to(str(device))
    print(f"  高斯数 : {avatar.num_gaussians}   基 K={avatar.num_basis}")

    plan = ParameterPlan.rgba_avatar_defaults(
        scene_extent=float(cfg.get("render.scene_extent", 1.0)))
    mask_weight = float(args.mask_weight if args.mask_weight is not None
                        else cfg.get("train.pipeline.mask_weight", 1.0))
    objective = Objective.rgba_avatar_defaults()
    objective.mask_weight = mask_weight
    print(f"  损失   : L1（背景权重 {mask_weight:.2f}）")

    def render_fn(mesh_verts, weight, batch, iteration):
        mesh = Mesh(verts=mesh_verts.to(device), faces=scene.faces.to(device),
                    uvs=scene.uvs.to(device), uv_faces=scene.uv_faces.to(device))
        gs = avatar.deform(mesh, None if weight is None else weight.to(device))
        bg = torch.zeros(batch.image.shape[0], 3, 1, 1, device=device)
        return rasterizer.render(gs, camera, bg, target_image=batch.image.to(device))

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
        return {"psnr": total / max(1, n)}

    out_dir.mkdir(parents=True, exist_ok=True)

    from live3dgsavatar.core.io.ply import save_ply

    tag = "_aug" if (args.augment and augment_weight > 0) else ""

    def save(iteration: int) -> None:
        save_ply(avatar, out_dir / f"model{tag}_{iteration:06d}.ply")

    def save_best(iteration: int, psnr: float) -> None:
        """验证指标刷新最好成绩时存一份 —— 后训变差也能拿回好的那份。"""
        save_ply(avatar, out_dir / f"model_best{tag}.ply")
        print(f"  [best] 第 {iteration} 步 验证 {psnr:.2f} dB → model_best.ply",
              flush=True)

    hooks = (
        LogHook(every=100),
        ValidationHook(source=val_src, evaluate=evaluate, every=eval_every,
                       early_stop=EarlyStopping(patience=patience,
                                                min_delta=min_delta),
                       on_improve=save_best),
        CheckpointHook(save=save, every=save_every),
    )

    from live3dgsavatar.data.sources import MixedSource, PrefetchSource

    train_source = train_src
    aug_dir = data_dir / str(cfg.get("train.pipeline.augment_dir", "augmented"))
    if args.augment:
        if not (aug_dir / "render_inputs.npz").is_file():
            raise SystemExit(
                f"[error] 找不到增强集 {aug_dir}/render_inputs.npz\n"
                "        先用 scripts/build_augmented.py 打包")
        if augment_weight <= 0:
            print("  [warn] --augment-weight=0：增强集不参与（等于没开 --augment）")
        else:
            aug_src = ManifestSource(aug_dir, split="train",
                                     test_ratio=test_ratio, device="cpu",
                                     seed=args.seed + 1)
            train_source = MixedSource([(train_src, 1.0),
                                        (aug_src, augment_weight)], seed=args.seed)
            print(f"  增强集 : {len(aug_src)} 帧（权重 {augment_weight:g}，"
                  f"占混合采样 {len(aug_src)/(len(aug_src)+len(train_src)):.0%}）")
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
    report = trainer.run()

    # 最终模型（训练结束时的那一份）—— 定期存档已在 CheckpointHook 里做过
    from live3dgsavatar.core.io.ply import save_ply

    final_ply = out_dir / f"model{tag}.ply"
    save_ply(avatar, final_ply)
    import shutil as _shutil

    _shutil.copyfile(model_cfg, out_dir / "config.yaml")
    print(f"\n最终模型：{final_ply}")

    print(f"\n训练结束：{report.iterations} 步，{report.seconds:.1f} s，"
          f"{report.iters_per_s:.2f} it/s")
    if report.best_metric is not None:
        print(f"  最佳验证 PSNR {report.best_metric:.2f} dB"
              f"（第 {report.best_iteration} 步）")
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps({
        "subject": subject, "work_name": work,
        "iterations": report.iterations, "seconds": report.seconds,
        "best_psnr": report.best_metric, "best_iteration": report.best_iteration,
        "stopped_early": report.stopped_early,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.promote:
        best_ply = out_dir / f"model_best{tag}.ply"
        src = args.promote_from or (best_ply if best_ply.is_file() else final_ply)
        print(f"  装入来源：{src.name}")
        name = args.promote_name or work
        target = models_dir / subject / name
        try:
            promote_model(source_ply=src, structure_config=model_cfg,
                          report=report_path, target=target, force=args.force,
                          filename=f"model{tag}.ply")
        except (FileExistsError, FileNotFoundError) as e:
            print(f"[error] {e}", file=sys.stderr)
            return 1
        print(f"已装入：{target}")
        print(f"  渲染：python scripts/render_test.py --subject {subject} "
              f"--work-name {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
