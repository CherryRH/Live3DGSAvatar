#!/usr/bin/env python
"""训练入口：在**我们自己的管线**上训一个头像模型（P0 骨架）。

与 RGBAvatar 的 `train_offline.py` 对应，但：

- 数据走 `data/sources.ManifestSource`（训练/验证按末尾 N 帧切分）
- 参数计划、损失、调度、钩子都是可配置对象（见 `training/`）
- 产出：`model.ply` + `report.json`（含验证集指标与耗时）

用法::

    conda activate live3dgs
    python scripts/train.py --subject crh_1 --work-name test

默认超参与 crh_1 的 `model.yaml` 一致（batch 10、50000 步、只剩 L1）。
「验证集」默认取末尾 350 帧（与 RGBAvatar 相同）。
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
                   help="第几步开始启用形变基（RGBAvatar 默认 3000）")
    p.add_argument("--test-frames", type=int, default=None,
                   help="末尾多少帧作验证集（默认 350）")
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-config", type=Path, default=None,
                   help="模型结构配置（默认：模型目录里的 config.yaml，"
                        "没有就用仓库模板 configs/model.yaml）")
    p.add_argument("--output-dir", type=Path, default=None)
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
    test_frames = int(args.test_frames if args.test_frames is not None
                      else cfg.get("train.pipeline.test_frames", 350))

    from live3dgsavatar.data.sources import ManifestSource

    train_src = ManifestSource(data_dir, split="train", test_frames=test_frames,
                               device="cpu", seed=args.seed)
    val_src = ManifestSource(data_dir, split="test", test_frames=test_frames,
                             device="cpu")

    print(f"训练 {subject}/{work}")
    print(f"  数据   : {data_dir}")
    print(f"  训练集 : {len(train_src)} 帧    验证集 : {len(val_src)} 帧")
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
    objective = Objective.rgba_avatar_defaults()

    def render_fn(mesh_verts, weight, batch, iteration):
        mesh = Mesh(verts=mesh_verts.to(device), faces=scene.faces.to(device),
                    uvs=scene.uvs.to(device), uv_faces=scene.uv_faces.to(device))
        gs = avatar.deform(mesh, None if weight is None else weight.to(device))
        bg = torch.zeros(batch.image.shape[0], 3, 1, 1, device=device)
        return rasterizer.render(gs, camera, bg, target_image=batch.image.to(device))

    @torch.no_grad()
    def evaluate(source) -> dict[str, float]:
        """验证集 PSNR（mask 内）。训练与评测用同一个 objective 口径。"""
        total, n = 0.0, 0
        for batch in source.iter_batches(max(1, batch_size)):
            out = render_fn(batch.mesh.to(device), batch.blend_weight.to(device),
                            batch, 0)
            pred, tgt, mask = out.color, batch.image.to(device), batch.mask.to(device)
            mse = (((pred - tgt) ** 2) * mask).sum() / (mask.sum() * 3).clamp(min=1)
            total += float(10.0 * torch.log10(1.0 / mse.clamp(min=1e-10))) * len(batch)
            n += len(batch)
        return {"psnr": total / max(1, n)}

    out_dir.mkdir(parents=True, exist_ok=True)

    def save(iteration: int) -> None:
        from live3dgsavatar.core.io.ply import save_ply

        save_ply(avatar, out_dir / f"model_{iteration:06d}.ply")

    hooks = (
        LogHook(every=100),
        ValidationHook(source=val_src, evaluate=evaluate, every=args.eval_every,
                       early_stop=EarlyStopping(patience=5)),
        CheckpointHook(save=save, every=args.save_every),
    )

    trainer = Trainer(source=train_src, plan=plan, objective=objective,
                      render_fn=render_fn, params=avatar.parameter_groups(),
                      iterations=iterations, batch_size=batch_size,
                      blend_start=blend_start, hooks=hooks, device=device)
    report = trainer.run()

    print(f"\n训练结束：{report.iterations} 步，{report.seconds:.1f} s，"
          f"{report.iters_per_s:.2f} it/s")
    if report.best_metric is not None:
        print(f"  最佳验证 PSNR {report.best_metric:.2f} dB"
              f"（第 {report.best_iteration} 步）")
    (out_dir / "report.json").write_text(json.dumps({
        "subject": subject, "work_name": work,
        "iterations": report.iterations, "seconds": report.seconds,
        "best_psnr": report.best_metric, "best_iteration": report.best_iteration,
        "stopped_early": report.stopped_early,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
