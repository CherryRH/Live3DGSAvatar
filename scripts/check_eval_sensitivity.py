#!/usr/bin/env python
"""检验评估管线对**结构缺陷**是否有足够的敏感度与特异性。

## 为什么需要它

后面的优化都要靠"缺陷指标"判断有没有用。如果指标对结构破坏不敏感，就无法
验收；如果它对颜色变化也敏感，就分不清"补了结构"和"调了颜色"。

做法是**受控扰动**：对一个基线模型施加已知类型的破坏，看指标是否按预期方向变化。

| 扰动 | 预期 hole | 预期 锐度 | 预期 与精修图的 PSNR | 说明 |
|---|---|---|---|---|
| clean | 基线 | 基线 | 基线 | — |
| 局部压不透明度 | **↑** | ↓ | ↓ | 结构信号 |
| 局部缩小尺度 | **↑** | ↓ | ↓ | 结构信号 |
| 局部改颜色 | ≈ 不变 | ≈ 不变 | ↓ | **特异性**：颜色变化不该被当成结构缺陷 |
| 全局放大尺度 | ↓ | ↓ | ↓ | 权衡：补住空洞但变糊 |
| 全局颜色噪声 | ≈ 不变 | ↑ | ↓ | 噪声也不该被当成结构缺陷 |

"局部"指绑定在下半张脸（下巴/脖子）的 texel —— 这是实际缺陷最集中的区域。

## 用法

```bash
python scripts/check_eval_sensitivity.py --subject crh_1 --work-name run
```

需要 GPU（光栅化）。产物：`<缺陷目录>/eval_sensitivity.json`。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import live3dgsavatar  # noqa: F401


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="评估管线的敏感度检查")
    p.add_argument("--subject", default=None)
    p.add_argument("--work-name", default="run")
    p.add_argument("--dataset", default=None)
    p.add_argument("--defects", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def measure(alpha, ids, image, refined=None):
    """单个视角的结构/外观指标（与训练期、离线评估同一口径）。"""
    import numpy as np

    from live3dgsavatar.evaluation.robustness import alpha_deficit, alpha_gradient

    surface = ids >= 0
    hole = float((alpha[surface] < 0.5).mean()) if bool(surface.any()) else 0.0
    deficit = alpha_deficit(alpha, surface)
    spike = alpha_gradient(alpha, surface)
    fg = alpha > 0.5
    if bool(fg.any()):
        grey = image.astype(np.float32).mean(axis=-1)
        gy, gx = np.gradient(grey)
        sharp = float(np.hypot(gx, gy)[fg].mean())
    else:
        sharp = 0.0
    psnr = None
    if refined is not None:
        from live3dgsavatar.core.metrics import psnr_masked

        psnr = psnr_masked(image, refined, refined.max(axis=-1) > 8)
    return hole, sharp, psnr, deficit, spike


def summarise(rows: list[dict]) -> dict:
    import numpy as np

    out = {}
    for key in ("hole", "sharp", "psnr", "deficit", "spike"):
        vals = [r[key] for r in rows if r.get(key) is not None]
        out[key] = round(float(np.mean(vals)), 4) if vals else None
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import numpy as np

    from live3dgsavatar.config import load_config
    from live3dgsavatar.data.datasets import resolve_dataset_dir
    from live3dgsavatar.data.models import models_root
    from live3dgsavatar.evaluation.views import parse_view_label

    cfg = load_config()
    subject = args.subject or str(cfg.get("subject") or "")
    work = args.work_name or str(cfg.get("work_name") or "run")
    if not subject:
        raise SystemExit("[error] 需要 --subject")
    cfg.set("subject", subject)
    cfg.set("work_name", work)
    dataset = args.dataset or subject
    try:
        data_dir = resolve_dataset_dir(cfg, dataset)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"[error] {e}") from None
    ply = models_root(cfg) / subject / work / "model.ply"
    if not ply.is_file():
        raise SystemExit(f"[error] 找不到模型：{ply}")

    defects_dir = Path(args.defects or (
        Path(str(cfg.get("paths.output_dir", "output"))) / "defects"
        / subject / work / dataset))
    selected = defects_dir / "selected.json"
    if not selected.is_file():
        raise SystemExit(f"[error] 找不到 {selected}（先跑 find_defects.py）")
    labels = json.loads(selected.read_text(encoding="utf-8"))["views"]["chosen"]

    print(f"模型   : {ply}")
    print(f"缺陷   : {defects_dir}（{len(labels)} 个视角）")
    if args.dry_run:
        print("\n[dry-run] 未执行渲染")
        return 0

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("[error] 需要 GPU")

    import nvdiffrast.torch as dr
    from PIL import Image

    from live3dgsavatar.core.camera import orbit_w2c
    from live3dgsavatar.core.io import load_ply
    from live3dgsavatar.core.io.ply import read_embedded_config
    from live3dgsavatar.core.render import SimpleRasterizer
    from live3dgsavatar.core.render.faceid import face_id_map
    from live3dgsavatar.core.types import Camera, Mesh
    from live3dgsavatar.data.scene import load_scene

    device = "cuda"
    with np.load(data_dir / "render_inputs.npz", allow_pickle=True) as d:
        total = int(d["mesh"].shape[0])
    ratio = float(cfg.get("train.pipeline.test_ratio", 0.1))
    train_end = total - max(1, int(round(total * ratio)))

    glctx = dr.RasterizeCudaContext()
    scene = load_scene(cfg, frames=train_end, data_dir=data_dir)
    rasterizer = SimpleRasterizer()
    avatar = load_ply(ply, config=read_embedded_config(ply), device=device)
    faces = scene.faces.to(torch.int32)
    w2c_data = torch.eye(4, dtype=torch.float32)
    w2c_data[:3, :3] = scene.R.detach().cpu()
    w2c_data[:3, 3] = scene.T.detach().cpu()
    bg = torch.tensor([float(x) for x in cfg.get("render.background", [0, 0, 0])],
                      dtype=torch.float32, device=device)

    # 观测：按 selected.json 的标签还原（帧、相机、精修图）
    obs = []
    for label in labels:
        info = parse_view_label(label)
        idx = int(info["expr"][5:]) if info["expr"].startswith("frame") else 0
        idx = min(max(idx, 0), scene.num_frames - 1)
        frame = scene.frames[idx]
        center = frame["mesh"].mean(dim=0).detach().cpu()
        view = orbit_w2c(w2c_data, center, info["yaw"], info["pitch"])
        cam = Camera(K=scene.K, w2c=view.unsqueeze(0).to(device),
                     width=scene.width, height=scene.height).to(device)
        refined = None
        path = defects_dir / "refined" / f"{label}.png"
        if path.is_file():
            refined = np.asarray(Image.open(path).convert("RGB"), np.float32)
        obs.append({"frame": frame, "cam": cam, "refined": refined})

    # 局部区域：绑定在"面重心 y 低于中位数"的 texel（下巴/脖子）
    with torch.no_grad():
        # 面重心（向量化）：mesh[faces] → [F,3,3] → 平均
        centroids = obs[0]["frame"]["mesh"][scene.faces].mean(dim=1)
        # 取下 20% 的面作为"局部区域"（下巴/脖子一带）——要够局部，
        # 否则扰动一大片，指标响应反而说明不了敏感度
        cut = float(centroids[:, 1].quantile(0.2))
        low_faces = centroids[:, 1] <= cut
        region = low_faces[avatar.binding_face_id]
        if int(region.sum()) == 0:                    # 兜底：随机 10% 的 texel
            region = torch.zeros_like(region)
            region[torch.randperm(region.numel(), device=device)[: region.numel() // 10]] = True
    print(f"  局部区域: {int(region.sum())} 个 texel（{region.float().mean():.1%}）")

    def evaluate() -> dict:
        rows = []
        for item in obs:
            mesh = Mesh(verts=item["frame"]["mesh"].unsqueeze(0), faces=scene.faces,
                        uvs=scene.uvs, uv_faces=scene.uv_faces)
            with torch.no_grad():
                gs = avatar.deform(mesh, item["frame"]["blend_weight"].unsqueeze(0))
                out = rasterizer.render(gs, item["cam"], bg)
                ids = face_id_map(item["frame"]["mesh"].unsqueeze(0), faces,
                                  item["cam"], glctx)
            image = (out.color[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                     * 255.0).astype(np.uint8)
            hole, sharp, psnr, deficit, spike = measure(
                out.alpha[0, 0].detach().cpu().numpy(),
                ids.detach().cpu().numpy(), image, item["refined"])
            rows.append({"hole": hole, "sharp": sharp, "psnr": psnr,
                         "deficit": deficit, "spike": spike})
        return summarise(rows)

    # 参数按语义分组取（`f_dc` 对应 `avatar.color`，不要猜属性名）
    groups = avatar.parameter_groups()
    opacity, scaling, color = groups["opacity"], groups["scaling"], groups["f_dc"]
    original = {k: p.detach().clone()
                for k, p in (("opacity", opacity), ("scaling", scaling),
                             ("f_dc", color))}

    def restore() -> None:
        opacity.data.copy_(original["opacity"])
        scaling.data.copy_(original["scaling"])
        color.data.copy_(original["f_dc"])

    results: dict[str, dict] = {}

    results["clean"] = evaluate()

    def scale_opacity(mask, factor: float) -> None:
        """把不透明度**乘**一个系数（在概率空间，而不是 logit 上相减）。

        logit 上减 4 对"已经很实"的高斯几乎无效（0.999 → 0.967），只会打击
        本来就不可见的那批 —— 之前的写法因此低估了不透明度的作用。
        """
        with torch.no_grad():
            p = torch.sigmoid(opacity[mask]) * factor
            p = p.clamp(1e-4, 1 - 1e-4)
            opacity[mask] = torch.log(p / (1 - p))

    scale_opacity(region, 0.05)                # 局部 ×0.05
    results["opacity_down(region)"] = evaluate()
    restore()

    scale_opacity(slice(None), 0.05)           # 全局 ×0.05
    results["opacity_down(global)"] = evaluate()
    restore()

    with torch.no_grad():
        scaling[region] -= 1.0                 # ×0.37
    results["scale_down(region)"] = evaluate()
    restore()

    with torch.no_grad():
        color[region] += 0.8                   # ≈ +58 级亮度
    results["color_shift(region)"] = evaluate()
    restore()

    with torch.no_grad():
        scaling += 0.6                         # ×1.82
    results["scale_up(global)"] = evaluate()
    restore()

    rng = np.random.default_rng(0)
    with torch.no_grad():
        noise = torch.as_tensor(rng.normal(0, 0.3, tuple(color.shape)),
                                device=device, dtype=color.dtype)
        color += noise
    results["color_noise(global)"] = evaluate()
    restore()

    clean = results["clean"]
    print(f"\n  {'扰动':24s} {'hole':>8s} {'Δhole':>8s} {'缺口':>8s} {'Δ缺口':>8s}"
          f" {'尖刺':>7s} {'锐度':>7s} {'PSNR':>7s}")
    for name, row in results.items():
        d_hole = (row["hole"] or 0) - (clean["hole"] or 0)
        d_def = (row["deficit"] or 0) - (clean["deficit"] or 0)
        psnr = "—" if row["psnr"] is None else f"{row['psnr']:.1f}"
        print(f"  {name:24s} {row['hole']:8.4f} {d_hole:+8.4f} "
              f"{row['deficit']:8.4f} {d_def:+8.4f} {row['spike']:7.4f} "
              f"{row['sharp']:7.2f} {psnr:>7s}")

    out_path = Path(args.out or (defects_dir / "eval_sensitivity.json"))
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\n产物：{out_path}")
    print("  判读：局部压不透明度/缩小尺度应使 hole 明显上升；局部改颜色/全局颜色噪声")
    print("        不应使 hole 上升。不满足则说明指标需要调整（见文件开头说明）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
