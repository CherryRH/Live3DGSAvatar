#!/usr/bin/env python
"""比较"微调前后，之前选出来的那些问题点到底改了多少"。

## 为什么不看精修图 PSNR

微调是在拟合精修图，拿"与精修图的 PSNR"当判据等于自己给自己打分 —— 它必然涨，
而且涨多少只反映拟合程度。真正要回答的是：**当初选出来的缺陷，结构上补上了吗？**

## 三个指标（都逐视角算，再聚合）

| 指标 | 定义 | 抓什么 |
|---|---|---|
| `hole` | 网格说"这里有表面"（面 id 图）却 `alpha < 0.5` 的像素比例 | **结构断裂/空洞**（脖子断裂这类） |
| `sharpness` | `alpha > 0.5` 内的平均梯度 | 糊 / 涂抹 |
| `psnr_refined` | 与该视角精修图的掩码内 PSNR | 拟合程度（次要） |

`hole` 是关键：它只依赖几何覆盖，不受颜色影响。**如果微调后 `hole` 没降，
说明那个缺陷不是"画得不好"，而是"没东西可画"** —— 补它需要新增高斯（加密），
不是调颜色/尺度能解决的。

## 用法

```bash
python scripts/compare_defects.py --subject crh_1 --work-name run_refined \\
    --base-work run --dataset crh_1
```
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
    p = argparse.ArgumentParser(description="逐视角比较微调前后的缺陷指标")
    p.add_argument("--subject", default=None)
    p.add_argument("--work-name", default=None, help="微调后的版本名")
    p.add_argument("--base-work", default="run", help="对照（微调前）的版本名")
    p.add_argument("--dataset", default=None)
    p.add_argument("--defects", type=Path, default=None,
                   help="find_defects.py 的产物目录（含 selected.json / refined）")
    p.add_argument("--out", type=Path, default=None, help="结果目录")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def _erode(mask, radius=2):
    """二值腐蚀：每轮收缩 1 像素（四邻域取与；用 padding 而非 roll，避免越界回绕）。"""
    import numpy as np

    out = np.asarray(mask, bool).copy()
    for _ in range(int(radius)):
        p = np.pad(out, 1, mode="constant", constant_values=False)
        out = (p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1]
               & p[1:-1, :-2] & p[1:-1, 2:])
    return out


def summarize_rows(rows: list[dict]) -> dict:
    """把逐视角指标聚合成均值/中位与相对变化（纯函数，便于单测）。"""
    import numpy as np

    if not rows:
        return {}
    def col(key: str) -> np.ndarray:
        return np.asarray([float(r[key]) for r in rows if key in r], dtype=np.float64)

    out: dict = {"n": len(rows)}
    for key in ("hole_base", "hole_new", "hole_in_base", "hole_in_new",
                "sharp_base", "sharp_new", "psnr_base", "psnr_new",
                "spike_base", "spike_new", "outside_base", "outside_new"):
        values = col(key)
        if values.size:
            out[key] = {"mean": round(float(values.mean()), 4),
                        "median": round(float(np.median(values)), 4)}
    for prefix in ("hole", "hole_in"):
        hb0, hn0 = col(f"{prefix}_base"), col(f"{prefix}_new")
        if not (hb0.size and hn0.size):
            continue
        out[f"{prefix}_delta"] = round(float((hn0 - hb0).mean()), 4)
        mask0 = hb0 > 1e-6
        out[f"{prefix}_rel"] = (round(float((hn0[mask0] - hb0[mask0]).mean()
                                          / hb0[mask0].mean()), 4)
                                if bool(mask0.any()) else 0.0)

    hb, hn = col("hole_base"), col("hole_new")
    if hb.size and hn.size:
        out["hole_delta"] = round(float((hn - hb).mean()), 4)
        # 相对变化：只在"本来就有缺口"的视角上算，避免 0/0
        mask = hb > 1e-6
        out["hole_rel"] = (round(float((hn[mask] - hb[mask]).mean()
                                       / hb[mask].mean()), 4)
                           if bool(mask.any()) else 0.0)
    sb, sn = col("sharp_base"), col("sharp_new")
    if sb.size and sn.size:
        out["sharp_rel"] = (round(float((sn.mean() - sb.mean()) / sb.mean()), 4)
                            if sb.mean() > 0 else 0.0)
    pb, pn = col("psnr_base"), col("psnr_new")
    if pb.size and pn.size:
        out["psnr_delta"] = round(float((pn - pb).mean()), 3)
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import numpy as np

    from live3dgsavatar.config import load_config
    from live3dgsavatar.evaluation.views import LABEL_RE, parse_view_label

    cfg = load_config()
    subject = args.subject or str(cfg.get("subject") or "")
    new_work = args.work_name or str(cfg.get("work_name") or "")
    if not (subject and new_work):
        raise SystemExit("[error] 需要 --subject 与 --work-name")
    cfg.set("subject", subject)
    cfg.set("work_name", new_work)
    dataset = args.dataset or subject

    from live3dgsavatar.data.datasets import resolve_dataset_dir
    from live3dgsavatar.data.models import models_root

    try:
        data_dir = resolve_dataset_dir(cfg, dataset)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"[error] {e}") from None

    root = models_root(cfg) / subject
    base_ply = root / args.base_work / "model.ply"
    new_ply = root / new_work / "model.ply"
    for ply in (base_ply, new_ply):
        if not ply.is_file():
            raise SystemExit(f"[error] 找不到模型：{ply}")

    # 默认用**对照版本**的缺陷目录：精修图正是在那里产出的；
    # 在微调后的模型上重跑 find_defects 会得到另一份 selected.json，那是另一件事。
    defects_dir = Path(args.defects or (
        Path(str(cfg.get("paths.output_dir", "output"))) / "defects"
        / subject / args.base_work / dataset))
    selected = defects_dir / "selected.json"
    refined_dir = defects_dir / "refined"
    if not selected.is_file():
        raise SystemExit(f"[error] 找不到 {selected}")
    chosen = json.loads(selected.read_text(encoding="utf-8"))["views"]["chosen"]
    # 精修图目录里可能比 selected.json 多（另一次选择的产物）；标签自描述，
    # 所以有精修图就用精修图为准，否则退回 selected.json。
    available = sorted(p.stem for p in refined_dir.glob("*.png")
                       if LABEL_RE.match(p.stem)) if refined_dir.is_dir() else []
    labels = available or chosen

    with np.load(data_dir / "render_inputs.npz", allow_pickle=True) as d:
        total = int(d["mesh"].shape[0])
    ratio = float(cfg.get("train.pipeline.test_ratio", 0.1))
    train_end = total - max(1, int(round(total * ratio)))

    print(f"对照 : {base_ply}")
    print(f"新模型: {new_ply}")
    print(f"缺陷 : {defects_dir}（{len(labels)} 个视角）")
    if args.dry_run:
        print("\n[dry-run] 未执行渲染")
        return 0

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("[error] 需要 GPU（nvdiffrast）")

    import nvdiffrast.torch as dr
    from PIL import Image

    from live3dgsavatar.core.camera import orbit_w2c, roll_w2c
    from live3dgsavatar.core.io import load_ply
    from live3dgsavatar.core.io.ply import read_embedded_config
    from live3dgsavatar.core.metrics import psnr_masked
    from live3dgsavatar.core.render import SimpleRasterizer
    from live3dgsavatar.core.render.faceid import face_id_map
    from live3dgsavatar.core.types import Camera, Mesh
    from live3dgsavatar.data.scene import load_scene

    device = "cuda"
    glctx = dr.RasterizeCudaContext()
    scene = load_scene(cfg, frames=train_end, frame_offset=0, data_dir=data_dir)
    rasterizer = SimpleRasterizer()
    w2c_data = torch.eye(4, dtype=torch.float32)
    w2c_data[:3, :3] = scene.R.detach().cpu()
    w2c_data[:3, 3] = scene.T.detach().cpu()
    bg = torch.tensor([float(x) for x in cfg.get("render.background", [0, 0, 0])],
                      dtype=torch.float32, device=device)
    faces = scene.faces.to(torch.int32)
    # 非 `frameXXXXX` 的表情（主轴极值）没有对应真实帧，落到中性帧上
    with np.load(data_dir / "render_inputs.npz", allow_pickle=True) as d:
        expr = np.asarray(d["blend_weight"][:train_end, :100], dtype=np.float32)
    neutral = int(np.argmin(np.linalg.norm(expr, axis=1)))

    models = {"base": load_ply(base_ply, config=read_embedded_config(base_ply),
                              device=device),
              "new": load_ply(new_ply, config=read_embedded_config(new_ply),
                             device=device)}

    def render_and_measure(avatar, frame, cam, refined):
        """返回 `(hole, sharpness, psnr_refined)`。"""
        mesh = Mesh(verts=frame["mesh"].unsqueeze(0), faces=scene.faces,
                    uvs=scene.uvs, uv_faces=scene.uv_faces)
        with torch.no_grad():
            gs = avatar.deform(mesh, frame["blend_weight"].unsqueeze(0))
            out = rasterizer.render(gs, cam, bg)
            ids = face_id_map(frame["mesh"].unsqueeze(0), faces, cam, glctx)
        alpha = out.alpha[0, 0].detach().cpu().numpy()
        ids_np = ids.detach().cpu().numpy()
        image = (out.color[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                 * 255.0).astype(np.uint8)

        surface = ids_np >= 0
        hole = float((alpha[surface] < 0.5).mean()) if bool(surface.any()) else 0.0
        # 内部缺口：把轮廓腐蚀掉，排除"边缘 1–2 px 对齐误差"造成的假缺口
        interior = _erode(surface, 2)
        hole_in = (float((alpha[interior] < 0.5).mean())
                   if bool(interior.any()) else 0.0)
        # 高斯云比网格"大"出去的比例：网格没有面、但 alpha 已经实了
        filled = alpha > 0.5
        outside = (float((filled & ~surface).sum() / max(1, int(filled.sum())))
                   if bool(filled.any()) else 0.0)
        # 尖刺：轮廓内 alpha 梯度能量，缩尺/碎裂时会上升
        gy, gx = np.gradient(alpha.astype(np.float64))
        spike = (float(np.hypot(gx, gy)[interior].mean())
                 if bool(interior.any()) else 0.0)
        fg = alpha > 0.5
        if bool(fg.any()):
            grey = image.astype(np.float32).mean(axis=-1)
            gy, gx = np.gradient(grey)
            sharp = float(np.hypot(gx, gy)[fg].mean())
        else:
            sharp = 0.0
        psnr = None
        if refined is not None:
            mask = refined.max(axis=-1) > 8
            psnr = psnr_masked(image, refined, mask)
        return hole, hole_in, sharp, psnr, spike, outside

    rows: list[dict] = []
    for label in labels:
        info = parse_view_label(label)
        base_idx = (int(info["expr"][5:]) if info["expr"].startswith("frame")
                    else neutral)
        base_idx = min(max(base_idx, 0), scene.num_frames - 1)
        frame = scene.frames[base_idx]
        center = frame["mesh"].mean(dim=0).detach().cpu()
        view = orbit_w2c(w2c_data, center, info["yaw"], info["pitch"])
        if info["roll"]:
            view = roll_w2c(view, info["roll"])
        cam = Camera(K=scene.K, w2c=view.unsqueeze(0).to(device),
                     width=scene.width, height=scene.height).to(device)

        refined = None
        path = refined_dir / f"{label}.png"
        if path.is_file():
            refined = np.asarray(Image.open(path).convert("RGB"), np.float32)

        hb, ib, sb, pb, kb, ob = render_and_measure(models["base"], frame, cam, refined)
        hn, inn, sn, pn, kn, on = render_and_measure(models["new"], frame, cam, refined)
        rows.append({"label": label, "hole_base": hb, "hole_new": hn,
                     "hole_in_base": ib, "hole_in_new": inn,
                     "sharp_base": sb, "sharp_new": sn,
                     "spike_base": kb, "spike_new": kn,
                     "outside_base": ob, "outside_new": on,
                     "psnr_base": pb, "psnr_new": pn})
        print(f"  {label[:38]:38s} 缺口 {hb:.3f}→{hn:.3f}"
              f"（内部 {ib:.3f}→{inn:.3f}）  云外溢 {ob:.3f}→{on:.3f}  "
              f"尖刺 {kb:.4f}→{kn:.4f}  "
              f"锐度 {sb:5.2f}→{sn:5.2f}  "
              f"拟合 {('—' if pb is None else f'{pb:.1f}')}→"
              f"{('—' if pn is None else f'{pn:.1f}')}")

    summary = summarize_rows(rows)
    out_dir = Path(args.out or defects_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "compare_defects.json").write_text(
        json.dumps({"rows": rows, "summary": summary}, indent=2,
                   ensure_ascii=False), encoding="utf-8")

    print("\n=== 汇总 ===")
    if summary:
        print(f"  结构缺口（含轮廓）：{summary['hole_base']['mean']:.4f} → "
              f"{summary['hole_new']['mean']:.4f}"
              f"（相对 {summary.get('hole_rel', 0):+.1%}）")
        if "hole_in_base" in summary:
            print(f"  结构缺口（内部）  ：{summary['hole_in_base']['mean']:.4f} → "
                  f"{summary['hole_in_new']['mean']:.4f}"
                  f"（相对 {summary.get('hole_in_rel', 0):+.1%}）")
        print(f"  锐度    ：{summary['sharp_base']['mean']:.2f} → "
              f"{summary['sharp_new']['mean']:.2f}"
              f"（相对 {summary.get('sharp_rel', 0):+.1%}）")
        if "psnr_delta" in summary:
            print(f"  拟合精修图：{summary['psnr_delta']:+.2f} dB"
                  "（渲染 vs SyncFix 输出，**不是**真值）")
    print(f"\n产物：{out_dir / 'compare_defects.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
