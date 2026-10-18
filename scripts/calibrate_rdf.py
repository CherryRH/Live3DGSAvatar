#!/usr/bin/env python
"""**标定**：用 RDF 的导出结果对齐到数据集的真值。

前置：先用 `scripts/rdf_dump.py`（在 realdenseface 环境里）导出
`x` 与 `verts`，再用本脚本（在 live3dgs 环境里）与数据集真值比对。

真值从哪来：`render_inputs.npz` 里有逐帧 `mesh` 与 `blend_weight`，
它们就是训练管线实际用的那套 —— 也就是模型"期望看到"的输入。

本脚本回答四个问题，且**按顺序**回答（前者不成立，后者的结论不可信）：

1. 表情/下颌/眼球段是否已经直接对得上？（若对不上，说明 RDF 输出口径
   与训练侧不同，必须在转换层修，而不是靠拟合坐标变换糊过去）
2. 头姿是 `R_global` 还是 `R_global @ R_neck`？
3. 坐标差是否**只是绕相机中心的缩放**？（1 个参数）
4. 若 3 不成立，完整的相似变换能做到多好？

用法::

    python scripts/calibrate_rdf.py \
        --rdf output/rdf/crh_2.npz \
        --data ~/Datasets/INSTA/crh_2 \
        --limit 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import live3dgsavatar  # noqa: E402,F401

from live3dgsavatar.config import load_config, resolve_data_dir  # noqa: E402
from live3dgsavatar.tracking import align as A  # noqa: E402
from live3dgsavatar.tracking import param_map as P  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RDF 参数与坐标的标定")
    p.add_argument("--rdf", type=Path, required=True, help="rdf_dump.py 的输出 npz")
    p.add_argument("--data", type=Path, default=None,
                   help="数据集目录（含 render_inputs.npz）；默认按配置推导")
    p.add_argument("--subject", default="crh_2", help="数据集 subject（默认 crh_2）")
    p.add_argument("--limit", type=int, default=0, help="最多用多少帧；0 = 全部")
    p.add_argument("--out", type=Path, default=None,
                   help="把标定结果写成 json（可选）")
    return p.parse_args()


def _stem(name: str) -> str:
    """只保留文件名主干：RDF 存 `00000.png`，数据集的 `names` 存 `00000`。"""
    base = str(name).replace("\\", "/").rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[0] if "." in base else base


def _match_indices(rdf_names: list[str], data_names: list[str]) -> tuple[list[int], list[int]]:
    """按文件名把 RDF 结果与数据集帧配对（两边都按名字，不靠顺序）。

    比较**主干**而不是完整文件名 —— 两边一个带扩展名一个不带。
    """
    index = {_stem(name): i for i, name in enumerate(data_names)}
    ra, rb = [], []
    for i, name in enumerate(rdf_names):
        j = index.get(_stem(name))
        if j is not None:
            ra.append(i)
            rb.append(j)
    return ra, rb


def main() -> int:
    args = parse_args()

    rdf = np.load(args.rdf)
    # `verts` 必需：坐标对齐要拿顶点做最小二乘。
    # RDF 自带的 track_video_offline.py 只存 `x`，不存顶点 —— 那种文件不够用。
    missing = {"x", "verts", "names"} - set(rdf.files)
    if missing:
        print(f"[error] {args.rdf} 缺少 {sorted(missing)}。\n"
              "        请用 `scripts/rdf_dump.py` 重新导出（它会保存顶点），\n"
              "        而不是 RDF 自带的 offline 脚本（只存参数）。",
              file=sys.stderr)
        return 1

    x = torch.from_numpy(np.asarray(rdf["x"], dtype=np.float32))
    verts_rdf = torch.from_numpy(np.asarray(rdf["verts"], dtype=np.float32))
    rdf_names = [str(n) for n in rdf["names"]]

    cfg = load_config()
    data_dir = args.data or (resolve_data_dir(cfg) if cfg.subject == args.subject
                             else None)
    if data_dir is None:
        # 显式按 subject 推导，避免依赖当前配置里的 subject
        cfg.set("subject", args.subject)
        data_dir = resolve_data_dir(cfg)
    data_dir = Path(data_dir)

    gt = np.load(data_dir / "render_inputs.npz")
    gt_mesh = torch.from_numpy(np.asarray(gt["mesh"], dtype=np.float32))
    gt_w = torch.from_numpy(np.asarray(gt["blend_weight"], dtype=np.float32))
    gt_names = [str(n) for n in gt["names"]] if "names" in gt.files else \
        [f"{i:05d}" for i in range(gt_mesh.shape[0])]

    ra, rb = _match_indices(rdf_names, gt_names)
    if not ra:
        print("[error] RDF 结果与数据集帧名对不上。\n"
              f"  RDF 前 3 个: {rdf_names[:3]}\n"
              f"  数据前 3 个: {gt_names[:3]}", file=sys.stderr)
        return 1
    if args.limit > 0:
        ra, rb = ra[: args.limit], rb[: args.limit]

    x = x[ra]
    verts_rdf = verts_rdf[ra]
    gt_mesh = gt_mesh[rb]
    gt_w = gt_w[rb]

    print(f"配对 {len(ra)} 帧（RDF {len(rdf_names)} 帧，数据 {len(gt_names)} 帧）")

    # RDF 用原生 FLAME（5023 顶点），我们的模型带 60 个牙齿顶点（5083）。
    # 两者的**公共前缀**是同一套拓扑（FLAME 2020/2023 的 `f` 逐位相同），
    # 所以对齐只用前缀；多出来的牙齿顶点单独处理（见 docs 的"牙齿"一节）。
    n_rdf, n_gt = verts_rdf.shape[1], gt_mesh.shape[1]
    n_common = min(n_rdf, n_gt)
    if n_gt - n_rdf == 60:
        print(f"  顶点数：RDF {n_rdf}，数据 {n_gt}（差 60 = 牙齿）；"
              f"只用公共前缀 {n_common} 做对齐")
    elif n_rdf != n_gt:
        print(f"[error] 顶点数差异无法解释：RDF {n_rdf} vs 数据 {n_gt}"
              "（预期差 0 或 60）", file=sys.stderr)
        return 1
    verts_rdf = verts_rdf[:, :n_common]
    gt_mesh = gt_mesh[:, :n_common]

    # ---- 1) 参数段：两种头姿假设 ----
    print("\n[1] 参数段对比（max|Δ|）")
    results = {}
    for compose in (True, False):
        w = P.to_blend_weight(x, compose_head=compose)
        err = P.segment_errors(w, gt_w)
        tag = "R_global @ R_neck" if compose else "仅 R_global"
        results[compose] = err
        seg = "  ".join(f"{k}={v:.4f}" for k, v in err.items())
        print(f"  头姿={tag:16s} {seg}")

    head_ok = results[True]["head"] < results[False]["head"]
    print(f"  → 头姿结论：{'复合（默认）' if head_ok else '仅 global'}"
          f"（{min(results[True]['head'], results[False]['head']):.4f}）")
    for name in ("expression", "jaw", "eyes", "eyelids"):
        best = min(results[True][name], results[False][name])
        flag = "✓" if best < 1e-3 else "✗ 需修正转换层"
        print(f"  → {name:10s} {best:.6f}  {flag}")

    compose = head_ok

    # ---- 2) 坐标：先试"绕相机中心缩放" ----
    print("\n[2] 坐标对齐")
    cam = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    # 用所有帧的所有顶点拟合，避免单帧的偶然性
    src = verts_rdf.reshape(-1, 3).to(torch.float64)
    dst = gt_mesh.reshape(-1, 3).to(torch.float64)

    s = A.fit_scale_about_camera(src, dst, cam)
    scaled = cam + s * (src - cam)
    r1 = A.residuals(scaled, dst)
    print(f"  单参数（绕相机中心缩放） s={s:.6f}  "
          f"max={r1['max']:.5f} mean={r1['mean']:.5f} p95={r1['p95']:.5f}")

    full = A.fit_similarity(src, dst)
    mapped = A.apply_similarity(src, full)
    r2 = A.residuals(mapped, dst)
    print(f"  完整相似变换          s={full['scale']:.6f}  "
          f"max={r2['max']:.5f} mean={r2['mean']:.5f} p95={r2['p95']:.5f}")

    # 判断"是否只是缩放"：两者差距在量级上是否可忽略
    ratio = r2["mean"] / max(r1["mean"], 1e-12)
    if ratio > 0.9:
        verdict = "单参数已足够（完整变换没有实质改善）"
    else:
        verdict = f"完整变换明显更好（mean 降到 {ratio:.0%}），说明存在旋转/平移分量"
    print(f"  → {verdict}")

    # ---- 3) 用标定结果重算平移段 ----
    print("\n[3] 用标定时的缩放修正平移段")
    trans_rdf = x[:, P.NUM_EXPRESSIONS + 15:].to(torch.float64)
    trans_scaled = cam + s * (trans_rdf - cam)
    w = P.to_blend_weight(x, translation=trans_scaled.to(torch.float32),
                          compose_head=compose)
    err = P.segment_errors(w, gt_w)
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in err.items()))

    if args.out:
        import json

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "frames": len(ra),
            "compose_head": bool(compose),
            "scale_about_camera": float(s),
            "camera_center": [0.0, 0.0, 1.0],
            "similarity": {
                "scale": float(full["scale"]),
                "rotation": full["rotation"].tolist(),
                "translation": full["translation"].tolist(),
            },
            "residuals": {"scale_only": r1, "similarity": r2},
            "segment_errors": err,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n已写出 {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
