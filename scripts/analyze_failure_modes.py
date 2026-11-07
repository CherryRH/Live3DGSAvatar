#!/usr/bin/env python
"""离线剖析"为什么某些地方会崩"：三角面、高斯与姿态暴露三者的关系。

只读 PLY + 数据集，不需要 GPU。回答三类问题：

1. **哪些三角面的高斯"消失"了**：分成"UV 采样就没有"与"有高斯但几乎全透明"两种，
   并按表面积加权，避免被大量小三角面误导；
2. **高斯有没有偏离表面（漂移）**：每个高斯的 `xyz` 参数就是它在绑定面局部坐标系里
   的偏移，直接取模长即可；
3. **姿态暴露**：用逐帧网格面法向与视线夹角判断每个面在多少帧里是"露出"的，
   并按头姿 yaw 分桶统计。

## 头姿 yaw 的取法

`blend_weight[100:106]` 是头姿 rot6d（旋转矩阵前两行）。用 Gram-Schmidt 还原矩阵后，
取 `yaw = atan2(-R[2,0], R[2,2])`。第 0 帧该值约为 0，说明朝向约定与数据一致。
这个 yaw 只用于**分桶**，不参与训练。

## 用法

```bash
python scripts/analyze_failure_modes.py --subject crh_1 --work-name run
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

#: 认为"这个高斯看不出来"的不透明度阈值
INVISIBLE_OPACITY = 0.05
#: 认为三角面"露出"的法向·视线阈值（1 = 正对，0 = 侧到极限）
FRONT_FACING = 0.2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="剖析失效机制")
    p.add_argument("--subject", default=None)
    p.add_argument("--work-name", default="run")
    p.add_argument("--dataset", default=None)
    p.add_argument("--defects", type=Path, default=None)
    p.add_argument("--frames", type=int, default=300,
                   help="用于姿态暴露统计的采样帧数")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args(argv)


def rot6d_to_matrix(d6):
    """`[..., 6]` → `[..., 3, 3]`（与 `matrix_to_rot6d` 互逆）。"""
    import numpy as np

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-2)


def yaw_from_rot6d(d6):
    """头姿 yaw（度）：`atan2(-R20, R22)`。"""
    import numpy as np

    R = rot6d_to_matrix(np.asarray(d6, dtype=np.float64))
    return np.degrees(np.arctan2(-R[..., 2, 0], R[..., 2, 2]))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import numpy as np
    import torch

    from live3dgsavatar.config import load_config
    from live3dgsavatar.core.io import load_ply
    from live3dgsavatar.core.io.ply import read_embedded_config
    from live3dgsavatar.data.datasets import resolve_dataset_dir
    from live3dgsavatar.data.flame_topology import load as load_topo
    from live3dgsavatar.data.models import models_root

    cfg = load_config()
    subject = args.subject or str(cfg.get("subject") or "")
    if not subject:
        raise SystemExit("[error] 需要 --subject")
    dataset = args.dataset or subject
    data_dir = resolve_dataset_dir(cfg, dataset)
    ply = models_root(cfg) / subject / args.work_name / "model.ply"
    if not ply.is_file():
        raise SystemExit(f"[error] 找不到模型：{ply}")
    defects_dir = Path(args.defects or (
        Path(str(cfg.get("paths.output_dir", "output"))) / "defects"
        / subject / "run" / dataset))
    mass_path = defects_dir / "mass.npz"

    topo = load_topo(REPO_ROOT / "data" / "FLAME2020",
                     Path(str(cfg.get("paths.output_dir", "output"))) / "cache", "cpu")
    faces = topo["faces"].numpy().astype(np.int64)
    avatar = load_ply(ply, config=read_embedded_config(ply), device="cpu")

    with np.load(data_dir / "render_inputs.npz", allow_pickle=True) as d:
        meshes = np.asarray(d["mesh"], dtype=np.float32)
        weights = np.asarray(d["blend_weight"], dtype=np.float32)
        w2v = np.asarray(d["w2v"], dtype=np.float32)
        if w2v.ndim == 2:
            w2v = np.broadcast_to(w2v, (len(meshes), 4, 4))
    n_gauss = avatar.num_gaussians
    face_id = avatar.binding_face_id.numpy()
    bary = avatar.binding_face_bary.numpy()
    opacity = torch.sigmoid(avatar.opacity.detach()).reshape(-1).numpy()
    # 偏移的世界方向：R_bind @ xyz（TBN 三列已归一化，近似正交）
    from live3dgsavatar.core.deform.tbn import compute_face_tbn

    with torch.no_grad():
        _tri = torch.from_numpy(meshes[0][faces]).float().unsqueeze(0)
        _tbn = compute_face_tbn(
            _tri, torch.from_numpy(topo["uv_faces"].numpy().astype(np.float32)
                                   if False else topo["uvs"].numpy()[
                                       topo["uv_faces"].numpy()]).float())
        _R = _tbn[0].numpy()
    offset_world = np.einsum("nij,nj->ni", _R[avatar.binding_face_id.numpy()],
                             avatar.xyz.detach().numpy()) * 1e3     # mm
    offset = np.linalg.norm(avatar.xyz.detach().numpy(), axis=1) * 1e3     # mm
    coverage = (np.load(mass_path)["coverage"]
                if mass_path.is_file() else np.zeros(n_gauss, np.int32))

    # ---- 面积与姿态暴露（采样帧） ----
    idx = np.linspace(0, len(meshes) - 1, min(args.frames, len(meshes))).astype(int)
    tri0 = meshes[idx[0]][faces]
    area = 0.5 * np.linalg.norm(np.cross(tri0[:, 1] - tri0[:, 0],
                                         tri0[:, 2] - tri0[:, 0]), axis=1)
    visible_count = np.zeros(len(faces), np.float64)
    yaw_of_visible: list[list[float]] = [[] for _ in range(len(faces))]
    for k, fi in enumerate(idx):
        tri = meshes[fi][faces]
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area_f = np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
        normal = normal / area_f
        centroid = tri.mean(axis=1)
        R = w2v[fi][:3, :3]
        t = w2v[fi][:3, 3]
        centroid_v = centroid @ R.T + t
        normal_v = normal @ R.T
        to_cam = -centroid_v
        to_cam /= (np.linalg.norm(to_cam, axis=1, keepdims=True) + 1e-12)
        facing = (normal_v * to_cam).sum(axis=1)
        yaw = yaw_from_rot6d(weights[fi, 100:106][None])[0]
        front = facing > FRONT_FACING
        visible_count[front] += 1
        for f in np.nonzero(front)[0]:
            yaw_of_visible[f].append(float(yaw))
    visible_frac = visible_count / len(idx)

    # ---- 1) 三角面层面的"消失" ----
    texels_per_face = np.bincount(face_id, minlength=len(faces))
    face_op_max = np.zeros(len(faces))
    np.maximum.at(face_op_max, face_id, opacity)
    no_texel = texels_per_face == 0
    transparent = (~no_texel) & (face_op_max < INVISIBLE_OPACITY)
    alive = ~(no_texel | transparent)

    # ---- 2) 高斯层面：偏移与不透明度 vs 覆盖 ----
    buckets = [("0", coverage == 0), ("1-9", (coverage >= 1) & (coverage < 10)),
               ("10-99", (coverage >= 10) & (coverage < 100)),
               ("100-999", (coverage >= 100) & (coverage < 1000)),
               ("1000+", coverage >= 1000)]
    gauss_rows = []
    for name, mask in buckets:
        if not bool(mask.any()):
            continue
        gauss_rows.append({
            "coverage": name, "n": int(mask.sum()),
            "offset_mm_median": round(float(np.median(offset[mask])), 4),
            "offset_mm_p90": round(float(np.percentile(offset[mask], 90)), 4),
            "opacity_median": round(float(np.median(opacity[mask])), 5),
            "opacity_gt_0.5": round(float((opacity[mask] > 0.5).mean()), 4),
            "face_visible_frac": round(float(visible_frac[face_id[mask]].mean()), 4),
            "offset_world_mm": {
                "x": round(float(np.median(offset_world[mask, 0])), 2),
                "y": round(float(np.median(offset_world[mask, 1])), 2),
                "z": round(float(np.median(offset_world[mask, 2])), 2),
                "toward_camera_share": round(float(
                    (offset_world[mask, 2] > 0).mean()), 4),
            },
        })

    # ---- 3) 姿态覆盖：每个面的"露出"比例与 yaw 范围 ----
    yaw_p90 = np.array([np.percentile(v, 90) if len(v) >= 5 else np.nan
                        for v in yaw_of_visible])
    never = visible_frac < 0.02
    wide = yaw_p90 > 30

    def share(mask) -> float:
        return round(float(area[mask].sum() / area.sum()), 4)

    report = {
        "model": str(ply),
        "frames_sampled": int(len(idx)),
        "gaussians": n_gauss,
        "faces": int(len(faces)),
        "texel_presence": {
            "no_texel": {"faces": int(no_texel.sum()), "area_share": share(no_texel),
                         "visible_frac_median": round(float(np.median(
                             visible_frac[no_texel])), 4) if no_texel.any() else None},
            "transparent": {"faces": int(transparent.sum()),
                            "area_share": share(transparent),
                            "visible_frac_median": round(float(np.median(
                                visible_frac[transparent])), 4) if transparent.any() else None},
            "alive": {"faces": int(alive.sum()), "area_share": share(alive),
                      "visible_frac_median": round(float(np.median(
                          visible_frac[alive])), 4) if alive.any() else None},
        },
        "by_coverage": gauss_rows,
        "pose_exposure": {
            "faces_never_visible": {"faces": int(never.sum()),
                                    "area_share": share(never)},
            "faces_visible_only_beyond_30deg": {
                "faces": int((wide & ~np.isnan(yaw_p90)).sum()),
                "area_share": share(wide & ~np.isnan(yaw_p90))},
            "visible_frac_median": round(float(np.median(visible_frac)), 4),
        },
        "correlations": {
            "offset_vs_coverage_spearman": round(float(_spearman(offset, coverage)), 4),
            "opacity_vs_coverage_spearman": round(float(_spearman(opacity, coverage)), 4),
        },
    }

    print(f"模型 : {ply}")
    print(f"高斯 : {n_gauss}   三角面 : {len(faces)}   采样帧 : {len(idx)}")
    print("\n[1] 三角面上的高斯")
    for key, label in (("no_texel", "UV 采样就没有"),
                       ("transparent", "有高斯但全透明"),
                       ("alive", "有可见高斯")):
        info = report["texel_presence"][key]
        print(f"  {label:14s} 面 {info['faces']:6d}  面积 {info['area_share']:6.1%}  "
              f"露出比例中位 {info['visible_frac_median']}")
    print("\n[2] 高斯偏移与不透明度（按观测覆盖）")
    print(f"  {'覆盖':8s} {'数量':>7s} {'偏移中位':>9s} {'偏移P90':>9s} "
          f"{'不透明度':>9s} {'>0.5':>6s} {'露出比例':>8s}")
    for row in gauss_rows:
        print(f"  {row['coverage']:8s} {row['n']:7d} {row['offset_mm_median']:9.3f} "
              f"{row['offset_mm_p90']:9.3f} {row['opacity_median']:9.5f} "
              f"{row['opacity_gt_0.5']:6.1%} {row['face_visible_frac']:8.1%}")
    print("\n  偏移方向（世界系，mm；相机在 +z，z>0 = 朝向相机）")
    for row in gauss_rows:
        d = row["offset_world_mm"]
        print(f"  {row['coverage']:8s} x {d['x']:+6.1f}  y {d['y']:+6.1f}  "
              f"z {d['z']:+6.1f}   z>0 占 {d['toward_camera_share']:.0%}")
    print("\n[3] 姿态暴露")
    pe = report["pose_exposure"]
    print(f"  从没露出过的面   {pe['faces_never_visible']['faces']:6d}  "
          f"面积 {pe['faces_never_visible']['area_share']:6.1%}")
    print(f"  只在 >30° 露出   "
          f"{pe['faces_visible_only_beyond_30deg']['faces']:6d}  "
          f"面积 {pe['faces_visible_only_beyond_30deg']['area_share']:6.1%}")
    print(f"  露出比例中位     {pe['visible_frac_median']:.1%}")
    print("\n[4] 相关（Spearman）")
    print(f"  偏移 vs 覆盖 {report['correlations']['offset_vs_coverage_spearman']:+.3f}"
          f"   不透明度 vs 覆盖 "
          f"{report['correlations']['opacity_vs_coverage_spearman']:+.3f}")

    out = Path(args.out or (defects_dir / "failure_modes.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n产物：{out}")
    return 0


def _spearman(a, b) -> float:
    """秩相关（不依赖 scipy）。"""
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3:
        return 0.0
    ar = np.argsort(np.argsort(a)).astype(np.float64)
    br = np.argsort(np.argsort(b)).astype(np.float64)
    ar -= ar.mean()
    br -= br.mean()
    denom = np.sqrt((ar ** 2).sum() * (br ** 2).sum())
    return float((ar * br).sum() / denom) if denom > 0 else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
