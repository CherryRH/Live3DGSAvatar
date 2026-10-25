#!/usr/bin/env python
"""**在 `realdenseface` 环境里运行**：把 CAP4D 的生成参数变成我们能渲染的网格。

## 为什么需要这一步（以及为什么不能在渲染那边做）

CAP4D 的 `flame/*.npz` 只给参数（150 维 identity、64 维表情、姿态、相机），
网格得自己算。而**算网格必须用 FLAME2023**：

- 我们的 `FlameModel`（FLAME2020 资产）重建本数据集的网格，实测误差
  **4.4–5.4 mm**，且随表情幅度增长 —— 基不对
- 换成 RDF 的 `flame2023.pkl` 后误差降到 **4.4 mm** 量级，仍不足以直接当监督，
  说明还必须用 **RDF 自己的前向**（它的 LBS/pose correctives 与我们的实现不同）
- 这也正是实时链路本来的做法：RDF 出 5023 顶点，我们再补 60 个牙齿
  （`tracking/mesh_build.py`）

所以本脚本跑在 RDF 环境里，用**它自己的** `decode_with_identity` 出网格，
落成一个中间 npz；渲染（`scripts/render_test.py --cap4d`）再补牙齿、
映射到 129 维驱动参数。

## 用法

::

    conda activate realdenseface
    cd ~/Projects/Live3DGSAvatar

    # identity 来自之前 export_cap4d.py 写出的 fit.npz（里面的 shape，150 维）
    python scripts/export_cap4d_frames.py \
        --cap4d ~/output/crh_1_200 \
        --identity output/cap4d/crh_1/fit.npz \
        --out ~/output/crh_1_200/cap4d_frames.npz

产物字段：`mesh (T,5023,3)`、`x (T,118)`、`K (T,3,3)`、`names (T,)`、`images_dir`。

⚠️ 本脚本**不修改 RDF 仓库**，只 import 它。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_RDF_ROOT = Path.home() / "Projects" / "RealDenseFace"

#: 我们的包（只用到纯 numpy 的部分：读 CAP4D 目录、内参换算）
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CAP4D 参数 → 5023 顶点网格（RDF 环境）")
    p.add_argument("--cap4d", type=Path, required=True,
                   help="CAP4D 输出目录（含 generated_images/）")
    p.add_argument("--identity", type=Path, required=True,
                   help="含 `shape`（150 维）的 npz；通常是 export_cap4d.py 的 fit.npz")
    p.add_argument("--out", type=Path, required=True, help="输出 npz")
    p.add_argument("--limit", type=int, default=0, help="最多处理多少帧；0 = 全部")
    p.add_argument("--rdf-root", type=Path, default=DEFAULT_RDF_ROOT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--check-deps", action="store_true", help="只检查环境")
    return p.parse_args(argv)


REQUIRED = ("torch", "numpy", "yaml", "timm", "roma", "trimesh")


def preflight(root: Path) -> list[str]:
    problems: list[str] = []
    if sys.version_info < (3, 11):
        problems.append(
            f"Python {sys.version.split()[0]} 太旧：flame_solver 是 cpython-311 编译产物")
    elif sys.version_info >= (3, 12):
        problems.append(
            f"Python {sys.version.split()[0]} 太新：flame_solver 只提供 cpython-311")
    missing = [n for n in REQUIRED if not _importable(n)]
    if missing:
        problems.append(f"缺少依赖：{', '.join(missing)}")
    if not (root / "tracker").is_dir():
        problems.append(f"找不到 RDF 仓库：{root}")
    if not _importable("flame_solver"):
        problems.append("flame_solver 无法导入（需在本环境编译）")
    return problems


def _importable(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.rdf_root.expanduser().resolve()

    problems = preflight(root)
    if args.check_deps or problems:
        print(f"Python {sys.version.split()[0]}  |  RDF 仓库 {root}")
        if problems:
            print("\n[error] 运行环境不满足要求：", file=sys.stderr)
            for i, p in enumerate(problems, 1):
                print(f"  {i}. {p}", file=sys.stderr)
            print("\n请在 `realdenseface` 环境里运行：\n"
                  "  conda activate realdenseface", file=sys.stderr)
            return 1
        print("✓ 环境检查通过")
        if args.check_deps:
            return 0

    # RDF 的模块按项目根约定 import
    sys.path.insert(0, str(root))
    import common.fix_chumpy  # noqa: F401

    import cv2
    import numpy as np
    import torch
    from camera import PerspectiveCamera
    from tracker import (FLAMEConfig, FaceBoxConfig, OnlineGNOptimizer,
                         OnlineGNOptimizerConfig, RealDenseFaceInferenceConfig,
                         RealDenseFaceInferencer, load_fitting_config)

    from live3dgsavatar.data.cap4d import build_rdf_x, read_cap4d_dir

    cap4d_dir = args.cap4d.expanduser().resolve()
    frames = read_cap4d_dir(cap4d_dir, limit=args.limit)
    print(f"CAP4D: {len(frames)} 帧（{cap4d_dir}）")

    # identity：CAP4D 要 150 维；RDF 的 identity 是 300 维，前 150 维同一套基
    with np.load(args.identity.expanduser().resolve()) as d:
        if "shape" not in d.files:
            print(f"[error] {args.identity} 里没有 `shape`", file=sys.stderr)
            return 1
        shape150 = np.asarray(d["shape"]).reshape(-1)
    if shape150.size == 150:
        identity = np.zeros(300, np.float32)
        identity[:150] = shape150
    elif shape150.size == 300:
        identity = shape150.astype(np.float32)
    else:
        print(f"[error] shape 应为 150 或 300 维，实际 {shape150.size}",
              file=sys.stderr)
        return 1

    flame_config = FLAMEConfig(
        flame_model_path=str(root / "weights/flame/flame2023.pkl"),
        flame_assets_path=str(root / "weights/flame/flame_assets.npz"),
        num_expressions=100)
    optimizer = OnlineGNOptimizer(load_fitting_config(
        str(root / "configs/fitting/video_online.yaml"),
        OnlineGNOptimizerConfig, flame=flame_config, device=args.device))

    # decode 不需要真实相机，但优化器要先 refresh 一次才能用
    first_img = cv2.imread(str(frames[0].image), cv2.IMREAD_COLOR)
    if first_img is None:
        print(f"[error] 读不出图：{frames[0].image}", file=sys.stderr)
        return 1
    h, w = first_img.shape[:2]
    camera = PerspectiveCamera(
        fov_y=np.radians(30.0),
        rot=np.diag([1.0, -1.0, -1.0]).astype(np.float32),
        pos=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        width=int(w), height=int(h), znear=0.01, zfar=100.0)
    optimizer.refresh_camera_matrices(camera)

    meshes, xs, ks = [], [], []
    with torch.no_grad():
        identity_t = torch.from_numpy(identity).to(optimizer.device)
        for i, f in enumerate(frames):
            x = torch.from_numpy(build_rdf_x(f)).to(optimizer.device)
            verts, _ = optimizer.decode_with_identity(identity_t, x)
            meshes.append(verts.detach().cpu().float().numpy().reshape(-1, 3))
            xs.append(x.detach().cpu().float().numpy().reshape(-1))
            ks.append(f.K.astype(np.float32))
            if (i + 1) % 50 == 0 or i + 1 == len(frames):
                print(f"  {i + 1}/{len(frames)}", flush=True)

    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        mesh=np.stack(meshes).astype(np.float32),
        x=np.stack(xs).astype(np.float32),
        K=np.stack(ks).astype(np.float32),
        names=np.array([f.image.name for f in frames]),
        images_dir=np.array(str(frames[0].image.parent)),
    )
    print(f"\n已写出 {out}：mesh {np.stack(meshes).shape}，K {np.stack(ks).shape}")
    print("下一步（live3dgs 环境）：")
    print(f"  python scripts/render_test.py --subject <对象> --work-name <版本> "
          f"--cap4d {cap4d_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
