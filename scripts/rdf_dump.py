#!/usr/bin/env python
"""**在 `realdenseface` 环境里运行**：把一段图像序列的 RDF 结果导出成 npz。

我们自己的环境是 py3.10，而 RDF 的 `flame_solver` 编译产物是 cpython-311，
两个环境无法互相导入 —— 所以 RDF 以**独立进程**的方式工作（见
`docs/LIVE_DRIVING.md`）。本脚本是它的"离线数据出口"：先在真实数据上
把参数口径和坐标对齐标定准，再谈实时。

用法（**必须用 realdenseface 环境**）::

    conda activate realdenseface
    cd ~/Projects/Live3DGSAvatar
    python scripts/rdf_dump.py \
        --images ~/Projects/cap4d/output/<任务>/generated_images/images \
        --out ~/Projects/cap4d/output/<任务>/generated_images

产出（写在 `--out` 目录下的 `render_inputs.npz`，与数据集同构）：
``x`` (N,118)、``verts`` (N,V,3)、``identity`` (300)、相机参数、
以及每帧对应的图像文件名。``verts`` 由 RDF 用**它自己的 FLAME2023** 算出，
我们的进程因此不需要 FLAME2023 —— 也就不会出现"用 2020 的基算 2023 的参数"。

⚠️ 本脚本**不修改 RDF 仓库**，只 import 它。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

#: RDF 仓库位置（可用 --rdf-root 覆盖）
DEFAULT_RDF_ROOT = Path.home() / "Projects" / "RealDenseFace"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="导出 RDF 跟踪结果（离线）")
    # 不用 `required=True`：`--check-deps` 只需要环境，不需要图像与输出。
    p.add_argument("--images", type=Path, default=None,
                   help="图像目录（按文件名排序）")
    p.add_argument("--out", type=Path, default=None,
                   help="输出目录（写 <dir>/render_inputs.npz，与数据集同构）；"
                        "给 .npz 路径则按该文件名写出")
    p.add_argument("--limit", type=int, default=0,
                   help="最多处理多少帧；0 表示全部")
    p.add_argument("--stride", type=int, default=1, help="抽帧步长")
    p.add_argument("--rdf-root", type=Path, default=DEFAULT_RDF_ROOT)
    p.add_argument("--model", choices=("vits", "vitb"), default="vits")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-compile", action="store_true",
                   help="禁用 torch.compile（首次运行更省时）")
    p.add_argument("--fov-y-deg", type=float, default=30.0)
    p.add_argument("--camera-distance", type=float, default=1.0)
    p.add_argument("--check-deps", action="store_true",
                   help="只检查运行环境，不处理图像（先用它确认环境对不对）")
    return p.parse_args()


#: 本脚本需要的第三方包（都在 realdenseface 环境的 requirements 里）
REQUIRED = ("torch", "cv2", "numpy", "yaml", "timm", "roma", "trimesh")


def preflight(root: Path) -> list[str]:
    """检查"是不是在正确的环境里"，返回问题列表（空表示没问题）。

    **为什么要在最前面查**：本脚本必须在 `realdenseface`（py3.11）里运行，
    因为 RDF 的 `flame_solver` 是 cpython-311 的编译产物。若在 py3.10 环境里跑，
    会先撞到缺 `cv2`、装了之后又撞 `flame_solver` 的 ImportError ——
    一路排查下去很容易误以为"只需要补个包"。这里一次把话说清楚。
    """
    problems: list[str] = []

    if sys.version_info < (3, 11):
        problems.append(
            f"Python {sys.version.split()[0]} 太旧：RDF 的 flame_solver 是 "
            "cpython-311 编译产物，需要 Python ≥ 3.11")
    elif sys.version_info >= (3, 12):
        problems.append(
            f"Python {sys.version.split()[0]} 太新：flame_solver 只提供了 "
            "cpython-311 的 .so，3.12+ 无法导入")

    missing = []
    for name in REQUIRED:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        problems.append(f"缺少依赖：{', '.join(missing)}"
                        "（在 realdenseface 环境里 pip install -r requirements.txt）")

    if not (root / "tracker").is_dir():
        problems.append(f"找不到 RDF 仓库：{root}（用 --rdf-root 指定）")

    try:
        import flame_solver  # noqa: F401  （关键：这是 ABI 绑定的部分）
    except ImportError:
        problems.append(
            "flame_solver 无法导入 —— 它需要在本环境里编译：\n"
            "           pip install -e ./cuda_extensions/flame_solver "
            "--no-build-isolation")

    return problems


def main() -> int:
    args = parse_args()

    root = args.rdf_root.expanduser().resolve()

    # 环境检查放在最前面（见 `preflight` 的说明）。
    problems = preflight(root)
    if args.check_deps or problems:
        print(f"Python {sys.version.split()[0]}  |  RDF 仓库 {root}")
        try:
            import torch
            print(f"torch {torch.__version__}")
        except ImportError:
            print("torch 未安装")
        if problems:
            print("\n[error] 运行环境不满足要求：", file=sys.stderr)
            for i, p in enumerate(problems, 1):
                print(f"  {i}. {p}", file=sys.stderr)
            print("\n请在 `realdenseface` 环境里运行本脚本：\n"
                  "  conda activate realdenseface", file=sys.stderr)
            return 1
        print("✓ 环境检查通过")
        if args.check_deps:
            return 0

    if args.images is None or args.out is None:
        print("[error] 需要 --images 与 --out（除非只做 --check-deps）",
              file=sys.stderr)
        return 1
    if not args.images.is_dir():
        print(f"[error] 图像目录不存在：{args.images}", file=sys.stderr)
        return 1

    images = sorted(
        p for p in args.images.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
    images = images[:: max(1, args.stride)]
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        print("[error] 目录里没有图像", file=sys.stderr)
        return 1

    # 放在 RDF 仓库根下 import：它的模块用的是相对项目根的路径约定
    sys.path.insert(0, str(root))
    import common.fix_chumpy  # noqa: F401  （必须在 FLAME 之前）

    import cv2
    import numpy as np
    import torch

    from camera import PerspectiveCamera
    from tracker import (
        FLAMEConfig,
        FaceBoxConfig,
        OnlineGNOptimizer,
        OnlineGNOptimizerConfig,
        RealDenseFaceInferenceConfig,
        RealDenseFaceInferencer,
        load_fitting_config,
    )

    first = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
    if first is None:
        print(f"[error] 读不出图像：{images[0]}", file=sys.stderr)
        return 1
    h, w = first.shape[:2]
    print(f"图像 {w}x{h}，{len(images)} 帧")

    camera = PerspectiveCamera(
        fov_y=np.radians(float(args.fov_y_deg)),
        rot=np.diag([1.0, -1.0, -1.0]).astype(np.float32),
        pos=np.array([0.0, 0.0, float(args.camera_distance)], dtype=np.float32),
        width=int(w), height=int(h), znear=0.01, zfar=100.0)

    flame_config = FLAMEConfig(
        flame_model_path=str(root / "weights/flame/flame2023.pkl"),
        flame_assets_path=str(root / "weights/flame/flame_assets.npz"),
        num_expressions=100)

    inferencer = RealDenseFaceInferencer(
        RealDenseFaceInferenceConfig(
            model_config_path=str(root / f"configs/model/{args.model}.yaml"),
            model_weights_path=str(root / f"weights/realdenseface/{args.model}.pth"),
            flame=flame_config, device=args.device,
            compile_model=not args.no_compile, output_type="torch"),
        FaceBoxConfig(model_weights_path=str(root / "weights/facebox/face_box.pth"),
                      device=args.device))

    optimizer = OnlineGNOptimizer(load_fitting_config(
        str(root / "configs/fitting/video_online.yaml"),
        OnlineGNOptimizerConfig, flame=flame_config, device=args.device))
    optimizer.refresh_camera_matrices(camera)

    xs, verts, names = [], [], []
    identity = None
    face_bbox = None

    for i, path in enumerate(images):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[warn] 跳过读不出的帧：{path.name}")
            continue
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        inference = inferencer.infer_image(rgb, face_bbox=face_bbox)
        face_bbox = inference.face_bbox
        x = optimizer.track(inference)

        with torch.no_grad():
            v, _ = optimizer.decode(x)

        xs.append(x.detach().cpu().float().numpy().reshape(-1))
        verts.append(v.detach().cpu().float().numpy().reshape(-1, 3))
        names.append(path.name)
        if identity is None:
            identity = optimizer.identity.detach().cpu().float().numpy().copy()

        if (i + 1) % 25 == 0 or i + 1 == len(images):
            print(f"  {i + 1}/{len(images)}  {path.name}", flush=True)

    if not xs:
        print("[error] 没有成功处理任何帧", file=sys.stderr)
        return 1

    # 与数据集统一：--out 给目录时写 `<dir>/render_inputs.npz`，
    # 这样该目录就是"可渲染目录"（images/ + render_inputs.npz），
    # render_test 可以直接拿它渲染，不必再关心文件名。
    out_path = args.out
    if out_path.suffix.lower() != ".npz":
        out_path = out_path / "render_inputs.npz"
        print(f"输出目录：{args.out}（帧文件 {out_path.name}）")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        x=np.stack(xs).astype(np.float32),
        verts=np.stack(verts).astype(np.float32),
        identity=(identity if identity is not None
                  else np.zeros(300, dtype=np.float32)),
        names=np.array(names),
        image_width=np.int32(w), image_height=np.int32(h),
        camera_fov_y=np.float32(args.fov_y_deg),
        camera_rot=np.diag([1.0, -1.0, -1.0]).astype(np.float32),
        camera_pos=np.array([0.0, 0.0, args.camera_distance], dtype=np.float32),
    )
    print(f"\n已写出 {out_path}：x {np.stack(xs).shape}，verts {np.stack(verts).shape}")
    print(f"渲染检查：python scripts/render_test.py --subject <对象> --work-name <版本> "
          f"--frames-dir {args.out if out_path.parent == args.out else out_path.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
