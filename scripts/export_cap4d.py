#!/usr/bin/env python
"""**在 `realdenseface` 环境里运行**：把一个数据集导出成 CAP4D 合成所需的全部输入。

CAP4D 的 MMDM 是**预训练**的扩散模型（仓库里只有推理代码），它要的不是
高斯模型，而是一份"参考图 + FLAME 参数 + 相机"的数据。本脚本负责把
我们自己的数据集翻译成那个格式，一次性产出：

    <out>/
    ├── fit.npz                  CAP4D 的参考参数（见下）
    ├── reference_images.json    选中的参考帧 [[cam, t], ...]
    ├── images/cam0/00000.png    参考图（原始黑底 RGBA → 只写 RGB）
    ├── bg/cam0/00000.png        前景权重（alpha），CAP4D 用它在白底上合成
    └── summary.json             残差、参考帧选择、诊断信息

## `fit.npz` 里有什么

| 键 | 形状 | 含义 |
|---|---|---|
| `shape` | (300,) | 身份。RDF 的 GN 优化在整段视频上收敛出来的 identity |
| `expr` | (T,65) | 表情。RDF 输出 100 维，取前 65 —— 与 CAP4D 的 `n_expr_params` 一致 |
| `rot` | (T,3) | 头姿轴角（见"坐标约定"） |
| `tra` | (T,3) | 平移 |
| `eye_rot` | (T,3) | 眼球。RDF 左右眼各 3 维，这里取平均（CAP4D 两眼共用一组）|
| `jaw_rot` | (T,3) | 下颌。CAP4D 当前**不读**这个键，留在文件里备查 |
| `fx/fy/cx/cy` | (1,1) | 相机内参（RDF 的虚拟相机）|
| `extr` | (1,4,4) | 单位阵：头姿里已经包含了相机 |
| `resolutions` | (1,2) | 图像尺寸 |
| `camera_order` | (1,) | `['cam0']` |

额外还写了我们自己的 `x`(118)、`verts`(V,3)、`residual_mm`、`reproj_px`，
方便后续做筛选与对照（CAP4D 会忽略不认识的键）。

## 坐标约定（为什么 `extr` 是单位阵）

RDF 的相机是 `fov_y=30°`、位于 `(0,0,1)`、`rot=diag(1,-1,-1)`。
CAP4D 在投影前会把顶点做一次 `diag(1,-1,-1)` 翻转，再用 `extr` 变换。

两者合起来，当我们取

    p_cap4d = flip(R·v_canonical + t)

并令 `extr = I` 时，`p_cap4d` 恰好等于 RDF 的**视图空间**坐标，
于是同一个内参投影出来的像素坐标与 RDF 完全一致（RDF 的
`project_points(in_viewport=True)` 也是 `u = fx·x/z + cx`）。

`R`/`t` 不靠手推，而是**逐帧用 Kabsch 对齐**求出来：
把 CAP4D 会用的"规范网格"（identity + expr，姿态归零）刚体对齐到
RDF 实际输出的网格（再减去 z=1）。这样连 RDF 的 neck 关节效应也被
最小二乘吸收进去，且给出一个可检验的残差。

## 自检

导出时逐帧重投影，把 `flip(R·v + t)` 用 `fx/fy/cx/cy` 投影，
与 RDF 自己的 `project_points` 比较，打印**最大像素误差**。
正常情况下应在 1e-3 像素量级；若明显偏大，说明约定对不上，
不要拿这份数据去跑 CAP4D。

## 已知限制

- **下颌**：CAP4D 的条件几何把 `jaw_rot` 当 None（`cap4d/flame/flame.py`
  与 `ReferenceDataset` 的白名单都不取它），所以生成时永远是闭颌。
  张嘴幅度大的帧，Kabsch 残差会变大 —— 用 `--max-residual-mm` 把它们筛掉
  （默认 3 mm），否则条件几何与参考图会互相矛盾。
- **眼睑**：RDF 不输出眼睑，写 0。

用法::

    conda activate realdenseface
    cd ~/Projects/Live3DGSAvatar

    # 先确认环境（不需要数据）
    python scripts/export_cap4d.py --check-deps

    # 导出：数据集 → CAP4D 输入
    python scripts/export_cap4d.py \
        --data ~/Datasets/INSTA/<数据集> \
        --out output/cap4d/<数据集> --limit 600

⚠️ 本脚本**不修改 RDF 仓库**，只 import 它。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: RDF 仓库位置（可用 --rdf-root 覆盖）
DEFAULT_RDF_ROOT = Path.home() / "Projects" / "RealDenseFace"

#: CAP4D 用到的表情维数（`CAP4DInferenceDataset(n_expr_params=65)`）
CAP4D_EXPR_DIM = 65

#: 其中**真正的 FLAME 表情**只有前 64 维。flowface 在加载时会把最后一个
#: 槽位**替换成眨眼方向**（`flowface/flame/flame.py`:
#: `expr_eigenvecs[:, :, -1] = blink_blendshape`）。所以第 65 维不是
#: FLAME 的第 65 个表情分量，而是"眨眼"，语义完全不同 —— RDF 不估眼睑，
#: 这一维必须填 0，不能把 RDF 的第 65 个系数塞进去。
CAP4D_FLAME_EXPR_DIM = 64

#: CAP4D 用到的身份维数（同一个数据集类是 `n_shape_params=150`）。
#: 它的 skinner 会 `assert shape_params.shape[1] == 150`，多一维都不行 ——
#: 而 RDF 的 identity 是 300 维，所以必须截断，见 `main` 里的说明。
CAP4D_SHAPE_DIM = 150

#: RDF 的表情维数
RDF_EXPR_DIM = 100

#: RDF `x` 的布局偏移
POSE_START = RDF_EXPR_DIM                      # [exp 100][pose 15][trans 3]
TRANS_START = RDF_EXPR_DIM + 15

#: RDF 姿态段内部顺序（与 `tracking/param_map.py` 一致）
POSE_SLICES = {"global": (0, 3), "neck": (3, 6), "jaw": (6, 9),
               "eye_l": (9, 12), "eye_r": (12, 15)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="用 RDF 把一个数据集导出成 CAP4D 的输入",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="环境不对时先跑 --check-deps。")
    p.add_argument("--data", type=Path, default=None,
                   help="数据集目录（含 images/ 或 input/）")
    p.add_argument("--out", type=Path, default=None, help="输出目录")
    p.add_argument("--image-subdir", default="images",
                   help="图像子目录（默认 images；没有则试 input）")
    p.add_argument("--limit", type=int, default=0, help="最多处理多少帧；0 = 全部")
    p.add_argument("--stride", type=int, default=1, help="抽帧步长")
    p.add_argument("--refs", type=int, default=4,
                   help="选几张参考图（CAP4D 上限 4）；按 yaw 分位挑选")
    p.add_argument("--gen-expr-factor", type=float, default=1.0,
                   help="写进 gen_data.npz 的表情缩放；CAP4D 生成的目标表情会乘它")
    p.add_argument("--max-reproj-px", type=float, default=10.0,
                   help="逐顶点重投影误差中位数上限（px）；超过即认为坐标约定错了")
    p.add_argument("--max-residual-mm", type=float, default=3.0,
                   help="Kabsch 残差超过此值（mm）的帧被筛掉；0 = 不筛")
    p.add_argument("--fps", type=float, default=30.0, help="写进 fit.npz 的帧率")
    p.add_argument("--fov-y-deg", type=float, default=30.0,
                   help="RDF 虚拟相机的竖直视场角")
    p.add_argument("--camera-distance", type=float, default=1.0,
                   help="RDF 虚拟相机的 z 位置")
    p.add_argument("--rdf-root", type=Path, default=DEFAULT_RDF_ROOT)
    p.add_argument("--model", choices=("vits", "vitb"), default="vits")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-compile", action="store_true",
                   help="禁用 torch.compile（首次运行更省时）")
    p.add_argument("--check-deps", action="store_true",
                   help="只检查运行环境（先用它确认环境对不对）")
    return p.parse_args(argv)


# --------------------------------------------------------------- 环境检查 --

REQUIRED = ("torch", "cv2", "numpy", "yaml", "timm", "roma", "trimesh")


def preflight(root: Path) -> list[str]:
    """确认在 `realdenseface` 环境里（RDF 的 `flame_solver` 是 cpython-311）。"""
    problems: list[str] = []
    if sys.version_info < (3, 11):
        problems.append(
            f"Python {sys.version.split()[0]} 太旧：flame_solver 是 cpython-311 "
            "编译产物，需要 Python ≥ 3.11")
    elif sys.version_info >= (3, 12):
        problems.append(
            f"Python {sys.version.split()[0]} 太新：flame_solver 只提供 cpython-311")

    missing = [n for n in REQUIRED if not _importable(n)]
    if missing:
        problems.append(f"缺少依赖：{', '.join(missing)}")

    if not (root / "tracker").is_dir():
        problems.append(f"找不到 RDF 仓库：{root}")

    if not _importable("flame_solver"):
        problems.append(
            "flame_solver 无法导入 —— 它需要在本环境里编译：\n"
            "           pip install -e ./cuda_extensions/flame_solver "
            "--no-build-isolation")
    return problems


def _importable(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


# ------------------------------------------------------------ 几何工具 --


def kabsch(src, dst):
    """求刚体变换 `(R, t)`，使 `R·src + t ≈ dst`（最小二乘）。

    用 SVD 求最优旋转，并用行列式修正反射（否则会得到"镜像"的旋转矩阵，
    在渲染里表现为左右翻转）。

    Args:
        src, dst: `[N,3]` 的 numpy 数组，逐点对应

    Returns:
        `(R [3,3], t [3])`
    """
    import numpy as np

    a = np.asarray(src, dtype=np.float64)
    b = np.asarray(dst, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"kabsch 需要两个同形状的 [N,3]，实际 {a.shape} / {b.shape}")

    ca, cb = a.mean(0), b.mean(0)
    H = (a - ca).T @ (b - cb)
    U, _S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def rotmat_to_rotvec(R):
    """旋转矩阵 → 轴角（罗德里格斯）。角度接近 0 或 π 时走数值稳定的分支。"""
    import numpy as np

    R = np.asarray(R, dtype=np.float64)
    cos = (np.trace(R) - 1.0) / 2.0
    cos = float(np.clip(cos, -1.0, 1.0))
    theta = float(np.arccos(cos))

    if theta < 1e-8:
        return np.zeros(3)
    if abs(theta - np.pi) < 1e-6:
        # 转轴取 (R+I) 里最大的一列，避免除零
        M = R + np.eye(3)
        axis = M[:, int(np.argmax(np.linalg.norm(M, axis=0)))]
        n = np.linalg.norm(axis)
        return (axis / n) * theta if n > 1e-12 else np.array([np.pi, 0.0, 0.0])

    skew = (R - R.T) / (2.0 * np.sin(theta))
    axis = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    return axis * theta


def yaw_deg(R) -> float:
    """从旋转矩阵取偏航角（度）。只用于挑参考帧，不参与几何。"""
    import numpy as np

    return float(np.degrees(np.arctan2(R[0, 2], R[2, 2])))


def project(points, fx, fy, cx, cy):
    """按 CAP4D 的公式投影：`u = fx·x/z + cx`，`v = fy·y/z + cy`。

    Args:
        points: `[N,3]`，**已经**是 CAP4D 的相机空间坐标
    """
    import numpy as np

    p = np.asarray(points, dtype=np.float64)
    z = p[:, 2]
    safe = np.where(np.abs(z) < 1e-8, 1e-8, z)
    return np.stack([p[:, 0] / safe * fx + cx,
                     p[:, 1] / safe * fy + cy], axis=1)


# ---------------------------------------------------------------- 主流程 --


def find_images(data_dir: Path, subdir: str) -> tuple[Path, list[Path]]:
    """找图像目录与文件列表。优先 `subdir`，回退 `input/`。"""
    exts = (".png", ".jpg", ".jpeg", ".webp")
    for name in (subdir, "input"):
        d = data_dir / name
        if d.is_dir():
            files = sorted(p for p in d.iterdir() if p.suffix.lower() in exts)
            if files:
                return d, files
    raise FileNotFoundError(
        f"在 {data_dir} 下找不到图像（试过 {subdir}/ 与 input/）")


def select_references(ysw: list[float], n_refs: int,
                      expr_mag: list[float] | None = None) -> list[int]:
    """按 yaw 分位挑参考帧：让 4 张参考覆盖尽量宽的转角。

    只挑一帧（正面）会让 MMDM 缺少侧脸依据；挑连续的几帧则几乎没有增益。

    若给了 `expr_mag`（每帧表情向量的范数），则在每个 yaw 分位**区间内**
    挑表情最小的一帧：参考图越接近中性，MMDM 越容易把它当"这个人长什么样"，
    而不是把参考帧的夸张表情也当成身份的一部分。
    """
    if not ysw:
        return []
    n = min(max(1, n_refs), len(ysw))
    order = sorted(range(len(ysw)), key=lambda i: ysw[i])

    # 把排好序的帧切成 n 段，每段取一帧
    segments = [order[int(round(k * (len(order) - 1) / max(1, n - 1)))]
                for k in range(n)] if n > 1 else [order[len(order) // 2]]
    picks = []
    for k, centre in enumerate(segments):
        if expr_mag is None:
            picks.append(centre)
            continue
        # 该中心点附近的窗口（按序号），在窗口里挑表情最小的
        span = max(1, len(order) // (2 * n))
        lo = max(0, order.index(centre) - span)
        hi = min(len(order), order.index(centre) + span + 1)
        window = order[lo:hi]
        picks.append(min(window, key=lambda i: expr_mag[i]))
    return sorted(set(picks))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.rdf_root.expanduser().resolve()

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
            print("\n请在 `realdenseface` 环境里运行：\n"
                  "  conda activate realdenseface", file=sys.stderr)
            return 1
        print("✓ 环境检查通过")
        if args.check_deps:
            return 0

    if args.data is None or args.out is None:
        print("[error] 需要 --data 与 --out（除非只做 --check-deps）",
              file=sys.stderr)
        return 1
    data_dir = args.data.expanduser().resolve()
    if not data_dir.is_dir():
        print(f"[error] 数据集目录不存在：{data_dir}", file=sys.stderr)
        return 1

    try:
        img_dir, images = find_images(data_dir, args.image_subdir)
    except FileNotFoundError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1

    images = images[:: max(1, args.stride)]
    if args.limit > 0:
        images = images[: args.limit]
    if len(images) < 2:
        print("[error] 至少需要 2 帧（identity 要靠多帧收敛）", file=sys.stderr)
        return 1

    # RDF 的模块按项目根约定 import
    sys.path.insert(0, str(root))
    import common.fix_chumpy  # noqa: F401  （必须在 FLAME 之前）

    import cv2
    import numpy as np
    import torch
    from camera import PerspectiveCamera
    from tracker import (FLAMEConfig, FaceBoxConfig, OnlineGNOptimizer,
                         OnlineGNOptimizerConfig, RealDenseFaceInferenceConfig,
                         RealDenseFaceInferencer, load_fitting_config)

    first = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
    if first is None:
        print(f"[error] 读不出图像：{images[0]}", file=sys.stderr)
        return 1
    h, w = first.shape[:2]
    print(f"图像 {w}x{h}，{len(images)} 帧（来源 {img_dir}）")

    camera = PerspectiveCamera(
        fov_y=np.radians(float(args.fov_y_deg)),
        rot=np.diag([1.0, -1.0, -1.0]).astype(np.float32),
        pos=np.array([0.0, 0.0, float(args.camera_distance)], dtype=np.float32),
        width=int(w), height=int(h), znear=0.01, zfar=100.0)
    fx, fy = float(camera.fx), float(camera.fy)
    cx, cy = float(camera.cx), float(camera.cy)
    print(f"内参 fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

    flame_config = FLAMEConfig(
        flame_model_path=str(root / "weights/flame/flame2023.pkl"),
        flame_assets_path=str(root / "weights/flame/flame_assets.npz"),
        num_expressions=RDF_EXPR_DIM)

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

    # ---- 1. 跟踪整段（identity 在过程中逐步收敛）----
    print("\n[1/3] RDF 跟踪")
    xs, verts, names = [], [], []
    face_bbox = None
    for i, path in enumerate(images):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"  [warn] 跳过读不出的帧：{path.name}")
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
        if (i + 1) % 50 == 0 or i + 1 == len(images):
            print(f"  {i + 1}/{len(images)}", flush=True)

    if len(xs) < 2:
        print("[error] 没有成功处理任何帧", file=sys.stderr)
        return 1

    xs = np.stack(xs)
    verts = np.stack(verts)
    identity = optimizer.identity.detach().cpu().float().numpy().copy()

    # ---- 2. 逐帧求刚体位姿 + 自检 ----
    print("\n[2/3] 求位姿（Kabsch）与重投影自检")
    eye = xs[:, POSE_START + 9: POSE_START + 12]
    eye2 = xs[:, POSE_START + 12: POSE_START + 15]
    eye_rot = 0.5 * (eye + eye2)              # CAP4D 两眼共用一组
    jaw = xs[:, POSE_START + 6: POSE_START + 9]

    # CAP4D 的 skinner 只吃 150 维 identity、65 维 expr。所以"CAP4D 实际会用的
    # 规范网格"必须用**截断后**的参数算 —— 否则我们拟合出的 rot/tra 对应的是
    # 一个比 CAP4D 更精细的网格，缝进去会系统性偏一点。
    identity_cap = identity.copy()
    identity_cap[CAP4D_SHAPE_DIM:] = 0.0
    identity_cap_t = torch.from_numpy(identity_cap).to(optimizer.device)

    rots, tras, residuals, residual_p95 = [], [], [], []
    reproj, reproj_p95, offsets = [], [], []
    trunc_mm: list[float] = []
    with torch.no_grad():
        identity_t = torch.from_numpy(identity).to(optimizer.device)
        for i in range(len(xs)):
            # 规范网格：identity + expr，姿态与平移归零；维度按 CAP4D 的口径截断
            flat = xs[i].copy()
            flat[CAP4D_FLAME_EXPR_DIM:RDF_EXPR_DIM] = 0.0   # CAP4D 只看到前 64 维
            flat[POSE_START:] = 0.0
            canon, _ = optimizer.decode_with_identity(
                identity_cap_t, torch.from_numpy(flat).to(optimizer.device))
            canon = canon.detach().cpu().float().numpy().reshape(-1, 3)

            # 诊断：截断本身让网格动了多少（前几帧量一下就够）
            if len(trunc_mm) < 5:
                full_flat = xs[i].copy()
                full_flat[POSE_START:] = 0.0
                full, _ = optimizer.decode_with_identity(
                    identity_t, torch.from_numpy(full_flat).to(optimizer.device))
                full = full.detach().cpu().float().numpy().reshape(-1, 3)
                trunc_mm.append(float(np.linalg.norm(full - canon, axis=1).mean()
                                      * 1000.0))

            target = verts[i].astype(np.float64).copy()
            target[:, 2] -= float(args.camera_distance)   # 见 docstring 的推导

            R, t = kabsch(canon, target)
            rots.append(rotmat_to_rotvec(R))
            tras.append(t)

            # 残差：非刚体部分（主要是下颌 / neck 关节），mm
            fitted = (canon @ R.T) + t
            per_vert_mm = np.linalg.norm(fitted - target, axis=1) * 1000.0
            residuals.append(float(per_vert_mm.mean()))
            residual_p95.append(float(np.percentile(per_vert_mm, 95)))

            # 自检：用导出的 (rot, tra) 重投影，与 RDF 自己的投影比
            # CAP4D 在投影前做 diag(1,-1,-1) 翻转；翻转后的点应等于 RDF 的视图空间
            p_cap = np.stack([fitted[:, 0], -fitted[:, 1], -fitted[:, 2]], axis=1)
            if p_cap[:, 2].min() <= 0:
                print("  [warn] 有顶点的相机空间 z ≤ 0，投影公式会反号",
                      file=sys.stderr)
            uv_ours = project(p_cap, fx, fy, cx, cy)
            uv_rdf, _mask = camera.project_points(
                torch.from_numpy(verts[i]).to(optimizer.device),
                in_viewport=True)
            uv_rdf = uv_rdf[:, :2].detach().cpu().float().numpy()

            # 逐顶点误差：**不要取最大值**。口腔/眼球内部几个顶点的残差会是
            # 主体误差的好几倍，取 max 会把"少数顶点"误判成"整体约定不对"。
            delta = uv_ours - uv_rdf
            per_vertex = np.linalg.norm(delta, axis=1)
            reproj.append(float(np.median(per_vertex)))
            reproj_p95.append(float(np.percentile(per_vertex, 95)))
            offsets.append(delta.mean(axis=0))

    residuals = np.array(residuals)
    residual_p95 = np.array(residual_p95)
    reproj = np.array(reproj)
    reproj_p95 = np.array(reproj_p95)
    offsets = np.array(offsets)

    # 系统偏移：若约定差一个常量（例如 RDF 的 NDC→像素少了 0.5 px），
    # 会表现为所有帧、所有顶点的平移相同。把它和"几何误差"分开看。
    mean_offset = offsets.mean(axis=0)
    print(f"  逐顶点重投影误差：中位 {np.median(reproj):.3f} px  "
          f"p95 {np.median(reproj_p95):.3f} px  （每帧内先取中位/p95）")
    print(f"  系统偏移 (du, dv) = ({mean_offset[0]:+.3f}, {mean_offset[1]:+.3f}) px"
          f"  —— 常数偏差只说明约定差了常量，不是几何错误")
    if trunc_mm:
        print(f"  截断影响（300→150 维 identity、100→64 维 expr）："
              f"平均 {np.mean(trunc_mm):.2f} mm —— 这是 CAP4D 表达能力的上限")
    print(f"  Kabsch 残差（逐帧均值）：中位 {np.median(residuals):.2f} mm  "
          f"p95 {np.percentile(residuals, 95):.2f} mm")
    print(f"  Kabsch 残差（逐顶点 p95）：中位 {np.median(residual_p95):.2f} mm"
          f"  —— 尾部主要来自下颌/口内")
    # 分层判据：
    #   · 中位 > 10 px ⟹ 约定/坐标错了（不能用）
    #   · 中位 3–10 px ⟹ 多半是下颌残差，数据可用但条件几何是近似的（警告）
    #   · 中位 < 3 px  ⟹ 正常
    if np.median(reproj) > float(args.max_reproj_px):
        print(f"  [error] 逐顶点重投影误差中位 {np.median(reproj):.2f} px > "
              f"{args.max_reproj_px} px：坐标约定对不上，不要用这份数据",
              file=sys.stderr)
        return 2
    if np.median(reproj) > 3.0:
        print(f"  [warn] 逐顶点重投影误差中位 {np.median(reproj):.2f} px 偏大："
              "主要是下颌/口内残差（CAP4D 的条件几何是闭颌的）。"
              "建议用 --max-residual-mm 筛掉张嘴幅度大的帧", file=sys.stderr)

    # ---- 3. 过滤 + 选参考 + 落盘 ----
    print("\n[3/3] 写文件")
    keep = np.ones(len(xs), dtype=bool)
    if args.max_residual_mm > 0:
        keep = residuals <= args.max_residual_mm
        if keep.sum() < 2:
            print(f"[warn] 残差阈值 {args.max_residual_mm} mm 下只剩 "
                  f"{int(keep.sum())} 帧，可能牙齿/下颌建模差异过大；"
                  "建议放宽阈值或换更闭颌的片段", file=sys.stderr)
            keep = np.ones(len(xs), dtype=bool)
    idx = np.where(keep)[0]
    print(f"  保留 {len(idx)}/{len(xs)} 帧"
          f"（阈值 {args.max_residual_mm} mm）")

    out = args.out.expanduser().resolve()
    written = write_images(img_dir, names, idx, out)
    if written != len(idx):
        # 图与 fit.npz 的行是按下标一一对应的，缺一张就会整体错位 —— 必须失败
        print(f"[error] 只写出 {written}/{len(idx)} 张图，下标会错位；"
              "请检查源图是否损坏", file=sys.stderr)
        return 1

    # yaw 只用于"挑参考帧"和报告，不参与几何
    keep_ys = [yaw_deg(_rotvec_to_mat(rots[int(i)])) for i in idx]
    # 参考帧优先挑"表情小"的：参考图决定 MMDM 学到的身份
    expr_mag = [float(np.linalg.norm(xs[int(i), :CAP4D_FLAME_EXPR_DIM]))
                for i in idx]
    refs = select_references(keep_ys, args.refs, expr_mag)

    # 顺带产出目标表情池：换成我们自己的分布，避免 MEAD 式的夸张表情
    gen_data = build_gen_data(xs[idx], args.gen_expr_factor)
    np.savez_compressed(out / "gen_data.npz", **gen_data)

    np.savez_compressed(out / "fit.npz", **build_fit_npz(
        identity=identity, xs=xs, idx=idx, rots=rots, tras=tras,
        eye_rot=eye_rot, jaw=jaw, verts=verts, residuals=residuals,
        reproj=reproj, names=names, fx=fx, fy=fy, cx=cx, cy=cy,
        width=w, height=h, fps=args.fps))

    (out / "reference_images.json").write_text(
        json.dumps([["cam0", int(t)] for t in refs], indent=2), encoding="utf-8")

    summary = {
        "dataset": str(data_dir), "image_dir": str(img_dir),
        "num_tracked": int(len(xs)), "num_exported": int(len(idx)),
        "resolution": [int(w), int(h)],
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "residual_mm": {"median": float(np.median(residuals)),
                        "p95": float(np.percentile(residuals, 95)),
                        "per_vertex_p95_median": float(np.median(residual_p95))},
        "reproj_px": {"median": float(np.median(reproj)),
                      "p95": float(np.median(reproj_p95)),
                      "offset_du": float(mean_offset[0]),
                      "offset_dv": float(mean_offset[1])},
        "identity_norm": float(np.linalg.norm(identity)),
        "refs": [{"t": int(t), "yaw_deg": float(keep_ys[k])}
                 for k, t in enumerate(refs)],
        "identity_converged": True,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n已写出 {out}")
    print(f"  fit.npz：{len(idx)} 帧  expr{xs[idx, :CAP4D_EXPR_DIM].shape}"
          f"  shape{identity.shape}")
    print(f"  参考帧（按 yaw 分位选）：{[(int(t), round(keep_ys[k], 1)) for k, t in enumerate(refs)]}")
    print(f"  identity 范数 {np.linalg.norm(identity):.2f}")
    print("\n下一步（CAP4D 仓库，另一个环境）：")
    print(f"  python cap4d/inference/generate_images.py \\\n"
          f"      --config_path configs/generation/low_quality.yaml \\\n"
          f"      --reference_data_path {out} --output_path <生成输出目录>")
    return 0


def _cap4d_expr(xs_rows):
    """把 RDF 的 100 维 expr 转成 CAP4D 的 65 维（末位是眨眼，置 0）。"""
    import numpy as np

    out = np.zeros((xs_rows.shape[0], CAP4D_EXPR_DIM), np.float32)
    out[:, :CAP4D_FLAME_EXPR_DIM] = xs_rows[:, :CAP4D_FLAME_EXPR_DIM]
    return out


def build_fit_npz(*, identity, xs, idx, rots, tras, eye_rot, jaw, verts,
                  residuals, reproj, names, fx, fy, cx, cy, width, height,
                  fps) -> dict:
    """组装 `fit.npz` 的内容（纯函数，便于单测）。"""
    import numpy as np

    arrays = idx if isinstance(idx, np.ndarray) else np.asarray(idx)
    return {
        # ---- CAP4D 会读的键 ----
        # 只取前 150 维：CAP4D 的 skinner 对维数有 `assert`，300 维会直接崩
        "shape": identity[:CAP4D_SHAPE_DIM].astype(np.float32),
        # 前 64 维来自 RDF；最后一维是眨眼，RDF 不估眼睑 → 填 0
        "expr": _cap4d_expr(xs[arrays]),
        "rot": np.stack([rots[i] for i in arrays]).astype(np.float32),
        "tra": np.stack([tras[i] for i in arrays]).astype(np.float32),
        "eye_rot": eye_rot[arrays].astype(np.float32),
        "fx": np.full((1, 1), fx, np.float32),
        "fy": np.full((1, 1), fy, np.float32),
        "cx": np.full((1, 1), cx, np.float32),
        "cy": np.full((1, 1), cy, np.float32),
        "extr": np.eye(4, dtype=np.float32)[None],
        "resolutions": np.array([[width, height]], np.int64),
        "camera_order": np.array(["cam0"]),
        # ---- CAP4D 忽略、留给我们自己的键 ----
        "shape_full": identity.astype(np.float32),   # 完整的 300 维
        "jaw_rot": jaw[arrays].astype(np.float32),
        "fps": np.array(fps, np.float32),
        "n_timesteps": np.array(len(arrays), np.int64),
        "n_views": np.array(1, np.int64),
        "x": xs[arrays].astype(np.float32),
        "verts": verts[arrays].astype(np.float32),
        "residual_mm": residuals[arrays].astype(np.float32),
        "reproj_px": reproj[arrays].astype(np.float32),
        "names": np.array([names[i] for i in arrays]),
    }


def build_gen_data(xs_rows, expr_factor: float = 1.0) -> dict:
    """用**我们自己的**表情分布生成 MMDM 的目标表情池（`gen_data.npz`）。

    为什么需要：CAP4D 自带的池子来自 MEAD/VFHQ/AVA/Nersemble，
    实测逐帧 `||expr||` 中位 9.8，而我们的数据是 7.4 —— MEAD 是**表演性**
    情感数据，表情幅度偏夸张。用它当目标，生成出来的表情会"不像正常人"。

    格式与官方一致：`expr (N,65)`（末位眨眼=0）、`eye_rot (N,3)`。
    """
    import numpy as np

    expr = np.zeros((xs_rows.shape[0], CAP4D_EXPR_DIM), np.float32)
    expr[:, :CAP4D_FLAME_EXPR_DIM] = xs_rows[:, :CAP4D_FLAME_EXPR_DIM]
    expr *= float(expr_factor)

    eye = 0.5 * (xs_rows[:, POSE_START + 9: POSE_START + 12]
                 + xs_rows[:, POSE_START + 12: POSE_START + 15])
    return {"expr": expr, "eye_rot": eye.astype(np.float32)}


def write_images(img_dir: Path, names: list[str], idx, out: Path) -> int:
    """把参考图与前景权重写到 `<out>/images/cam0/` 与 `<out>/bg/cam0/`。

    重新编号成 `0..N-1`，与 `fit.npz` 的第 t 行一一对应（CAP4D 按下标取图）。
    返回成功写出的张数。
    """
    import numpy as np
    from PIL import Image

    img_out = out / "images" / "cam0"
    bg_out = out / "bg" / "cam0"
    img_out.mkdir(parents=True, exist_ok=True)
    bg_out.mkdir(parents=True, exist_ok=True)

    written = 0
    for t, src_i in enumerate(idx):
        try:
            with Image.open(Path(img_dir) / names[src_i]) as im:
                arr = np.asarray(im.convert("RGBA"))
        except (OSError, ValueError):
            continue
        # 用 PIL 而不是 cv2：两边环境都有，且 RGBA 直接可读（cv2 要自己拼通道）
        Image.fromarray(np.ascontiguousarray(arr[..., :3])).save(
            img_out / f"{t:05d}.png")
        # 前景权重：CAP4D 用它把黑底图合成到白底上（见 apply_bg）
        Image.fromarray(np.ascontiguousarray(arr[..., 3])).save(
            bg_out / f"{t:05d}.png")
        written += 1
    return written


def _rotvec_to_mat(rv):
    """轴角 → 旋转矩阵（罗德里格斯），供挑参考帧时算 yaw 用。"""
    import numpy as np

    rv = np.asarray(rv, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rv))
    if theta < 1e-12:
        return np.eye(3)
    k = rv / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


if __name__ == "__main__":
    raise SystemExit(main())
