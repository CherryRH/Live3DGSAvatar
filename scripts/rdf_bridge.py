#!/usr/bin/env python
"""**在 `realdenseface` 环境里运行**：把 RDF 的实时结果推给本项目。

本进程是 TCP **服务端**，本项目（`live3dgs`）作为客户端连上来。
两边的环境不兼容（RDF 的 `flame_solver` 是 cpython-311），所以必须分进程；
细节见 `docs/LIVE_DRIVING.md`。

## 为什么先做"图像序列回放"

真摄像头只能跑一次、不可重复，出错时也分不清是采集还是链路的问题。
图像序列（`crh_2/input`）可重复、可逐帧核对，而且 RDF 的
`OnlineGNOptimizer` 本来就是按帧喂的 —— 回放与摄像头的区别只在**取图**这一层。
所以 `--images` 先跑通，加摄像头时只需要再写十几行。

用法::

    conda activate realdenseface
    cd ~/Projects/Live3DGSAvatar

    # 等本项目连上来，然后按 30 fps 推送
    python scripts/rdf_bridge.py \
        --images ~/Datasets/INSTA/crh_2/input --port 8980 --fps 30 --loop

    # 只跑一遍就退出（调试用）
    python scripts/rdf_bridge.py --images ... --limit 300 --once

⚠️ 本脚本**不修改 RDF 仓库**，只 import 它。
"""

from __future__ import annotations

import argparse
import contextlib
import socket
import sys
import time
from pathlib import Path

DEFAULT_RDF_ROOT = Path.home() / "Projects" / "RealDenseFace"

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

#: 本脚本需要的第三方包（都在 realdenseface 环境的 requirements 里）
REQUIRED = ("torch", "cv2", "numpy", "timm", "roma", "trimesh")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RDF → 本项目的实时桥接（TCP 服务端）")
    # 不用 `required=True`：`--check-deps` 只需环境。
    src = p.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, default=None,
                     help="摄像头索引（0 = 默认摄像头）。实时驱动的正式用法")
    src.add_argument("--images", type=Path, default=None,
                     help="图像目录回放（可重复、便于调试）")
    p.add_argument("--camera-width", type=int, default=0,
                   help="请求的采集宽度（0 = 摄像头默认）")
    p.add_argument("--camera-height", type=int, default=0,
                   help="请求的采集高度（0 = 摄像头默认）")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8980)
    p.add_argument("--fps", type=float, default=30.0,
                   help="推送帧率上限；0 表示不限（以 RDF 的速度为准）")
    # `--limit` 在 main 里就把帧表截断了，**不在这里提前返回** ——
    # 否则 `--limit 300 --loop` 会只跑一遍就停（`--loop` 失效）。
    p.add_argument("--limit", type=int, default=0,
                   help="从目录里最多取多少帧；配合 --loop 时会反复回放这些帧")
    p.add_argument("--stride", type=int, default=1, help="抽帧步长")
    p.add_argument("--loop", action="store_true", help="循环回放")
    p.add_argument("--once", action="store_true",
                   help="一个客户端断开后退出（默认继续等下一个）")
    p.add_argument("--start-delay", type=float, default=0.0,
                   help="接上客户端后延迟多少秒开始（给前端留出准备时间）")
    p.add_argument("--rdf-root", type=Path, default=DEFAULT_RDF_ROOT)
    p.add_argument("--model", choices=("vits", "vitb"), default="vits")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--fov-y-deg", type=float, default=30.0)
    p.add_argument("--camera-distance", type=float, default=1.0)
    p.add_argument("--check-deps", action="store_true",
                   help="只检查环境，不启动")
    return p.parse_args()


def preflight(root: Path) -> list[str]:
    """确认在正确的环境里（理由同 `rdf_dump.py`）。"""
    problems: list[str] = []
    if sys.version_info < (3, 11):
        problems.append(
            f"Python {sys.version.split()[0]} 太旧：flame_solver 是 cpython-311 "
            "编译产物，需要 Python ≥ 3.11")
    elif sys.version_info >= (3, 12):
        problems.append(
            f"Python {sys.version.split()[0]} 太新：flame_solver 只提供 cpython-311")

    missing = []
    for name in REQUIRED:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        problems.append(f"缺少依赖：{', '.join(missing)}")

    if not (root / "tracker").is_dir():
        problems.append(f"找不到 RDF 仓库：{root}")

    try:
        import flame_solver  # noqa: F401
    except ImportError:
        problems.append("flame_solver 无法导入（需在本环境编译）")
    return problems


# ------------------------------------------------------------- 输入源 --


class ImageSequence:
    """图像目录回放（可循环）。"""

    kind = "images"

    def __init__(self, images: list[Path], loop: bool) -> None:
        self.images = images
        self.loop = loop
        self._i = 0
        self._cv2 = __import__("cv2")

    def read(self):
        """返回 RGB `[H,W,3]` uint8；回放结束返回 ``None``。"""
        if self._i >= len(self.images):
            if not self.loop:
                return None
            self._i = 0
            print("  循环回放…", flush=True)
        path = self.images[self._i]
        self._i += 1
        bgr = self._cv2.imread(str(path), self._cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        pass


class Camera:
    """摄像头采集。

    ⚠️ **只在有客户端时读**：没人看的时候持续读会让采集缓冲积压，
    下一次连上来会先看到一批过期帧。
    """

    kind = "camera"

    def __init__(self, index: int, width: int = 0, height: int = 0) -> None:
        import cv2

        self._cv2 = cv2
        self.cap = cv2.VideoCapture(int(index))
        if not self.cap.isOpened():
            raise RuntimeError(
                f"打不开摄像头 {index}。检查：是否被其它程序占用、"
                "权限（Linux 下当前用户是否在 video 组）、索引是否正确")
        if width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        if height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        # 缓冲区设 1：宁可丢帧也不要延迟（实时场景下"旧帧"没有价值）
        with contextlib.suppress(Exception):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    @property
    def size(self) -> tuple[int, int]:
        w = int(self.cap.get(self._cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(self._cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def read(self):
        ok, bgr = self.cap.read()
        if not ok or bgr is None:
            return None
        return self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self.cap.release()


def make_source(args, images: list[Path] | None):
    """按参数构造输入源，并返回 `(source, width, height)`。"""
    if args.camera is not None:
        cam = Camera(args.camera, args.camera_width, args.camera_height)
        w, h = cam.size
        if w <= 0 or h <= 0:
            # 有些后端在未读到第一帧前返回 0
            frame = cam.read()
            if frame is None:
                cam.close()
                raise RuntimeError("摄像头已打开但读不到画面")
            h, w = frame.shape[:2]
        return cam, w, h

    if not images:
        raise SystemExit("[error] 需要 --camera 或 --images")

    import cv2

    first = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise SystemExit(f"[error] 读不出图像：{images[0]}")
    h, w = first.shape[:2]
    return ImageSequence(images, args.loop), w, h


# ------------------------------------------------------------ 构建 RDF --


def build_tracker(args, width: int, height: int):
    """构造 RDF 的 inferencer / optimizer / camera。"""
    root = args.rdf_root.expanduser().resolve()
    # ⚠️ RDF 的 `common/` `camera/` `tracker/` 是**仓库内模块**（没有安装），
    #    必须先把它的根目录放进 sys.path，否则 `import common` 直接 ModuleNotFoundError。
    #    `flame_solver` 是 pip 安装的，不依赖这一行 —— 所以 `--check-deps` 能过、
    #    真跑起来才炸，很容易误判成"环境没问题"。
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import common.fix_chumpy  # noqa: F401  （必须在 FLAME 之前）

    import numpy as np

    from camera import PerspectiveCamera
    from tracker import (
        FLAMEConfig, FaceBoxConfig, OnlineGNOptimizer, OnlineGNOptimizerConfig,
        RealDenseFaceInferenceConfig, RealDenseFaceInferencer, load_fitting_config,
    )

    camera = PerspectiveCamera(
        fov_y=np.radians(float(args.fov_y_deg)),
        rot=np.diag([1.0, -1.0, -1.0]).astype(np.float32),
        pos=np.array([0.0, 0.0, float(args.camera_distance)], dtype=np.float32),
        width=int(width), height=int(height), znear=0.01, zfar=100.0)

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
    return inferencer, optimizer, flame_config


def serve_one(args, conn: socket.socket, source,
              tracker: tuple, width: int, height: int) -> int:
    """对一个已连接的客户端推送帧，直到发完 / 断开。返回发送帧数。"""
    from live3dgsavatar.tracking import rdf_wire as W

    import numpy as np
    import torch

    inferencer, optimizer, _flame = tracker
    w, h = width, height
    hello = W.hello_bytes(num_vertices=optimizer.canonical[0].shape[0],
                          num_expressions=100, width=w, height=h,
                          identity=optimizer.identity.cpu().numpy())
    conn.sendall(hello)
    print(f"客户端已连接，开始推送（{w}x{h}，来源={source.kind}，"
          f"fps={args.fps or '不限'}）")

    if args.start_delay > 0:
        time.sleep(args.start_delay)

    interval = (1.0 / args.fps) if args.fps and args.fps > 0 else 0.0
    face_bbox = None
    sent = 0
    t_start = time.perf_counter()
    seq = 0

    while True:
        while True:
            t0 = time.perf_counter()
            frame = source.read()
            if frame is None:
                break                       # 回放结束（或摄像头断流）
            rgb = np.ascontiguousarray(frame)

            inference = inferencer.infer_image(rgb, face_bbox=face_bbox)
            face_bbox = inference.face_bbox
            x = optimizer.track(inference)
            with torch.no_grad():
                verts, _ = optimizer.decode(x)

            conn.sendall(W.frame_bytes(
                seq, time.perf_counter() - t_start,
                x.detach().cpu().float().numpy().reshape(-1),
                verts.detach().cpu().float().numpy().reshape(-1, 3)))
            seq += 1
            sent += 1

            if sent % 30 == 0:
                elapsed = time.perf_counter() - t_start
                print(f"  {sent} 帧  {sent / elapsed:.1f} fps", flush=True)

            if interval > 0:
                rest = interval - (time.perf_counter() - t0)
                if rest > 0:
                    time.sleep(rest)

        if source.kind != "images" or not args.loop:
            if source.kind == "camera":
                print("  摄像头读不到画面，结束本次推送", flush=True)
            break

    return sent


def main() -> int:
    args = parse_args()
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

    if args.camera is None and args.images is None:
        print("[error] 需要 --camera（实时）或 --images（回放）", file=sys.stderr)
        return 1

    images = None
    if args.images is not None:
        if not args.images.is_dir():
            print(f"[error] 图像目录不存在：{args.images}", file=sys.stderr)
            return 1
        images = sorted(p for p in args.images.iterdir()
                        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
        images = images[:: max(1, args.stride)]
        if args.limit > 0:
            images = images[: args.limit]
        if not images:
            print("[error] 目录里没有图像", file=sys.stderr)
            return 1

    try:
        source, img_w, img_h = make_source(args, images)
    except (RuntimeError, SystemExit) as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1

    server = None
    try:
        # **先加载模型再监听**：RDF 首次构建（含 torch.compile）常要几十秒。
        # 先监听会让客户端连上后干等，看起来像"卡住了"；端口不通更好排查。
        print(f"{source.kind} {img_w}x{img_h}，正在加载 RDF 模型"
              f"{'（首次含 torch.compile，约 30s）' if not args.no_compile else ''}…",
              flush=True)
        tracker = build_tracker(args, img_w, img_h)
        print("模型就绪", flush=True)

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        print(f"监听 {args.host}:{args.port}", flush=True)

        while True:
            print("等待本项目连接…", flush=True)
            conn, addr = server.accept()
            print(f"来自 {addr[0]}:{addr[1]}")
            try:
                serve_one(args, conn, source, tracker, img_w, img_h)
            except (BrokenPipeError, ConnectionResetError):
                print("  客户端断开")
            finally:
                conn.close()
            if args.once:
                break
            print("等待下一个客户端…（Ctrl-C 退出）")
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        if server is not None:
            server.close()
        source.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
