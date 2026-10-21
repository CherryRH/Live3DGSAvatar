#!/usr/bin/env python
"""**在 `realdenseface` 环境里运行**：通用的"画面进、参数出"追踪服务。

本项目（`live3dgs`）把一帧画面发过来，本进程做追踪，回一帧参数与顶点。
两边环境不兼容（RDF 的 `flame_solver` 是 cpython-311），所以必须分进程；
细节见 `docs/LIVE_DRIVING.md`。

## 通用性

服务**不关心画面从哪来** —— 数据集回放、摄像头、单张图片都行，只要按协议发。
取帧由客户端负责，所以加新画面来源时这里一行都不用改。

## 用法

::

    conda activate realdenseface
    cd ~/Projects/Live3DGSAvatar

    # 起来等客户端（先加载模型再监听，端口不通说明还没好）
    python scripts/rdf_bridge.py --port 8980

    # 首次跑得太慢可以关掉 torch.compile（起得快、每帧慢一点）
    python scripts/rdf_bridge.py --port 8980 --no-compile

    # 只做一个会话就退出（调试用）
    python scripts/rdf_bridge.py --port 8980 --once

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


#: 追踪服务启动所需的 RDF 资产（相对 RDF 仓库根）。
#: 缺任何一个都会在**构建模型时**才崩，报错信息还埋在三层调用栈里 ——
#: 所以在预检阶段就查一遍，一次说清缺什么、该放哪。
def required_assets(root: Path, model: str) -> list[tuple[Path, str]]:
    """返回 `[(路径, 说明)]`：跑起来之前必须存在的文件。"""
    return [
        (root / f"weights/realdenseface/{model}.pth",
         f"{model} 的推理权重"),
        (root / "weights/flame/flame2023.pkl", "FLAME 2023 模型"),
        (root / "weights/flame/flame_assets.npz", "FLAME 资产（拓扑/UV）"),
        (root / "weights/facebox/face_box.pth", "人脸检测权重"),
        (root / f"configs/model/{model}.yaml", "模型配置"),
        (root / "configs/fitting/video_online.yaml", "在线拟合配置"),
    ]


def preflight(root: Path, model: str = "vits") -> list[tuple[str, str]]:
    """确认在正确的环境里、且资产齐备。

    **环境与资产分开报**：前者说明"该用哪个解释器"，后者说明"该拷哪些文件" ——
    部署到新机器时最常缺的就是后者，而它原来只在构建模型时才暴露。
    """
    problems: list[tuple[str, str]] = []
    if sys.version_info < (3, 11):
        problems.append(("env",
            f"Python {sys.version.split()[0]} 太旧：flame_solver 是 cpython-311 "
            "编译产物，需要 Python ≥ 3.11"))
    elif sys.version_info >= (3, 12):
        problems.append(("env",
            f"Python {sys.version.split()[0]} 太新：flame_solver 只提供 cpython-311"))

    missing = []
    for name in REQUIRED:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        problems.append(("env", f"缺少依赖：{', '.join(missing)}"))

    if not (root / "tracker").is_dir():
        problems.append(("env", f"找不到 RDF 仓库：{root}"))

    try:
        import flame_solver  # noqa: F401
    except ImportError:
        problems.append(("env", "flame_solver 无法导入（需在本环境编译）"))

    # 资产检查：只在仓库目录存在时才有意义
    if (root / "tracker").is_dir():
        missing = [(p, why) for p, why in required_assets(root, model)
                   if not p.is_file()]
        if missing:
            lines = "；".join(f"{p.relative_to(root)}（{why}）" for p, why in missing)
            problems.append(("asset",
                f"RDF 资产不完整，缺 {len(missing)} 项：{lines}。"
                f"请把 RealDenseFace 的 weights/ 与 configs/ 一并部署到 {root}"))
    return problems


# --------------------------------------------------------- 追踪（惰性）--


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


class Tracker:
    """把 RDF 包装成"**一帧画面进、参数与顶点出**"的纯函数式服务。

    为什么按尺寸惰性重建：RDF 的相机由 `width/height` 决定。客户端可能送不同
    尺寸的画面（数据集 512²、摄像头 640×480）。尺寸一变就重建相机并清掉
    人脸框缓存 —— 人脸框是**上一帧**的搜索结果，跨尺寸没有意义。
    """

    def __init__(self, args) -> None:
        self.args = args
        self._size: tuple[int, int] | None = None
        self._parts = None
        self._face_bbox = None
        self.frames = 0

    def ensure(self, width: int, height: int) -> None:
        """（重新）构建追踪器到指定尺寸。启动时就调一次，避免首个客户端干等。"""
        size = (int(width), int(height))
        if self._size == size:
            return
        if self._size is not None:
            print(f"画面尺寸 {self._size[0]}x{self._size[1]} → "
                  f"{size[0]}x{size[1]}，重建相机", flush=True)
        self._parts = build_tracker(self.args, size[0], size[1])
        self._size = size
        self._face_bbox = None

    def track(self, rgb):
        """`rgb [H,W,3] uint8` → `(x [118], verts [V,3])`（numpy）。"""
        import numpy as np
        import torch

        h, w = rgb.shape[:2]
        self.ensure(w, h)
        inferencer, optimizer, _flame = self._parts

        inference = inferencer.infer_image(np.ascontiguousarray(rgb),
                                           face_bbox=self._face_bbox)
        self._face_bbox = inference.face_bbox
        x = optimizer.track(inference)
        with torch.no_grad():
            verts, _ = optimizer.decode(x)

        self.frames += 1
        return (x.detach().cpu().float().numpy().reshape(-1),
                verts.detach().cpu().float().numpy().reshape(-1, 3))


# --------------------------------------------------------------- 服务 --


def _recv_exact(conn: socket.socket, n: int, *, allow_eof: bool = False):
    """读满 `n` 字节；对端在**消息边界**关闭时返回 ``None``。"""
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            if allow_eof and not buf:
                return None
            raise ConnectionError(f"连接在中途断开（已读 {len(buf)}/{n} 字节）")
        buf.extend(chunk)
    return bytes(buf)


def serve_client(conn: socket.socket, tracker: Tracker) -> int:
    """处理一个客户端的请求，直到它断开。返回处理的帧数。

    协议是**纯请求/响应**：客户端发一帧画面，本进程回一帧结果。
    谁都可以连 —— 数据集回放、摄像头、甚至单张图片，只要按格式发即可。
    """
    from live3dgsavatar.tracking import rdf_wire as W

    import numpy as np

    head_n = W.request_head_size()
    handled = 0
    t_start = time.perf_counter()

    while True:
        head = _recv_exact(conn, head_n, allow_eof=True)
        if head is None:
            print("  客户端断开", flush=True)
            return handled
        seq, w, h = W.unpack_request_head(head)
        body = _recv_exact(conn, w * h * 3, allow_eof=True)
        if body is None:
            print("  客户端在帧体中间断开", flush=True)
            return handled

        rgb = np.frombuffer(body, np.uint8).reshape(h, w, 3)
        x, verts = tracker.track(rgb)
        conn.sendall(W.frame_bytes(seq, time.perf_counter() - t_start, x, verts))

        handled += 1
        if handled % 30 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  {handled} 帧  {handled / elapsed:.1f} fps  {w}x{h}",
                  flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RDF 通用追踪服务：接收一帧画面，返回参数与网格")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8980)
    p.add_argument("--width", type=int, default=512,
                   help="启动时预构建用的画面宽度（客户端尺寸不同会自动重建）")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--once", action="store_true",
                   help="一个客户端断开后退出（默认继续等下一个）")
    p.add_argument("--rdf-root", type=Path, default=DEFAULT_RDF_ROOT)
    p.add_argument("--model", choices=("vits", "vitb"), default="vits")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--fov-y-deg", type=float, default=30.0,
                   help="相机竖直视场角；需与训练该数据集时的相机一致")
    p.add_argument("--camera-distance", type=float, default=1.0)
    p.add_argument("--check-deps", action="store_true",
                   help="只检查环境，不启动")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = args.rdf_root.expanduser().resolve()

    problems = preflight(root, args.model)
    if args.check_deps or problems:
        print(f"Python {sys.version.split()[0]}  |  RDF 仓库 {root}")
        try:
            import torch
            print(f"torch {torch.__version__}")
        except ImportError:
            print("torch 未安装")
        if problems:
            print("\n[error] 无法启动追踪服务：", file=sys.stderr)
            for i, (_kind, text) in enumerate(problems, 1):
                print(f"  {i}. {text}", file=sys.stderr)
            kinds = {k for k, _ in problems}
            if "env" in kinds:
                print("\n环境问题：请在 `realdenseface` 环境里运行\n"
                      "  conda activate realdenseface", file=sys.stderr)
            if "asset" in kinds:
                print("\n资产问题：把 RealDenseFace 的 weights/ 与 configs/ 一起部署，\n"
                      "详见 docs/ENVIRONMENT.md「RDF 资产」", file=sys.stderr)
            return 1
        print("✓ 环境检查通过")
        if args.check_deps:
            return 0

    tracker = Tracker(args)
    server = None
    try:
        # **先加载模型再监听**：RDF 首次构建（含 torch.compile）常要几十秒。
        # 先监听会让客户端连上后干等，看起来像"卡住了"。
        print(f"正在加载 RDF 模型（预构建 {args.width}x{args.height}）"
              f"{'（首次含 torch.compile，约 30s）' if not args.no_compile else ''}…",
              flush=True)
        tracker.ensure(args.width, args.height)
        print("模型就绪", flush=True)

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        print(f"追踪服务已启动：{args.host}:{args.port}", flush=True)
        print("  协议：发一帧画面 → 回一帧参数与网格（见 docs/LIVE_DRIVING.md）",
              flush=True)

        while True:
            print("等待客户端连接…", flush=True)
            conn, addr = server.accept()
            print(f"来自 {addr[0]}:{addr[1]}", flush=True)
            try:
                serve_client(conn, tracker)
            except (BrokenPipeError, ConnectionResetError, ConnectionError) as e:
                print(f"  连接中断：{e}", flush=True)
            finally:
                conn.close()
            if args.once:
                break
            print("等待下一个客户端…（Ctrl-C 退出）", flush=True)
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        if server is not None:
            server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
