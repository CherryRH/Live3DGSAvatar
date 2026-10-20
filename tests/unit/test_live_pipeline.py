"""实时驱动的链路（**不需要 GPU / RDF**）。

追踪服务是独立的"画面进、参数出"进程，所以两端可以分别测：这里用真实的回环
socket 起一个假追踪服务，验证「请求帧 → 客户端 → 转换 → Session」整条链路。

这里守的六件事：

1. **协议自描述**：回复头带顶点数/表情维数，客户端不必预先握手
2. **分片读取**：`recv` 返回片段是常态，按 `recv(n)` 一次到位会偶发失败
3. **取帧线程**：追踪慢于取帧时不积压，只保留最新一帧
4. **过期判定**：服务卡住时要能识别（否则画面冻住却仍在渲染）
5. **补牙齿与参数转换**：5023 → 5083、118 → 129
6. **驱动数据集切换**：换数据集后相机/帧数/取帧器都要跟着变
"""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from live3dgsavatar.config import load_config, repo_root
from live3dgsavatar.tracking import frame_source as FS
from live3dgsavatar.tracking import live_source as LS
from live3dgsavatar.tracking import mesh_build as MB
from live3dgsavatar.tracking import param_map as PM
from live3dgsavatar.tracking import rdf_client as RC
from live3dgsavatar.tracking import rdf_wire as W

NUM_VERTS = 5023
NUM_EXP = 100
IMG_W, IMG_H = 64, 48


# --------------------------------------------------------------- 协议 --


def _rgb(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (IMG_H, IMG_W, 3), np.uint8)


def test_request_roundtrip() -> None:
    rgb = _rgb()
    buf = W.request_bytes(7, rgb)
    assert len(buf) == W.request_size(IMG_W, IMG_H)

    seq, w, h = W.unpack_request_head(buf[:W.request_head_size()])
    assert (seq, w, h) == (7, IMG_W, IMG_H)
    back = W.unpack_request(buf[W.request_head_size():], width=w, height=h)["rgb"]
    assert back.shape == (IMG_H, IMG_W, 3)
    assert np.array_equal(back, rgb)


def test_request_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="H,W,3"):
        W.request_bytes(0, np.zeros((4, 4), np.uint8))


def test_reply_is_self_describing() -> None:
    """回复头必须带 n_verts / n_exp：客户端不握手也能解析。"""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(W.x_dim(NUM_EXP)).astype(np.float32)
    verts = rng.standard_normal((NUM_VERTS, 3)).astype(np.float32)

    buf = W.frame_bytes(42, 1.25, x, verts)
    assert len(buf) == W.frame_size(NUM_VERTS, NUM_EXP)

    seq, ts, n_verts, n_exp = W.unpack_frame_head(buf[:W.frame_head_size()])
    assert seq == 42 and abs(ts - 1.25) < 1e-12
    assert (n_verts, n_exp) == (NUM_VERTS, NUM_EXP)

    body = buf[W.frame_head_size():]
    assert len(body) == W.body_size(n_verts, n_exp)
    payload = W.unpack_frame(body, num_vertices=n_verts, num_expressions=n_exp)
    assert np.array_equal(payload["x"], x)
    assert np.array_equal(payload["verts"], verts)


def test_bad_magic_is_rejected() -> None:
    rgb = _rgb()
    req = bytearray(W.request_bytes(1, rgb))
    req[0:4] = b"XXXX"
    with pytest.raises(ValueError, match="魔数"):
        W.unpack_request_head(bytes(req[:W.request_head_size()]))

    rng = np.random.default_rng(0)
    rep = bytearray(W.frame_bytes(
        1, 0.0, rng.standard_normal(W.x_dim(NUM_EXP)).astype(np.float32),
        rng.standard_normal((NUM_VERTS, 3)).astype(np.float32)))
    rep[0:4] = b"XXXX"
    with pytest.raises(ValueError, match="魔数"):
        W.unpack_frame_head(bytes(rep[:W.frame_head_size()]))


# ------------------------------------------------------- 假追踪服务 --


class _FakeTracker:
    """最小追踪服务：收一帧请求，回一帧确定性结果。

    故意**分片发送**回复：真实网络就是这样，读端必须循环读满。
    """

    def __init__(self, *, chunk: int = 4096, close_after: int = 0,
                 bind_port: int = 0) -> None:
        self.chunk = chunk
        self.close_after = close_after
        self.bind_port = bind_port
        self.port = 0
        self.requests: list[tuple[int, int, int]] = []
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> int:
        self._thread.start()
        self._ready.wait(timeout=5.0)
        return self.port

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)

    def _run(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", self.bind_port))
        self.port = server.getsockname()[1]
        server.listen(1)
        self._ready.set()
        try:
            while not self._stop.is_set():
                conn, _ = server.accept()
                with conn:
                    handled = 0
                    while not self._stop.is_set():
                        head = _read_exact(conn, W.request_head_size())
                        if head is None:
                            break
                        seq, w, h = W.unpack_request_head(head)
                        body = _read_exact(conn, w * h * 3)
                        if body is None:
                            break
                        rgb = W.unpack_request(body, width=w, height=h)["rgb"]
                        self.requests.append((seq, int(rgb.shape[1]),
                                              int(rgb.shape[0])))
                        handled += 1
                        buf = W.frame_bytes(seq, float(seq), _reply_x(seq),
                                            _reply_verts(seq))
                        for i in range(0, len(buf), self.chunk):
                            conn.sendall(buf[i:i + self.chunk])
                        # 先回复再断开：否则客户端连第一帧都收不到
                        if self.close_after and handled >= self.close_after:
                            return
        finally:
            server.close()


def _read_exact(conn: socket.socket, n: int):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _reply_x(seq: int) -> np.ndarray:
    return np.full(W.x_dim(NUM_EXP), float(seq), np.float32)


def _reply_verts(seq: int) -> np.ndarray:
    return np.full((NUM_VERTS, 3), float(seq) * 0.001, np.float32)


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


# ------------------------------------------------------------- 客户端 --


def test_client_requests_and_receives() -> None:
    server = _FakeTracker(chunk=8192)
    port = server.start()
    client = RC.RdfClient("127.0.0.1", port, connect_timeout_s=2.0)
    try:
        obs = client.request(_rgb(1), seq=5)
        assert obs.seq == 5, "seq 必须原样回填"
        assert obs.x.shape == (W.x_dim(NUM_EXP),)
        assert obs.verts.shape == (NUM_VERTS, 3)
        assert client.status()["requests"] == 1
        assert client.connected is True
    finally:
        client.close()
        server.stop()


def test_client_handles_fragmented_replies() -> None:
    """回复被切成 4 KB 片段发送，仍应完整收到。"""
    server = _FakeTracker(chunk=4096)
    port = server.start()
    client = RC.RdfClient("127.0.0.1", port)
    try:
        for i in range(3):
            obs = client.request(_rgb(i), seq=i)
            assert obs.seq == i
            assert obs.verts.shape == (NUM_VERTS, 3)
        assert len(server.requests) == 3
    finally:
        client.close()
        server.stop()


def test_client_reports_connection_error_without_server() -> None:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()                      # 立刻释放，确保没人监听

    client = RC.RdfClient("127.0.0.1", port, connect_timeout_s=0.2)
    with pytest.raises(ConnectionError):
        client.request(_rgb(), seq=0)
    assert client.connected is False
    assert client.status()["error"], "失败必须记录原因"


def test_client_reconnects_after_peer_closes() -> None:
    """服务端关闭连接后，下一次请求应重连而不是永久失效。"""
    first = _FakeTracker(close_after=1)
    port = first.start()
    client = RC.RdfClient("127.0.0.1", port)
    try:
        client.request(_rgb(), seq=0)
        first.stop()
        # 第二次请求必然失败（对端已关）——失败后连接状态要清掉
        with pytest.raises(ConnectionError):
            client.request(_rgb(), seq=1)
        assert client.connected is False

        # 新服务绑**同一端口**，客户端应能自动重连
        second = _FakeTracker(chunk=8192, bind_port=port)
        try:
            assert second.start() == port
            obs = client.request(_rgb(), seq=2)
            assert obs.seq == 2
            assert client.connected is True
        finally:
            second.stop()
    finally:
        client.close()


# ------------------------------------------------------------ 取帧器 --


def test_dataset_frames_reads_input_in_order(tmp_path) -> None:
    from PIL import Image

    src = tmp_path / "input"
    src.mkdir()
    for i in range(3):
        Image.fromarray(np.full((8, 8, 3), i * 10, np.uint8)).save(src / f"{i:05d}.png")

    frames = FS.DatasetFrames(tmp_path, fps=0)
    assert frames.num_frames == 3
    values = []
    for _ in range(3):
        rgb = frames.read()
        assert rgb is not None and rgb.shape == (8, 8, 3)
        values.append(int(rgb[0, 0, 0]))
    assert values == [0, 10, 20], f"应按顺序读，实际 {values}"


def test_dataset_frames_loops_by_default(tmp_path) -> None:
    from PIL import Image

    src = tmp_path / "input"
    src.mkdir()
    Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(src / "00000.png")

    frames = FS.DatasetFrames(tmp_path, fps=0)
    assert frames.read() is not None
    assert frames.read() is not None, "默认应循环"

    once = FS.DatasetFrames(tmp_path, fps=0, loop=False)
    assert once.read() is not None
    assert once.read() is None, "loop=False 读完应返回 None"


def test_dataset_frames_prefers_input_then_images(tmp_path) -> None:
    """`auto`：有 input/ 用 input/，没有就退到 images/。

    为什么要回退：有些数据集（如 duda）只有抠图后的 images/。
    没有回退时那些数据集根本无法用于实时驱动。
    """
    from PIL import Image

    for sub in ("input", "images"):
        d = tmp_path / sub
        d.mkdir()
        Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(d / "00000.png")

    auto = FS.DatasetFrames(tmp_path, fps=0)
    assert auto.subdir == "input", "auto 应优先 input/"

    only_images = tmp_path / "imgs_only"
    (only_images / "images").mkdir(parents=True)
    Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(only_images / "images" / "00000.png")
    fallback = FS.DatasetFrames(only_images, fps=0)
    assert fallback.subdir == "images"
    assert fallback.read() is not None


def test_dataset_frames_subdir_override(tmp_path) -> None:
    """`tracking.frame_subdir` 可以强制指定目录。"""
    from PIL import Image

    for sub in ("input", "images"):
        d = tmp_path / sub
        d.mkdir()
        Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(d / "00000.png")

    assert FS.DatasetFrames(tmp_path, fps=0, subdir="images").subdir == "images"
    assert FS.DatasetFrames(tmp_path, fps=0, subdir="input").subdir == "input"


def test_dataset_frames_reports_missing_dirs() -> None:
    """两个目录都没有时，错误信息要指出找过哪些目录。"""
    with pytest.raises(FileNotFoundError, match="input"):
        FS.DatasetFrames(repo_root() / "no-such-dataset")

    with pytest.raises(ValueError, match="frame_subdir"):
        FS.DatasetFrames(repo_root(), fps=0, subdir="native")


# --------------------------------------------------------- 驱动源 --


class _FakeClient:
    """注入 `LiveSource` 的假客户端（不碰网络）。"""

    def __init__(self, obs: RC.Observation | None = None,
                 *, error: Exception | None = None) -> None:
        self._obs = obs
        self._error = error
        self.requests = 0
        self.connected = False

    def request(self, rgb, seq=None):        # noqa: ANN001
        self.requests += 1
        if self._error is not None:
            raise self._error
        assert self._obs is not None
        return RC.Observation(seq=self._obs.seq, timestamp=0.0,
                              x=self._obs.x, verts=self._obs.verts,
                              received_at=time.monotonic())

    def close(self) -> None:
        self.connected = False

    def status(self) -> dict:
        return {"connected": self.connected, "requests": self.requests,
                "fps": 0.0, "error": None}


class _FakeFrames:
    """无限取帧器（每帧 4×4）。"""

    kind = "fake"

    def __init__(self) -> None:
        self.reads = 0

    num_frames = 100

    def read(self):
        self.reads += 1
        time.sleep(0.001)
        return np.zeros((4, 4, 3), np.uint8)

    def read_at(self, index):        # noqa: ANN001
        return self.read()


def _observation(seq: int = 0) -> RC.Observation:
    rng = np.random.default_rng(7)
    return RC.Observation(
        seq=seq, timestamp=0.0,
        x=rng.standard_normal(W.x_dim(NUM_EXP)).astype(np.float32),
        verts=rng.standard_normal((NUM_VERTS, 3)).astype(np.float32) * 0.05,
        received_at=time.monotonic())


def _cfg(**over):
    cfg = load_config()
    cfg.set("tracking.enabled", True)
    # 这些测试都不需要 GPU；默认 cuda 会让 `Camera.to(device)` 尝试初始化 CUDA
    cfg.set("runtime.device", "cpu")
    for k, v in over.items():
        cfg.set(k, v)
    return cfg


def test_live_source_produces_full_topology() -> None:
    """5023 顶点必须补成 5083，129 维参数必须组装正确。"""
    if not (repo_root() / "data" / "FLAME2020").is_dir():
        pytest.skip("缺少 FLAME 资产")

    src = LS.LiveSource(_cfg(), "cpu", client=_FakeClient())
    src.ingest(_observation())
    mesh, weight = src.observation()

    assert tuple(mesh.shape) == (1, MB.NUM_FULL_VERTS, 3)
    assert tuple(weight.shape) == (1, PM.DIM_BLEND_WEIGHT)
    assert mesh.dtype == torch.float32 and weight.dtype == torch.float32
    assert torch.isfinite(mesh).all()


def test_live_source_caches_by_sequence() -> None:
    """同一 seq 只转换一次（渲染比追踪快时不能每帧重算）。"""
    if not (repo_root() / "data" / "FLAME2020").is_dir():
        pytest.skip("缺少 FLAME 资产")

    src = LS.LiveSource(_cfg(), "cpu", client=_FakeClient())
    src.ingest(_observation(seq=3))
    a = src.observation()
    b = src.observation()
    assert a[0] is b[0], "同一 seq 应复用同一批张量"


def test_live_source_stale_observation_is_unavailable() -> None:
    """过期判定：服务卡住后超过 max_age 的观测不可用。"""
    if not (repo_root() / "data" / "FLAME2020").is_dir():
        pytest.skip("缺少 FLAME 资产")

    src = LS.LiveSource(_cfg(**{"tracking.max_age_s": 0.5}), "cpu",
                        client=_FakeClient())
    obs = _observation()
    src.ingest(obs)
    assert src.available is True
    obs.received_at -= 5.0            # 人为做旧
    assert src.available is False
    assert src.observation() is None
    assert src.current_seq() is None


def test_live_source_disabled_returns_none() -> None:
    src = LS.LiveSource(_cfg(**{"tracking.enabled": False}), "cpu",
                        client=_FakeClient())
    src.ingest(_observation())
    assert src.observation() is None
    assert src.available is False


def test_live_source_thread_pulls_frames() -> None:
    """取帧线程应持续"取帧 → 请求"，并把最新结果交给渲染侧。"""
    client = _FakeClient(_observation(seq=1))
    frames = _FakeFrames()
    src = LS.LiveSource(_cfg(), "cpu", client=client, frames=frames)
    try:
        src.start()
        assert _wait_for(lambda: client.requests >= 3), (
            f"只发出 {client.requests} 次请求")
        assert src.running is True
    finally:
        src.stop()
    assert src.running is False
    assert frames.reads >= 3


def test_live_source_survives_client_errors() -> None:
    """追踪服务没起时线程必须活着重试，而不是退出。"""
    client = _FakeClient(error=ConnectionError("connection refused"))
    src = LS.LiveSource(_cfg(), "cpu", client=client, frames=_FakeFrames())
    try:
        src.start()
        assert _wait_for(lambda: client.requests >= 2, timeout=5.0), (
            f"只重试了 {client.requests} 次")
        assert src.running is True, "失败不应让线程退出"
        assert src.observation() is None
    finally:
        src.stop()


# ------------------------------------------------------- Session 分支 --


def test_session_live_branch_uses_observation() -> None:
    """`_frame_inputs` 在 live 模式下应直接用观测，而不是读数据集。"""
    if not (repo_root() / "data" / "FLAME2020").is_dir():
        pytest.skip("缺少 FLAME 资产")

    from live3dgsavatar.app.session import Session

    session = Session(_cfg())
    session.params.source_kind = "live"
    session.live = LS.LiveSource(_cfg(), "cpu", client=_FakeClient())
    session.live.ingest(_observation())

    mesh, weight = session._frame_inputs()
    assert tuple(mesh.shape) == (1, MB.NUM_FULL_VERTS, 3)
    assert tuple(weight.shape) == (1, PM.DIM_BLEND_WEIGHT)


def _tiny_scene(num_frames: int = 1, verts: int = MB.NUM_FULL_VERTS):
    from live3dgsavatar.data.scene import Scene

    faces = torch.zeros(1, 3, dtype=torch.int32)
    return Scene(
        frames=[{"mesh": torch.zeros(verts, 3),
                 "blend_weight": torch.zeros(PM.DIM_BLEND_WEIGHT)}
                for _ in range(num_frames)],
        faces=faces, uvs=torch.zeros(1, 2), uv_faces=faces,
        K=torch.eye(3), R=torch.eye(3), T=torch.zeros(3),
        width=8, height=8, dataset_frames=num_frames)


def test_session_live_falls_back_to_first_frame() -> None:
    """live 没有数据时**回退到数据集首帧**（追踪服务可能还没起）。

    回退必须**可见**：`live_fallback` 置真、`reason` 给出可执行提示，
    否则用户会以为实时驱动在工作。
    """
    from live3dgsavatar.app.session import Session

    session = Session(_cfg())
    session.params.source_kind = "live"
    session.live = LS.LiveSource(_cfg(), "cpu", client=_FakeClient())
    session.scene = _tiny_scene()

    mesh, weight = session._frame_inputs()
    assert tuple(mesh.shape) == (1, MB.NUM_FULL_VERTS, 3)
    assert tuple(weight.shape) == (1, PM.DIM_BLEND_WEIGHT)
    assert session.live_fallback is True
    reason = session.live_brief()["reason"]
    assert reason and ("rdf_bridge" in reason or "未连接" in reason), (
        f"回退时必须给出可执行原因，实际：{reason!r}")


def test_live_frame_pending_detects_new_content() -> None:
    """同一 seq 不应重复渲染；seq 变了或回退状态变了才重渲。"""
    from live3dgsavatar.app.session import Session

    session = Session(_cfg())
    session.params.source_kind = "live"
    session.live = LS.LiveSource(_cfg(), "cpu", client=_FakeClient())
    session.live.ingest(_observation(seq=1))
    session._last_input_key = ("live", 1)
    assert session.live_frame_pending() is False, "同 seq 不该重渲"

    session._last_input_key = ("live", 0)
    assert session.live_frame_pending() is True, "seq 变了应重渲"

    session.live = LS.LiveSource(_cfg(), "cpu", client=_FakeClient())
    session._last_input_key = ("live-fallback",)
    assert session.live_frame_pending() is False, "回退帧也不该反复重渲"


def test_dataset_mode_always_renders() -> None:
    """非 live 模式保持原语义："暂停 ≠ 停止渲染"（链路始终活着）。"""
    from live3dgsavatar.app.session import Session

    session = Session(_cfg())
    session.params.source_kind = "checkpoint"
    assert session.live_frame_pending() is True


# ------------------------------------------------- 驱动数据集切换 --


def test_set_drive_dataset_swaps_scene_and_frame_count() -> None:
    """切换驱动数据集要同时换掉：scene、相机、帧号上限、取帧器。"""
    from live3dgsavatar.app import session as S
    from live3dgsavatar.data import scene as scene_mod

    session = S.Session(_cfg())
    session.scene = _tiny_scene(num_frames=5)
    session.timeline = S.FrameTimeline(5)
    session.live = LS.LiveSource(_cfg(), "cpu", client=_FakeClient())
    session._sync_live_frames = lambda: None      # 取帧器单独测

    called: dict = {}

    def fake_load_scene(cfg, frames=None, device=None, data_dir=None):  # noqa: ANN001
        called["data_dir"] = data_dir
        return _tiny_scene(num_frames=11)

    # `set_drive_dataset` 在函数内部 `from ..data.scene import load_scene`，
    # 所以替换**模块属性**即可（不需要 monkeypatch 夹具）。
    original = scene_mod.load_scene
    scene_mod.load_scene = fake_load_scene
    session._dataset_dir = lambda name: Path("/tmp/fake")
    try:
        state = session.set_drive_dataset("crh_2")
    finally:
        scene_mod.load_scene = original

    assert session.params.drive_dataset == "crh_2"
    assert session.scene.total_frames == 11, "帧数必须跟着数据集变"
    assert session.timeline.total == 11, "帧号上限必须跟着变"
    assert state["dataset"]["num_frames"] == 11
    assert called["data_dir"] == Path("/tmp/fake")


def test_set_drive_dataset_rejects_topology_mismatch() -> None:
    """顶点数不匹配必须**报出来**，而不是画出错位的图。"""
    from live3dgsavatar.app import session as S
    from live3dgsavatar.data import scene as scene_mod

    session = S.Session(_cfg())
    # 模型自带数据集是 5083 顶点；待载入的（假）数据集只有 5023
    session._model_topology = (MB.NUM_FULL_VERTS, 10032)
    session._dataset_dir = lambda name: Path("/tmp/fake")
    session._sync_live_frames = lambda: None

    original = scene_mod.load_scene
    scene_mod.load_scene = lambda *a, **k: _tiny_scene(verts=MB.NUM_BASE_VERTS)
    try:
        with pytest.raises(RuntimeError, match="顶点"):
            session.set_drive_dataset("duda")
    finally:
        scene_mod.load_scene = original


def test_sync_drive_mode_starts_and_stops_tracking() -> None:
    """数据集模式不该让追踪进程空转（它和渲染抢同一块 GPU）。"""
    from live3dgsavatar.app.session import Session

    session = Session(_cfg())
    session.scene = _tiny_scene()
    started: list[str] = []

    class _Live:
        enabled = False

        def set_enabled(self, value):        # noqa: ANN001
            self.enabled = value
            started.append("enable" if value else "disable")

        def start(self):
            started.append("start")

        def stop(self):
            started.append("stop")

        def status(self):
            return {"connected": False, "requests": 0, "fps": 0.0,
                    "error": None, "num_vertices": None}

        available = False

        def set_frames(self, frames):        # noqa: ANN001
            pass

    session.live = _Live()
    session._live_frames_error = None
    session._sync_live_frames = lambda: None       # 取帧器单独测

    session.params.source_kind = "checkpoint"
    session.sync_drive_mode()
    assert "stop" in started, "数据集模式应停掉追踪"

    started.clear()
    session.params.source_kind = "live"
    session.sync_drive_mode()
    assert started[:2] == ["enable", "start"], f"应启用并启动，实际 {started}"


# ------------------------------------------------- 播放 / 暂停 / 跳帧 --


class _IndexedFrames:
    """按帧号取图的假取帧器（记录被请求过哪些帧号）。"""

    num_frames = 10
    subdir = "input"

    def __init__(self) -> None:
        self.reads: list[int] = []

    def read_at(self, index):        # noqa: ANN001
        self.reads.append(int(index))
        # 稍微慢一点，模拟真实追踪耗时；否则循环会瞬间跑几千轮、帧号反复回绕
        time.sleep(0.002)
        return np.zeros((4, 4, 3), np.uint8)


def _indexed_client() -> _FakeClient:
    client = _FakeClient(_observation(seq=0))
    client.seqs = []
    original = client.request

    def request(rgb, seq=None):      # noqa: ANN001
        client.seqs.append(int(seq))
        return original(rgb, seq)

    client.request = request
    return client


def _live_with_index(playing: bool = True):
    client = _indexed_client()
    frames = _IndexedFrames()
    src = LS.LiveSource(_cfg(), "cpu", client=client, frames=frames)
    messages: list[int] = []
    src.on_index = messages.append
    src.seek(0)
    src.set_playing(playing)
    return src, client, frames, messages


def test_live_playback_advances_frames() -> None:
    """播放时**追踪一帧 = 推进一帧**（播放条与内容不会脱节）。"""
    src, client, frames, messages = _live_with_index(playing=True)
    try:
        src.start()
        assert _wait_for(lambda: len(client.seqs) >= 3), client.seqs
    finally:
        src.stop()

    assert client.seqs[:3] == [0, 1, 2], f"应按帧号递增，实际 {client.seqs[:3]}"
    assert messages[:2] == [1, 2], f"应回调上层推进，实际 {messages[:3]}"
    # 帧号会回绕，所以不能断言"≥ 某值"；断言不变量：当前帧号 = 最后一次回调值
    assert src.index == messages[-1], (
        f"当前帧号应与最后一次回调一致：index={src.index} last={messages[-1]}")


def test_live_pause_keeps_frame_and_stops_requesting() -> None:
    """暂停时帧号不动，也不应满负荷重送（否则白烧 GPU）。"""
    src, client, frames, messages = _live_with_index(playing=False)
    try:
        src.start()
        assert _wait_for(lambda: len(client.seqs) >= 1)
        first = len(client.seqs)
        time.sleep(0.3)
        extra = len(client.seqs) - first
        assert extra <= 1, f"暂停期间不该反复请求，实际 +{extra}"
        assert src.index == 0, "暂停时帧号必须停住"
    finally:
        src.stop()


def test_live_jump_pushes_target_frame() -> None:
    """跳帧要立刻生效（下一次取帧就是目标帧）。"""
    src, client, frames, messages = _live_with_index(playing=False)
    try:
        src.start()
        assert _wait_for(lambda: len(client.seqs) >= 1)
        src.seek(7)
        assert _wait_for(lambda: 7 in client.seqs), client.seqs
        assert frames.reads[-1] == 7
    finally:
        src.stop()


def test_session_play_pause_and_jump_reach_live_source() -> None:
    """Session 的播放/跳帧接口必须转给取帧线程（两种模式语义一致）。"""
    from live3dgsavatar.app.protocol import FrameTimeline
    from live3dgsavatar.app.session import Session

    session = Session(_cfg())
    session.scene = _tiny_scene(num_frames=10)
    session.timeline = FrameTimeline(10)

    calls: list[tuple] = []

    class _Live:
        enabled = False

        def set_playing(self, value):        # noqa: ANN001
            calls.append(("playing", value))

        def seek(self, index):               # noqa: ANN001
            calls.append(("seek", index))

        def set_enabled(self, value): ...    # noqa: D401,ANN001
        def start(self): ...
        def stop(self): ...
        def set_frames(self, frames): ...    # noqa: ANN001

        def status(self):
            return {"connected": False, "requests": 0, "fps": 0.0, "error": None}

        available = False

    session.live = _Live()
    session._live_frames_error = None

    session.set_playing(True)
    assert ("playing", True) in calls
    assert session.timeline.playing is True

    index = session.jump_frame(6)
    assert index == 6
    assert ("seek", 6) in calls
    assert session.params.frame == 6

    step = session.step_frame(+1)
    assert step == 7
    assert ("seek", 7) in calls


def test_dataset_switch_resets_to_paused_first_frame() -> None:
    """换数据集后回到第 0 帧并暂停：新数据集时长与内容都不同。"""
    from live3dgsavatar.app import session as S
    from live3dgsavatar.data import scene as scene_mod

    session = S.Session(_cfg())
    session.scene = _tiny_scene(num_frames=5)
    session.timeline = S.FrameTimeline(5)
    session.timeline.playing = True
    session.params.playing = True

    seeks: list[int] = []
    playings: list[bool] = []

    class _Live:
        enabled = True

        def set_playing(self, v): playings.append(v)      # noqa: ANN001
        def seek(self, i): seeks.append(i)                # noqa: ANN001
        def set_frames(self, f): ...                      # noqa: ANN001
        def start(self): ...
        def stop(self): ...
        def status(self):
            return {"connected": False, "requests": 0, "fps": 0.0, "error": None}
        available = False

    session.live = _Live()
    session._sync_live_frames = lambda: (seeks.append(0), playings.append(False))
    session._dataset_dir = lambda name: Path("/tmp/fake")

    original = scene_mod.load_scene
    scene_mod.load_scene = lambda *a, **k: _tiny_scene(num_frames=9)
    try:
        session.set_drive_dataset("crh_2")
    finally:
        scene_mod.load_scene = original

    assert session.timeline.index == 0
    assert session.timeline.playing is False
    assert session.params.playing is False
    assert playings[-1] is False, "必须通知取帧线程暂停"
