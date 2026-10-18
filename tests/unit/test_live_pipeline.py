"""实时驱动的链路（**不需要 GPU / RDF**）。

用真实的回环 socket 起一个假桥接进程，验证"帧格式 → 客户端 → 转换"整条链路。
RDF 本身只能在 realdenseface 环境里跑，但**链路**不依赖它 ——
这正是把桥接做成独立进程的好处之一：两端可以分别测。

这里守的四件事：

1. **分片读取**：`recv` 返回片段是常态，按 `recv(n)` 一次到位解析会偶发失败
2. **latest-wins**：渲染比 RDF 快时不该积压，只保留最新一帧
3. **过期判定**：桥接卡住时要能识别（否则画面冻住却仍在渲染）
4. **补牙齿与参数转换**：5023 → 5083、118 → 129
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
from live3dgsavatar.tracking import live_source as LS
from live3dgsavatar.tracking import mesh_build as MB
from live3dgsavatar.tracking import param_map as PM
from live3dgsavatar.tracking import rdf_client as RC
from live3dgsavatar.tracking import rdf_wire as W

NUM_VERTS = 5023
NUM_EXP = 100


# ------------------------------------------------------------- 帧格式 --


def test_hello_roundtrip() -> None:
    ident = np.arange(300, dtype=np.float32)
    buf = W.hello_bytes(num_vertices=NUM_VERTS, num_expressions=NUM_EXP,
                        width=512, height=512, identity=ident)
    assert len(buf) == W.hello_size()

    info = W.unpack_hello(buf)
    assert info["num_vertices"] == NUM_VERTS
    assert info["num_expressions"] == NUM_EXP
    assert info["width"] == 512
    assert np.array_equal(info["identity"], ident)


def test_frame_roundtrip() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(W.x_dim(NUM_EXP)).astype(np.float32)
    verts = rng.standard_normal((NUM_VERTS, 3)).astype(np.float32)

    buf = W.frame_bytes(42, 1.25, x, verts)
    assert len(buf) == W.frame_size(NUM_VERTS, NUM_EXP)

    head = buf[:W.frame_head_size()]
    body = buf[W.frame_head_size():]
    seq, ts = W.unpack_frame_head(head)
    assert seq == 42 and abs(ts - 1.25) < 1e-12

    payload = W.unpack_frame(body, num_vertices=NUM_VERTS, num_expressions=NUM_EXP)
    assert np.array_equal(payload["x"], x)
    assert np.array_equal(payload["verts"], verts)


def test_wrong_version_is_rejected_loudly() -> None:
    """协议版本不一致必须立刻报错，而不是画出错位的图。"""
    buf = bytearray(W.hello_bytes(num_vertices=NUM_VERTS, num_expressions=NUM_EXP,
                                  width=512, height=512))
    buf[4:6] = (W.VERSION + 1).to_bytes(2, "little")
    with pytest.raises(ValueError, match="版本"):
        W.unpack_hello(bytes(buf))


def test_bad_magic_is_rejected() -> None:
    buf = bytearray(W.hello_bytes(num_vertices=NUM_VERTS, num_expressions=NUM_EXP,
                                  width=512, height=512))
    buf[0:4] = b"XXXX"
    with pytest.raises(ValueError, match="魔数"):
        W.unpack_hello(bytes(buf))


# ------------------------------------------------------- 真实 socket --


class _FakeBridge:
    """最小的桥接进程：监听 → 接受 → 连发 N 帧。"""

    def __init__(self, frames: int = 5, *, chunk: int = 4096) -> None:
        self.frames = frames
        self.chunk = chunk
        self.port = 0
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.sent = 0

    def start(self) -> int:
        self._thread.start()
        self._ready.wait(timeout=5.0)
        return self.port

    def _run(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        self.port = server.getsockname()[1]
        server.listen(1)
        self._ready.set()
        try:
            conn, _ = server.accept()
            with conn:
                conn.sendall(W.hello_bytes(
                    num_vertices=NUM_VERTS, num_expressions=NUM_EXP,
                    width=512, height=512))
                rng = np.random.default_rng(1)
                for seq in range(self.frames):
                    buf = W.frame_bytes(
                        seq, float(seq),
                        rng.standard_normal(W.x_dim(NUM_EXP)).astype(np.float32),
                        rng.standard_normal((NUM_VERTS, 3)).astype(np.float32))
                    # **故意切片发送**：真实网络就是这样，读端必须循环读满
                    for i in range(0, len(buf), self.chunk):
                        conn.sendall(buf[i:i + self.chunk])
                    self.sent += 1
                    time.sleep(0.002)
                time.sleep(0.05)
        finally:
            server.close()


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_client_receives_fragmented_frames() -> None:
    """帧被切成 4 KB 片段发送，读端仍应完整收到每一帧。"""
    bridge = _FakeBridge(frames=4, chunk=4096)
    client = RC.RdfClient("127.0.0.1", bridge.start())
    try:
        client.start()
        assert _wait_for(lambda: client.frames_received >= 4), (
            f"只收到 {client.frames_received} 帧")
        obs = client.latest()
        assert obs is not None
        assert obs.verts.shape == (NUM_VERTS, 3)
        assert obs.x.shape == (W.x_dim(NUM_EXP),)
        assert client.status()["connected"] is True
    finally:
        client.stop()


def test_client_keeps_only_the_latest_frame() -> None:
    """latest-wins：读得慢时中间的帧被丢弃，但拿到的始终是最新的。"""
    bridge = _FakeBridge(frames=10)
    client = RC.RdfClient("127.0.0.1", bridge.start())
    try:
        client.start()
        assert _wait_for(lambda: client.frames_received >= 10)
        obs = client.latest()
        assert obs is not None and obs.seq == 9, f"最新帧应为 9，实际 {obs.seq}"
    finally:
        client.stop()


def test_client_reports_stale_frames_as_unavailable() -> None:
    """过期判定：桥接停止后，超过 max_age 的帧应视为不可用。"""
    bridge = _FakeBridge(frames=1)
    client = RC.RdfClient("127.0.0.1", bridge.start())
    try:
        client.start()
        assert _wait_for(lambda: client.frames_received >= 1)
        obs = client.latest(max_age_s=10.0)
        assert obs is not None, "刚收到的帧不应过期"
        # 把它人为做旧
        obs.received_at -= 5.0
        assert client.latest(max_age_s=1.0) is None, "超过 max_age 应为 None"
        assert client.latest(max_age_s=10.0) is not None
    finally:
        client.stop()


def test_client_survives_no_server() -> None:
    """没有桥接时不应崩溃，只是 `connected=False` 并重试。"""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()                      # 立刻释放，确保没人监听

    client = RC.RdfClient("127.0.0.1", port, connect_timeout_s=0.2)
    try:
        client.start()
        assert _wait_for(lambda: client.last_error is not None, timeout=3.0)
        assert client.latest() is None
        assert client.status()["connected"] is False
    finally:
        client.stop()


# ------------------------------------------------------------ 转换 --


class _FakeClient:
    """注入 `LiveSource` 的假客户端（不碰网络）。"""

    def __init__(self, obs: RC.Observation | None) -> None:
        self._obs = obs
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        pass

    def latest(self, max_age_s=None):
        return self._obs

    def status(self) -> dict:
        return {"connected": self._obs is not None, "frames_received": 1,
                "fps": 30.0, "seq": self._obs.seq if self._obs else None,
                "age_s": 0.0, "error": None}


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
    for k, v in over.items():
        cfg.set(k, v)
    return cfg


def test_live_source_produces_full_topology() -> None:
    """5023 顶点必须补成 5083，129 维参数必须组装正确。"""
    flame = repo_root() / "data" / "FLAME2020"
    if not flame.is_dir():
        pytest.skip("缺少 FLAME 资产")

    src = LS.LiveSource(_cfg(), "cpu", client=_FakeClient(_observation()))
    mesh, weight = src.observation()

    assert tuple(mesh.shape) == (1, MB.NUM_FULL_VERTS, 3)
    assert tuple(weight.shape) == (1, PM.DIM_BLEND_WEIGHT)
    assert mesh.dtype == torch.float32 and weight.dtype == torch.float32
    assert torch.isfinite(mesh).all()


def test_live_source_caches_by_sequence() -> None:
    """同一 seq 只转换一次（渲染比 RDF 快时不能每帧重算）。"""
    flame = repo_root() / "data" / "FLAME2020"
    if not flame.is_dir():
        pytest.skip("缺少 FLAME 资产")

    src = LS.LiveSource(_cfg(), "cpu", client=_FakeClient(_observation(seq=3)))
    a = src.observation()
    b = src.observation()
    assert a[0] is b[0], "同一 seq 应复用同一批张量"


def test_live_source_returns_none_when_disabled() -> None:
    src = LS.LiveSource(_cfg(**{"tracking.enabled": False}),
                        "cpu", client=_FakeClient(_observation()))
    assert src.observation() is None
    assert src.available is False


def test_live_source_returns_none_without_frames() -> None:
    src = LS.LiveSource(_cfg(), "cpu", client=_FakeClient(None))
    assert src.observation() is None
    assert src.status()["available"] is False


# ------------------------------------------------------- Session 分支 --


def test_session_live_branch_uses_observation() -> None:
    """`_frame_inputs` 在 live 模式下应直接用观测，而不是读数据集。"""
    flame = repo_root() / "data" / "FLAME2020"
    if not flame.is_dir():
        pytest.skip("缺少 FLAME 资产")

    from live3dgsavatar.app.session import Session

    session = Session(_cfg())
    session.params.source_kind = "live"
    session.live = LS.LiveSource(_cfg(), "cpu", client=_FakeClient(_observation()))

    mesh, weight = session._frame_inputs()
    assert tuple(mesh.shape) == (1, MB.NUM_FULL_VERTS, 3)
    assert tuple(weight.shape) == (1, PM.DIM_BLEND_WEIGHT)


def test_session_live_falls_back_to_first_frame() -> None:
    """live 没有数据时**回退到数据集首帧**（摄像头可能还没就绪）。

    为什么不报错：通话场景里黑屏/断流比"停在首帧"糟得多。
    但回退必须**可见** —— `live_fallback` 置真、`reason` 给出可执行提示，
    否则用户会以为摄像头在工作。
    """
    from live3dgsavatar.app.session import Session

    session = Session(_cfg())
    session.params.source_kind = "live"
    session.live = LS.LiveSource(_cfg(), "cpu", client=_FakeClient(None))
    session.load_model("crh_1", "test") if False else None

    # 未加载数据集时无从回退 —— 构造一个只含一帧的最小 scene
    from live3dgsavatar.data.scene import Scene
    from live3dgsavatar.core.types import Camera

    v = MB.NUM_FULL_VERTS
    faces = torch.zeros(1, 3, dtype=torch.int32)
    session.scene = Scene(
        frames=[{"mesh": torch.zeros(v, 3),
                 "blend_weight": torch.zeros(PM.DIM_BLEND_WEIGHT)}],
        faces=faces, uvs=torch.zeros(1, 2), uv_faces=faces,
        K=torch.eye(3), R=torch.eye(3), T=torch.zeros(3),
        width=8, height=8, dataset_frames=1)

    mesh, weight = session._frame_inputs()
    assert tuple(mesh.shape) == (1, v, 3)
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
    session.live = LS.LiveSource(_cfg(), "cpu", client=_FakeClient(_observation(seq=1)))
    session._last_input_key = ("live", 1)
    assert session.live_frame_pending() is False, "同 seq 不该重渲"

    session._last_input_key = ("live", 0)
    assert session.live_frame_pending() is True, "seq 变了应重渲"

    session.live = LS.LiveSource(_cfg(), "cpu", client=_FakeClient(None))
    session._last_input_key = ("live-fallback",)
    assert session.live_frame_pending() is False, "回退帧也不该反复重渲"


def test_dataset_mode_always_renders() -> None:
    """非 live 模式保持原语义："暂停 ≠ 停止渲染"（链路始终活着）。"""
    from live3dgsavatar.app.session import Session

    session = Session(_cfg())
    session.params.source_kind = "checkpoint"
    assert session.live_frame_pending() is True
