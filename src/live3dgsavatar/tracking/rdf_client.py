"""追踪服务的客户端：发一帧画面，拿回一帧观测。

## 为什么不再有"后台收帧线程"

服务改成**请求/响应**之后，节奏由本端决定：发什么画面、多久发一次，都是本端的事。
所以这里没有接收线程、没有"最新一帧"缓存 —— 只有一个阻塞的 `request()`。
由谁来驱动它由上层决定（见 `live_source`：一个取帧线程）。

好处是换画面来源（数据集 / 摄像头 / 单张图片）时，**客户端一行都不用改**。

## 断线处理

`request()` 在连接断开时抛 `ConnectionError` 并清掉连接状态，下次调用自动重连。
上层（`LiveSource`）负责重试；连续拿不到观测时渲染会回退到数据集首帧，
不会卡死或黑屏。
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from dataclasses import dataclass

import numpy as np

from . import rdf_wire as W

log = logging.getLogger(__name__)

#: 单次请求的超时（秒）。RDF 单帧约 60–70 ms，这里给足余量，主要防挂死。
_REQUEST_TIMEOUT_S = 10.0


@dataclass
class Observation:
    """一帧追踪结果。"""

    seq: int
    #: 服务侧时间戳（仅用于日志/诊断）
    timestamp: float
    #: `[n_exp+18]` 追踪参数
    x: "np.ndarray"
    #: `[n_verts, 3]` 用服务侧 FLAME2023 算出的顶点
    verts: "np.ndarray"
    #: 本端收到它的时刻（`time.monotonic()`）
    received_at: float

    def age_s(self, now: float | None = None) -> float:
        return (time.monotonic() if now is None else now) - self.received_at


class RdfClient:
    """追踪服务的长连接客户端。

    `request()` 用一把锁串行化：协议靠"一问一答"保序，并发发帧会让回复错位。
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8980, *,
                 connect_timeout_s: float = 2.0,
                 request_timeout_s: float = _REQUEST_TIMEOUT_S) -> None:
        self.host = host
        self.port = int(port)
        self.connect_timeout_s = float(connect_timeout_s)
        self.request_timeout_s = float(request_timeout_s)

        self._sock: socket.socket | None = None
        self._lock = threading.Lock()

        self.requests = 0
        self.last_error: str | None = None
        self.last_vertices: int | None = None
        self.last_expressions: int | None = None
        self._timestamps: list[float] = []

    # ------------------------------------------------------------- 连接 --

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def connect(self) -> None:
        """建立连接（已连接则直接返回）。失败抛 `ConnectionError`。"""
        if self._sock is not None:
            return
        try:
            sock = socket.create_connection(
                (self.host, self.port), timeout=self.connect_timeout_s)
            sock.settimeout(self.request_timeout_s)
            # 低延迟：小包立刻发，不要攒
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError as e:
            self.last_error = f"{type(e).__name__}: {e}"
            raise ConnectionError(self.last_error) from e
        self._sock = sock
        log.info("已连接追踪服务 %s:%d", self.host, self.port)

    def close(self) -> None:
        sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    # ------------------------------------------------------------- 请求 --

    def request(self, rgb: np.ndarray, seq: int | None = None) -> Observation:
        """发送一帧画面，阻塞等待结果。

        Args:
            rgb: `[H,W,3]` uint8
            seq: 帧号；``None`` 时用内部计数

        Raises:
            ConnectionError: 连不上 / 中途断开 / 超时（连接已清掉，可重连）
        """
        if seq is None:
            seq = self.requests

        with self._lock:
            self.connect()
            sock = self._sock
            assert sock is not None
            try:
                sock.sendall(W.request_bytes(seq, rgb))
                head = self._recv_exact(sock, W.frame_head_size())
                rseq, ts, n_verts, n_exp = W.unpack_frame_head(head)
                body = self._recv_exact(sock, W.body_size(n_verts, n_exp))
                payload = W.unpack_frame(body, num_vertices=n_verts,
                                         num_expressions=n_exp)
            except (OSError, ValueError) as e:
                # I/O 或解析错误一律当作"连接不可用"：清掉，下次重连。
                self.last_error = f"{type(e).__name__}: {e}"
                self.close()
                raise ConnectionError(self.last_error) from e

            now = time.monotonic()
            self.requests += 1
            self.last_vertices = n_verts
            self.last_expressions = n_exp
            self._timestamps.append(now)
            cutoff = now - 3.0
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.pop(0)

        return Observation(seq=int(rseq), timestamp=float(ts),
                           x=payload["x"], verts=payload["verts"],
                           received_at=now)

    @staticmethod
    def _recv_exact(sock: socket.socket, n: int) -> bytes:
        """读满 `n` 字节。

        `recv` 可以返回任意小的片段；直接 `recv(n)` 再解析是经典的
        "偶发解析失败"来源，这里循环到底。
        """
        if n <= 0:
            return b""
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("服务端关闭了连接")
            buf.extend(chunk)
        return bytes(buf)

    # ------------------------------------------------------------- 状态 --

    def fps(self) -> float:
        """最近 3 秒的完成速率。"""
        if len(self._timestamps) < 2:
            return 0.0
        span = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / span if span > 1e-6 else 0.0

    def status(self) -> dict:
        return {
            "connected": self.connected,
            "requests": self.requests,
            "fps": round(self.fps(), 1),
            "error": self.last_error,
            "num_vertices": self.last_vertices,
            "num_expressions": self.last_expressions,
        }
