"""连接 RDF 桥接进程，维护"最新一帧观测"。

## 为什么是后台线程而不是 asyncio 任务

渲染循环跑在 uvicorn 的事件循环线程里（见 `app/session.py` 的单线程约定）。
把 socket 读也放进事件循环会让"RDF 卡住"直接变成"渲染卡住" ——
两者必须解耦。所以这里用一个 daemon 线程收帧，只通过一把锁与主线程交换
**最新一帧**。

## 背压：只保留最新

RDF 30 fps、渲染 60+ fps，两边速率不一致。做法与推流的帧通道一致：
**只留最新一帧**，渲染方取走即清空。这样不会积压，代价是跳帧 ——
比"延迟越积越大"好得多。

过期判断用**到达时间**（`time.monotonic()`），不用桥接发来的时间戳：
两端在不同进程（可能不同机器），墙钟不可比。
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from dataclasses import dataclass

from . import rdf_wire as W

log = logging.getLogger(__name__)

#: 读 socket 的超时（秒）。用于让线程能及时响应 stop 请求。
_RECV_TIMEOUT_S = 0.5


@dataclass
class Observation:
    """一帧 RDF 观测。"""

    seq: int
    #: 桥接侧时间戳（仅用于日志/诊断）
    timestamp: float
    #: `[118]` RDF 参数
    x: "object"
    #: `[5023, 3]` RDF 用自己 FLAME2023 算出的顶点
    verts: "object"
    #: 本端收到它的时刻（`time.monotonic()`）
    received_at: float

    def age_s(self, now: float | None = None) -> float:
        return (time.monotonic() if now is None else now) - self.received_at


class RdfClient:
    """桥接进程的客户端：后台线程收帧，主线程只取最新一帧。

    断线会自动重连（退避 0.5 → 5 s）。本对象**永不阻塞**渲染循环：
    `latest()` 只做一次加锁读取。
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8980,
                 *, connect_timeout_s: float = 2.0) -> None:
        self.host = host
        self.port = int(port)
        self.connect_timeout_s = float(connect_timeout_s)

        self._lock = threading.Lock()
        self._latest: Observation | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        # 状态（读的地方也都在锁内或只写一次）
        self.connected = False
        self.frames_received = 0
        self.last_error: str | None = None
        self.hello: dict | None = None
        self._frames_total = 0
        self._rate_window: list[float] = []

    # ------------------------------------------------------------ 生命周期 --

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="rdf-client",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._thread = None

    def __enter__(self) -> "RdfClient":
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()

    # -------------------------------------------------------------- 取帧 --

    def latest(self, max_age_s: float | None = None) -> Observation | None:
        """取最新一帧；超过 `max_age_s` 视为过期，返回 ``None``。

        取走不清空 —— 渲染循环可能比 RDF 快，同一帧可以复用多次
        （画面保持不动，而不是变成黑屏）。
        """
        with self._lock:
            obs = self._latest
        if obs is None:
            return None
        if max_age_s is not None and obs.age_s() > float(max_age_s):
            return None
        return obs

    def status(self) -> dict:
        """给前端的状态摘要。"""
        with self._lock:
            obs = self._latest
            rate = self._rate()
        return {
            "connected": self.connected,
            "frames_received": self.frames_received,
            "fps": round(rate, 1),
            "seq": obs.seq if obs else None,
            "age_s": round(obs.age_s(), 3) if obs else None,
            "num_vertices": (W.FrameCodec(self.hello).num_vertices
                             if self.hello else None),
            "error": self.last_error,
        }

    def _rate(self) -> float:
        """最近 1 秒的到达帧率（近似；只用于显示）。"""
        now = time.monotonic()
        cutoff = now - 1.0
        self._rate_window = [t for t in self._rate_window if t >= cutoff]
        if len(self._rate_window) < 2:
            return 0.0
        span = self._rate_window[-1] - self._rate_window[0]
        return (len(self._rate_window) - 1) / span if span > 0 else 0.0

    # ------------------------------------------------------------ 线程主体 --

    def _run(self) -> None:
        backoff = 0.5
        while not self._stop.is_set():
            try:
                self._serve_once()
                backoff = 0.5
            except (OSError, ValueError) as e:
                self.last_error = f"{type(e).__name__}: {e}"
                log.debug("RDF 连接中断：%s", e)
            finally:
                self.connected = False
            if self._stop.wait(backoff):
                break
            backoff = min(backoff * 2, 5.0)

    def _serve_once(self) -> None:
        """连上去 → 读握手 → 收帧，直到断线或收到 stop。"""
        with socket.create_connection((self.host, self.port),
                                      timeout=self.connect_timeout_s) as sock:
            sock.settimeout(_RECV_TIMEOUT_S)
            hello = W.unpack_hello(_recv_exact(sock, W.hello_size()))
            codec = W.FrameCodec(hello)
            with self._lock:
                self.hello = hello
                self.connected = True
                self.last_error = None
            log.info("已连接 RDF 桥接 %s:%d（%d 顶点，%dx%d）",
                     self.host, self.port, codec.num_vertices,
                     codec.width, codec.height)

            head_size = W.frame_head_size()
            while not self._stop.is_set():
                head = _recv_exact(sock, head_size)
                body = _recv_exact(sock, codec.body_size)
                frame = codec.decode(head, body)
                obs = Observation(seq=frame["seq"], timestamp=frame["timestamp"],
                                  x=frame["x"], verts=frame["verts"],
                                  received_at=time.monotonic())
                with self._lock:
                    self._latest = obs
                    self.frames_received += 1
                    self._rate_window.append(obs.received_at)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """从 socket 读满 `n` 字节。

    `recv` 可以返回任意小的片段；直接 `recv(n)` 再用结果解析是经典的
    "偶发解析失败"来源。这里循环到底，超时则抛 `socket.timeout`。
    """
    if n <= 0:
        return b""
    chunks: list[bytes] = []
    got = 0
    while got < n:
        chunk = sock.recv(n - got)
        if not chunk:
            raise ConnectionError("对端关闭了连接")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)
