"""WebSocket 写入必须串行 —— 并发 `send` 会直接断开连接。

真实事故：服务端同时跑三个任务（渲染推流 / 帧发送 / 控制消息），
它们各自直接调 `ws.send_*`。`websockets` 的底层用单个 waiter 表示
"正在等下一位"，两个协程同时进来就触发

    assert waiter is None or waiter.cancelled()   （legacy/protocol.py:308）

连接随即被强制关闭 —— 现象是"点一下界面就断"：控制消息要回 status，
恰好撞上正在发帧。

这里用一个"重入即报错"的假 socket 验证 `_Sender` 真的串行化了，
并验证断开后不再尝试写入。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import live3dgsavatar  # noqa: E402, F401

from live3dgsavatar.app.server import _Sender  # noqa: E402


class _ConcurrentDetector:
    """假 WebSocket：同一时刻只允许一个 send 在飞。

    检测方式贴近真实原因：真实实现里并发 send 会命中 waiter 断言，
    这里用一个"正在发送"标记模拟同一处竞态。
    """

    def __init__(self, delay: float = 0.002) -> None:
        self.delay = delay
        self._busy = False
        self.texts: list[str] = []
        self.blobs: list[bytes] = []
        self.fail_after: int | None = None     # 第 n 次写入后抛错（模拟断开）
        self._calls = 0

    async def _guard(self, store, payload) -> None:
        if self._busy:
            raise AssertionError(
                "并发写入：真实 websockets 会在这里 assert waiter is None")
        self._busy = True
        try:
            await asyncio.sleep(self.delay)     # 模拟网络 drain
            self._calls += 1
            if self.fail_after is not None and self._calls > self.fail_after:
                raise ConnectionResetError("client gone")
            store.append(payload)
        finally:
            self._busy = False

    async def send_text(self, payload: str) -> None:
        await self._guard(self.texts, payload)

    async def send_bytes(self, payload: bytes) -> None:
        await self._guard(self.blobs, payload)


def test_concurrent_sends_are_serialized() -> None:
    """三个任务各发一批：一次都不能重叠。"""
    ws = _ConcurrentDetector()
    sender = _Sender(ws)

    async def flood(tag: str) -> None:
        for _ in range(10):
            await sender.text(tag)
            await sender.bytes(tag.encode())

    async def run() -> None:
        await asyncio.gather(flood("produce"), flood("consume"),
                             flood("control"))

    asyncio.run(run())

    assert len(ws.texts) == 30, len(ws.texts)
    assert len(ws.blobs) == 30, len(ws.blobs)


def test_failure_marks_closed_and_stops_writing() -> None:
    """发送失败后应停止尝试：否则每帧都会炸一次、日志被刷满。"""
    ws = _ConcurrentDetector()
    ws.fail_after = 1
    sender = _Sender(ws)

    async def run() -> list[bool]:
        results = [await sender.text("a")]
        results.append(await sender.text("b"))      # 应失败
        results.append(await sender.text("c"))      # 已关闭，直接 False
        return results

    results = asyncio.run(run())
    assert results == [True, False, False], results
    assert sender.closed is True
    assert ws.texts == ["a"], ws.texts


def test_json_helper_encodes_chinese() -> None:
    """中文不能变成 `\\uXXXX`：日志与前端排查都靠它可读。"""
    ws = _ConcurrentDetector(delay=0)
    sender = _Sender(ws)

    async def run() -> None:
        await sender.json({"type": "status", "msg": "渲染中"})

    asyncio.run(run())
    assert "渲染中" in ws.texts[0], ws.texts[0]
