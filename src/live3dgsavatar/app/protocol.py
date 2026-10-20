"""GUI 后端的协议编解码与背压工具。

**刻意不依赖 FastAPI / torch**：这些是纯函数，可以在没有 GPU、
没有 Web 依赖的环境里直接单元测试。`docs/GUI_PROTOCOL.md` 是权威定义。
"""

from __future__ import annotations

import struct

#: 二进制帧头长度（字节）：一个 uint32 的 seq。
#: 二进制帧头：`<B 类型><I seq><f psnr>`。
#: 为什么把 PSNR 塞进帧头：对比图与 PSNR 必须**同步**（同一个帧号算出来的），
#: 走 status 会因为周期推送而与画面错位。
FRAME_HEADER_BYTES = 9
#: 渲染结果帧
MSG_RENDER = 1
#: 对比用的目标帧（数据集 `images/` 的同一帧）
MSG_TARGET = 2
_FRAME_HEAD = struct.Struct("<BIf")
#: 协议版本；与前端 `web/js/protocol.js` 的 `PROTOCOL_VERSION` 必须一致。
PROTOCOL_VERSION = 1

#: 数据源：驱动参数从哪来。
#: - `checkpoint`：数据集里已有的采样（当前可用）
#: - `live`     ：实时 tracking（**空钩子，待接入**）
SOURCE_KINDS = ("checkpoint", "live")
#: `live` 数据源下帧号与驱动相关字段**不适用**，收到时忽略而不是报错。
#: `live` 模式下仍然无意义的字段。
#: **帧号与播放态不在其中** —— 现在两种驱动方式都支持播放/暂停/跳帧：
#: 暂停时取帧线程重送同一帧，跳帧时立刻送目标帧。
LIVE_IGNORED_FIELDS = ("perturb", "amplitude")

#: 分辨率比例允许的范围（前端控件与后端校验共用）。
MIN_SCALE = 0.05
MAX_SCALE = 1.0
#: 扰动幅度允许的范围。
MAX_AMPLITUDE = 1.0


def pack_frame(seq: int, rgb, *, kind: int = MSG_RENDER,
               psnr: float = float("nan")) -> bytes:
    """按协议打包一帧：9 字节头（类型/seq/psnr）+ RGB8。

    Args:
        seq: 帧序号；超出 uint32 会回绕（客户端靠跳号统计丢帧，
            回绕会导致一次误报，但 2^32 帧后才发生，可忽略）
        rgb: 形状 `[H, W, 3]` 的 uint8 张量或数组，行优先
        kind: `MSG_RENDER`（渲染结果）或 `MSG_TARGET`（对比目标）
        psnr: 与目标帧的 PSNR；渲染帧上无意义，填 NaN
    """
    header = _FRAME_HEAD.pack(kind & 0xFF, seq & 0xFFFFFFFF, float(psnr))
    if hasattr(rgb, "numpy"):          # torch.Tensor
        rgb = rgb.numpy()
    if hasattr(rgb, "tobytes"):
        body = rgb.tobytes()
    else:                              # pragma: no cover - 防御性
        raise TypeError(f"不支持的图像类型：{type(rgb)!r}")
    return header + body


def unpack_frame(buffer: bytes) -> tuple[int, int, float, bytes]:
    """`pack_frame` 的逆运算，返回 `(kind, seq, psnr, rgb_bytes)`。"""
    if len(buffer) < FRAME_HEADER_BYTES:
        raise ValueError(f"帧太短：{len(buffer)} 字节")
    kind, seq, psnr = _FRAME_HEAD.unpack(buffer[:FRAME_HEADER_BYTES])
    return kind, seq, psnr, buffer[FRAME_HEADER_BYTES:]


def expected_frame_bytes(width: int, height: int, channels: int = 3) -> int:
    """一帧二进制消息应有的总长度。"""
    return FRAME_HEADER_BYTES + width * height * channels


class LatestSlot:
    """**只保留最新一项**的槽位（背压的核心）。

    容量为 1：满了就丢最旧的。渲染方永不阻塞，消费方永远拿到最新帧 ——
    代价是跳帧，好过延迟越积越大。

    刻意不依赖 `asyncio`，这样可以在同步测试里验证语义。
    真正的异步队列在 `server.py` 里用 `asyncio.Queue(maxsize=1)` 实现同样语义。
    """

    def __init__(self) -> None:
        self._item = None
        self.dropped = 0

    def offer(self, item) -> None:
        """放入一项；已有旧项则丢弃并计数。"""
        if self._item is not None:
            self.dropped += 1
        self._item = item

    def take(self):
        """取走当前项（取后为空）；无项返回 `None`。"""
        item, self._item = self._item, None
        return item

    @property
    def is_empty(self) -> bool:
        return self._item is None


def merge_params(current: dict, msg: dict) -> dict:
    """把一条 `params` 消息合并进当前参数字典，返回**新的**字典。

    只处理并校验协议里声明的字段；非法值抛 `ValueError`（由调用方转成
    给前端的错误消息，而不是让服务崩掉）。

    Args:
        current: 当前参数（至少含 `background` / `scaling_modifier` /
            `scale` / `target_fps` / `frame` / `playing` / `perturb` / `amplitude`）
        msg: 客户端发来的 `{"type":"params", "render":{...}, "drive":{...}}`
    """
    out = dict(current)
    render = msg.get("render") or {}
    drive = msg.get("drive") or {}

    if "background" in render:
        bg = [float(x) for x in render["background"]]
        if len(bg) != 3:
            raise ValueError(f"background 必须是 3 个数，实际 {len(bg)} 个")
        if any(not (0.0 <= c <= 1.0) for c in bg):
            raise ValueError(f"background 各分量必须在 [0, 1]，实际 {bg}")
        out["background"] = bg
    if "scaling_modifier" in render:
        v = float(render["scaling_modifier"])
        if not (0.05 <= v <= 5.0):
            raise ValueError(f"scaling_modifier 应在 [0.05, 5.0]，实际 {v}")
        out["scaling_modifier"] = v
    if "scale" in render:
        v = float(render["scale"])
        if not (MIN_SCALE <= v <= MAX_SCALE):
            raise ValueError(f"scale 应在 [{MIN_SCALE}, {MAX_SCALE}]，实际 {v}")
        out["scale"] = v
    if "target_fps" in render:
        v = float(render["target_fps"])
        if v < 0 or v > 480:
            raise ValueError(f"target_fps 应在 [0, 480]（0 表示不限），实际 {v}")
        out["target_fps"] = v

    if "compare" in msg:
        out["compare"] = bool(msg["compare"])

    source = msg.get("source") or {}
    if "kind" in source:
        kind = str(source["kind"])
        if kind not in SOURCE_KINDS:
            raise ValueError(
                f"source.kind 只能是 {' | '.join(SOURCE_KINDS)}，实际 {kind!r}")
        out["source_kind"] = kind
    # ⚠️ 数据集切换**刻意不走这里**：它要重载整个 scene（几百毫秒到几秒），
    #    属于"慢且可能失败"的操作，走 `POST /api/dataset` 让前端能显示
    #    加载浮窗并拿到明确的成功/失败结果。WS 的 params 只承载即时参数。

    # `live` 下只有调试扰动无意义（它的驱动来自追踪结果，不来自数据集帧）。
    # 帧号/播放态**保留**：live 模式同样要能暂停与跳帧。
    if out.get("source_kind") == "live":
        for field in LIVE_IGNORED_FIELDS:
            drive.pop(field, None)

    # ⚠️ 刻意**没有** `drive.mode`。是否推进帧号完全由 `playing` 表达：
    #    暂停 ≠ 停止渲染，只是帧号停住（渲染仍按帧数上限全速进行）。
    #    原本的 still/play 与播放按钮语义重复，已删除。
    if "frame" in drive:
        frame = int(drive["frame"])
        if frame < 0:
            raise ValueError(f"drive.frame 不能为负，实际 {frame}")
        out["frame"] = frame
    if "amplitude" in drive:
        v = float(drive["amplitude"])
        if not (0.0 <= v <= MAX_AMPLITUDE):
            raise ValueError(f"drive.amplitude 应在 [0, {MAX_AMPLITUDE}]，实际 {v}")
        out["amplitude"] = v
    if "playing" in drive:
        out["playing"] = bool(drive["playing"])
    if "perturb" in drive:
        out["perturb"] = bool(drive["perturb"])

    return out


def output_size(width: int, height: int, scale: float) -> tuple[int, int]:
    """按比例算出输出分辨率（至少 1×1）。"""
    if not (MIN_SCALE <= scale <= MAX_SCALE):
        raise ValueError(f"scale 应在 [{MIN_SCALE}, {MAX_SCALE}]，实际 {scale}")
    w = max(1, int(round(width * scale)))
    h = max(1, int(round(height * scale)))
    return w, h


def config_message(width: int, height: int, background, num_frames: int,
                   channels: int = 3) -> dict:
    """构造连接后的第一条 `config` 消息。"""
    return {
        "type": "config",
        "version": PROTOCOL_VERSION,
        "width": int(width),
        "height": int(height),
        "channels": int(channels),
        "background": [float(x) for x in background],
        "num_frames": int(num_frames),
    }


# --------------------------------------------------------- 帧号推进 --


class FrameTimeline:
    """数据集帧号的推进状态（播放 / 单步）。

    只有 `playing` 决定"每次渲染后是否推进一帧"；
    **暂停不代表停止渲染** —— 渲染循环仍按帧数上限全速工作，
    只是帧号停住（画面内容因此保持不变，但链路一直是活的）。

    **刻意做成纯逻辑**：帧号推进最容易出现「越界」「回绕到 0」「播放与显示不同步」
    这类错误，而它本身不需要 GPU —— 放在这里就能被单元测试覆盖。

    为什么要放在**后端**推进：渲染循环每出一帧就推进一次，
    「渲染的帧」与「显示的帧」天然一致；若交给前端定时器发消息，
    两者会因消息延迟与丢帧而漂移。
    """

    def __init__(self, total: int) -> None:
        if total < 1:
            raise ValueError(f"total 必须 ≥ 1，实际 {total}")
        self.total = total
        self.index = 0
        self.playing = False

    def set_index(self, index: int) -> int:
        """直接跳到某帧；越界会被夹到合法范围（不允许回绕）。"""
        self.index = max(0, min(int(index), self.total - 1))
        return self.index

    def advance(self) -> int:
        """播放时前进一步；到末尾**回绕到 0**。返回新帧号。"""
        self.index = (self.index + 1) % self.total
        return self.index

    def step(self, delta: int) -> int:
        """单步前进/后退；**在边界处夹住，不回绕**（便于逐帧审查）。

        与 `advance()` 的回绕语义刻意不同：播放要循环，单步要能停在两端。
        """
        self.index = max(0, min(self.index + int(delta), self.total - 1))
        return self.index


# --------------------------------------------------------- 帧率控制 --


def frame_interval(target_fps: float) -> float:
    """帧数上限 → 相邻两次渲染的**最小间隔**（秒）。

    `target_fps <= 0` 表示不限速，返回 0.0。
    """
    if target_fps <= 0:
        return 0.0
    return 1.0 / target_fps


def sleep_after(elapsed: float, interval: float, tolerance: float = 0.5) -> float:
    """按「帧数上限 + 等待」语义算出本轮还应睡多久（秒）。

    语义（见 docs/GUI_PROTOCOL.md §1）：

    - 上一轮耗时 < 间隔 → 睡满差额，使触发间隔恒为 `interval`
    - 上一轮耗时 ≥ 间隔 → **不睡**，立刻开始下一轮

    后者即"渲染慢于间隔时自动降级为渲染完就继续"，
    绝不补帧、绝不排队 —— 实际帧率 = `min(target_fps, 1/渲染耗时)`。

    Args:
        elapsed: 上一轮从开始到出帧的耗时（秒）
        interval: `frame_interval()` 的结果
        tolerance: 把极小的剩余睡眠视作 0，避免无意义的
            `asyncio.sleep(1e-6)` 让出调度却几乎不睡

    Returns:
        应睡眠的秒数；0.0 表示立刻继续。
    """
    if interval <= 0.0:
        return 0.0
    remaining = interval - elapsed
    if remaining <= interval * tolerance:
        return 0.0
    return remaining


class RollingWindow:
    """固定时间窗的滚动统计（用于渲染吞吐、前端 FPS）。

    刻意用**时间窗**而不是累计平均：改变帧数上限后，累计平均要很久才收敛，
    而时间窗在 1–2 个窗口内就能反映新状态。

    纯逻辑，不依赖 asyncio / GPU，可直接单测。
    """

    def __init__(self, window_s: float = 3.0) -> None:
        if window_s <= 0:
            raise ValueError(f"window_s 必须 > 0，实际 {window_s}")
        self.window_s = window_s
        self._events: list[float] = []      # 单调时钟时间戳

    def add(self, now: float) -> None:
        """登记一次事件（一帧）。"""
        self._events.append(now)
        self._trim(now)

    def rate(self, now: float) -> float:
        """当前窗口内的**每秒事件数**（FPS）。窗口为空返回 0。"""
        self._trim(now)
        if not self._events:
            return 0.0
        span = now - self._events[0]
        if span <= 0:
            # 窗口里只有 1 个事件、或全部同一时刻：无法算速率，
            # 返回 0 而不是 inf/NaN（上层按"还没有稳定速率"处理）
            return 0.0
        return (len(self._events) - 1) / span

    @property
    def count(self) -> int:
        return len(self._events)

    def clear(self) -> None:
        self._events.clear()

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_s
        # 事件按时间递增，线性丢弃过期的即可
        n_drop = 0
        for t in self._events:
            if t < cutoff:
                n_drop += 1
            else:
                break
        if n_drop:
            del self._events[:n_drop]


class IntervalLogger:
    """按时间间隔决定"该不该打一行日志"。

    与 `RollingWindow` 配套：窗口给数值，本类给节奏。
    """

    def __init__(self, interval_s: float = 10.0) -> None:
        if interval_s <= 0:
            raise ValueError(f"interval_s 必须 > 0，实际 {interval_s}")
        self.interval_s = interval_s
        self._last: float | None = None

    def due(self, now: float) -> bool:
        """到点了就返回 True 并推进计时；否则 False。"""
        if self._last is None:
            self._last = now
            return False
        if now - self._last >= self.interval_s:
            self._last = now
            return True
        return False
