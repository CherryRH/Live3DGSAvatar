"""GUI 后端协议层测试（**无需 GPU、无需 FastAPI**）。

覆盖 `live3dgsavatar/app/protocol.py`：帧打包、参数校验、分辨率计算、背压槽位。
这些是纯函数，正是后端里最容易出错、也最值得测的部分 ——
HTTP/WebSocket 的粘合层交给手工联调，不写重量级的 ASGI 测试。
"""

from __future__ import annotations

import numpy as np

from live3dgsavatar.app import protocol as P


def _raises(exc_type):
    """零依赖的 `pytest.raises` 替代（运行器的 pytest 桩不支持 raises）。"""
    import contextlib

    class _Holder:
        value = None

    @contextlib.contextmanager
    def _cm():
        holder = _Holder()
        try:
            yield holder
        except exc_type as e:
            holder.value = e
            return
        except Exception as e:                       # noqa: BLE001
            raise AssertionError(
                f"期望 {exc_type.__name__}，实际 {type(e).__name__}: {e}") from e
        raise AssertionError(f"期望 {exc_type.__name__}，但未抛出")

    return _cm()


# ------------------------------------------------------------- 帧打包 --


def test_pack_frame_layout_matches_frontend() -> None:
    """帧布局必须与前端 `web/js/protocol.js::parseFrame` 一致。

    头是 `<B 类型><I seq><f psnr>`（9 字节），随后是行优先 RGB。
    """
    rgb = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)   # 1×2
    buf = P.pack_frame(0x01020304, rgb)

    assert len(buf) == P.expected_frame_bytes(2, 1), "总长度"
    assert buf[0] == P.MSG_RENDER, "类型字节"
    assert buf[1:5] == bytes([0x04, 0x03, 0x02, 0x01]), "seq 应为小端"
    assert buf[9:] == bytes([10, 20, 30, 40, 50, 60]), "像素应为行优先 RGB"


def test_pack_unpack_roundtrip() -> None:
    rgb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    kind, seq, psnr, body = P.unpack_frame(P.pack_frame(12345, rgb))
    assert kind == P.MSG_RENDER
    assert seq == 12345
    assert body == rgb.tobytes()


def test_target_frame_carries_psnr() -> None:
    """对比帧带 PSNR：它必须与画面**同一帧号**，走 status 会错位。"""
    rgb = np.zeros((1, 2, 3), np.uint8)
    kind, seq, psnr, body = P.unpack_frame(
        P.pack_frame(9, rgb, kind=P.MSG_TARGET, psnr=21.5))
    assert kind == P.MSG_TARGET
    assert seq == 9
    assert abs(psnr - 21.5) < 1e-6
    assert body == rgb.tobytes()


def test_pack_frame_accepts_torch_tensor() -> None:
    torch = __import__("torch")
    t = torch.arange(2 * 2 * 3, dtype=torch.uint8).reshape(2, 2, 3)
    buf = P.pack_frame(7, t)
    assert P.unpack_frame(buf)[1] == 7
    assert len(buf) == P.expected_frame_bytes(2, 2)


def test_pack_frame_seq_wraps_at_uint32() -> None:
    """seq 超出 uint32 应回绕而不是抛错（2^32 帧后才发生）。"""
    rgb = np.zeros((1, 1, 3), dtype=np.uint8)
    assert P.unpack_frame(P.pack_frame(2**32 + 5, rgb))[1] == 5


def test_merge_params_carries_compare_flag() -> None:
    """对比开关是运行期参数：走 params，不持久化。"""
    out = P.merge_params(_base(), {"compare": True})
    assert out["compare"] is True
    out = P.merge_params(out, {"compare": False})
    assert out["compare"] is False


def test_unpack_frame_rejects_short_buffer() -> None:
    with _raises(ValueError) as exc:
        P.unpack_frame(b"\x01\x02")
    assert "太短" in str(exc.value)


def test_protocol_version_matches_frontend() -> None:
    """前后端协议版本必须一致，否则前端会拒绝 config。"""
    import re
    from pathlib import Path

    from support import REPO_ROOT

    js = (REPO_ROOT / "web" / "js" / "protocol.js").read_text(encoding="utf-8")
    m = re.search(r"PROTOCOL_VERSION\s*=\s*(\d+)", js)
    assert m, "前端 protocol.js 里应定义 PROTOCOL_VERSION"
    assert int(m.group(1)) == P.PROTOCOL_VERSION, (
        f"协议版本不一致：前端 {m.group(1)}，后端 {P.PROTOCOL_VERSION}")


# ------------------------------------------------------------- 参数合并 --


def pytest_approx(value, tol=1e-9):
    """零依赖的近似比较（运行器的 pytest 桩没有 approx）。"""
    class _Approx:
        def __eq__(self, other):
            return abs(float(other) - value) <= tol
    return _Approx()


def _base() -> dict:
    return {
        "source_kind": "checkpoint",
        "background": [0.0, 0.0, 0.0],
        "scaling_modifier": 1.0,
        "scale": 1.0,
        "target_fps": 60.0,
        "frame": 0,
        "playing": False,
        "perturb": False,
        "amplitude": 0.1,
    }


def test_merge_params_partial_update() -> None:
    """只发改动字段，其余保持不变（协议允许部分更新）。"""
    out = P.merge_params(_base(), {"render": {"scale": 0.5}})
    assert out["scale"] == 0.5
    assert out["target_fps"] == 60.0, "未提到的字段不应被改"
    assert out["perturb"] is False, "未提及的字段不应被改"


def test_merge_params_returns_new_dict() -> None:
    """不得就地修改传入的字典（避免调用方的状态被悄悄改掉）。"""
    base = _base()
    P.merge_params(base, {"render": {"scale": 0.5}})
    assert base["scale"] == 1.0


def test_merge_params_rejects_bad_values() -> None:
    cases = [
        ({"render": {"background": [0, 0]}}, "background 个数"),
        ({"render": {"background": [0, 0, 2]}}, "background 越界"),
        ({"render": {"scale": 0.0}}, "scale 过小"),
        ({"render": {"scale": 1.5}}, "scale 过大"),
        ({"render": {"target_fps": -1}}, "fps 为负"),
        ({"drive": {"frame": -3}}, "frame 为负"),
        ({"drive": {"amplitude": 2.0}}, "amplitude 越界"),
    ]
    for msg, why in cases:
        with _raises(ValueError):
            P.merge_params(_base(), msg)


def test_merge_params_empty_message_is_noop() -> None:
    base = _base()
    assert P.merge_params(base, {"type": "params"}) == base


# ------------------------------------------------------------- 尺寸 --


def test_output_size_rounds_and_clamps() -> None:
    assert P.output_size(512, 512, 1.0) == (512, 512)
    assert P.output_size(512, 512, 0.5) == (256, 256)
    assert P.output_size(500, 500, 0.5) == (250, 250)
    # 极端缩小也要保证至少 1 像素，不能出现 0
    assert P.output_size(10, 10, 0.05) == (1, 1)


def test_output_size_rejects_bad_scale() -> None:
    for bad in (0.0, -1.0, 1.01, 2.0):
        with _raises(ValueError):
            P.output_size(512, 512, bad)


# ------------------------------------------------------------- 背压 --


def test_latest_slot_keeps_only_newest() -> None:
    """容量 1 的语义：新项覆盖旧项并计数丢弃。"""
    slot = P.LatestSlot()
    assert slot.is_empty
    assert slot.take() is None

    slot.offer("a")
    slot.offer("b")
    slot.offer("c")
    assert slot.dropped == 2, "前两项应被丢弃"
    assert slot.take() == "c"
    assert slot.is_empty, "取走后应为空"
    assert slot.take() is None


def test_latest_slot_no_drop_when_drained() -> None:
    """取一放一不应计为丢弃（否则丢帧统计会虚高）。"""
    slot = P.LatestSlot()
    for i in range(5):
        slot.offer(i)
        assert slot.take() == i
    assert slot.dropped == 0


# ------------------------------------------------------------- config --


def test_config_message_shape() -> None:
    msg = P.config_message(256, 128, [1, 1, 1], 254)
    assert msg["type"] == "config"
    assert msg["version"] == P.PROTOCOL_VERSION
    assert (msg["width"], msg["height"]) == (256, 128)
    assert msg["channels"] == 3
    assert msg["background"] == [1.0, 1.0, 1.0]
    assert msg["num_frames"] == 254


def test_config_message_matches_frontend_requirements() -> None:
    """前端 `geometryFromConfig` 要求的字段必须齐全，且宽高为整数。"""
    msg = P.config_message(512, 512, [0, 0, 0], 254)
    for key in ("width", "height", "background", "num_frames"):
        assert key in msg, f"缺少前端需要的字段 {key}"
    assert isinstance(msg["width"], int) and isinstance(msg["height"], int)


# --------------------------------------------------------- 帧号推进 --


def test_frame_timeline_advance_wraps() -> None:
    """播放推进：到末尾**回绕**（循环播放）。"""
    tl = P.FrameTimeline(5)
    assert tl.index == 0
    assert [tl.advance() for _ in range(5)] == [1, 2, 3, 4, 0]
    assert tl.index == 0


def test_frame_timeline_step_clamps_not_wraps() -> None:
    """单步：边界**夹住**，不回绕。

    与 `advance()` 的语义刻意不同 —— 播放要循环，单步要能停在两端，
    否则"上一帧"在开头会突然跳到结尾，逐帧审查时极易迷失。
    """
    tl = P.FrameTimeline(5)
    tl.set_index(0)
    assert tl.step(-1) == 0, "开头再后退应停在 0"
    assert tl.step(+1) == 1
    assert tl.step(+1) == 2
    for _ in range(10):
        tl.step(+1)
    assert tl.index == 4, "末尾再前进应停在最后一帧"
    assert tl.step(-1) == 3


def test_frame_timeline_set_index_clamps() -> None:
    tl = P.FrameTimeline(10)
    assert tl.set_index(-5) == 0
    assert tl.set_index(99) == 9
    assert tl.set_index(4) == 4


def test_frame_timeline_rejects_empty_dataset() -> None:
    with _raises(ValueError):
        P.FrameTimeline(0)


def test_frame_timeline_playing_flag_defaults_off() -> None:
    tl = P.FrameTimeline(3)
    assert tl.playing is False


def test_merge_params_carries_playing() -> None:
    """播放状态必须能经 `params` 传到后端（前端只发状态，不自己推进）。"""
    out = P.merge_params(_base(), {"drive": {"playing": True}})
    assert out["playing"] is True
    out = P.merge_params(out, {"drive": {"playing": False}})
    assert out["playing"] is False


def test_timeline_covers_full_dataset_not_preload_limit() -> None:
    """**回归测试**：帧号能走满**整个数据集**，不被预加载上限卡住。

    曾用 `render.frames`(=20) 作为 GUI 的预加载上限，而「数据集帧」滑块
    反映的是总数（254）—— 于是拖到 20 之后画面不再变化。
    根因是「预加载多少帧」与「能选到第几帧」被混为一谈。
    现在 GUI 用独立的 `app.preload_frames`（默认 -1 全部）。
    """
    from live3dgsavatar.config import load_config

    cfg = load_config()
    preload = int(cfg.get("app.preload_frames", -1))
    assert preload == -1, (
        f"GUI 应默认预载全部帧（app.preload_frames=-1），实际 {preload}；"
        "否则「数据集帧」滑块会被夹在已载入范围内")

    # 若预载了 N 帧，timeline 的可达范围就是 N
    total_loaded = 254
    tl = P.FrameTimeline(total_loaded)
    tl.set_index(total_loaded - 1)
    assert tl.index == total_loaded - 1, "应能选到最后一帧"


# ------------------------------------------------------- 帧率控制 --


def test_frame_interval() -> None:
    assert P.frame_interval(60) == 1 / 60
    assert P.frame_interval(30) == 1 / 30
    assert P.frame_interval(0) == 0.0, "0 表示不限速"
    assert P.frame_interval(-1) == 0.0


def test_sleep_after_waits_when_fast_enough() -> None:
    """渲染快于间隔 → 睡满差额，触发间隔恒为 interval。"""
    interval = 1 / 60                      # 16.67 ms
    assert P.sleep_after(0.005, interval) == pytest_approx(interval - 0.005)
    # 端到端：5 ms 渲染 + 睡眠 = 恒定 16.67 ms ⇒ 恰好 60 FPS
    total = 0.005 + P.sleep_after(0.005, interval)
    assert abs(total - interval) < 1e-9


def test_sleep_after_does_not_wait_when_too_slow() -> None:
    """渲染慢于间隔 → **不睡**，立刻开始下一轮（绝不补帧）。

    这是"上一帧没渲染完就等待"的真实含义：不堆积待渲染的帧，
    实际帧率自然降为 `1/渲染耗时`。
    """
    interval = 1 / 60
    assert P.sleep_after(0.050, interval) == 0.0, "慢于间隔时不应再睡"
    assert P.sleep_after(interval, interval) == 0.0, "刚好等于间隔也不睡"


def test_sleep_after_ignores_tiny_remainder() -> None:
    """极小的剩余睡眠视作 0 —— 否则会让出调度却几乎不睡，反而抖动。"""
    interval = 1 / 60
    # 剩余仅为间隔的 0.1%
    assert P.sleep_after(interval * 0.999, interval) == 0.0


def test_sleep_after_unlimited_fps_never_sleeps() -> None:
    assert P.sleep_after(0.001, 0.0) == 0.0
    assert P.sleep_after(1.0, 0.0) == 0.0


def test_effective_fps_is_min_of_cap_and_render_speed() -> None:
    """核心语义：实际帧率 = min(帧数上限, 1/渲染耗时)。"""
    def simulate(cap_fps: float, render_s: float, steps: int = 10) -> float:
        interval = P.frame_interval(cap_fps)
        total = 0.0
        for _ in range(steps):
            total += render_s + P.sleep_after(render_s, interval)
        return steps / total

    # 渲染 6 ms：60 上限 → 60；30 上限 → 30；不限 → 受渲染耗时限制
    assert abs(simulate(60, 0.006) - 60) < 0.5
    assert abs(simulate(30, 0.006) - 30) < 0.5
    assert abs(simulate(0, 0.006) - 1 / 0.006) < 0.5
    # 渲染 50 ms（超出 60 的间隔）→ 被渲染耗时限制，而不是硬撑 60
    assert abs(simulate(60, 0.050) - 20) < 0.5


def test_rolling_window_rate() -> None:
    """滑动时间窗：改变帧率后 1–2 个窗口内收敛（累计平均做不到）。"""
    w = P.RollingWindow(window_s=1.0)
    assert w.rate(0.0) == 0.0, "空窗口"
    # 在 [0,1) 内每秒 10 次
    t = 0.0
    for i in range(10):
        t = i * 0.1
        w.add(t)
    assert abs(w.rate(t) - 10.0) < 1.0
    # 窗口推进到 2s，不再有事件 → 速率落到 0
    assert w.rate(3.0) == 0.0


def test_rolling_window_single_event_is_zero_not_inf() -> None:
    """窗口里只有 1 个事件时无法算速率，应返回 0 而不是 inf/NaN。"""
    w = P.RollingWindow(window_s=1.0)
    w.add(5.0)
    assert w.rate(5.0) == 0.0


def test_rolling_window_rejects_bad_window() -> None:
    with _raises(ValueError):
        P.RollingWindow(window_s=0)


def test_interval_logger_fires_on_schedule() -> None:
    lg = P.IntervalLogger(interval_s=10.0)
    assert lg.due(0.0) is False, "第一次只登记起点"
    assert lg.due(9.9) is False
    assert lg.due(10.0) is True
    assert lg.due(15.0) is False, "触发后重新计时"
    assert lg.due(20.1) is True


def test_interval_logger_rejects_bad_interval() -> None:
    with _raises(ValueError):
        P.IntervalLogger(interval_s=0)


def test_merge_params_source_kind() -> None:
    out = P.merge_params(_base(), {"source": {"kind": "live"}})
    assert out["source_kind"] == "live"
    with _raises(ValueError):
        P.merge_params(_base(), {"source": {"kind": "bogus"}})


def test_live_keeps_frame_and_playing_but_drops_perturb() -> None:
    """`live` 下**帧号与播放态仍然有效**（要能暂停/跳帧），只有调试扰动无意义。

    为什么变了：取帧线程现在按帧号送图 —— 暂停时重送同一帧、跳帧时立刻送目标帧。
    所以播放条在两种驱动方式下语义一致。
    """
    base = _base()
    out = P.merge_params(base, {
        "source": {"kind": "live"},
        "drive": {"frame": 9, "amplitude": 0.3, "playing": True, "perturb": True},
    })
    assert out["source_kind"] == "live"
    assert out["frame"] == 9, "live 下帧号必须生效（播放条要能跳帧）"
    assert out["playing"] is True, "live 下播放态必须生效（要能暂停）"
    assert out["perturb"] == base["perturb"], "扰动开关仍应被忽略"
    assert out["amplitude"] == base["amplitude"], "扰动幅度仍应被忽略"


def test_no_drive_mode_field() -> None:
    """**回归测试**：协议里**不应再有** `drive.mode`。

    原本的 `still`/`play` 与播放按钮语义重复（"定住"就是"暂停"），
    且容易误解为"暂停就不渲染了"。
    现在只由 `playing` 表达"是否推进帧号"；
    **暂停不代表停止渲染** —— 渲染循环仍按帧数上限全速工作。
    """
    assert not hasattr(P, "DRIVE_MODES"), "DRIVE_MODES 应已删除"
    out = P.merge_params(_base(), {"drive": {"mode": "play"}})
    assert "drive_mode" not in out, "不应再产生 drive_mode 字段"
    assert out.get("drive_mode") is None


def test_playing_is_the_only_advance_switch() -> None:
    """推进帧号只由 `playing` 决定，没有第二个开关。"""
    assert P.merge_params(_base(), {"drive": {"playing": True}})["playing"] is True
    assert P.merge_params(_base(), {"drive": {"playing": False}})["playing"] is False


def test_perturb_is_independent_of_playing() -> None:
    """扰动与播放正交：可以暂停着看扰动结果，也可以播着不扰动。"""
    out = P.merge_params(_base(), {"drive": {"perturb": True, "playing": False}})
    assert out["perturb"] is True and out["playing"] is False
    out = P.merge_params(_base(), {"drive": {"perturb": False, "playing": True}})
    assert out["perturb"] is False and out["playing"] is True
