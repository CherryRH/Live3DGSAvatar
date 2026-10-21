"""GUI 后端：FastAPI 应用（HTTP + WebSocket）+ 推流循环。

层次：本模块只做**消息编解码与调度**，不碰张量；
所有 GPU 相关工作在 `session.py` 里。协议见 `docs/GUI_PROTOCOL.md`。

并发模型（刻意简单）：

- 全部在 uvicorn 的**单个事件循环线程**里跑。PyTorch 在事件循环线程里提交
  GPU kernel 是安全的：提交会释放 GIL，HTTP 仍能响应。
  **不做多线程渲染** —— 那会引入 CUDA 上下文竞态，收益也有限。
- 渲染循环是事件循环里的一个 asyncio 任务。**只在有客户端时渲染**，
  没人看就别占 GPU。
- 每帧推给一个**容量 1** 的队列：满则丢旧帧。渲染永不阻塞，
  客户端永远拿到最新画面（代价是跳帧，好过延迟累积）。
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import socket
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import (Config, load_config, model_selected,
                     resolve_model_ply)
from .protocol import (
    IntervalLogger,
    RollingWindow,
    config_message,
    frame_interval,
    MSG_RENDER_JPEG,
    MSG_TARGET,
    MSG_TARGET_JPEG,
    pack_blob,
    pack_frame,
    sleep_after,
)
from ..tracking.live_source import LiveUnavailable

from . import settings as settings_mod
from .session import Session

logger = logging.getLogger("live3dgsavatar.app")

WEB_DIR = Path(__file__).resolve().parents[3] / "web"


def _install_routes(app: FastAPI, session: Session) -> None:
    """注册 HTTP 与 WebSocket 路由。

    单独成函数便于测试时用假 session 装配，不依赖真实 GPU。
    """

    @app.get("/healthz")
    async def healthz() -> dict:
        import torch

        return {
            "ok": session.ready,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "model": session.state().get("model"),
        }

    @app.get("/api/state")
    async def api_state() -> dict:
        return session.state()

    @app.get("/api/models")
    async def api_models() -> dict:
        """可用模型列表（供面板的两个下拉框）。

        只检查 `model.ply` 是否存在 —— 深度校验留给加载时的报错，
        否则既慢又容易把能跑的目录误判成不可用。
        """
        from ..data.models import list_models

        def collect() -> dict:
            infos = list_models(session.cfg)
            return {
                "models": [i.to_dict() for i in infos],
                "current": {
                    "subject": session.model_subject or str(
                        session.cfg.get("subject") or ""),
                    "work_name": session.model_work_name or str(
                        session.cfg.get("work_name") or ""),
                },
            }

        return await asyncio.to_thread(collect)

    @app.get("/api/datasets")
    async def api_datasets() -> dict:
        """可用数据集列表（供界面选择"用什么驱动"）。

        只依赖目录结构和一个 npz 的 shape，不加载任何张量，
        所以这个接口是毫秒级的、也不需要 GPU。
        """
        from ..data.datasets import list_datasets

        def collect() -> dict:
            infos = list_datasets(session.cfg)
            model_subject = (session.model_subject
                             or str(session.cfg.get("subject") or ""))
            return {
                "datasets": [{**i.to_dict(), "is_model": i.name == model_subject}
                             for i in infos],
                "model_subject": model_subject,
                "current": session.params.drive_dataset,
            }

        return await asyncio.to_thread(collect)

    @app.get("/api/settings")
    async def api_settings() -> dict:
        """全部配置项（含**锁定项**）+ 分组元数据。

        全量下发是刻意的：用户需要看到"有哪些参数、为什么不能改"，
        而不是在面板里找不到它们。
        """
        return {
            "groups": settings_mod.groups_payload(),
            "fields": settings_mod.snapshot(session.cfg),
            "overrides_path": str(settings_mod.overrides_path()),
            "overrides": settings_mod.load_overrides(),
        }

    @app.post("/api/settings")
    async def api_settings_save(payload: dict) -> JSONResponse:
        """校验 → 即时应用 → 持久化到 `configs/local.yaml`。

        三步都要做：只应用不持久化，重启就丢；只持久化不应用，
        用户改了却看不到变化。校验失败的项**整批拒绝**（不做部分应用），
        避免"改了一半"的中间状态。
        """
        patch = (payload or {}).get("patch")
        if not isinstance(patch, dict) or not patch:
            raise HTTPException(status_code=400,
                                detail="需要 {\"patch\": {键: 值}} 且不能为空")
        try:
            clean = settings_mod.validate_patch(patch)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        applied = session.apply_settings(clean)

        saved_to = None
        warning = None
        try:
            path = await asyncio.to_thread(settings_mod.save_overrides, clean)
            saved_to = str(path)
        except OSError as e:
            # 存盘失败（只读文件系统等）不该让"已经生效的修改"回滚，
            # 但要明确告诉用户"这次重启后会丢"。
            warning = f"已应用，但保存失败：{e}"
            logger.warning("保存设置失败：%s", e)

        return JSONResponse({
            "ok": True,
            "applied": applied,
            "saved_to": saved_to,
            "warning": warning,
            "groups": settings_mod.groups_payload(),
            "fields": settings_mod.snapshot(session.cfg),
            # 渲染参数的**权威值**（与 /api/state 的 render 同构）。
            # 前端据此同步任何镜像控件；不塞进 4 Hz 的 status ——
            # 这些值很少变，每次推是浪费。
            "render": session.params.to_dict(),
            "status": _status_payload(session),
        })

    @app.post("/api/dataset")
    async def api_dataset(payload: dict) -> JSONResponse:
        """切换**驱动数据集**（慢操作，走 HTTP 而不是 WS）。

        为什么单独一个接口：换数据集要重载整个 scene（磁盘 IO + 搬张量），
        几百毫秒到几秒，还可能失败。前端需要"显示加载浮窗 → 等结果 →
        成功才刷新界面"，这要求一次明确的请求/响应；WS 的 params 是即时的、
        不保证顺序的，承载不了这个语义。

        成功后**回到第 0 帧并暂停**：新数据集的时长与内容都不同。
        """
        name = str((payload or {}).get("dataset") or "")
        try:
            state = await asyncio.to_thread(session.set_drive_dataset, name)
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:  # noqa: BLE001
            logger.exception("切换驱动数据集失败")
            raise HTTPException(status_code=400,
                                detail=f"{type(e).__name__}: {e}") from e

        # 帧数变了 → 让渲染循环补发 config（帧条要重排）；对比帧也要重发
        app.state.stream_total = None
        app.state.target_key = None
        return JSONResponse(state)

    @app.post("/api/model")
    async def api_model(payload: dict) -> JSONResponse:
        subject = (payload or {}).get("subject")
        work_name = (payload or {}).get("work_name")
        if not subject or not work_name:
            raise HTTPException(status_code=400,
                                detail="subject 与 work_name 都不能为空")
        try:
            # 加载会阻塞事件循环（含磁盘 IO 与 GPU 初始化）。
            # 对本地单用户工具可接受；换模型是低频操作。
            state = await asyncio.to_thread(session.load_model, subject, work_name)
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=400,
                detail=f"{e}\n本项目模型应放在：{session.model_dir_hint()}") from e
        except Exception as e:  # noqa: BLE001
            logger.exception("加载模型失败")
            raise HTTPException(status_code=400,
                                detail=f"{type(e).__name__}: {e}") from e
        _restart_stream(app)
        return JSONResponse(state)

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        await _serve_client(app, ws, session)


def _restart_stream(app: FastAPI) -> None:
    task = getattr(app.state, "stream_task", None)
    if task is not None and not task.done():
        task.cancel()
    app.state.stream_task = None


async def _serve_client(app: FastAPI, ws: WebSocket, session: Session) -> None:
    """一个客户端的完整生命周期：发 config → 推帧 + 并发收控制消息。"""
    # ⚠️ 所有写入都必须经过它：`produce`/`consume`/`control` 三个任务并发跑，
    #    直接 `ws.send_*` 会让 websockets 底层的 waiter 断言失败并断开连接。
    sender = _Sender(ws)

    # 模型可能还没加载：**不要断开连接**（断开会让前端反复重连、日志刷满）。
    # 提示一次，然后轮询等待；用户在界面里点「加载模型」后自动接上。
    if not session.ready:
        await sender.json({
            "type": "error", "transient": True,
            "message": "还没有加载模型：在右侧选择后点「加载模型」"})
        ready_log = IntervalLogger(interval_s=10.0)
        while not session.ready:
            await asyncio.sleep(0.3)
            # 心跳，用来及时发现客户端已断开
            if ready_log.due(asyncio.get_running_loop().time()):
                if not await sender.json({"type": "status"}):
                    return
        if not session.ready:  # pragma: no cover - 防御
            return

    width, height = session.output_size()
    await sender.json(config_message(
        width, height, session.params.background,
        session.scene.total_frames))

    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    stop = asyncio.Event()
    # 本连接的发送字节统计：`total` 为累计，`logged` 是上次打日志时的值。
    # ⚠️ 速率必须用**区间增量**算：拿累计值除以固定间隔会得到"只增不减"的假速率
    #    （曾经把 4722 MiB 当成 472 MiB/s 打了出来）。
    sent = {"total": 0, "logged": 0, "at": asyncio.get_running_loop().time()}
    stats_log_interval_s = float(
        session.cfg.get("app.stats_log_interval_s", 10.0))
    stats_window_s = max(3.0, stats_log_interval_s)
    # `status` 的推送节奏。**必须周期性推** —— 否则播放时前端的帧号条不会动
    # ⚠️ 必须周期性推：只在控制消息后回 status 会让播放时的帧号条不动。
    # 4 Hz：人手感知足够，又不至于刷满控制通道。
    status_interval_s = max(0.05, float(
        session.cfg.get("render.status_interval_s", 0.25)))

    # 后端自己的吞吐统计（**与前端解耦**：前端算它看到的，后端记它渲染的）。
    # 前端 FPS 由 web/js 自己按帧到达间隔计算；后端指标走 stdout 周期日志。
    live_warn = IntervalLogger(interval_s=5.0)
    window = RollingWindow(window_s=stats_window_s)
    logger_ = IntervalLogger(interval_s=stats_log_interval_s)
    status_logger = IntervalLogger(interval_s=status_interval_s)

    async def produce() -> None:
        """渲染循环：**帧数上限 + 等待**语义。

        ```
        loop:
            t0 = now
            render()                      # 用最新参数，不读任何渲染缓存
            推进播放（若在播放）           # 渲染一帧 = 播放一帧
            打包推入队列（容量 1，满则丢旧）
            sleep(interval - elapsed)     # 上一轮没超时就等满间隔
        ```

        两个刻意的性质：

        1. **实际帧率 = min(帧数上限, 1/渲染耗时)**。渲染慢于间隔时自动降级为
           "渲染完立刻继续"，**绝不补帧、绝不排队**。
        2. **渲染一帧 = 播放一帧**。推进与渲染在同一处，二者不可能漂移。
        """
        loop = asyncio.get_running_loop()
        while not stop.is_set():
            t0 = loop.time()

            # live 模式下"没有新内容就不重渲"。
            # RDF 约 15 fps、渲染可到 45 fps，照常跑会把同一份内容渲染 3 次，
            # 白白占用 GPU —— 而 GPU 正是 RDF 的瓶颈。跳过后帧率会诚实地
            # 掉到内容更新率（15 fps），这是**正确**的读数。
            if not session.live_frame_pending():
                await asyncio.sleep(0.005)
                # 状态照常推（前端要看到"摄像头未就绪/等待中"）
                status_logger.interval_s = max(0.05, float(
                    session.cfg.get("render.status_interval_s", 0.25)))
                now = loop.time()
                if status_logger.due(now):
                    await sender.json({"type": "status",
                                       **_status_payload(session)})
                continue

            try:
                rgb, target, timings = await asyncio.to_thread(session.render)
            except LiveUnavailable as e:
                # 兜底：正常情况下 live 无数据会回退到数据集首帧（不抛异常），
                # 走到这里说明是"连接了但取不到转换结果"这类异常情形。
                # **等一会儿再试，不要断开连接** —— 断开会让前端反复重连。
                if live_warn.due(loop.time()):
                    logger.info("实时驱动等待中：%s", e)
                await sender.json({"type": "error", "message": str(e),
                                   "transient": True})
                await asyncio.sleep(0.1)
                continue
            except Exception as e:  # noqa: BLE001
                logger.exception("渲染失败")
                await sender.json({"type": "error",
                                   "message": f"渲染失败：{type(e).__name__}: {e}"})
                break

            # 分辨率可能因 scale 变化而变；总帧数也可能因切换数据集而变。
            # 两者任一变化都要补发 config，否则前端帧号条会错。
            h, w = rgb.shape[0], rgb.shape[1]
            total = int(session.scene.total_frames) if session.scene else 0
            if ((w, h) != (app.state.stream_size or (0, 0))
                    or total != app.state.stream_total):
                app.state.stream_size = (w, h)
                app.state.stream_total = total
                await sender.json(config_message(
                    w, h, session.params.background,
                    session.scene.total_frames))

            # 播放推进：**渲染一帧 = 播放一帧**（由渲染循环负责，前端不参与）
            if (session.timeline is not None and session.timeline.playing
                    and not session.params.is_live):
                session.advance_frame()

            psnr = timings.get("psnr", float("nan"))
            # ---- 编码：raw 或 JPEG ----
            # 每帧从 cfg 读：设置面板改完立即生效，不必重连。
            # 编码放到线程里做（512² 约 0.7 ms），不占事件循环。
            use_jpeg = str(session.cfg.get("render.encode", "raw")).lower() == "jpeg"
            if use_jpeg:
                quality = int(session.cfg.get("render.encode_quality", 85))
                render_kind, target_kind = MSG_RENDER_JPEG, MSG_TARGET_JPEG
                blob = await asyncio.to_thread(_encode_jpeg, rgb, quality)
                if blob is None:                # 编码失败：退回 raw，别断流
                    use_jpeg = False
            if use_jpeg:
                frame = pack_blob(session.frames_served, blob,
                                  kind=render_kind, psnr=psnr)
            else:
                frame = pack_frame(session.frames_served, rgb, psnr=psnr)

            # 对比帧**只在变化时发**：它随驱动帧号变，而渲染可能 45 fps
            # 刷同一帧 —— 每帧都发会让数据量无谓翻倍。开关切换也要重发。
            target_key = None
            if target is not None:
                target_key = (session._current_index(),)
            if target is not None and target_key != app.state.target_key:
                app.state.target_key = target_key
                if use_jpeg:
                    t_blob = await asyncio.to_thread(
                        _encode_jpeg, target.numpy(), quality)
                if use_jpeg and t_blob is not None:
                    pair = (frame, pack_blob(session.frames_served, t_blob,
                                             kind=MSG_TARGET_JPEG, psnr=psnr))
                else:
                    pair = (frame, pack_frame(session.frames_served, target,
                                              kind=MSG_TARGET, psnr=psnr))
            else:
                if target is None:
                    app.state.target_key = None
                pair = (frame, None)

            # 只保留最新**一对**：满了就丢最旧的（容量 1 的队列即背压）。
            # 已被前端接收过的一律作废 —— 这是"视频流"而非"离线播放"的关键。
            while True:
                try:
                    queue.put_nowait(pair)
                    break
                except asyncio.QueueFull:
                    # 丢最旧的，并**计数**：客户端带宽不够时这里会持续触发，
                    # 是"服务端 55 FPS、前端只有 30"最直接的解释。
                    try:
                        queue.get_nowait()
                        session.frames_dropped += 1
                    except asyncio.QueueEmpty:  # pragma: no cover
                        pass

            # 周期性推 status：前端据此更新帧号条、耗时与显存
            # ⚠️ 每帧从 cfg 重读间隔：设置面板改了要**立刻**生效，
            #    而不是等下一次重建连接。
            status_logger.interval_s = max(0.05, float(
                session.cfg.get("render.status_interval_s", 0.25)))
            logger_.interval_s = max(1.0, float(
                session.cfg.get("app.stats_log_interval_s", 10.0)))
            now = loop.time()
            if status_logger.due(now):
                await sender.json({"type": "status",
                                   **_status_payload(session)})

            # 后端吞吐统计与周期日志
            window.add(now)
            if logger_.due(now):
                # `overhead` = 整帧耗时 − (deform + raster)。
                # 它应当是 0.1–0.5 ms 量级（to_thread 派发 + 打包 + 入队）。
                # 若显著偏大，说明**事件循环被别的东西占住**，而不是渲染慢 ——
                # 这是判断"要不要把某段搬到别的线程"的唯一依据。
                gpu_ms = session.last_deform_ms + session.last_rasterize_ms
                overhead_ms = session.last_frame_ms - gpu_ms
                now_t = loop.time()
                elapsed_s = max(0.5, now_t - sent["at"])
                mib_s = _throughput_mib_s(sent["total"], sent["logged"],
                                          elapsed_s)
                sent["logged"] = sent["total"]
                sent["at"] = now_t
                sent_mib = sent["total"] / 2 ** 20
                logger.info(
                    "[stats] %.1fs  渲染 %d 帧  平均 %.1f FPS  平均 %.2f ms/帧"
                    "（deform %.2f + raster %.2f + 其它 %.2f）"
                    "发送 %.1f MiB/s（累计 %.0f MiB）峰值显存 %.0f MiB  丢帧 %d",
                    stats_log_interval_s, session.frames_rendered,
                    window.rate(now), session.last_frame_ms,
                    session.last_deform_ms, session.last_rasterize_ms,
                    overhead_ms, mib_s, sent_mib,
                    session.peak_memory_mib, session.frames_dropped)

            # 帧数上限：上一轮没超时就等满间隔；超时则立刻继续（不补帧）
            later = loop.time()
            delay = sleep_after(later - t0, frame_interval(session.params.target_fps))
            if delay > 0:
                await asyncio.sleep(delay)

    async def consume() -> None:
        """取最新帧发出去。发送慢时自然丢帧（队列容量 1）。"""
        while not stop.is_set():
            render_bytes, target_bytes = await queue.get()
            if not await sender.bytes(render_bytes):
                return                       # 已断开：让其它任务收手
            sent["total"] += len(render_bytes)
            # 对比帧紧跟在渲染帧之后：前端按类型分发，顺序保证同一帧号成对
            if target_bytes is not None and not await sender.bytes(target_bytes):
                return
            sent["total"] += len(target_bytes or b"")
            session.frames_served += 1

    async def control() -> None:
        """收控制消息。与 produce/consume 并发跑。"""
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("收到非 JSON 控制消息，已忽略")
                continue
            await _handle_control(sender, session, msg)

    tasks = [asyncio.create_task(f()) for f in (produce, consume, control)]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        for t in done:
            with contextlib.suppress(WebSocketDisconnect, asyncio.CancelledError):
                t.result()
    except WebSocketDisconnect:
        pass
    finally:
        stop.set()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("客户端断开")


class _Sender:
    """串行化对同一个 WebSocket 的所有写。

    ## 为什么必须有它

    Starlette/websockets 的 `send` **不支持并发调用**。连接建立后我们同时跑三个
    任务（渲染推流 / 帧发送 / 控制消息），它们都可能写：
    底层的 `_drain_helper` 用单个 waiter 表示"正在等下一位"，两个协程同时进来
    就触发 `assert waiter is None or waiter.cancelled()` —— 连接随即被强制关闭。

    表现就是"点一下界面就断开"：控制消息要回 status，恰好撞上正在发帧。
    这把锁是**唯一**的写入入口，不要绕过它直接 `ws.send_*`。

    写失败（客户端已断开）时置 `closed` 并返回 ``False``，调用方据此收手，
    不必各自 try/except。
    """

    def __init__(self, ws: WebSocket) -> None:
        self._ws = ws
        self._lock = asyncio.Lock()
        self.closed = False

    async def text(self, payload: str) -> bool:
        async with self._lock:
            if self.closed:
                return False
            try:
                await self._ws.send_text(payload)
                return True
            except Exception:                                   # noqa: BLE001
                self.closed = True
                return False

    async def bytes(self, payload: bytes) -> bool:
        async with self._lock:
            if self.closed:
                return False
            try:
                await self._ws.send_bytes(payload)
                return True
            except Exception:                                   # noqa: BLE001
                self.closed = True
                return False

    async def json(self, payload: dict) -> bool:
        return await self.text(json.dumps(payload, ensure_ascii=False))


async def _handle_control(sender: "_Sender", session: Session,
                         msg: dict) -> None:
    kind = msg.get("type")

    if kind == "params":
        before = session.params.source_kind
        try:
            session.params.merge(msg)
        except (ValueError, TypeError) as e:
            await sender.json({"type": "error", "message": f"参数非法：{e}"})
            return

        # ---- 驱动方式变了：启停追踪线程 ----
        if session.params.source_kind != before:
            await asyncio.to_thread(session.sync_drive_mode)
            await _send_config(sender, session)

        # `frame` / `playing` 由 timeline 权威保存，params 只是镜像
        if session.timeline is not None:
            drive = msg.get("drive") or {}
            if "frame" in drive:
                session.jump_frame(int(drive["frame"]))
            if "playing" in drive:
                session.set_playing(bool(drive["playing"]))
        await _send_status(sender, session)

    elif kind == "step":
        # 单帧前进/后退。用**独立的 op**（而不是 frame=frame+delta），
        # 这样前端不必知道当前帧号，也不会因丢帧而算错。
        delta = int(msg.get("delta", 1))
        index = session.step_frame(delta)
        await _send_status(sender, session)
        logger.debug("单步 delta=%+d → 第 %d 帧", delta, index)

    elif kind == "stream":
        session.paused = bool(msg.get("paused"))

    elif kind == "capture":
        await _capture(sender, session, msg.get("name"))

    else:
        logger.debug("未知控制消息 type=%r，已忽略", kind)


async def _send_config(sender: "_Sender", session: Session) -> None:
    """补发一条 config：分辨率或**总帧数**变了，前端据此重排帧号条。"""
    if session.scene is None:
        return
    w, h = session.output_size()
    await sender.json(config_message(
        w, h, session.params.background, session.scene.total_frames))


async def _send_status(sender: "_Sender", session: Session) -> None:
    """回一条 status，让前端立即看到 frame / playing 的权威值。"""
    await sender.json({"type": "status", **_status_payload(session)})


async def _capture(sender: "_Sender", session: Session,
                   name: str | None) -> None:
    """把当前帧以无损 PNG 落到 output/gui_captures/。"""
    import time

    from PIL import Image

    out_dir = Path(session.cfg.paths.output_dir) / "gui_captures"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = name or time.strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"{stem}.png"

    rgb, _target, _timings = await asyncio.to_thread(session.render)
    await asyncio.to_thread(Image.fromarray(rgb.numpy()).save, path)
    logger.info("已截取：%s", path)
    await sender.json({"type": "captured", "path": str(path)})


def _throughput_mib_s(total_bytes: int, logged_bytes: int,
                      elapsed_s: float) -> float:
    """区间吞吐（MiB/s）。用**增量**，不是累计值。

    参数里的两个数都是累计字节数；`elapsed_s` 是距上次统计的真实耗时
    （而不是标称间隔 —— 渲染循环可能被拖慢，标称间隔会算偏）。
    新连接会把累计值清零，所以增量可能为负，按 0 处理。
    """
    if elapsed_s <= 0:
        return 0.0
    delta = max(0, int(total_bytes) - int(logged_bytes))
    return delta / 2 ** 20 / elapsed_s


_encode_warned = False


def _encode_jpeg(rgb, quality: int):
    """JPEG 编码；失败返回 ``None``（调用方回退 raw）。

    单独包一层是为了**只警告一次**：Pillow 缺失或图像异常时若每帧都打日志，
    终端会被刷满，反而看不出真正的问题。
    """
    global _encode_warned
    try:
        from ..streaming import encode_jpeg

        return encode_jpeg(rgb.numpy() if hasattr(rgb, "numpy") else rgb, quality)
    except Exception as e:                                      # noqa: BLE001
        if not _encode_warned:
            logger.warning("JPEG 编码不可用，回退 raw：%s: %s", type(e).__name__, e)
            _encode_warned = True
        return None


def _status_payload(session: Session) -> dict:
    return {
        "seq": session.frames_served,
        # ⚠️ 刻意**不含 fps**：前端 FPS 由前端按帧到达间隔自行计算
        #    （那反映"对方实际看到的画面"）；后端指标走 stdout 周期日志。
        #    此前这里塞了 `fps: 0.0` 并注明"由前端计算"，但前端并未实现，
        #    导致顶栏恒显示 0 —— 两端都不要为对方负责。
        "ms": {
            "frame": round(session.last_frame_ms, 3),
            "deform": round(session.last_deform_ms, 3),
            "rasterize": round(session.last_rasterize_ms, 3),
        },
        "dropped": session.frames_dropped,
        "peak_memory_mib": round(session.peak_memory_mib, 1),
        "paused": getattr(session, "paused", False),
        # 播放状态与当前帧号由**后端**权威给出
        "frame": session.timeline.index if session.timeline else 0,
        "playing": session.timeline.playing if session.timeline else False,
        "total_frames": session.scene.total_frames if session.scene else 0,
        # 实时驱动的**紧凑**状态：4 Hz 推送，只放前端要显示的四项。
        # 完整状态（含 identity / 顶点数等）走 `/api/state`。
        "live": _live_brief(session),
    }


def _live_brief(session: Session) -> dict:
    return session.live_brief()


# --------------------------------------------------------------- 应用装配 --


class NoCacheStaticFiles(StaticFiles):
    """静态文件**禁用缓存**。

    开发期频繁改 `web/` 下的前端，而 `StaticFiles` 只发 `ETag` /
    `Last-Modified`、**不发 `Cache-Control`**，浏览器会走启发式缓存 ——
    表现为「改了前端却看不到变化」。宁可每次多传几十 KB。
    """

    def file_response(self, *args, **kwargs):        # type: ignore[override]
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        return response


#: 这些 host 视为"本机回环"。它们会改用 dual-stack socket 监听，
#: 使得 `localhost` / `127.0.0.1` / `::1` 三种写法都能访问。
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "", "::"}


def make_dualstack_socket(port: int, backlog: int = 128) -> socket.socket:
    """建一个**同时接受 IPv4 与 IPv6** 的本机回环监听 socket。

    为什么需要它：单 host 只能覆盖一个地址族 —— 实测本机

    ==================  ==================  ==================
    host                127.0.0.1           ::1
    ==================  ==================  ==================
    ``127.0.0.1``       ✓                   ✗
    ``localhost``       ✓                   ✗
    ``::``              ✗（v6only=1）        ✓
    ==================  ==================  ==================

    而 `localhost` 在 /etc/hosts 里**优先解析为 `::1`**，浏览器打开
    `http://localhost:8000` 会先试 `::1` —— 只绑 IPv4 时 Windows 浏览器
    报「无法连接」，而 VS Code 内置浏览器（走 127.0.0.1）却正常。

    做法：绑 `::` 并显式关闭 `IPV6_V6ONLY`，得到一个 dual-stack socket。
    此时 IPv4 连接以 v4-mapped 地址到达，两种写法都通。
    """
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
    except OSError as e:                       # pragma: no cover - 平台差异
        sock.close()
        raise RuntimeError(f"该平台不支持关闭 IPV6_V6ONLY：{e}") from e
    sock.bind(("::", port))
    sock.listen(backlog)
    sock.set_inheritable(True)
    return sock


def create_app(cfg: Config | None = None, session: Session | None = None) -> FastAPI:
    """构建应用。`session` 可注入（测试用假 session）。"""
    cfg = cfg or load_config()
    session = session or Session(cfg)

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        """应用启停时管理实时驱动的连接。

        **为什么在这里而不是 `Session.__init__`**：构造 `Session` 的场合很多
        （脚本、测试），不该顺手起一个后台线程去连网络。只有真正跑服务时才连。

        启动时不无条件起线程：默认是数据集模式，追踪服务可能压根没跑。
        由 `sync_drive_mode()` 按当前模式决定。
        """
        session.sync_drive_mode()
        try:
            yield
        finally:
            session.live.stop()

    app = FastAPI(title="Live3DGSAvatar GUI", docs_url=None, redoc_url=None,
                  lifespan=lifespan)
    app.state.session = session
    app.state.stream_task = None
    app.state.stream_size = None
    #: 上次广播给前端的**总帧数**（切换数据集后要重发 config）
    app.state.stream_total = None
    #: 上次发送的对比帧键（帧号）；None 表示还没发过或已关闭对比
    app.state.target_key = None

    _install_routes(app, session)

    if WEB_DIR.is_dir():
        # 静态前端。挂在最后，避免 "/" 抢走 /api 与 /ws。
        app.mount("/", NoCacheStaticFiles(directory=str(WEB_DIR), html=True),
                  name="web")
    else:
        @app.get("/")
        async def missing_web() -> JSONResponse:  # pragma: no cover
            return JSONResponse(
                {"error": f"前端目录不存在：{WEB_DIR}"}, status_code=500)
    return app


# ------------------------------------------------------------------- CLI --


def _pick_startup_model(cfg) -> tuple[str, str] | None:
    """决定启动时加载哪个模型。

    规则：配置里写了的优先；否则用列表里的**第一个**可用模型 ——
    界面上下拉框默认也选第一个，两者保持一致，避免"界面显示 crh_1、
    后端却没加载"的割裂。

    返回 `(subject, work_name)`；一个都没有时返回 ``None``（允许空启动）。
    """
    from ..data.models import list_models

    if model_selected(cfg):
        try:
            resolve_model_ply(cfg)
            return str(cfg.get("subject")), str(cfg.get("work_name"))
        except (FileNotFoundError, ValueError) as e:
            logger.warning("配置里的模型不可用（%s），改用列表里的第一个", e)

    infos = list_models(cfg)
    if not infos:
        return None
    first = infos[0]
    return first.subject, first.work_name


def _missing_flame_assets(cfg) -> list[str]:
    """启动自检：FLAME 资产是否齐备。返回缺失的文件名（空 = 没问题）。

    只查这一项：模型与数据集都可能还没有 —— 那不影响服务启动，
    用户可以先打开界面再选。但缺 FLAME 时渲染一定失败，提前拦下更省事。
    """
    from ..config import repo_root
    from ..data.flame_topology import asset_paths

    explicit = cfg.get("paths.flame_dir")
    flame_dir = Path(str(explicit)) if explicit else repo_root() / "data" / "FLAME2020"
    return [p.name for p in asset_paths(flame_dir) if not p.is_file()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """命令行 = 对 `configs/*.yaml` 的覆盖层（见 docs/CONFIG.md）。"""
    p = argparse.ArgumentParser(
        prog="python -m live3dgsavatar.app",
        description="Live3DGSAvatar GUI 服务（浏览器实时预览）",
        epilog="默认值来自 configs/system.yaml 的 app 节；"
               "用 python scripts/show_config.py 查看实际生效值。")
    p.add_argument("--config-dir", type=Path, default=None)
    p.add_argument("--host", default=None, help="监听地址（默认 app.host）")
    p.add_argument("--port", type=int, default=None, help="端口（默认 app.port）")
    p.add_argument("--subject", default=None)
    p.add_argument("--work-name", default=None)
    p.add_argument("--target-fps", type=float, default=None)
    p.add_argument("--no-preload-model", action="store_true",
                   help="启动时不加载模型（之后在界面里选）")
    p.add_argument("--reload", action="store_true",
                   help="代码变更自动重载（开发用；需 uvicorn[standard]）")
    p.add_argument("--log-level", default="info")
    return p.parse_args(argv if argv is not None else sys.argv[1:])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S")

    cfg = load_config(args.config_dir)
    if args.subject is not None:
        cfg.set("subject", args.subject)
    if args.work_name is not None:
        cfg.set("work_name", args.work_name)
    if args.target_fps is not None:
        cfg.set("render.target_fps", args.target_fps)

    host = args.host or str(cfg.get("app.host", "127.0.0.1"))
    port = args.port or int(cfg.get("app.port", 8000))

    # 启动前只检查**必要资产**（FLAME）：模型与数据集允许为空 ——
    # 用户可以先起服务，再在界面里选。缺 FLAME 则渲染必然失败，直接拦下。
    missing = _missing_flame_assets(cfg)
    if missing:
        logger.error("缺少 FLAME 资产：%s", "、".join(missing))
        logger.error("请把 generic_model.pkl / flame_uv.npz 放到 %s",
                     cfg.get("paths.flame_dir") or "data/FLAME2020/")
        return 1

    session = Session(cfg)
    if not args.no_preload_model:
        chosen = _pick_startup_model(cfg)
        if chosen is not None:
            subject, work_name = chosen
            logger.info("加载模型：%s/%s", subject, work_name)
            try:
                session.load_model(subject, work_name)
                logger.info("就绪：%s", session.describe())
            except Exception as e:  # noqa: BLE001
                # **不退出**：用户可以在界面里换一个
                logger.warning("启动时加载模型失败（可在界面里重新选择）：%s: %s",
                               type(e).__name__, e)
        else:
            logger.info("没有可用模型：请在浏览器里选择后点「加载模型」")

    app = create_app(cfg, session)

    import uvicorn

    logger.info("前端：http://localhost:%d", port)
    logger.info("（也可用 http://127.0.0.1:%d）", port)

    config = uvicorn.Config(app, host=host, port=port,
                            log_level=args.log_level, reload=False,
                            access_log=False)

    sockets = None
    if host in _LOOPBACK_HOSTS:
        # 本机回环：用 dual-stack socket，让 localhost / 127.0.0.1 / ::1 都能访问。
        try:
            sockets = [make_dualstack_socket(port)]
            logger.info("监听：IPv4 127.0.0.1 + IPv6 ::1（dual-stack，端口 %d）", port)
        except OSError as e:
            logger.warning("dual-stack 监听失败（%s），回退到 host=%s", e, host)
            sockets = None
    else:
        logger.warning("监听在非回环地址 %s —— 本服务无鉴权，仅限可信网络", host)

    server = uvicorn.Server(config)
    if sockets is not None:
        server.run(sockets=sockets)
    else:
        server.run()
    return 0
