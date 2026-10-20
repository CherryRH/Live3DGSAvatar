"""实时驱动源：把「取帧 → 追踪服务 → 渲染输入」串起来。

## 数据流

```
取帧器（数据集 input/ 或将来摄像头）
      │  一帧 RGB
      ▼
RdfClient.request()  ──►  追踪服务（另一个进程）
      │  参数 x[118] + 顶点 verts[5023,3]
      ▼
补牙齿（5023 → 5083）+ 参数转换（118 → 129）
      │
      ▼
渲染线程取走：`observation()` → (mesh, blend_weight)
```

## 为什么取帧要单独一个线程

追踪一次约 60–70 ms，而渲染可能 5–10 ms。若在渲染线程里做请求，
渲染帧率会被追踪速度直接拖死。所以这里用一个 daemon 线程循环"取帧→请求"，
只把**最新结果**交出去；渲染线程只做一次加锁读取，永不阻塞。

## 背压：只保留最新

与推流的帧通道同一策略：只留最新一帧。追踪慢于取帧时不会积压，
代价是跳帧 —— 比"延迟越积越大"好得多。

## 播放/暂停/跳帧（与数据集模式统一）

取帧是**按帧号**的（`read_at`），帧号由本类持有，规则与数据集模式一致：

| 操作 | 行为 |
|---|---|
| 播放 | 每次追踪成功后帧号 +1；**渲染一帧 = 追踪一帧**（因此播放速度=追踪速度）|
| 暂停 | 帧号不动，**重复送同一帧**（保持观测新鲜），序号不变 → 渲染不重算 |
| 跳帧 | `seek(i)` 立即生效，下一轮就送第 i 帧 |

暂停时为什么不干脆停掉请求：观测有 `max_age` 过期判定，停了就会"看不见"，
画面会回退到首帧。所以保持低频重送（过半衰期才重送一次，约 2 次/秒），
既不会过期，也不至于让追踪满负荷空转。
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import torch

from ..config import Config, repo_root
from . import mesh_build, param_map
from .rdf_client import Observation, RdfClient

log = logging.getLogger(__name__)

#: 连续失败到这个次数才打日志，避免服务没起时刷屏
_WARN_EVERY = 20


class LiveUnavailable(RuntimeError):
    """live 数据源暂时没有可用观测。

    单独一个异常类型（而不是泛泛的 `RuntimeError`）是为了让渲染循环能区分
    "暂时没帧、等一会儿就好"与"真的渲染失败"：前者不该断开连接。
    """


class LiveSource:
    """实时驱动源：后台线程取帧并请求追踪，渲染线程取最新结果。

    Args:
        cfg: 配置（读 `tracking.*`）
        device: 目标设备（与 `Session.device` 一致）
        client: 可注入的客户端（测试用假客户端）
        frames: 可注入的取帧器；``None`` 时由 `set_frames()` 提供
    """

    def __init__(self, cfg: Config, device: str, *,
                 client: RdfClient | None = None,
                 frames=None) -> None:
        self.cfg = cfg
        self.device = device

        self.enabled = bool(cfg.get("tracking.enabled", False))
        self.max_age_s = float(cfg.get("tracking.max_age_s", 1.0))

        self.client = client or RdfClient(
            host=str(cfg.get("tracking.host", "127.0.0.1")),
            port=int(cfg.get("tracking.port", 8980)))
        self.frames = frames

        self._teeth: mesh_build.TeethBuilder | None = None
        self._cache_seq: int | None = None
        self._cache: tuple[torch.Tensor, torch.Tensor] | None = None

        self._lock = threading.Lock()
        self._latest: Observation | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._failures = 0
        #: 当前目标帧号（由 `seek()` / 播放推进；播放条与它一一对应）
        self._index = 0
        #: 是否自动推进帧号
        self._playing = False
        #: 帧号变化回调（Session 用它同步 timeline 与 params.frame）
        self.on_index = None

    # --------------------------------------------------------------- 配置 --

    def set_frames(self, frames) -> None:
        """设置取帧器（切换驱动数据集时调用）。"""
        self.frames = frames

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def set_playing(self, playing: bool) -> None:
        """播放 / 暂停。暂停时帧号停住，但仍以低频重送同一帧。"""
        self._playing = bool(playing)

    def seek(self, index: int) -> None:
        """跳到指定帧。下一次取帧立刻用它。"""
        with self._lock:
            self._index = max(0, int(index))

    @property
    def index(self) -> int:
        with self._lock:
            return self._index

    def _advance(self) -> None:
        """播放推进一帧（到头回绕），并通知上层。"""
        frames = self.frames
        total = int(getattr(frames, "num_frames", 0)) or 1
        with self._lock:
            self._index = (self._index + 1) % total
            index = self._index
        if self.on_index is not None:
            self.on_index(index)

    # ----------------------------------------------------------- 生命周期 --

    def start(self) -> None:
        """启动取帧线程（幂等）。没有取帧器时不做任何事。"""
        if not self.enabled or self.frames is None:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="rdf-pull",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=3.0)
        self._thread = None
        if self.client.connected:
            self.client.close()
        with self._lock:
            self._latest = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # --------------------------------------------------------------- 取帧 --

    def _run(self) -> None:
        """取帧线程主体：按帧号取图 → 请求追踪 → 存为最新。"""
        last_index: int | None = None
        last_obs: Observation | None = None
        while not self._stop.is_set():
            frames = self.frames
            if frames is None:
                break

            index = self.index
            # 暂停且观测还新鲜 → 不发请求（省 GPU）。过半衰期才重送一次，
            # 否则观测会过期，画面会掉回首帧。
            if (not self._playing and index == last_index
                    and last_obs is not None
                    and last_obs.age_s() < self.max_age_s * 0.5):
                self._stop.wait(0.02)
                continue

            try:
                rgb = frames.read_at(index)
            except Exception as e:                                  # noqa: BLE001
                log.warning("取帧失败：%s", e)
                rgb = None
            if rgb is None:
                # 不循环且读完了：结束（保持最后一帧，不报错）
                log.info("取帧结束，停止请求追踪")
                break

            try:
                # 序号用**帧号**：暂停时重复送同一帧 → 序号不变 → 渲染不重算。
                obs = self.client.request(rgb, seq=index)
            except ConnectionError as e:
                self._failures += 1
                if self._failures % _WARN_EVERY == 1:
                    log.info("追踪服务不可用（第 %d 次）：%s",
                             self._failures, e)
                # 退避，别把 CPU 打满
                self._stop.wait(min(0.2 * self._failures, 2.0))
                continue

            self._failures = 0
            last_index = index
            last_obs = obs
            self.ingest(obs)
            if self._playing:
                self._advance()

    def ingest(self, obs: Observation) -> None:
        """把一帧观测放进"待渲染槽"（只保留最新）。

        取帧线程调它；单元测试也直接调它 —— 转换逻辑不该只有起线程才能测。
        """
        with self._lock:
            self._latest = obs

    def _fresh(self) -> Observation | None:
        with self._lock:
            obs = self._latest
        if obs is None:
            return None
        if obs.age_s() > self.max_age_s:
            return None
        return obs

    # -------------------------------------------------------------- 取输入 --

    @property
    def available(self) -> bool:
        """是否有一帧足够新的观测。"""
        return self.enabled and self._fresh() is not None

    def current_seq(self) -> int | None:
        """当前可用观测的序号（**不触发转换**）。

        渲染循环用它判断"有没有新内容"，比 `observation()` 便宜得多
        （后者要做补牙齿 + 参数转换）。
        """
        if not self.enabled:
            return None
        obs = self._fresh()
        return None if obs is None else obs.seq

    def observation(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """返回 `(mesh [1,V,3], blend_weight [1,129])`，无可用帧时返回 ``None``。

        同一 `seq` 的结果会被缓存复用。
        """
        if not self.enabled:
            return None
        obs = self._fresh()
        if obs is None:
            return None
        if obs.seq == self._cache_seq and self._cache is not None:
            return self._cache

        mesh, weight = self._convert(obs)
        self._cache_seq = obs.seq
        self._cache = (mesh, weight)
        return mesh, weight

    def _convert(self, obs: Observation) -> tuple[torch.Tensor, torch.Tensor]:
        builder = self._teeth_builder()
        base = torch.from_numpy(obs.verts).to(torch.float32)
        full = builder.build(base)                       # [V,3]
        x = torch.from_numpy(obs.x).to(torch.float32)
        weight = param_map.to_blend_weight(x)            # [129]

        mesh = full.unsqueeze(0).to(self.device)
        weight = weight.unsqueeze(0).to(self.device)
        return mesh, weight

    def _teeth_builder(self) -> mesh_build.TeethBuilder:
        if self._teeth is None:
            self._teeth = mesh_build.TeethBuilder(self._flame_dir())
        return self._teeth

    def _flame_dir(self) -> Path:
        explicit = self.cfg.get("paths.flame_dir")
        return Path(str(explicit)) if explicit else repo_root() / "data" / "FLAME2020"

    # ---------------------------------------------------------------- 状态 --

    def status(self) -> dict:
        info = self.client.status()
        obs = self._fresh()
        return {
            "enabled": self.enabled,
            "running": self.running,
            "available": self.available,
            "playing": self._playing,
            "frame": self.index,
            "max_age_s": self.max_age_s,
            "age_s": round(obs.age_s(), 3) if obs else None,
            **info,
        }
