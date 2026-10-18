"""把 RDF 观测变成渲染输入：`(mesh, blend_weight)`。

夹在 `rdf_client`（收帧）与 `app/session`（渲染）之间，做三件事：

1. **补牙齿**：5023 → 5083（`mesh_build`）
2. **参数转换**：118 → 129（`param_map`）
3. **缓存**：同一 `seq` 只转换一次

第 3 点不是过早优化：渲染可能 60–175 fps，而 RDF 只有约 30 fps，
不做缓存的话每渲染一帧都要重算一次牙齿的 Kabsch（约 1 ms）。
缓存键用 `seq` 而不是时间戳 —— 序号单调且由发送方保证唯一。
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from ..config import Config, repo_root
from . import mesh_build, param_map
from .rdf_client import Observation, RdfClient

log = logging.getLogger(__name__)


class LiveUnavailable(RuntimeError):
    """live 数据源暂时没有可用观测。

    单独一个异常类型（而不是泛泛的 `RuntimeError`）是为了让渲染循环能区分
    "暂时没帧、等一会儿就好"与"真的渲染失败"：前者不该断开连接。
    """


class LiveSource:
    """实时驱动源。参数与顶点都来自 `RdfClient`。

    设备约定：返回的张量已经在本类的 `device` 上，调用方不必再搬。

    Args:
        cfg: 配置（读 `tracking.*`）
        device: 目标设备（与 `Session.device` 一致）
        client: 可注入的客户端（测试用 `FakeClient`）
    """

    def __init__(self, cfg: Config, device: str, *,
                 client: RdfClient | None = None) -> None:
        self.cfg = cfg
        self.device = device

        self.enabled = bool(cfg.get("tracking.enabled", False))
        self.max_age_s = float(cfg.get("tracking.max_age_s", 1.0))

        self.client = client or RdfClient(
            host=str(cfg.get("tracking.host", "127.0.0.1")),
            port=int(cfg.get("tracking.port", 8980)))

        self._teeth: mesh_build.TeethBuilder | None = None
        self._cache_seq: int | None = None
        self._cache: tuple[torch.Tensor, torch.Tensor] | None = None

    # ------------------------------------------------------------- 生命周期 --

    def start(self) -> None:
        if self.enabled:
            self.client.start()

    def stop(self) -> None:
        self.client.stop()

    # --------------------------------------------------------------- 取输入 --

    @property
    def available(self) -> bool:
        """是否有一帧足够新的观测。"""
        return self.enabled and self.client.latest(self.max_age_s) is not None

    def current_seq(self) -> int | None:
        """当前可用观测的序号（**不触发转换**）。

        渲染循环用它判断"有没有新内容"，比 `observation()` 便宜得多
        （后者要做补牙齿 + 参数转换）。
        """
        if not self.enabled:
            return None
        obs = self.client.latest(self.max_age_s)
        return None if obs is None else obs.seq

    def observation(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """返回 `(mesh [1,V,3], blend_weight [1,129])`，无可用帧时返回 ``None``。

        同一 `seq` 的结果会被缓存复用。
        """
        if not self.enabled:
            return None
        obs = self.client.latest(self.max_age_s)
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

    # ----------------------------------------------------------------- 状态 --

    def status(self) -> dict:
        info = self.client.status()
        return {
            "enabled": self.enabled,
            "available": self.available,
            "max_age_s": self.max_age_s,
            **info,
        }
