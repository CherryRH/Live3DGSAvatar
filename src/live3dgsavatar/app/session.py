"""渲染会话：持有模型与数据集，负责"按当前参数出一帧"。

**这是 GUI 后端里唯一接触 GPU 的部分**，也是唯一接触 `core/` 的部分。
HTTP/WebSocket 层只做消息编解码，不碰张量 —— 便于把并发问题挡在外面。

设计要点：

- **单线程假设**：所有方法都在同一个线程（uvicorn 的事件循环线程）里调用。
  PyTorch 在事件循环线程里跑 GPU 前向是安全的：kernel 提交会释放 GIL，
  服务器仍能响应其它请求。**不做多线程渲染** —— 那会引入 CUDA 上下文竞态，
  而收益只是把已经够快的 167 FPS 再压一点。
- **参数更新只改状态**，真正的渲染发生在出帧时。避免"改一个滑块就重渲一次"。
- **分辨率缩放**在渲染后用 Pillow 缩，而不是让光栅化器输出小图：
  后者需要改相机内参，容易与数据集内参打架。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..config import Config, model_dir, resolve_model_ply

# ⚠️ 直接从 `..core.types` 导入 `Mesh`。若写成 `import ..core.types as types`
#    会遮蔽标准库的 `types` 模块，间接依赖随之行为异常且极难排查。
from ..core.types import Mesh
from .protocol import FrameTimeline, merge_params, output_size


@dataclass
class RenderParams:
    """前端可实时调整的参数（对应 docs/GUI_PROTOCOL.md §3.2 的 `params`）。

    校验与合并逻辑放在 `protocol.merge_params` —— 那是**纯函数**，
    没有 GPU、没有 Web 依赖，可以直接单元测试。
    """

    background: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    scaling_modifier: float = 1.0
    scale: float = 1.0          # 输出分辨率比例（1.0 = 原生）
    target_fps: float = 60.0    # **帧数上限**；0 表示不限
    source_kind: str = "checkpoint"  # checkpoint | live
    frame: int = 0
    #: 是否推进帧号。**暂停不代表停止渲染**，只是帧号停住。
    playing: bool = False
    #: 调试：以当前帧为基准加随机扰动（探查训练分布外行为）。
    perturb: bool = False
    #: `perturb` 的幅度。
    amplitude: float = 0.1

    def as_dict(self) -> dict:
        return {
            "background": list(self.background),
            "scaling_modifier": self.scaling_modifier,
            "scale": self.scale,
            "target_fps": self.target_fps,
            "source_kind": self.source_kind,
            "frame": self.frame,
            "amplitude": self.amplitude,
            "playing": self.playing,
        }

    def merge(self, msg: dict) -> None:
        """按字段合并一条 `params` 消息；非法值抛 `ValueError`。"""
        merged = merge_params(self.as_dict(), msg)
        for key, value in merged.items():
            setattr(self, key, value)

    def to_dict(self) -> dict:
        """给前端的 `render` 子集。

        含 `source_kind`：否则刷新页面后数据源会退回"数据集采样"，
        而实际后端还在按 live 驱动 —— 界面与状态不一致。
        """
        return {
            "background": list(self.background),
            "scaling_modifier": self.scaling_modifier,
            "scale": self.scale,
            "target_fps": self.target_fps,
            "source_kind": self.source_kind,
        }

    def to_drive_dict(self) -> dict:
        """给前端的 `drive` 子集（含播放状态与当前帧号）。"""
        return {
            "frame": self.frame,
            "playing": self.playing,
            "perturb": self.perturb,
            "amplitude": self.amplitude,
        }

    @property
    def is_live(self) -> bool:
        """`live` 数据源：实时 tracking，**待接入**。"""
        return self.source_kind == "live"


class Session:
    """一个渲染会话：模型 + 数据集 + 当前参数。

    生命周期：进程内**单例**。切换模型走 `load_model()`，不做多会话。
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.params = RenderParams(
            background=[float(x) for x in cfg.get("render.background", [0, 0, 0])],
            scaling_modifier=float(cfg.get("render.scaling_modifier", 1.0)),
            scale=float(cfg.get("render.scale", 1.0)),
            target_fps=float(cfg.get("render.target_fps", 60)),
        )
        # 运行设备：模型/数据集/相机/背景必须**全在同一个设备上**，
        # 否则 `mesh.verts[:, mesh.faces[...]]` 这类索引会抛设备不匹配。
        self.device = str(cfg.get("runtime.device", "cuda"))
        self.avatar = None
        self.scene = None
        # 帧号推进状态（播放/单步）。total 在模型载入后确定。
        self.timeline: FrameTimeline | None = None
        # 实时驱动源（live 数据源）。**构造即创建**，但不 enabled 时不连。
        self.live = None            # type: ignore[assignment]
        self.camera = None
        self._rasterizer = None
        self._mesh_cache: dict[int, object] = {}
        #: 最近一次 `_frame_inputs()` 用的内容键（用于"没变就不重渲"）
        self._last_input_key: tuple | None = None
        #: live 模式下当前是否在显示**回退帧**（摄像头未就绪）
        self.live_fallback = False

        from ..tracking.live_source import LiveSource  # noqa: PLC0415

        self.live = LiveSource(cfg, self.device)
        # 统计（供 status 消息）
        self.peak_memory_mib = 0.0
        self.last_frame_ms = 0.0
        self.last_deform_ms = 0.0
        self.last_rasterize_ms = 0.0
        self.frames_rendered = 0
        self.frames_served = 0
        self.frames_dropped = 0
        self.model_path: Path | None = None

    # ------------------------------------------------------------ 加载 --

    def load_model(self, subject: str | None = None,
                   work_name: str | None = None) -> dict:
        """载入模型与数据集。切换模型时调用。

        Raises:
            FileNotFoundError: 模型或数据集不存在（消息里给出解析路径）
            RuntimeError: 需要 GPU 但不可用
        """
        if subject is not None:
            self.cfg.set("subject", subject)
        if work_name is not None:
            self.cfg.set("work_name", work_name)

        if not torch.cuda.is_available():
            raise RuntimeError(
                "GUI 后端需要可用的 NVIDIA GPU（渲染无法降级到 CPU）。"
                "请在能访问 GPU 的终端运行。")

        from ..core.avatar import AvatarConfig
        from ..core.io import load_ply
        from ..core.render import SimpleRasterizer
        from ..core.types import Camera
        from ..data.scene import load_scene

        cfg = self.cfg
        model_ply = resolve_model_ply(cfg)
        # 预载帧数：默认 -1（全部）。帧数据很小（254 帧约 15 MiB），
        # 载入全部才能让「数据集帧」滑块走满全程。
        # ⚠️ 不要改用 `render.frames`（那是渲染测试的采样量，会让滑块被夹住）。
        preload = int(cfg.get("app.preload_frames", -1))
        # ⚠️ 显式给出设备：`load_scene` 会把**帧、拓扑、相机**统一搬到该设备，
        #    下面的 `load_ply(device="cuda")` 与 `Camera.to("cuda")` 才与之一致。
        #    曾经这里漏传，`scene.faces` 留在 CPU，而 `mesh.verts` 在 GPU，
        #    `mesh.verts[:, mesh.faces[used]]` 抛设备不匹配错误。
        scene = load_scene(cfg, frames=preload, device=self.device)

        av_cfg = AvatarConfig(
            tex_size=int(cfg.get("model.network.tex_size")),
            num_basis_in=int(cfg.get("model.network.num_basis_in")),
            num_basis_blend=int(cfg.get("model.network.num_basis_blend")),
            mlp_hidden=tuple(cfg.get("model.network.mlp_hidden") or ()),
            use_weight_proj=bool(cfg.get("model.network.use_weight_proj", True)),
        )
        avatar = load_ply(model_ply, config=av_cfg, device=self.device)

        # 状态一致性：先算完再替换，避免中途失败留下半初始化状态
        self.avatar = avatar
        self.scene = scene
        self.model_path = model_ply
        self.camera = Camera.from_intrinsics_extrinsics(
            scene.K, scene.R, scene.T, scene.width, scene.height).to(self.device)
        self._rasterizer = SimpleRasterizer(
            sh_degree=int(cfg.get("render.sh_degree", 0)),
            scaling_modifier=self.params.scaling_modifier)
        self._mesh_cache.clear()
        self.timeline = FrameTimeline(scene.num_frames)
        self.params.frame = 0
        self._reset_counters()
        return self.state()

    def live_brief(self) -> dict:
        """给 `status` 的紧凑 live 状态。"""
        st = self.live.status()
        return {
            "enabled": st["enabled"],
            "connected": st["connected"],
            "available": st["available"],
            "fps": st["fps"],
            "age_s": st["age_s"],
            "max_age_s": st["max_age_s"],
            "fallback": self.live_fallback,
            "reason": (self._live_unavailable_reason()
                       if self.live_fallback else None),
        }

    def state(self) -> dict:
        """/api/state 的返回体。"""
        cfg = self.cfg
        out: dict = {
            "subject": str(cfg.subject),
            "work_name": str(cfg.work_name),
            "render": self.params.to_dict(),
            "drive": self.params.to_drive_dict(),
            "live": {**self.live.status(),
                     "fallback": self.live_fallback,
                     "reason": (self._live_unavailable_reason()
                                if self.live_fallback else None)},
        }
        if self.avatar is not None:
            out["model"] = {
                "path": str(self.model_path) if self.model_path else None,
                "num_gaussians": int(self.avatar.num_gaussians),
                "num_basis": int(self.avatar.num_basis),
                "num_basis_in": int(self.avatar.config.num_basis_in),
                "tex_size": int(self.avatar.config.tex_size),
            }
        if self.scene is not None:
            out["dataset"] = {
                "num_frames": self.scene.total_frames,
                "loaded_frames": self.scene.num_frames,
                "image_subdir": str(cfg.get("paths.image_subdir", "images")),
            }
        out["stats"] = {
            "frames_rendered": self.frames_rendered,
            "frames_served": self.frames_served,
            "frames_dropped": self.frames_dropped,
            "peak_memory_mib": self.peak_memory_mib,
        }
        return out

    # ------------------------------------------------------------ 出帧 --

    # ---------------------------------------------------------- 设置应用 --
    #: 设置项 → `RenderParams` 字段的映射。
    #: 只有这些**同时**写进 cfg 与运行参数，其余只改 cfg（下次读取时生效）。
    _RENDER_MIRROR = {
        "render.background": "background",
        "render.scaling_modifier": "scaling_modifier",
        "render.scale": "scale",
        "render.target_fps": "target_fps",
    }

    def apply_settings(self, patch: dict) -> list[str]:
        """把一批**已校验**的设置写进配置，并让可即时生效的项立刻起作用。

        调用方负责校验（`app.settings.validate_patch`）—— 这里不再重复。

        Args:
            patch: 点分路径 → 值（已转换类型）

        Returns:
            实际即时生效的键（供前端提示"哪些立刻变了"）
        """
        applied: list[str] = []
        for key, value in patch.items():
            self.cfg.set(key, value)
            field = self._RENDER_MIRROR.get(key)
            if field is not None:
                setattr(self.params, field,
                        list(value) if field == "background" else float(value))
                applied.append(key)
        # 背景色与缩放会在下一帧的 `render()` 里被读取；
        # `scaling_modifier` 由 `render()` 每帧同步到光栅化器，无需重建。
        return applied

    @property
    def ready(self) -> bool:
        return self.avatar is not None and self.scene is not None

    def output_size(self) -> tuple[int, int]:
        """当前参数下的输出分辨率（已应用 `scale`）。"""
        if self.scene is None:
            return (0, 0)
        return output_size(self.scene.width, self.scene.height, self.params.scale)

    def render(self) -> tuple[torch.Tensor, dict]:
        """按当前参数渲染一帧。

        Returns:
            `(rgb_uint8_hwc, timing)`；`rgb` 为 CPU 上的 `[H, W, 3]` uint8，
            即可以直接塞进协议帧的形状。
        """
        if not self.ready:
            raise RuntimeError("会话尚未加载模型，无法渲染")

        timings: dict[str, float] = {}
        t_all = time.perf_counter()

        mesh, blend_weight = self._frame_inputs()

        m = Mesh(verts=mesh, faces=self.scene.faces,
                 uvs=self.scene.uvs, uv_faces=self.scene.uv_faces)
        bg = torch.tensor(self.params.background, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            t0 = time.perf_counter()
            gs = self.avatar.deform(m, blend_weight)
            torch.cuda.synchronize()
            timings["deform"] = (time.perf_counter() - t0) * 1000.0

            # 相机与缩放系数可能在运行期改过
            self._rasterizer.scaling_modifier = self.params.scaling_modifier
            t0 = time.perf_counter()
            out = self._rasterizer.render(gs, self.camera, bg)
            torch.cuda.synchronize()
            timings["rasterize"] = (time.perf_counter() - t0) * 1000.0

        rgb = (out.color[0].clamp(0, 1).permute(1, 2, 0) * 255).byte().cpu()

        w, h = self.output_size()
        if (w, h) != (rgb.shape[1], rgb.shape[0]):
            rgb = self._downscale(rgb, w, h)

        timings["frame"] = (time.perf_counter() - t_all) * 1000.0
        self.last_deform_ms = timings["deform"]
        self.last_rasterize_ms = timings["rasterize"]
        self.last_frame_ms = timings["frame"]
        self.frames_rendered += 1
        self.peak_memory_mib = max(
            self.peak_memory_mib, torch.cuda.max_memory_allocated() / 2**20)
        return rgb, timings

    # ------------------------------------------------------------ 内部 --

    def _frame_inputs(self):
        """按当前驱动模式取出 `(mesh [1,V,3], blend_weight [1,D])`。

        `live` 与 `checkpoint` 的**唯一**差别就在这里：前者从桥接进程取
        最新观测，后者从数据集的当前帧取。下游（deform/rasterize）完全一样 ——
        这正是把驱动来源做成可替换层的目的。

        **live 没有数据时回退到数据集首帧**（而不是报错）：通话时摄像头
        可能未就绪或断流，此时让画面停在首帧比黑屏/断流合理。
        `self.live_fallback` 会置真，前端据此显示原因。
        """
        if self.params.is_live:
            obs = self.live.observation()
            if obs is not None:
                self.live_fallback = False
                # ⚠️ `observation()` 返回的是 `(mesh, weight)`，不是 Observation ——
                #    内容键要另外取序号。
                self._last_input_key = ("live", self.live.current_seq())
                return obs
            # 回退：停在数据集首帧
            self.live_fallback = True
            self._last_input_key = ("live-fallback",)
            return self._dataset_frame(0)

        assert self.timeline is not None
        idx = self.timeline.index
        self.live_fallback = False
        self._last_input_key = ("dataset", idx, self.params.perturb,
                                round(float(self.params.amplitude), 6))
        return self._dataset_frame(idx)

    def _dataset_frame(self, idx: int):
        """取数据集某帧的 `(mesh, blend_weight)`，并应用调试扰动。"""
        assert self.scene is not None
        idx = max(0, min(int(idx), self.scene.num_frames - 1))
        cached = self._mesh_cache.get(idx)
        if cached is None:
            frame = self.scene.frames[idx]
            cached = (
                frame["mesh"].unsqueeze(0),
                frame["blend_weight"].unsqueeze(0).clone(),
            )
            self._mesh_cache[idx] = cached
        mesh, weight = cached

        # `perturb` 是与驱动模式正交的调试开关。
        # **用帧号做种子**：同一帧号重复请求给出同一扰动，
        # 否则"审查"时画面会一直抖，无法判断是模型问题还是随机性。
        if self.params.perturb:
            g = torch.Generator(device=weight.device)
            g.manual_seed(0x5EED + idx)
            noise = torch.randn(weight.shape, generator=g,
                                device=weight.device, dtype=weight.dtype)
            weight = weight + self.params.amplitude * noise
        return mesh, weight

    def live_frame_pending(self) -> bool:
        """live 模式下是否有一帧**尚未渲染过**的内容。

        为什么需要它：RDF 约 15 fps、渲染可以 45 fps。若照常按帧数上限渲染，
        **同一份内容会被重复渲染 3 次**，白白占用 GPU —— 而 GPU 正是 RDF 的
        瓶颈，等于自己在拖慢自己。

        非 live 模式恒返回 ``True``：数据集模式保持原有语义
        （"暂停 ≠ 停止渲染"，链路始终是活的）。
        """
        if not self.params.is_live:
            return True
        seq = self.live.current_seq()
        key = ("live", seq) if seq is not None else ("live-fallback",)
        return key != self._last_input_key

    def _live_unavailable_reason(self) -> str:
        """live 不可用时的**可执行**提示（而不是只报"无数据"）。"""
        st = self.live.status()
        if not st["enabled"]:
            return "实时驱动已关闭（tracking.enabled = false）"
        if not st["connected"]:
            err = st.get("error")
            return ("未连接 RDF 桥接进程"
                    + (f"（{err}）" if err else "")
                    + "。请先运行：conda activate realdenseface && "
                      "python scripts/rdf_bridge.py --camera 0")
        return ("已连接，但最近 "
                f"{self.live.max_age_s:.1f}s 内没有新帧（RDF 可能卡住）")

    @staticmethod
    def _downscale(rgb: torch.Tensor, w: int, h: int) -> torch.Tensor:
        """`[H, W, 3]` uint8 → `[h, w, 3]`。用 Pillow（在 CPU 上，开销很小）。"""
        import numpy as np
        from PIL import Image

        img = Image.fromarray(rgb.numpy())
        img = img.resize((w, h), Image.BILINEAR)
        return torch.from_numpy(np.array(img))

    def _reset_counters(self) -> None:
        self.frames_rendered = 0
        self.frames_served = 0
        self.frames_dropped = 0
        self.peak_memory_mib = 0.0
        torch.cuda.reset_peak_memory_stats()

    def describe(self) -> str:
        """一行摘要，用于启动日志。"""
        if not self.ready:
            return "（未加载）"
        w, h = self.output_size()
        return (f"{self.cfg.subject}/{self.cfg.work_name} · "
                f"{self.avatar.num_gaussians} 高斯 · K={self.avatar.num_basis} · "
                f"输出 {w}×{h} · 模型 {self.model_path}")

    # -------------------------------------------------- 播放 / 单步 --

    def step_frame(self, delta: int) -> int:
        """单步前进/后退（边界夹住，不回绕）。返回新帧号。"""
        if self.timeline is None:
            return 0
        index = self.timeline.step(delta)
        self.params.frame = index
        return index

    def set_playing(self, playing: bool) -> None:
        if self.timeline is None:
            return
        self.timeline.playing = bool(playing)
        self.params.playing = bool(playing)

    def advance_frame(self) -> int:
        """播放推进一帧（回绕）。

        **由渲染循环在每次出帧之后调用** —— 推进与渲染在同一处，
        因此"渲染一帧 = 播放一帧"是结构性的，不会漂移。
        """
        if self.timeline is None:
            return 0
        index = self.timeline.advance()
        self.params.frame = index
        return index

    def model_dir_hint(self) -> str:
        """模型缺失时给出的提示路径。"""
        return str(model_dir(self.cfg))
