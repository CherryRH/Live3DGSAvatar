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

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..config import Config, model_dir, resolve_data_dir, resolve_model_ply

# ⚠️ 直接从 `..core.types` 导入 `Mesh`。若写成 `import ..core.types as types`
#    会遮蔽标准库的 `types` 模块，间接依赖随之行为异常且极难排查。
from ..core.types import Mesh
from .protocol import FrameTimeline, merge_params, output_size

log = logging.getLogger(__name__)


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
    #: 用哪个数据集驱动。空串 = 跟随模型（`paths.data_root/<subject>`）。
    #: 只在 `source_kind == "checkpoint"` 时用于取网格与参数；
    #: live 模式下它决定"送什么画面给追踪服务"。
    drive_dataset: str = ""
    frame: int = 0
    #: 是否推进帧号。**暂停不代表停止渲染**，只是帧号停住。
    playing: bool = False
    #: 是否并排显示"数据集原图"（`images/`）并计算 PSNR。
    #: 运行期开关，不持久化 —— 它只影响显示与一个指标，不影响渲染结果。
    compare: bool = False
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
            "drive_dataset": self.drive_dataset,
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
            "drive_dataset": self.drive_dataset,
            "compare": self.compare,
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
        """`live`：画面送给追踪服务，用返回的参数驱动。"""
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
        #: live 模式下当前是否在显示**回退帧**（追踪服务未就绪）
        self.live_fallback = False
        #: 当前 scene 对应的数据集目录（判断"要不要重载"）
        self._active_dataset_dir: Path | None = None
        #: 取帧器构造失败的原因（例如数据集缺画面目录）
        self._live_frames_error: str | None = None
        #: 最近一次的对比帧缓存 `(key, rgb_u8, mask)`（给计时口径用）
        self._target_cache = None
        #: 对比帧池：`key -> (rgb_u8, mask)`。顺序播放时下一帧由后台线程
        #: 预先解好，渲染线程不必付 PNG 解码成本（约 7–10 ms/帧）
        self._target_store: dict = {}
        self._target_inflight: set = set()
        self._target_lock = threading.Lock()
        #: PSNR 缓存 `(key, value, at)` —— 同一帧短期内不重算
        self._psnr_cache = None
        #: 模型自带数据集的拓扑 `(顶点数, 面数)` —— 切换驱动数据集时用它校验。
        #: ⚠️ 不能用 `avatar.num_basis`：那是约简基的**个数 K**（20），
        #:    不是顶点数。曾经拿它跟 5083 比，导致换任何数据集都报"不匹配"。
        self._model_topology: tuple[int, int] | None = None

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
        #: 已加载模型的对象/版本（未加载时为空串）。
        #: 不从 `cfg.subject` 反推：配置里可能还没选，或刚被改过。
        self.model_subject: str = ""
        self.model_work_name: str = ""

    # ------------------------------------------------------------ 加载 --

    def load_model(self, subject: str | None = None,
                   work_name: str | None = None) -> dict:
        """载入模型与数据集。切换模型时调用。

        Raises:
            FileNotFoundError: 模型或数据集不存在（消息里给出解析路径）
            RuntimeError: 需要 GPU 但不可用
        """
        subject = (subject or self.cfg.get("subject") or "").strip() or None
        work_name = (work_name or self.cfg.get("work_name") or "").strip() or None
        if subject is None or work_name is None:
            raise ValueError(
                "还没有选择模型。请在右侧「模型」里选对象与版本，"
                "或用 POST /api/model 指定 subject 与 work_name。")
        self.cfg.set("subject", subject)
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
        # ⚠️ **赋值顺序决定并发安全**：`ready` 由 `avatar` / `scene` / `timeline`
        #    共同决定，三者必须"一次性"就绪。渲染循环会轮询 `ready`，
        #    若先设 `scene` 再设 `timeline`，中间那个窗口里 `render()` 会在
        #    `self.timeline is None` 上崩掉（真实发生过的竞态）。
        #    所以：其余全部先备好，**`scene` 最后赋值**，它一到位就全到位。
        self.avatar = avatar
        self.model_path = model_ply
        self.model_subject = str(subject)
        self.model_work_name = str(work_name)
        self._active_dataset_dir = resolve_data_dir(cfg)
        if scene.num_frames > 0:
            self._model_topology = (int(scene.frames[0]["mesh"].shape[0]),
                                    int(scene.faces.shape[0]))
        self.camera = Camera.from_intrinsics_extrinsics(
            scene.K, scene.R, scene.T, scene.width, scene.height).to(self.device)
        self._rasterizer = SimpleRasterizer(
            sh_degree=int(cfg.get("render.sh_degree", 0)),
            scaling_modifier=self.params.scaling_modifier)
        self._mesh_cache.clear()
        self._target_cache = None
        self._psnr_cache = None
        with self._target_lock:
            self._target_store.clear()
            self._target_inflight.clear()
        self.timeline = FrameTimeline(scene.num_frames)
        self.params.frame = 0
        self.params.drive_dataset = ""
        # ↓ 最后一步：`ready` 从这一刻起为真
        self.scene = scene
        self._reset_counters()
        self._sync_live_frames()
        self._push_live_index(0)
        return self.state()

    # ------------------------------------------------------ 驱动数据集 --

    def _dataset_dir(self, name: str) -> Path:
        """按名字取数据集目录；空名字表示跟随模型自带的数据集。"""
        from ..data.datasets import resolve_dataset_dir

        if not name:
            return resolve_data_dir(self.cfg)
        return resolve_dataset_dir(self.cfg, name)

    def set_drive_dataset(self, name: str) -> dict:
        """切换**驱动数据集**（模型不变）。

        为什么需要重载整个 scene 而不是只换帧：数据集带自己的**相机**与
        **帧数**。渲染的相机必须与被驱动参数的坐标系一致，帧号上限也必须
        跟着变 —— 否则要么画错，要么滑块越界。

        Raises:
            FileNotFoundError / ValueError: 数据集不存在或不可驱动
            RuntimeError: 拓扑与本模型不兼容
        """
        from ..core.types import Camera
        from ..data.scene import load_scene

        name = str(name or "")
        data_dir = self._dataset_dir(name)
        if self.scene is not None and data_dir == self._active_dataset_dir:
            # 同一个数据集：只更新参数，不做无谓重载
            self.params.drive_dataset = name
            return self.state()

        preload = int(self.cfg.get("app.preload_frames", -1))
        scene = load_scene(self.cfg, frames=preload, device=self.device,
                           data_dir=data_dir)

        # 拓扑必须与模型训练时的一致：每个高斯都绑在"某个面 + 重心坐标"上，
        # 面数或顶点数一变，绑定就指向了错的三角形。
        # 校验对象是**模型自带的数据集**，不是 `avatar.num_basis`（那是 K）。
        if self._model_topology is not None and scene.num_frames > 0:
            got = (int(scene.frames[0]["mesh"].shape[0]), int(scene.faces.shape[0]))
            if got != self._model_topology:
                mine_v, mine_f = self._model_topology
                raise RuntimeError(
                    f"数据集拓扑与模型不匹配：{data_dir.name} 是 "
                    f"{got[0]} 顶点 / {got[1]} 面，模型训练用的是 "
                    f"{mine_v} 顶点 / {mine_f} 面。\n"
                    "驱动数据集必须与模型同拓扑（同一套 FLAME + 牙齿）。")

        self.scene = scene
        self._active_dataset_dir = data_dir
        self.camera = Camera.from_intrinsics_extrinsics(
            scene.K, scene.R, scene.T, scene.width, scene.height).to(self.device)
        self._mesh_cache.clear()
        self._target_cache = None
        self._psnr_cache = None
        with self._target_lock:
            self._target_store.clear()
            self._target_inflight.clear()
        self.timeline = FrameTimeline(scene.num_frames)
        self.params.frame = 0
        self.params.drive_dataset = name
        # 换了数据集就回到开头并暂停：新数据集的时长与内容都不同，
        # 继续播下去只会让人以为"切成功了但画面乱跳"。
        self.timeline.playing = False
        self.params.playing = False
        if self.live is not None:
            self.live.set_playing(False)
        self._sync_live_frames()
        self._push_live_index(0)
        return self.state()

    def _sync_live_frames(self) -> None:
        """把"送什么画面给追踪服务"与当前驱动数据集对齐。"""
        from ..tracking.frame_source import DatasetFrames

        if self.scene is None or self.live is None:
            return
        if not getattr(self, "_active_dataset_dir", None):
            return
        try:
            frames = DatasetFrames(
                self._active_dataset_dir,
                fps=float(self.cfg.get("tracking.source_fps", 30.0)),
                subdir=str(self.cfg.get("tracking.frame_subdir", "auto")))
        except FileNotFoundError as e:
            # 没有 input/ 的数据集（如 duda）不能喂 RDF —— 记录原因，
            # 由 `_live_unavailable_reason()` 报给前端，而不是静默失败。
            self.live.set_frames(None)
            self._live_frames_error = str(e)
            return
        self._live_frames_error = None
        self.live.set_frames(frames)
        # 帧号回调：取线程推进后同步 timeline（播放条才不会与内容脱节）
        self.live.on_index = self._on_live_index
        self.live.seek(self.timeline.index if self.timeline else 0)
        # 切到 live 时要带上当前的播放态，否则"正在播放"会变成"停住"
        self.live.set_playing(bool(self.timeline.playing) if self.timeline else False)
        # **只在 live 模式下真的开始请求追踪**：RDF 推理与渲染抢同一块 GPU，
        # 数据集模式下没必要让它空转。取帧器照常挂上，切模式时立刻可用。
        if self.params.is_live:
            self.live.set_enabled(True)
            self.live.start()

    def sync_drive_mode(self) -> None:
        """按 `source_kind` 启动/停止追踪线程。

        只在需要时请求追踪：RDF 推理与渲染抢同一块 GPU，
        数据集模式下没必要让它空转。
        """
        if self.live is None:
            return
        if self.params.is_live:
            self.live.set_enabled(True)
            self._sync_live_frames()
            self.live.start()
        else:
            self.live.set_enabled(False)
            self.live.stop()

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
            #: 实际送的是哪个目录的画面（input / images），前端据此说明
            "frames_subdir": (getattr(self.live.frames, "subdir", None)
                              if self.live else None),
            "fallback": self.live_fallback,
            "reason": (self._live_unavailable_reason()
                       if self.live_fallback else None),
        }

    def state(self) -> dict:
        """/api/state 的返回体。"""
        cfg = self.cfg
        out: dict = {
            "subject": self.model_subject or str(cfg.get("subject") or ""),
            "work_name": self.model_work_name or str(cfg.get("work_name") or ""),
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
        """模型、数据集、时间线三者齐备才算就绪。

        `timeline` 也纳入判定：它是渲染循环每帧都要用的，缺了会在渲染中途崩，
        不如让"就绪"这个概念本身就不成立。
        """
        return (self.avatar is not None and self.scene is not None
                and self.timeline is not None)

    def output_size(self) -> tuple[int, int]:
        """当前参数下的输出分辨率（已应用 `scale`）。"""
        if self.scene is None:
            return (0, 0)
        return output_size(self.scene.width, self.scene.height, self.params.scale)

    def render(self) -> tuple[torch.Tensor, "torch.Tensor | None", dict]:
        """按当前参数渲染一帧。

        Returns:
            `(rgb, target, timings)`：

            - `rgb`：CPU 上的 `[H, W, 3]` uint8，可直接塞进协议帧
            - `target`：**对比用**的数据集原图（同尺寸 uint8）。
              未开启 `params.compare`、或该帧没有 `images/` 时返回 ``None``
            - `timings`：各阶段耗时（毫秒）

        `timings` 里额外带 `psnr`（与 target 的掩码内 PSNR，无法计算时缺省）。
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

        # ---- 对比图与 PSNR（只在开启时做）----
        target = None
        if self.params.compare:
            target, psnr = self._compare_target(rgb)
            if psnr is not None:
                timings["psnr"] = psnr

        timings["frame"] = (time.perf_counter() - t_all) * 1000.0
        self.last_deform_ms = timings["deform"]
        self.last_rasterize_ms = timings["rasterize"]
        self.last_frame_ms = timings["frame"]
        self.frames_rendered += 1
        self.peak_memory_mib = max(
            self.peak_memory_mib, torch.cuda.max_memory_allocated() / 2**20)
        return rgb, target, timings

    def _current_index(self) -> int:
        """当前**驱动帧号**：live 模式由取帧线程持有，其余看 timeline。"""
        if self.params.is_live and self.live is not None:
            return int(self.live.index)
        return int(self.timeline.index) if self.timeline is not None else 0

    def _compare_target(self, rgb: torch.Tensor):
        """取当前帧的抠图原图，返回 `(uint8[H,W,3], psnr)`；不可用时 `(None, None)`。

        - 用 `images/` 而不是 `input/`：渲染是黑底的，两者同背景才有可比性
        - **分辨率对齐**：渲染可能被 `scale` 缩小，目标图要缩到同一尺寸，
          否则并排显示与 PSNR 都是错的
        - 缓存按帧号：同一帧重复渲染时不重复读盘
        - PSNR **不每帧都算**：同一帧号且刚算过（< 0.5 s）就复用。
          它是个读数，不是渲染输入；而计算本身要遍历 100 万个数（优化后 ~3 ms）
        """
        import numpy as np
        from PIL import Image

        if self._active_dataset_dir is None or self.scene is None:
            return None, None
        index = self._current_index()
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        key = (index, w, h, str(self._active_dataset_dir))

        item = self._get_target(index, w, h, key)
        if item is None:
            self._target_cache = None
            return None, None
        target, mask = item
        self._target_cache = (key, target, mask)

        # PSNR：同帧号短期内复用。播放时帧号每帧都变，就每帧都算 —— 那是
        # 指标本身的成本（~3 ms），不是重复浪费。
        now = time.perf_counter()
        cached_psnr = self._psnr_cache
        if (cached_psnr is not None and cached_psnr[0] == key
                and now - cached_psnr[2] < 0.5):
            psnr = cached_psnr[1]
        else:
            full, masked = _psnr_pair(rgb.numpy(), target, mask)
            psnr = masked if masked is not None else full
            self._psnr_cache = (key, psnr, now)
        return torch.from_numpy(target), psnr

    # ------------------------------------------------- 对比帧（解码/预取） --

    def _decode_target(self, index: int, w: int, h: int):
        """解码一帧目标图，返回 `(rgb_u8, mask)`；失败返回 ``None``。

        只读磁盘与配置，**不碰实例状态** —— 所以可以安全地在后台线程里调用。
        """
        import numpy as np
        from PIL import Image

        sub = str(self.cfg.get("paths.image_subdir", "images"))
        path = Path(self._active_dataset_dir) / sub / f"{index:05d}.png"
        try:
            with Image.open(path) as im:
                rgba = np.asarray(im.convert("RGBA"))
        except (OSError, ValueError) as e:
            log.debug("对比帧不可用 %s：%s", path, e)
            return None

        if (rgba.shape[1], rgba.shape[0]) != (w, h):
            im = Image.fromarray(rgba).resize((w, h), Image.LANCZOS)
            rgba = np.asarray(im)
        return np.ascontiguousarray(rgba[..., :3]), rgba[..., 3] > 127

    def _get_target(self, index: int, w: int, h: int, key: tuple):
        """取目标帧：先查内存池，未命中则同步解码并**预取下一帧**。

        为什么要预取：解码一张 512² PNG 要 7–10 ms，比渲染本身（~3 ms）还慢。
        顺序播放时每帧都要新图，若都压在渲染线程里，「对比原图」会把帧率砍半。
        放到后台线程后，渲染线程只在**第一次**遇到某帧时付代价。
        """
        with self._target_lock:
            hit = self._target_store.pop(key, None)
            if hit is not None:
                self._target_store[key] = hit      # 重新插入 = 最近使用

        item = hit
        if item is None:
            item = self._decode_target(index, w, h)
            if item is None:
                return None
            with self._target_lock:
                self._target_store[key] = item
                while len(self._target_store) > 8:
                    self._target_store.pop(next(iter(self._target_store)))

        # **命中也要预取**：否则会退化成"命中 → 不预取 → 下帧未命中"的锯齿，
        # 每两帧就要同步解一次（实测 3 ms / 13 ms 交替）。
        self._prefetch_target(index + 1, w, h)
        return item

    def _prefetch_target(self, index: int, w: int, h: int) -> None:
        """后台解码 `index` 放进池子（同一帧只排一次）。"""
        if self._active_dataset_dir is None:
            return
        key = (index, w, h, str(self._active_dataset_dir))
        with self._target_lock:
            if key in self._target_store or key in self._target_inflight:
                return
            self._target_inflight.add(key)

        def work() -> None:
            item = None
            try:
                item = self._decode_target(index, w, h)
            except Exception as e:                              # noqa: BLE001
                log.debug("预取对比帧失败 %s：%s", index, e)
            finally:
                with self._target_lock:
                    self._target_inflight.discard(key)
            if item is not None:
                with self._target_lock:
                    self._target_store[key] = item
                    while len(self._target_store) > 8:
                        self._target_store.pop(next(iter(self._target_store)))

        threading.Thread(target=work, name="target-prefetch", daemon=True).start()

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

        if self.timeline is None:          # 防御：正常路径不会到这里
            raise RuntimeError("会话尚未就绪（没有时间线），请先加载模型")
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
        if self._live_frames_error:
            return "这个数据集没有可用画面"
        st = self.live.status()
        if not st["enabled"]:
            return "实时驱动已关闭"
        if not st["connected"]:
            return "未连接追踪服务"
        return "已连接，等待画面"

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
        return (f"{self.model_subject}/{self.model_work_name} · "
                f"{self.avatar.num_gaussians} 高斯 · K={self.avatar.num_basis} · "
                f"输出 {w}×{h} · 模型 {self.model_path}")

    # -------------------------------------------------- 播放 / 单步 --

    def step_frame(self, delta: int) -> int:
        """单步前进/后退（边界夹住，不回绕）。返回新帧号。"""
        if self.timeline is None:
            return 0
        index = self.timeline.step(delta)
        self.params.frame = index
        self._push_live_index(index)
        return index

    def set_playing(self, playing: bool) -> None:
        """播放 / 暂停。**两种驱动方式语义一致**。

        数据集模式：渲染循环推进帧号（渲染一帧 = 播放一帧）。
        live 模式：由取帧线程推进（**追踪一帧 = 播放一帧**）—— 因为画面内容
        只随追踪结果变化，按渲染帧率推进会让显示与内容对不上。
        """
        if self.timeline is None:
            return
        self.timeline.playing = bool(playing)
        self.params.playing = bool(playing)
        if self.live is not None:
            self.live.set_playing(bool(playing))

    def jump_frame(self, index: int) -> int:
        """跳到指定帧（直播模式下也立即生效）。"""
        if self.timeline is None:
            return 0
        index = self.timeline.set_index(int(index))
        self.params.frame = index
        self._push_live_index(index)
        return index

    def _push_live_index(self, index: int) -> None:
        """把帧号告诉取帧线程（它在 live 模式下决定送哪一帧）。"""
        if self.live is not None:
            self.live.seek(int(index))

    def _on_live_index(self, index: int) -> None:
        """取帧线程推进后的回调：同步 timeline 与 params。"""
        if self.timeline is not None:
            self.timeline.index = int(index) % self.timeline.total
            self.params.frame = self.timeline.index
        else:
            self.params.frame = int(index)

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


def _psnr_pair(render, target, mask) -> tuple[float | None, float | None]:
    """返回 `(全图 PSNR, 掩码内 PSNR)`（dB）。MSE 为 0 时给 99.0。

    与离线评估口径一致：0–255、按 alpha>0.5 取前景。

    ## 为什么不用 `((a - b) ** 2).mean()`

    那样每帧要分配 3 个 float32 大数组（512² RGB ≈ 3 MB 一个），实测
    **13 ms/帧**，直接变成开启对比时的瓶颈（曾经让 FPS 从 60 掉到 30）。
    改成"一次转换 + 原地减法 + einsum 求平方和"后 **2.8 ms**，
    且结果与旧写法**逐位相同**（同样是 float32 求和）。
    """
    import numpy as np

    a = np.asarray(render)
    b = np.asarray(target)
    if a.shape != b.shape:
        return None, None

    def _psnr(mse: float) -> float:
        return 99.0 if mse <= 0.0 else float(10.0 * np.log10((255.0 ** 2) / mse))

    # 原地转 float32 再原地相减：只占一块临时内存
    diff = a.astype(np.float32)
    np.subtract(diff, b, out=diff)

    full = _psnr(float(np.einsum("ijk,ijk->", diff, diff)) / diff.size)
    if mask is None or not bool(np.any(mask)):
        return full, None

    sub = diff[mask]
    mse = float(np.einsum("ij,ij->", sub, sub)) / sub.size
    return full, _psnr(mse)
