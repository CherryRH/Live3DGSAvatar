"""在线更新（热更新）：用一小段观测**微调已有模型**，而不是重新训练。

## 定位

预训练模型是"这个人、这类光照、这个裁剪"下的平均解。真正通话时，
**相机、光照、姿态分布**都与训练时不同 —— 必然存在系统性偏差。
本模块在通话开始的一小段时间内（目标 ~10 s）把这个偏差压下去。

## 刻意只更新「外观」，不碰「几何与驱动」

| 可更新 | 冻结 | 理由 |
|---|---|---|
| 高斯颜色 `color` | `xyz` / `rotation` / `scaling` | 几何一动，绑定关系与多帧一致性都会变 |
| 高斯不透明度 `opacity` | 融合基 `*_b` | 基是训练学到的表达空间，小数据量下改它只会破坏 |
| — | `weight_module`(MLP) | 10 s 内迭代量不足，改它容易学坏 |

**这个边界是刻意的**：热更新解决的是"外观偏差"，不是"重建一个更好的模型"。
几何/驱动层面的缺陷（如头顶没拍到）应当用对称先验或补拍来解决。

## 分层

本模块只依赖 `core/`（渲染与类型）与 `training/` 内部组件，
**不依赖 `data/`、`app/`** —— 观测以 `FrameObservation` 传入，
所以数据集与摄像头两条来源走同一份更新逻辑。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from ..core.avatar import GaussianAvatar
from ..core.render import SimpleRasterizer
from ..core.types import Camera, GaussianSet
from .losses import PhotometricLoss
from .observations import BatchObservation, FrameObservation


@dataclass
class UpdateConfig:
    """热更新的超参数。**默认值以"10 秒内跑完"为目标**。

    Attributes:
        steps: 最大迭代次数（与 `time_budget_s` 取先到者）
        time_budget_s: 墙钟预算（秒）。超出即停，保证通话不被卡住
        lr_color: 颜色学习率（作用于 SH DC）
        lr_opacity: 不透明度学习率（作用于 sigmoid 前的 logit）
        update_opacity: 是否更新不透明度
        mask_weight: 掩码外像素的权重（0 = 完全忽略背景）
        log_every: 每多少步记录一次
        frame_chunk: 每步用几帧（1 表示逐帧；受显存限制时调小）
        color_l1: 优化目标用 L1（对离群像素稳健，收敛更慢）
        update_color: 是否更新颜色。与 `update_opacity` 不能同时为 False
            （那样在线更新是空操作，应在上层就跳过）
    """

    steps: int = 300
    time_budget_s: float = 10.0
    lr_color: float = 2e-3
    lr_opacity: float = 2e-2
    update_opacity: bool = True
    mask_weight: float = 0.0
    log_every: int = 50
    frame_chunk: int = 1
    color_l1: bool = False
    update_color: bool = True

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError(f"steps 必须 > 0，实际 {self.steps}")
        if self.time_budget_s <= 0:
            raise ValueError(f"time_budget_s 必须 > 0，实际 {self.time_budget_s}")
        if self.frame_chunk <= 0:
            raise ValueError(f"frame_chunk 必须 > 0，实际 {self.frame_chunk}")
        if not self.update_color and not self.update_opacity:
            raise ValueError(
                "update_color 与 update_opacity 不能同时为 False —— "
                "那样没有任何参数可优化；调用方应直接跳过在线更新")


@dataclass
class UpdateState:
    """可训练的**增量**（相对预训练模型），挂在模型之外。

    为什么不直接改 `avatar.parameters()`：

    1. 预训练模型应保持可复现 —— 更新失败要能"回到原样"
    2. 增量可随时丢弃/回滚，也能只保存增量（体积远小于整个模型）
    3. 更新逻辑与模型存储解耦：以后换表示也不用动这里
    """

    color_delta: torch.Tensor           # [N, 1, 3] 加到 SH DC
    opacity_delta: torch.Tensor | None  # [N, 1] 加到 opacity logit
    step: int = 0

    @classmethod
    def zeros(cls, avatar: GaussianAvatar, *, update_opacity: bool) -> "UpdateState":
        dev, dt = avatar.color.device, avatar.color.dtype
        n = avatar.num_gaussians
        return cls(
            color_delta=torch.zeros(n, 1, 3, device=dev, dtype=dt, requires_grad=True),
            opacity_delta=(torch.zeros(n, 1, device=dev, dtype=dt, requires_grad=True)
                           if update_opacity else None),
        )

    def parameters(self) -> list[torch.Tensor]:
        ps = [self.color_delta]
        if self.opacity_delta is not None:
            ps.append(self.opacity_delta)
        return ps

    def detach_(self) -> "UpdateState":
        """去掉计算图（保存或跨轮复用时调用），返回自身。"""
        self.color_delta = self.color_delta.detach().requires_grad_(True)
        if self.opacity_delta is not None:
            self.opacity_delta = self.opacity_delta.detach().requires_grad_(True)
        return self

    def snapshot(self) -> "UpdateState":
        """复制一份（脱离计算图），用于返回给调用方存档。"""
        return UpdateState(
            color_delta=self.color_delta.detach().clone().requires_grad_(True),
            opacity_delta=(self.opacity_delta.detach().clone().requires_grad_(True)
                           if self.opacity_delta is not None else None),
            step=self.step,
        )

    def to_dict(self) -> dict:
        """只保存增量（体积远小于整个模型，便于会话间传递）。"""
        d = {"color_delta": self.color_delta.detach().cpu(), "step": self.step}
        if self.opacity_delta is not None:
            d["opacity_delta"] = self.opacity_delta.detach().cpu()
        return d

    @classmethod
    def from_dict(cls, d: dict, device="cpu") -> "UpdateState":
        return cls(
            color_delta=d["color_delta"].to(device).requires_grad_(True),
            opacity_delta=(d["opacity_delta"].to(device).requires_grad_(True)
                           if "opacity_delta" in d else None),
            step=int(d.get("step", 0)),
        )


@dataclass
class UpdateReport:
    """一次热更新的指标（用于日志与验收）。"""

    before: dict[str, float]
    after: dict[str, float]
    steps: int
    elapsed_s: float
    history: list[dict[str, float]] = field(default_factory=list)

    @property
    def psnr_delta(self) -> float:
        return self.after["psnr"] - self.before["psnr"]

    def summary(self) -> str:
        return (f"热更新 {self.steps} 步 / {self.elapsed_s:.2f}s："
                f"PSNR {self.before['psnr']:.2f} → {self.after['psnr']:.2f} dB "
                f"({self.psnr_delta:+.2f})")


class OnlineUpdater:
    """把 `BatchObservation` 拟合成对模型的一组增量。

    典型用法（10 秒预算）::

        updater = OnlineUpdater(avatar, rasterizer, config=UpdateConfig())
        report, state = updater.fit(batch, background=bg)
        print(report.summary())
        gs = updater.apply(state, batch.frames[0], background)   # 带增量的渲染
    """

    def __init__(
        self,
        avatar: GaussianAvatar,
        rasterizer: SimpleRasterizer,
        config: UpdateConfig | None = None,
        loss_fn: PhotometricLoss | None = None,
    ) -> None:
        self.avatar = avatar
        self.rasterizer = rasterizer
        self.config = config or UpdateConfig()
        self.loss_fn = loss_fn or PhotometricLoss(
            mask_weight=(config.mask_weight if config else 0.0),
            l1=(config.color_l1 if config else False))

    # ------------------------------------------------------------ 对外 --
    def fit(
        self,
        batch: BatchObservation,
        background: torch.Tensor,
        *,
        state: UpdateState | None = None,
        eval_batch: BatchObservation | None = None,
    ) -> tuple[UpdateReport, UpdateState]:
        """在 `batch` 上迭代优化外观增量。

        Args:
            batch: 观测集合（同一相机、同一分辨率）
            background: `[3]` 背景色
            state: 从已有增量继续优化（流式场景）；``None`` 从零开始
            eval_batch: 用于计算 before/after 的帧集合；``None`` 用 `batch`

        Returns:
            `(报告, 增量状态)`。报告里含 before/after PSNR —— **这就是验收判据**。
        """
        cfg = self.config
        # ⚠️ **复制**传入的 state，绝不原地修改调用方的对象。
        #    否则调用方持有的"上一轮结果"会跟着本轮迭代一起变，
        #    表现为两轮的 `state.step` 混在一起、无法比较前后差异。
        st = (state.snapshot() if state is not None
              else UpdateState.zeros(self.avatar, update_opacity=cfg.update_opacity))
        # 只把**开启的**参数放进优化器：Adam 会更新 groups 里的全部张量，
        # 若把关闭项也放进去，它照样会被改（学习率只是 0 才不动）。
        params = []
        if cfg.update_color:
            params.append({"params": [st.color_delta], "lr": cfg.lr_color})
        if st.opacity_delta is not None:
            params.append({"params": [st.opacity_delta], "lr": cfg.lr_opacity})
        opt = torch.optim.Adam(params)

        eval_on = eval_batch or batch
        with torch.no_grad():
            before = self._evaluate(eval_on, background, st)

        history: list[dict[str, float]] = []
        t0 = time.perf_counter()
        steps = 0
        for step in range(cfg.steps):
            if time.perf_counter() - t0 > cfg.time_budget_s:
                break
            opt.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=background.device)
            # 逐帧（真实场景下帧是陆续到达的，不该假设能一次看完全部）
            chunk = batch.frames[: cfg.frame_chunk]
            for obs in chunk:
                out = self._forward(obs, batch.camera, background, st)
                l, _ = self.loss_fn(out, obs)
                loss = loss + l / len(chunk)
            loss.backward()
            opt.step()
            st.step += 1
            steps += 1
            if cfg.log_every and (step + 1) % cfg.log_every == 0:
                with torch.no_grad():
                    cur = self._evaluate(eval_on, background, st)
                history.append({"step": step + 1, "loss": float(loss.detach()),
                                "psnr": cur["psnr"]})

        with torch.no_grad():
            after = self._evaluate(eval_on, background, st)
        report = UpdateReport(before=before, after=after, steps=steps,
                              elapsed_s=time.perf_counter() - t0, history=history)
        # ⚠️ 返回**快照**而非 `st` 本身：传入 `state=` 时 `st` 就是调用方的对象，
        #    原地返回它会让上一轮的引用"跟着变"（曾因此让两轮的 `state.step` 混在一起）。
        return report, st.snapshot()

    def apply(
        self,
        state: UpdateState,
        obs: FrameObservation,
        camera: Camera,
        background: torch.Tensor,
    ):
        """用增量渲染单帧（不修改模型本身）。"""
        with torch.no_grad():
            return self._forward(obs, camera, background, state)

    # ------------------------------------------------------------ 内部 --
    def _forward(self, obs: FrameObservation, camera: Camera,
                 background: torch.Tensor, state: UpdateState):
        """带增量的前向。顺序与 `core/render/pipeline.render_frame` 一致，
        只是把增量注入到高斯属性上。"""
        gs = self.avatar.deform(obs.mesh, obs.blend_weight)
        color = gs.color + state.color_delta
        opacity = gs.opacity
        if state.opacity_delta is not None:
            opacity = opacity + state.opacity_delta
        patched = GaussianSet(xyz=gs.xyz, rotation=gs.rotation, scaling=gs.scaling,
                              opacity=opacity, color=color, space=gs.space)
        return self.rasterizer.render(patched, camera, background)

    def _evaluate(self, batch: BatchObservation, background, state: UpdateState) -> dict[str, float]:
        psnrs, losses = [], []
        for obs in batch.frames:
            out = self._forward(obs, batch.camera, background, state)
            _, m = self.loss_fn(out, obs)
            psnrs.append(m["psnr"])
            losses.append(m["loss"])
        return {"psnr": sum(psnrs) / len(psnrs), "loss": sum(losses) / len(losses)}
