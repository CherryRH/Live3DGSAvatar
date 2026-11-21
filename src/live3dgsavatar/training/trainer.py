"""训练循环与钩子：**全量训练与在线微调共用的那一份代码**。

## 结构

```
Trainer(source, plan, objective, render_fn, hooks)
    for it in range(iterations):
        plan.apply_lr(optimizer, it)           # 学习率曲线
        batch = source.random_batch(size)      # 取数（含背景合成）
        out = render_fn(mesh, weight, batch)   # 注入：真实渲染 or 假渲染
        loss = objective(pred=out.color, ...)  # 加权损失
        loss.backward(); optimizer.step()
        hooks.on_step(it, ...)                 # 验证/日志/checkpoint
```

`render_fn` 是**依赖注入**：真实训练传 `GaussianAvatar + SimpleRasterizer`，
测试传一个纯 torch 的假函数。这样训练逻辑（顺序、调度、钩子）能脱离
CUDA 扩展单独验证 —— 训练顺序、调度、早停这些逻辑不必起 GPU 就能测。

## 为什么这样切

- 学习率按组查表（`ParameterPlan.apply_lr`），加一组参数不用改循环
- 损失是对象（`Objective`），权重进配置，返回值带分项
- 验证/存档/日志都是钩子，训练、微调、冒烟测试共用一套
- 定期存档（`CheckpointHook`），中断不丢，也能按验证指标挑最佳
- 验证集驱动早停（`ValidationHook`）
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import torch

from .objectives import Objective
from .plans import ParameterPlan
from .schedule import EarlyStopping, blend_active


class Hook(Protocol):
    """训练钩子。全部可选实现。"""

    def on_start(self, trainer: "Trainer") -> None: ...
    def on_step(self, trainer: "Trainer", iteration: int,
                parts: dict[str, float]) -> None: ...
    def on_eval(self, trainer: "Trainer", iteration: int,
                metrics: dict[str, float]) -> None: ...
    def on_end(self, trainer: "Trainer") -> None: ...


def _to_device(batch, device):
    """把 batch 里的张量搬到 `device`。

    数据源通常是 CPU 建的（图像很大，没必要常驻显存），而渲染在 GPU 上。
    这个搬运必须**统一在 trainer 里做**：否则每个 `render_fn` 都要自己记得
    搬 image/mask，漏一个就是 "Expected all tensors to be on the same device"。
    """
    import torch as _torch

    if not isinstance(batch, (list, tuple)) and hasattr(batch, "__dict__"):
        fields = vars(batch)
        moved = {
            k: (v.to(device) if isinstance(v, _torch.Tensor) else v)
            for k, v in fields.items()
        }
        return type(batch)(**moved)
    return batch


@dataclass
class TrainReport:
    """一次训练的摘要（写 JSON 用）。"""

    iterations: int = 0
    seconds: float = 0.0
    last_parts: dict[str, float] = field(default_factory=dict)
    best_metric: float | None = None
    best_iteration: int | None = None
    #: 是否真的落盘过 best 模型（`best_metric` 只反映落盘的那一份，
    #: 不再用"history 最大值"冒充 —— 评审 B3）
    best_saved: bool = False
    stopped_early: bool = False

    @property
    def iters_per_s(self) -> float:
        return self.iterations / self.seconds if self.seconds > 0 else 0.0


class Trainer:
    """通用训练循环。

    Args:
        source: 提供 `random_batch(n)` 的数据源
        plan: 参数计划（决定优化器里有哪些组、各自学习率）
        objective: 损失
        render_fn: `(mesh, blend_weight, batch, iteration) -> RenderOutput`
            —— 返回对象需有 `.color`、`.alpha`，可选 `.est_color` / `.est_weight`
        params: `GaussianAvatar.parameter_groups()` 的结果，或**取参函数**
            （`params=avatar.parameter_groups`）。会分裂/剪枝的管线必须传函数：
            那些操作会替换 `nn.Parameter`，重建优化器时要重新取参，否则优化器
            会一直持有旧张量，模型从第一次分裂起就不再更新。
        iterations: 总步数
        batch_size: 每步样本数
        blend_start: 第几步开始启用 `blend_weight`（暖启动，见 `schedule.blend_active`）
        hooks: 钩子列表
        device: 渲染/损失所在设备
        optimizer_factory: 便于替换（默认 Adam，`eps=1e-15` —— 参数里 logit/对数尺度差很大，
            默认 1e-8 会在小梯度上过早停止更新）
    """

    def __init__(self, *, source, plan: ParameterPlan, objective: Objective,
                 render_fn: Callable, params, iterations: int,
                 batch_size: int = 1, blend_start: int = 0,
                 hooks: tuple[Hook, ...] = (), device: str | torch.device = "cuda",
                 optimizer_factory: Callable | None = None,
                 post_step: Callable[[], None] | None = None) -> None:
        if iterations <= 0:
            raise ValueError(f"iterations 必须为正，实际 {iterations}")
        if batch_size <= 0:
            raise ValueError(f"batch_size 必须为正，实际 {batch_size}")

        self.source = source
        self.plan = plan
        self.objective = objective
        self.render_fn = render_fn
        # `params` 既可以是 `parameter_groups()` 的结果，也可以是一个**取参函数**。
        # 后者是加密/生长场景的正确用法：split/prune 会用新的 `nn.Parameter`
        # 整体替换张量，旧 dict 里的引用随即失效 —— 重建优化器时必须重新取一次，
        # 否则之后的梯度全落在没人用的旧张量上，模型从第一次分裂起就静止。
        self._params_fn = params if callable(params) else None
        self.params = self._params_fn() if callable(params) else params
        self.optimizer_factory = optimizer_factory
        self.iterations = int(iterations)
        self.batch_size = int(batch_size)
        self.blend_start = int(blend_start)
        self.hooks = tuple(hooks)
        self.device = device
        self.post_step = post_step

        groups = plan.opt_groups(self.params)
        if optimizer_factory is None:
            self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
        else:
            self.optimizer = optimizer_factory(groups)

        self.report = TrainReport()
        self._stop = False
        self.last_batch = None
        self.last_out = None

    # ------------------------------------------------------------ 控制 --

    def rebuild_optimizer(self) -> None:
        """高斯集合分裂/剪枝后重建优化器（参数张量已被替换）。

        有取参函数时必须**先重新取参数**，再建优化器；只重建 Adam 而不换
        参数引用是"分裂后不再学习"的根因（见 `_params_fn` 的注释）。
        """
        if self._params_fn is not None:
            self.params = self._params_fn()
        groups = self.plan.opt_groups(self.params)
        if self.optimizer_factory is None:
            self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
        else:
            self.optimizer = self.optimizer_factory(groups)

    def request_stop(self) -> None:
        """请求在下一步之后停止（钩子用它做时间预算/早停）。"""
        self._stop = True

    @property
    def stopped(self) -> bool:
        return self._stop

    # ------------------------------------------------------------ 主循环 --

    def run(self) -> TrainReport:
        self._emit("on_start")

        t0 = time.perf_counter()
        executed = 0
        try:
            for iteration in range(self.iterations):
                if self._stop:
                    break
                parts = self.step(iteration)
                executed += 1
                self._emit("on_step", iteration, parts)
        finally:
            # 只记**真正跑过**的步数：早停时 for 变量会先 +1 再 break，
            # 用 `iteration + 1` 会多算一步（评审 B4）。
            self.report.iterations = executed
            self.report.seconds = time.perf_counter() - t0
            self._emit("on_end")
            close = getattr(self.source, "close", None)
            if callable(close):
                close()
        return self.report

    def step(self, iteration: int) -> dict[str, float]:
        """走一步：取数 → 渲染 → 损失 → 反传。返回损失分项。"""
        self.plan.apply_lr(self.optimizer, iteration)

        batch = _to_device(self.source.random_batch(self.batch_size), self.device)
        weight = batch.blend_weight if blend_active(iteration, self.blend_start) \
            else None

        out = self.render_fn(batch.mesh, weight, batch, iteration)
        self.last_batch = batch
        self.last_out = out
        loss_mask = getattr(out, "loss_mask", batch.mask)
        loss, parts = self.objective(
            pred=out.color, target=batch.image, mask=loss_mask,
            sample_weight=getattr(batch, "weight", None),
            weight_map=getattr(batch, "weight_map", None),
            alpha=getattr(out, "alpha", None),
            target_alpha=loss_mask.to(out.color.dtype),
            iteration=iteration,
            extras=self._extras(out),
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        if self.post_step is not None:
            self.post_step()

        self.report.last_parts = parts
        return parts

    # ------------------------------------------------------------ 内部 --

    def _emit(self, name: str, *args) -> None:
        """调用钩子的**可选**方法。

        钩子只实现自己关心的方法（`LogHook` 没有 `on_end`、`TimeBudgetHook`
        没有 `on_start` 之外的东西），所以这里必须按需取，不能假设齐全。
        """
        for hook in self.hooks:
            fn = getattr(hook, name, None)
            if fn is not None:
                fn(self, *args)

    def _extras(self, out) -> dict[str, torch.Tensor]:
        """把模型自带的正则挂进来（由 `render_fn` 决定提供哪些）。

        `render_fn` 可以在返回对象上附带 `sparsity_loss` / `orth_loss`
        属性；没有就当 0（`Objective` 按权重跳过）。
        """
        extras = {}
        for name, attr in (("sparsity_loss", "sparsity_loss"),
                           ("orth_loss", "orth_loss"),
                           ("nv_hole", "nv_hole_loss"),
                           ("nv_spill", "nv_spill_loss"),
                           ("surface_spill", "surface_spill_loss"),
                           ("surface_point", "surface_point_loss"),
                           ("surface_normal", "surface_normal_loss"),
                           ("surface_scale", "surface_scale_loss"),
                           ("surface_size", "surface_size_loss"),
                           ("surface_iso", "surface_iso_loss"),
                           ("surface_outside", "surface_outside_loss"),
                           ("surface_bg", "surface_bg_loss")):
            value = getattr(out, attr, None)
            if callable(value):
                extras[name] = value()
            elif isinstance(value, torch.Tensor):
                extras[name] = value
        return extras


class ValidationHook:
    """定期在验证集上算指标，可早停。

    Args:
        source: 验证数据源（`iter_batches`）
        evaluate: `(batch) -> dict[str, float]`，通常是 PSNR/SSIM
        every: 每多少步验证一次
        early_stop: 早停器（按 `metric` 越大越好）；``None`` 表示不早停
        metric: 用作早停判据的指标名
    """

    def __init__(self, *, source, evaluate: Callable, every: int = 2000,
                 early_stop: EarlyStopping | None = None,
                 metric: str = "psnr",
                 on_improve: Callable[[int, float], None] | None = None,
                 start_after: int = 0) -> None:
        self.source = source
        self.evaluate = evaluate
        self.every = int(every)
        self.early_stop = early_stop
        self.metric = metric
        #: 该步之前只记录/选最佳，不允许早停（形变基暖启动阶段）
        self.start_after = int(start_after)
        #: 验证指标刷新最好成绩时调用 `(iteration, value)` —— 存"最佳模型"用
        self.on_improve = on_improve
        self.history: list[tuple[int, dict[str, float]]] = []
        self.best_iteration: int | None = None

    def on_start(self, trainer: "Trainer") -> None:
        self.history.clear()

    def on_step(self, trainer: "Trainer", iteration: int,
                parts: dict[str, float]) -> None:
        if self.every <= 0 or (iteration + 1) % self.every != 0:
            return
        metrics = self.evaluate(self.source)
        self.history.append((iteration, metrics))
        if self.early_stop is not None and self.metric in metrics:
            stop = self.early_stop.update(metrics[self.metric])
            improved = self.early_stop.improved
            if iteration < self.start_after and not improved:
                # 暖启动阶段不累计"无提升"，否则可能在 blend 打开前就停
                self.early_stop.bad = 0
            if stop and iteration >= self.start_after:
                trainer.request_stop()
                trainer.report.stopped_early = True
            elif improved:
                trainer.report.best_metric = self.early_stop.best
                trainer.report.best_iteration = iteration
                trainer.report.best_saved = True
                self.best_iteration = iteration
                if self.on_improve is not None:
                    self.on_improve(iteration, float(self.early_stop.best))

    def on_end(self, trainer: "Trainer") -> None:
        # 不再把 best_metric 改成 history 的最大值：那可能不是落盘的 checkpoint
        # （评审 B3）。没触发过 on_improve 时保持 None + best_saved=False。
        return None

    @property
    def best(self) -> float | None:
        if not self.history:
            return None
        return max(m.get(self.metric, float("-inf")) for _, m in self.history)


class CheckpointHook:
    """定期把模型存盘。

    Args:
        save: `(iteration) -> None`，由调用方决定怎么存（PLY/state_dict）
        every: 每多少步存一次
        on_end: 结束时是否再存一次
    """

    def __init__(self, *, save: Callable[[int], None], every: int = 5000,
                 on_end: bool = True) -> None:
        self.save = save
        self.every = int(every)
        self.save_at_end = bool(on_end)
        self.saved: list[int] = []

    def on_step(self, trainer: "Trainer", iteration: int,
                parts: dict[str, float]) -> None:
        if self.every > 0 and (iteration + 1) % self.every == 0:
            self.save(iteration)
            self.saved.append(iteration)

    def on_end(self, trainer: "Trainer") -> None:
        if self.save_at_end:
            self.save(trainer.report.iterations)
            self.saved.append(trainer.report.iterations)


class LogHook:
    """按间隔打印一行（含分项损失与速度）。"""

    def __init__(self, *, every: int = 100, prefix: str = "[train]") -> None:
        self.every = int(every)
        self.prefix = prefix
        self._t0 = 0.0

    def on_start(self, trainer: "Trainer") -> None:
        self._t0 = time.perf_counter()

    def on_step(self, trainer: "Trainer", iteration: int,
                parts: dict[str, float]) -> None:
        if self.every <= 0 or (iteration + 1) % self.every != 0:
            return
        dt = time.perf_counter() - self._t0
        rate = (iteration + 1) / dt if dt > 0 else 0.0
        # 总损失单独标出来，其余项放括号里 —— 一眼能看出"现在在优化什么"。
        # 这些是**当前 batch 的训练损失**（像素空间、0~1），不是 PSNR；
        # 每步的 batch 是随机采的，所以数值会随样本难度上下跳。
        total = parts.get("total")
        terms = " ".join(f"{k} {v:.4f}" for k, v in parts.items()
                         if k != "total")
        head = f"损失 {total:.4f}" if total is not None else "损失 —"
        print(f"{self.prefix} {iteration + 1}/{trainer.iterations}  "
              f"{head}（{terms}）  {rate:.1f} it/s", flush=True)


class TimeBudgetHook:
    """墙钟预算：到点请求停止（在线微调用）。"""

    def __init__(self, seconds: float) -> None:
        if seconds <= 0:
            raise ValueError(f"seconds 必须为正，实际 {seconds}")
        self.seconds = float(seconds)
        self._t0 = 0.0

    def on_start(self, trainer: "Trainer") -> None:
        self._t0 = time.perf_counter()

    def on_step(self, trainer: "Trainer", iteration: int,
                parts: dict[str, float]) -> None:
        if time.perf_counter() - self._t0 >= self.seconds:
            trainer.request_stop()
