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
        params: `GaussianAvatar.parameter_groups()` 的结果
        iterations: 总步数
        batch_size: 每步样本数
        blend_start: 第几步开始启用 `blend_weight`（暖启动，见 `schedule.blend_active`）
        hooks: 钩子列表
        device: 渲染/损失所在设备
        optimizer_factory: 便于替换（默认 Adam，`eps=1e-15` —— 参数里 logit/对数尺度差很大，
            默认 1e-8 会在小梯度上过早停止更新）
    """

    def __init__(self, *, source, plan: ParameterPlan, objective: Objective,
                 render_fn: Callable, params: dict, iterations: int,
                 batch_size: int = 1, blend_start: int = 0,
                 hooks: tuple[Hook, ...] = (), device: str | torch.device = "cuda",
                 optimizer_factory: Callable | None = None) -> None:
        if iterations <= 0:
            raise ValueError(f"iterations 必须为正，实际 {iterations}")
        if batch_size <= 0:
            raise ValueError(f"batch_size 必须为正，实际 {batch_size}")

        self.source = source
        self.plan = plan
        self.objective = objective
        self.render_fn = render_fn
        self.iterations = int(iterations)
        self.batch_size = int(batch_size)
        self.blend_start = int(blend_start)
        self.hooks = tuple(hooks)
        self.device = device

        groups = plan.opt_groups(params)
        if optimizer_factory is None:
            self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
        else:
            self.optimizer = optimizer_factory(groups)

        self.report = TrainReport()
        self._stop = False

    # ------------------------------------------------------------ 控制 --

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
        iteration = 0
        try:
            for iteration in range(self.iterations):
                if self._stop:
                    break
                parts = self.step(iteration)
                self._emit("on_step", iteration, parts)
        finally:
            # `iteration` 是最后一次进入循环的值，实际执行了 iteration+1 步
            self.report.iterations = iteration + 1 if self.iterations > 0 else 0
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
        loss, parts = self.objective(
            pred=out.color, target=batch.image, mask=batch.mask,
            alpha=getattr(out, "alpha", None),
            target_alpha=batch.mask.to(out.color.dtype),
            iteration=iteration,
            extras=self._extras(out),
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

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
        for name in ("sparsity_loss", "orth_loss"):
            fn = getattr(out, name, None)
            if callable(fn):
                extras[name] = fn()
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
                 on_improve: Callable[[int, float], None] | None = None) -> None:
        self.source = source
        self.evaluate = evaluate
        self.every = int(every)
        self.early_stop = early_stop
        self.metric = metric
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
            if self.early_stop.update(metrics[self.metric]):
                trainer.request_stop()
                trainer.report.stopped_early = True
            elif self.early_stop.improved:
                trainer.report.best_metric = self.early_stop.best
                trainer.report.best_iteration = iteration
                self.best_iteration = iteration
                if self.on_improve is not None:
                    self.on_improve(iteration, float(self.early_stop.best))

    def on_end(self, trainer: "Trainer") -> None:
        if self.best is not None:
            trainer.report.best_metric = self.best

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
