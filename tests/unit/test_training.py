"""训练层（`training/plans.py`、`schedule.py`、`objectives.py`、`trainer.py`）。

渲染要 CUDA 扩展，这里不碰它：训练循环通过 **注入的 `render_fn`** 验证，
所以"学习率怎么走、损失怎么组、钩子什么时候触发"这些都能离线测。
RGBAvatar 把这几件事写死在 `Reconstruction.step` 里，做不到这样的测试 ——
这也是我们重写这一层的主要理由之一。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import live3dgsavatar  # noqa: E402, F401

from live3dgsavatar.training import objectives as obj_mod  # noqa: E402
from live3dgsavatar.training.objectives import Objective  # noqa: E402
from live3dgsavatar.training.plans import ALL_GROUPS, ParameterPlan  # noqa: E402
from live3dgsavatar.training.schedule import (EarlyStopping,  # noqa: E402
                                              blend_active, expon_lr)
from live3dgsavatar.training.trainer import (CheckpointHook,  # noqa: E402
                                             LogHook, Trainer,
                                             ValidationHook)


# ------------------------------------------------------------------ 学习率 --


def test_expon_lr_hits_endpoints() -> None:
    assert expon_lr(0, 8e-4, 8e-6, 30000) == pytest.approx(8e-4)
    assert expon_lr(30000, 8e-4, 8e-6, 30000) == pytest.approx(8e-6)
    # 超过 max_steps 后不再变化
    assert expon_lr(99999, 8e-4, 8e-6, 30000) == pytest.approx(8e-6)


def test_expon_lr_is_monotone_decreasing() -> None:
    values = [expon_lr(i, 1e-3, 1e-5, 1000) for i in range(0, 1001, 50)]
    assert all(a >= b - 1e-12 for a, b in zip(values, values[1:]))


def test_expon_lr_delay_is_noop_without_delay_steps() -> None:
    """`delay_mult` 只有配 `delay_steps > 0` 才生效（上游语义）。

    RGBAvatar 的配置里写了 `position_lr_delay_mult: 0.01` 却没设
    `lr_delay_steps`，所以那一项实际上不起作用 —— 这里把这个事实锁住。
    """
    for i in (0, 100, 1000):
        assert expon_lr(i, 1e-3, 1e-5, 1000, delay_steps=0, delay_mult=0.01) \
            == pytest.approx(expon_lr(i, 1e-3, 1e-5, 1000, delay_steps=0))
    # 给了窗口才压低
    assert expon_lr(0, 1e-3, 1e-5, 1000, delay_steps=100, delay_mult=0.01) \
        == pytest.approx(1e-5)


def test_expon_lr_rejects_bad_args() -> None:
    with pytest.raises(ValueError, match="max_steps"):
        expon_lr(0, 1e-3, 1e-5, 0)
    with pytest.raises(ValueError, match="lr"):
        expon_lr(0, 0.0, 1e-5, 100)
    with pytest.raises(ValueError, match="delay_steps"):
        expon_lr(0, 1e-3, 1e-5, 100, delay_steps=-1)


def test_blend_active_switches_at_start() -> None:
    assert blend_active(2999, 3000) is False
    assert blend_active(3000, 3000) is True
    assert blend_active(0, 0) is True


def test_early_stopping_stops_after_patience() -> None:
    """`patience` 是"连续多少次无提升就停"。"""
    es = EarlyStopping(patience=2, min_delta=0.0)
    assert es.update(10.0) is False         # 首次 → 记录最好值
    assert es.update(9.0) is False          # 第 1 次无提升
    assert es.update(9.5) is True           # 第 2 次 → 停


def test_early_stopping_improvement_resets_patience() -> None:
    es = EarlyStopping(patience=2, min_delta=0.0)
    es.update(10.0)
    es.update(9.0)
    assert es.update(11.0) is False         # 提升 → 计数清零
    assert es.bad == 0


# ---------------------------------------------------------------- 参数计划 --


def test_rgba_defaults_cover_all_groups() -> None:
    plan = ParameterPlan.rgba_avatar_defaults()
    assert set(plan.enabled) == set(ALL_GROUPS)


def test_opt_groups_requires_matching_model_params() -> None:
    """计划里启用了模型没有的组时必须报错，而不是静默少优化一组。"""
    plan = ParameterPlan.appearance_only()
    with pytest.raises(KeyError, match="opacity"):
        plan.opt_groups({"f_dc": torch.nn.Parameter(torch.zeros(1))})


def test_appearance_only_excludes_geometry() -> None:
    plan = ParameterPlan.appearance_only()
    assert set(plan.enabled) == {"f_dc", "opacity"}
    assert "xyz" not in plan.enabled


def test_plan_apply_lr_writes_into_optimizer() -> None:
    from live3dgsavatar.training.plans import GroupPlan

    params = {"f_dc": torch.nn.Parameter(torch.zeros(3))}
    plan = ParameterPlan(groups={"f_dc": GroupPlan(
        lr=1e-2, final_lr=1e-4, max_steps=100)})
    opt = torch.optim.Adam(plan.opt_groups(params), lr=0.0)
    plan.apply_lr(opt, 50)
    assert opt.param_groups[0]["lr"] < 1e-2
    plan.apply_lr(opt, 100)
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-4)


# ------------------------------------------------------------------ 损失 --


def test_objective_l1_only_matches_masked_l1() -> None:
    pred = torch.rand(1, 3, 4, 4)
    target = torch.rand(1, 3, 4, 4)
    mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    mask[:, :, 1:3, 1:3] = True

    loss, parts = Objective(l1=1.0)(pred=pred, target=target, mask=mask)
    expect = (pred - target).abs()[mask.expand_as(pred)].mean()
    assert parts["l1"] == pytest.approx(float(expect), rel=1e-5)
    assert loss == pytest.approx(expect, rel=1e-5)


def test_objective_mask_weight_includes_background() -> None:
    """mask_weight > 0 时背景也参与（按权重），0 时完全不看背景。"""
    pred = torch.zeros(1, 3, 4, 4)
    target = torch.zeros(1, 3, 4, 4)
    target[:, :, :, 2:] = 1.0                     # 右半边是"背景误差"
    mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    mask[:, :, :, :2] = True                      # 只把左半边算前景

    _, p0 = Objective(l1=1.0, mask_weight=0.0)(pred=pred, target=target, mask=mask)
    _, p1 = Objective(l1=1.0, mask_weight=1.0)(pred=pred, target=target, mask=mask)
    assert p0["l1"] == 0.0
    assert p1["l1"] == pytest.approx(0.5, rel=1e-5)


def test_objective_alpha_term() -> None:
    pred = torch.zeros(1, 3, 4, 4)
    target = torch.zeros(1, 3, 4, 4)
    alpha = torch.full((1, 1, 4, 4), 0.5)
    _, parts = Objective(l1=0.0, alpha=1.0)(
        pred=pred, target=target, alpha=alpha,
        target_alpha=torch.ones(1, 1, 4, 4))
    assert parts["alpha"] == pytest.approx(0.5, rel=1e-5)
    assert parts["total"] == pytest.approx(0.5, rel=1e-5)


def test_objective_ssim_runs_on_cpu() -> None:
    pred = torch.rand(1, 3, 16, 16)
    _, parts = Objective(l1=0.0, ssim=1.0)(pred=pred, target=pred.clone())
    assert parts["ssim"] < 1e-3, parts      # 完全一致 → 损失接近 0


def test_objective_reports_per_term_values() -> None:
    __, parts = Objective(l1=1.0, alpha=0.5)(
        pred=torch.zeros(1, 3, 4, 4), target=torch.zeros(1, 3, 4, 4),
        alpha=torch.zeros(1, 1, 4, 4), target_alpha=torch.zeros(1, 1, 4, 4))
    assert set(parts) == {"l1", "alpha", "total"}


# ---------------------------------------------------------------- 训练循环 --


class _FakeSource:
    """返回固定形状的合成 batch；不读盘。"""

    def __init__(self, n: int = 2, size: int = 8) -> None:
        self.n, self.size = n, size
        self.calls = 0

    def random_batch(self, batch_size: int):
        self.calls += 1
        n, s = batch_size, self.size
        return SimpleNamespace(
            mesh=torch.zeros(n, 4, 3),
            blend_weight=torch.zeros(n, 129),
            image=torch.full((n, 3, s, s), 0.5),
            mask=torch.ones(n, 1, s, s, dtype=torch.bool),
        )

    def iter_batches(self, batch_size: int, shuffle: bool = False):
        yield self.random_batch(batch_size)


def _trainer(*, iterations=3, params=None, render_fn=None, hooks=(), plan=None,
             source=None):
    color = torch.nn.Parameter(torch.zeros(1, 3, 1, 1))
    params = params if params is not None else {"f_dc": color}
    if render_fn is None:
        def render_fn(mesh, weight, batch, it):
            return SimpleNamespace(color=color.expand(batch.image.shape[0], 3, 8, 8),
                                   alpha=torch.ones(batch.image.shape[0], 1, 8, 8))
    return Trainer(source=source or _FakeSource(),
                   plan=plan or ParameterPlan.appearance_only(with_opacity=False),
                   objective=Objective(l1=1.0), render_fn=render_fn,
                   params=params, iterations=iterations, batch_size=2,
                   device="cpu", hooks=hooks)


def test_trainer_runs_and_updates_parameters() -> None:
    color = torch.nn.Parameter(torch.zeros(1, 3, 1, 1))

    def render_fn(mesh, weight, batch, it):
        return SimpleNamespace(color=color.expand(batch.image.shape[0], 3, 8, 8),
                               alpha=torch.ones(batch.image.shape[0], 1, 8, 8))

    t = _trainer(iterations=4, params={"f_dc": color}, render_fn=render_fn)
    report = t.run()
    assert report.iterations == 4
    assert report.last_parts["l1"] > 0
    assert float(color.detach().abs().sum()) > 0, "参数应当被更新"


def test_trainer_respects_blend_start() -> None:
    """`blend_start` 之前必须传 `weight=None`（RGBAvatar 的暖启动）。"""
    seen: list[bool] = []

    color = torch.nn.Parameter(torch.zeros(1, 3, 1, 1))

    def render_fn(mesh, weight, batch, it):
        seen.append(weight is not None)
        return SimpleNamespace(color=color.expand(batch.image.shape[0], 3, 8, 8),
                               alpha=torch.ones(batch.image.shape[0], 1, 8, 8))

    t = _trainer(iterations=4, render_fn=render_fn, params={"f_dc": color})
    t.blend_start = 2
    t.run()
    assert seen == [False, False, True, True], seen


def test_trainer_calls_hooks_in_order() -> None:
    events: list[str] = []

    class H:
        def on_start(self, trainer): events.append("start")
        def on_step(self, trainer, it, parts): events.append(f"step{it}")
        def on_end(self, trainer): events.append("end")

    _trainer(iterations=2, hooks=(H(),)).run()
    assert events == ["start", "step0", "step1", "end"]


def test_trainer_tolerates_partial_hooks() -> None:
    """钩子只实现部分方法（LogHook 没有 on_end）时不能崩。"""
    _trainer(iterations=2, hooks=(LogHook(every=100),)).run()


def test_trainer_stop_request_breaks_loop() -> None:
    class StopNow:
        def on_step(self, trainer, it, parts):
            trainer.request_stop()

    t = _trainer(iterations=100, hooks=(StopNow(),))
    report = t.run()
    assert report.iterations < 100


def test_checkpoint_hook_saves_at_intervals_and_end() -> None:
    saved: list[int] = []
    t = _trainer(iterations=5, hooks=(
        CheckpointHook(save=saved.append, every=2),))
    report = t.run()
    # 第 2、4 步（0 基 → iteration+1 % every == 0）+ 结束
    assert saved == [1, 3, report.iterations], saved


def test_validation_hook_early_stops_on_plateau() -> None:
    values = iter([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    def evaluate(_source):
        return {"psnr": next(values, 1.0)}

    t = _trainer(iterations=100, hooks=(
        ValidationHook(source=_FakeSource(), evaluate=evaluate, every=1,
                       early_stop=EarlyStopping(patience=2, min_delta=0.1)),))
    report = t.run()
    assert report.stopped_early is True
    assert report.best_metric == pytest.approx(1.0)


def test_trainer_rejects_bad_args() -> None:
    with pytest.raises(ValueError, match="iterations"):
        _trainer(iterations=0)
    with pytest.raises(ValueError, match="batch_size"):
        Trainer(source=_FakeSource(), plan=ParameterPlan.appearance_only(
            with_opacity=False), objective=Objective(l1=1.0),
            render_fn=lambda *a: None, params={"f_dc": torch.nn.Parameter(
                torch.zeros(1))}, iterations=1, batch_size=0, device="cpu")
