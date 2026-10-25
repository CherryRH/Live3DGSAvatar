"""参数计划：**哪些参数参与优化、各自用什么学习率曲线**。

一次训练由"优化什么"和"怎么优化"两部分描述。本模块负责前者：
把模型的可训练参数按语义分组，每组给一条学习率曲线。这样：

- 完整训练：打开全部组，位置组用指数衰减，迭代几万步
- 在线微调：只开外观组（颜色/不透明度），常数学习率，几百步内结束

两者共用同一个训练循环（见 `trainer.py`），区别只是这里的计划不同。

## 默认值

与 `models/crh_1/test/config.yaml` 里的训练配置一致：

| 组 | lr | 说明 |
|---|---|---|
| `xyz` | 8e-4 → 8e-6，30000 步 | 位置（乘 `scene_extent`）|
| `f_dc` | 0.0125 | 颜色 DC |
| `opacity` | 0.25 | logit |
| `scaling` | 0.025 | |
| `rotation` | 0.005 | |
| `xyz_b` | 8e-4 × 0.05 → 8e-6 × 0.05 | 位置基 |
| `rotation_b` | 0.005 × 0.5 | 旋转基 |
| `f_dc_b` | 0.0125 × 0.5 | 颜色基 |
| `weight_module` | 1e-3 | 权重投影 MLP |

位置类学习率只乘**一次** `scene_extent`：它在 `GroupPlan` 构造时就并入了
起点/终点值，循环里不再重复缩放。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .schedule import expon_lr

#: 模型里可优化的参数组（与 `GaussianAvatar.parameter_groups()` 一致）
ALL_GROUPS = ("xyz", "opacity", "scaling", "rotation", "f_dc",
              "xyz_b", "rotation_b", "f_dc_b", "weight_module")


@dataclass(frozen=True)
class GroupPlan:
    """单个参数组的学习率计划。

    Attributes:
        lr: 初始学习率
        final_lr: 终止学习率；``None`` 表示常数
        max_steps: 指数衰减的步数上限
        delay_steps: 起步延迟步数；**0 表示不延迟**（此时 delay_mult 无效）
        delay_mult: 延迟窗口内的起始倍率
    """

    lr: float
    final_lr: float | None = None
    max_steps: int = 30_000
    delay_steps: int = 0
    delay_mult: float = 1.0

    def __post_init__(self) -> None:
        if self.lr < 0:
            raise ValueError(f"lr 不能为负，实际 {self.lr}")
        if self.final_lr is not None and self.final_lr < 0:
            raise ValueError(f"final_lr 不能为负，实际 {self.final_lr}")
        if self.delay_steps < 0:
            raise ValueError(f"delay_steps 不能为负，实际 {self.delay_steps}")

    def at(self, iteration: int) -> float:
        """第 `iteration` 步的学习率。"""
        if self.final_lr is None:
            return self.lr
        return expon_lr(iteration, self.lr, self.final_lr, self.max_steps,
                        self.delay_steps, self.delay_mult)


def _as_param_list(name: str, value) -> list:
    """把"一组参数"规整成 `Parameter` 列表。"""
    import torch.nn as nn

    if isinstance(value, nn.Parameter):
        return [value]
    if isinstance(value, nn.Module):
        found = [p for p in value.parameters() if p.requires_grad]
        if not found:
            raise ValueError(f"参数组 {name!r} 是 Module，但里面没有可训练参数")
        return found
    try:
        items = list(value)
    except TypeError:
        raise TypeError(
            f"参数组 {name!r} 应为 Parameter / Module / Parameter 序列，"
            f"实际 {type(value).__name__}") from None
    bad = [type(v).__name__ for v in items if not isinstance(v, nn.Parameter)]
    if bad:
        raise TypeError(f"参数组 {name!r} 的序列里混入了非 Parameter：{bad}")
    if not items:
        raise ValueError(f"参数组 {name!r} 是空序列")
    return items


@dataclass
class ParameterPlan:
    """整模型的计划：启用哪些组 + 每组的曲线。

    Attributes:
        groups: 组名 → 计划。**只包含启用的组** —— 不在表里的组不参与优化。
        scene_extent: 位置类学习率的缩放系数（场景尺度归一化用）
    """

    groups: dict[str, GroupPlan] = field(default_factory=dict)
    scene_extent: float = 1.0

    # ------------------------------------------------------------ 构造 --

    @classmethod
    def rgba_avatar_defaults(cls, *, scene_extent: float = 1.0) -> "ParameterPlan":
        """复现 crh_1 的 `config.yaml` 里那套学习率。"""
        return cls(scene_extent=float(scene_extent), groups={
            "xyz": GroupPlan(lr=8e-4 * scene_extent,
                             final_lr=8e-6 * scene_extent,
                             max_steps=30_000),
            "f_dc": GroupPlan(lr=0.0125),
            "opacity": GroupPlan(lr=0.25),
            "scaling": GroupPlan(lr=0.025),
            "rotation": GroupPlan(lr=0.005),
            "xyz_b": GroupPlan(lr=8e-4 * scene_extent * 0.05,
                               final_lr=8e-6 * scene_extent * 0.05,
                               max_steps=30_000),
            "rotation_b": GroupPlan(lr=0.005 * 0.5),
            "f_dc_b": GroupPlan(lr=0.0125 * 0.5),
            "weight_module": GroupPlan(lr=1e-3),
        })

    @classmethod
    def appearance_only(cls, *, lr_color: float = 2e-3,
                        lr_opacity: float = 2e-2,
                        with_opacity: bool = True) -> "ParameterPlan":
        """只调外观：在线微调用。常数学习率，不碰几何与驱动。"""
        groups = {"f_dc": GroupPlan(lr=float(lr_color))}
        if with_opacity:
            groups["opacity"] = GroupPlan(lr=float(lr_opacity))
        return cls(groups=groups)

    @classmethod
    def from_groups(cls, names, *, lr: float) -> "ParameterPlan":
        """给定组名，统一用同一个常数学习率（调试用）。"""
        unknown = [n for n in names if n not in ALL_GROUPS]
        if unknown:
            raise ValueError(f"未知参数组：{unknown}；可用 {list(ALL_GROUPS)}")
        return cls(groups={n: GroupPlan(lr=float(lr)) for n in names})

    # ------------------------------------------------------------ 查询 --

    @property
    def enabled(self) -> tuple[str, ...]:
        return tuple(self.groups)

    def lr_at(self, group: str, iteration: int) -> float:
        plan = self.groups.get(group)
        return 0.0 if plan is None else plan.at(iteration)

    def opt_groups(self, params: dict) -> list[dict]:
        """按 `parameter_groups()` 生成优化器参数组（只含启用的）。

        一组的值可以是三种东西，这里统一展开成 `Parameter` 列表：

        - 单个 `nn.Parameter`（大多数高斯属性）
        - 一个 `nn.Module`（`weight_module` 是 `nn.Sequential`，
          之前的实现把它整个塞给 Adam，报
          "optimizer can only optimize Tensors"）
        - 可迭代的 `Parameter`

        Args:
            params: `GaussianAvatar.parameter_groups()` 的返回

        Raises:
            KeyError: 计划里启用了模型没有的组 —— 配置与模型对不上时必须报错，
                而不是静默少优化一组
            TypeError: 组的值不是上面三种之一
        """
        import torch.nn as nn

        out = []
        for name, plan in self.groups.items():
            if name not in params:
                raise KeyError(
                    f"参数计划里启用了 {name!r}，但模型没有这一组；"
                    f"模型有 {sorted(params)}")
            out.append({"params": _as_param_list(name, params[name]),
                        "name": name, "lr": plan.at(0)})
        return out

    def apply_lr(self, optimizer, iteration: int) -> None:
        """把第 `iteration` 步的学习率写回优化器。"""
        for group in optimizer.param_groups:
            name = group.get("name")
            if name in self.groups:
                group["lr"] = self.lr_at(name, iteration)
