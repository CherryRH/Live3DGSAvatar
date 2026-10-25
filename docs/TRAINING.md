# 训练管线

> 对应 RGBAvatar 的 `train_offline.py`，但把"参数计划 / 损失 / 调度 / 钩子"
> 拆成可配置对象，并让**在线微调与全量训练共用同一条循环**。

---

## 1. 模块与依赖

```
app/  ──►  training/  ──►  core/            （单向）
             ▲
data/sources ┘  （只提供样本，不依赖 training）
```

| 模块 | 职责 | 不做什么 |
|---|---|---|
| `data/sources.py` | 读 `render_inputs.npz` + `images/`，切分、采样、背景合成 | 不碰渲染、不建模型 |
| `training/plans.py` | 参数计划：哪些组优化、各自 lr 曲线 | 不建优化器 |
| `training/schedule.py` | lr 曲线、blend 暖启动、早停判据 | 不读写状态 |
| `training/objectives.py` | 加权损失，返回分项 | 不反传 |
| `training/trainer.py` | 训练循环 + 钩子（验证/存档/日志/时间预算） | 不知道模型长什么样 |
| `training/densify.py` | 自适应加密/剪枝（**P1，待实现**）| — |

`trainer` 通过注入 `render_fn` 与渲染解耦。因此训练逻辑（学习率怎么写回、
钩子何时触发、早停是否生效）可以在**没有 GPU** 的情况下单测 ——
RGBAvatar 把这一切写在 `Reconstruction.step` 里，做不到。

## 2. 一次训练由四样东西组成

```python
Trainer(
    source    = ManifestSource(data_dir, split="train"),   # 数据
    plan      = ParameterPlan.rgba_avatar_defaults(),      # 优化什么
    objective = Objective(l1=1.0),                         # 优化目标
    render_fn = lambda mesh, w, batch, it: ...,            # 怎么渲染
    iterations=50000, batch_size=10, blend_start=3000,
    hooks     = [LogHook(), ValidationHook(...), CheckpointHook(...)],
)
```

在线微调 = 换三个参数：

```python
Trainer(source=LiveSource(...),
        plan=ParameterPlan.appearance_only(),
        objective=Objective.appearance_only(),
        hooks=[TimeBudgetHook(10.0)], ...)
```

## 3. 与 RGBAvatar 的对应与差异

| RGBAvatar | 本项目 | 原因 |
|---|---|---|
| `config/offline.yaml` 里的学习率 | `ParameterPlan.rgba_avatar_defaults()` | 组 → 计划，加一组不用改循环 |
| `Reconstruction.update_learning_rate` 的 if/elif | `ParameterPlan.apply_lr` | 同上 |
| 损失写成一串加法 | `Objective` | 权重进配置，返回分项 |
| `if iteration > 20000` 才开 LPIPS | `Objective.lpips_from` | 变成参数 |
| `blend_start_iter` 隐式约定 | `Trainer(blend_start=...)` | 显式 |
| `fast_forward` 死写在 `step` 里 | 默认关（可作钩子） | 它改写颜色参数，调参时干扰判断 |
| 数据加载器无 epoch/验证集 | `ManifestSource` 支持 train/test | 早停/选模型需要 |
| 只有最终 `save_ply` | `CheckpointHook` | 中断不丢、可选最佳 |

### 一处刻意的偏差

RGBAvatar 的位置学习率把 `scene_extent` 乘了**两次**（构造调度器时一次、
`update_learning_rate` 里又乘一次）。它的配置里 `scene_extent = 1.0`，所以
数值上无差别；这里只乘一次，避免换数据时出现隐式平方。

### 一处上游"空转"的参数

`position_lr_delay_mult: 0.01` 只有配 `lr_delay_steps > 0` 才生效，
而 RGBAvatar 没有设后者 → 该项实际不起作用。我们把 `delay_steps` 显式暴露在
`GroupPlan` 里，默认 0（与上游行为一致），并在测试里锁住了这个语义。

## 4. 数据切分

沿用 RGBAvatar：**末尾 350 帧为验证集**（`train.pipeline.test_frames`）。
它在 `dataset/flame_dataset.py` 里写成 `images[:-350]` / `images[-350:]`。

## 5. 模型结构从哪来

训练新建模型时，结构取自 `<模型目录>/config.yaml`（RGBAvatar 的格式）：

```yaml
model:
  tex_size: 256
  num_basis_in: 129
  num_basis_blend: 20
  use_blend: true          # 兼容：False 等价于不做权重投影
  use_mlp_proj: true       # 兼容：False 等价于单层线性
```

`AvatarConfig.from_yaml` 负责读取，缺的键用默认值（老配置可能没有
`mlp_hidden`），并接受 `use_blend` / `use_mlp_proj` 这两个旧开关。

## 6. 用法

```bash
# 只检查数据与配置（不需要 GPU）
python scripts/train.py --subject crh_1 --work-name test --dry-run

# 正式训练（默认 batch 10、50000 步、blend_start 3000，与 RGBAvatar 一致）
python scripts/train.py --subject crh_1 --work-name test

# 覆盖步数/批大小
python scripts/train.py --subject crh_1 --work-name test --iterations 5000 --batch-size 4
```

产物：`output/train/<subject>/<work_name>/`

```
model_000000.ply …        定期存档（CheckpointHook）
report.json               步数、耗时、最佳验证 PSNR
```

## 7. 阶段与验收

| 阶段 | 内容 | 验收 |
|---|---|---|
| **P0**（已做骨架）| sources / plans / schedule / objectives / trainer / hooks | 在 crh_1 上训到与现成模型**同量级 PSNR**（±0.5 dB）|
| **P1** | 加密 + 剪枝（重心坐标空间） | 同迭代数下 PSNR 提升，高斯数可解释 |
| **P2** | 随机背景/掩码、逐帧内参、fused-ssim | 训练速度提升；白底合成数据可训 |
| **P3** | CAP4D 数据做增广微调 | 在 crh_2（同人不同采集）上提升，crh_1 不降 |
| **P4** | `updater` 收敛为 trainer 的薄壳 | 热更新的数字与 `finetune` 一致 |
