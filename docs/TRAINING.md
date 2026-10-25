# 训练管线

> 把"参数计划 / 损失 / 调度 / 钩子"拆成可配置对象，
> 让**在线微调与全量训练共用同一条循环**。

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
钩子何时触发、早停是否生效）可以在**没有 GPU** 的情况下单测，
不必等真跑起来才发现调度写错。

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

## 3. 设计要点

| 关注点 | 我们怎么做 |
|---|---|
| 学习率 | 组 → 计划的查表（`ParameterPlan`），加一组参数不用改循环 |
| 损失 | `Objective` 对象，权重进配置，返回值带分项 |
| 暖启动 | `blend_start` 之前只用模板网格，之后才启用形变基 |
| 验证/存档/日志 | 都是钩子，训练、微调、冒烟测试共用 |
| 中断 | 定期存 `model_XXXXXX.ply`，不必一次跑完 |
| 选模型 | 验证集 PSNR 驱动早停（`EarlyStopping`） |
| 训练态专有逻辑（如颜色回写）| 默认关闭，需要时做成钩子 |

两处与直觉不同的地方，刻意写在这里：

- **位置学习率的 `scene_extent` 只乘一次**。它在 `GroupPlan` 构造时并入
  起点/终点值，循环里不再重复缩放 —— 否则换数据时会隐式平方。
- **起步延迟默认关闭**。`delay_mult` 只有配 `delay_steps > 0` 才生效；
  默认计划是 `delay_steps=0, delay_mult=1.0`，需要时显式打开。

## 4. 数据切分

**末尾 350 帧为验证集**（`train.pipeline.test_frames`），其余为训练集。
按文件名排序后取最后 N 帧 —— 视频是连续的，末尾一段与开头最不相似，
用它当验证集比随机采样更能暴露过拟合。

## 5. 模型结构从哪来

训练新建模型时，结构取自模型目录里的 `config.yaml`：

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

**训练新模型时** `<模型目录>/config.yaml` 还不存在，所以 `scripts/train.py`
按优先级回退（`resolve_structure_config`）：

1. `--model-config` 显式给出
2. `<models_dir>/<subject>/<work>/config.yaml`（续训/复现）
3. `<models_dir>/<subject>/*/config.yaml`（同对象任一版本）
4. 仓库模板 `configs/model.yaml`（新建模型的默认结构）

第 4 条是有意存在的：否则"还没训过"会被报成"还没有选择模型"，
让人以为数据没配好。

## 6. 用法

```bash
# 只检查数据与配置（不需要 GPU）
python scripts/train.py --subject crh_1 --work-name test --dry-run

# 正式训练（默认 batch 10、50000 步、blend_start 3000）
python scripts/train.py --subject crh_1 --work-name test

# 覆盖步数/批大小
python scripts/train.py --subject crh_1 --work-name test --iterations 5000 --batch-size 4
```

产物：`output/train/<subject>/<work_name>/`

```
model.ply                 训练结束时的模型（PLY 头内嵌结构自描述）
model_000000.ply …        定期存档（CheckpointHook）
config.yaml               结构配置副本（可读，便于换机器）
report.json               步数、耗时、最佳验证 PSNR
```

### 让渲染用上新模型

```bash
# 训练结束后直接装进 models/<对象>/<版本>/
python scripts/train.py --subject crh_1 --work-name run2 --promote

# 覆盖同名模型要显式确认
python scripts/train.py ... --promote --force

# 也可以只装某个存档点
python scripts/train.py ... --promote --promote-from output/train/crh_1/run2/model_030000.ply
```

`promote_model` 会拷 `model.ply` + `config.yaml` + `report.json`，目标已存在时
默认**拒绝覆盖**。

### 在留出集上评估

训练用末尾 350 帧作验证集，评估时也要落在同一段上：

```bash
python scripts/render_test.py --subject crh_1 --work-name run2 \
    --frame-offset 2513 --frames 350 --dump-diff
```

`--frame-offset` 指定从第几帧开始取（crh_1 总 2863 帧、验证集从 2513 起）。

## 7. 阶段与验收

| 阶段 | 内容 | 验收 |
|---|---|---|
| **P0**（已做骨架）| sources / plans / schedule / objectives / trainer / hooks | 在 crh_1 上训到与现成模型**同量级 PSNR**（±0.5 dB）|
| **P1** | 加密 + 剪枝（重心坐标空间） | 同迭代数下 PSNR 提升，高斯数可解释 |
| **P2** | 随机背景/掩码、逐帧内参、fused-ssim | 训练速度提升；白底合成数据可训 |
| **P3** | CAP4D 数据做增广微调 | 在 crh_2（同人不同采集）上提升，crh_1 不降 |
| **P4** | `updater` 收敛为 trainer 的薄壳 | 热更新的数字与 `finetune` 一致 |
