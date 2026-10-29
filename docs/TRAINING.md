# 训练管线

> 参数计划 / 损失 / 调度 / 钩子都是可配置对象；在线微调与全量训练共用同一条循环。

```
app/  ──►  training/  ──►  core/            （单向）
             ▲
data/sources ┘  （只提供样本，不依赖 training）
```

| 模块 | 职责 |
|---|---|
| `data/sources.py` | 读 `render_inputs.npz` + `images/`：切分、采样、背景合成、预取、混训 |
| `training/plans.py` | 哪些参数组优化、各自学习率曲线 |
| `training/schedule.py` | lr 曲线、形变基暖启动、早停判据 |
| `training/objectives.py` | 加权损失（L1/SSIM/LPIPS/alpha），返回分项 |
| `training/trainer.py` | 训练循环 + 钩子（验证/存档/日志/时间预算） |
| `training/densify.py` | 自适应加密/剪枝（**待实现**）|

`trainer` 通过注入 `render_fn` 与渲染解耦，所以调度、钩子、早停都能**离线单测**。

在线微调就是换三个参数：

```python
Trainer(source=LiveSource(...),
        plan=ParameterPlan.appearance_only(),
        objective=Objective.appearance_only(),
        hooks=[TimeBudgetHook(10.0)], ...)
```

## 用法

```bash
python scripts/train.py --subject crh_1 --work-name run5 --dry-run   # 只查数据与配置
python scripts/train.py --subject crh_1 --work-name run5 --promote   # 训练并装进 models/
```

常用覆盖：`--iterations` `--batch-size` `--blend-start` `--test-ratio`
`--mask-weight` `--eval-every` `--patience` `--prefetch-workers` `--augment`。

产物（`output/train/<对象>/<版本>/`）：

```
model.ply                 训练结束时的模型（PLY 头内嵌结构自描述）
model_best.ply            验证最好的一次（--promote 优先用它）
model_XXXXXX.ply          定期存档
config.yaml / report.json 结构副本与指标
```

`promote_model` 把 `model.ply` + `config.yaml` + `report.json` 拷进
`models/<对象>/<版本>/`；目标已存在时**默认拒绝覆盖**（`--force` 才覆盖）。

> 一个 work-name 对应一个 `model.ply`。界面与脚本都按这个约定找模型，
> 所以增强训练也用普通文件名，靠 work-name 区分（`run5` / `run5_aug`）。

## 切分

`train.pipeline.test_ratio`（默认 `0.1`）表示**末尾 10% 作验证集**。
crh_1 共 2863 帧 → 训练 2577 / 验证 286。增强集按同一比例切分。

用比例而非帧数：换数据集不必重算参数，也让增强集能自动对齐。

## 模型结构

训练新模型时结构取自模型目录的 `config.yaml`（没有就用 `configs/model.yaml`）。
`AvatarConfig.from_yaml` 负责读取，缺的键用默认值，并接受 `use_blend` /
`use_mlp_proj` 这两个旧开关。按优先级回退：

1. `--model-config` 显式给出
2. `<models_dir>/<对象>/<版本>/config.yaml`（续训/复现）
3. `<models_dir>/<对象>/*/config.yaml`（同对象任一版本）
4. 仓库模板 `configs/model.yaml`

第 4 条是必要的：否则"还没训过"会被报成"还没有选择模型"。

## 三个刻意的取舍

- **背景参与损失**（`mask_weight=1`）：只在前景 mask 内算损失时，落在人像
  之外的高斯拿不到梯度，推理时表现为"头周围发散"。
- **位置学习率的 `scene_extent` 只乘一次**：并入 `GroupPlan` 的起止值，
  循环里不再缩放，避免换数据时隐式平方。
- **起步延迟默认关闭**：`delay_mult` 只有配 `delay_steps > 0` 才生效，
  默认 `delay_steps=0, delay_mult=1.0`。

## 速度：GPU 利用率上不去时先查取数

取数（解 PNG）常比一步计算还慢，三档手段按见效顺序：

1. **预取**（默认开，`prefetch_workers=2`）：实测 85.8 → 42.2 ms/batch（2.0×）
2. **图像缓存**：`python scripts/cache_images.py --subject crh_1`，训练走 memmap 直读
3. **加大 batch**：`batch_size` 10 → 16–24，摊薄每步固定开销

判断瓶颈：加大 batch 后 `it/s × batch_size` 上升 → 固定开销大；不变 → 数据瓶颈。

## 早停与最佳模型

```yaml
train.pipeline:
  eval_every: 1000    # 每 1000 步验证
  patience: 3         # 连续 3 次无提升就停（= 3000 步）
  min_delta: 0.05     # 提升小于 0.05 dB 不算
```

- 验证刷新成绩时另存 `model_best.ply`，`--promote` 优先装它
- `patience × eval_every` 才是"容忍多少步没有进步"；想更短就先减 patience
- `min_delta` 太小会把噪声当提升，0.05 dB 一般合适

## 增强数据（可选）

外部生成的图可作补充训练样本，放在 `<数据集>/augmented/`，
格式与主数据集一致。生成分两步（跨环境）：

```bash
# 一条命令：需要时先跑 RDF，再整理进 <数据集>/augmented/
./build_dataset.sh <图像目录> --into crh_1
```

`build_dataset.sh` 会检测目录里是否已有跟踪结果：有就跳过 RDF 直接整理，没有就
先在 `realdenseface` 环境里跟踪（环境名可用 `RDF_ENV` / `MAIN_ENV` 覆盖）。

目录约定见 [`CONVENTIONS.md`](CONVENTIONS.md) §6：`images/` + `render_inputs.npz`，
与数据集完全一致。

训练时加 `--augment` 才启用：

```bash
python scripts/train.py --subject crh_1 --work-name run5_aug --augment --promote
```

- `train.pipeline.augment_weight`（默认 1.0）：增强样本的损失权重，0 = 完全不考虑
- 采样按帧数占比自然混入；**验证集永远只含真实数据**

判断有没有用，两条都要看：留出集**不下降**（底线），然后是否**上升**（增益）。
只满足前者说明"没坏但没用"，应调小权重或放弃。

⚠️ 对比必须同切分：`test_ratio` 一变，验证集就变，数字不可比。

## 验收阶段

| 阶段 | 内容 | 验收 |
|---|---|---|
| **P0**（已通）| sources / plans / schedule / objectives / trainer | crh_1 留出集 PSNR 与基线同量级 |
| **P1** | 加密 + 剪枝（重心坐标空间） | 同迭代数下 PSNR 提升 |
| **P2** | 随机背景、逐帧内参、fused-ssim | 训练更快；白底合成数据可训 |
| **P3** | 局部修复式增广（见 `CAP4D.md` 的判据）| crh_2 提升且 crh_1 不降 |
| **P4** | `updater` 收敛为 trainer 薄壳 | 热更新数字与 `finetune` 一致 |
