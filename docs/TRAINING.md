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

## 大角度验证与早停

默认开启：验证时在 24 帧上加渲染 ±20/30/40° 视角，统计轮廓内 `alpha < 0.5` 的面积
比例，用 `PSNR − 10·缺口` 作为早停指标。只按正脸 PSNR 早停会选出"只在正脸好"的模型。

```bash
python scripts/train.py --subject crh_1 --work-name run --no-robust-eval   # 关掉
```

指标含义见 [`REDESIGN.md`](REDESIGN.md)。

## 模型增强流程（一条命令）

从找缺陷到落地评测，四个阶段串成一条命令：

```bash
python scripts/enhance_model.py --subject crh_1 --work-name run \
    --syncfix-dir ~/Projects/SyncFix --syncfix-model <权重> \
    --promote-name run_refined
```

| 阶段 | 调用 | 产物 |
|---|---|---|
| 1 找缺陷 | `find_defects.py` | `<缺陷目录>/{selected.json, views/, refs/, mass.npz}` |
| 2 外部精修 | SyncFix `inference.py` | `syncfix_raw/` → 整理成 `refined/` |
| 3 微调 | `refine_from_images.py` | `<缺陷目录>/finetune/model.ply` + `models/<对象>/<名字>_refined/` |
| 4 评测 | `render_test.py` ×2、`compare_defects.py` | 留出集 PSNR / 锐度比 / 结构缺口 |

- 每步都能单独重跑：`--skip-find` / `--skip-syncfix` / `--skip-train`；
- 状态写在 `<缺陷目录>/enhance_pipeline.json`，重跑自动跳过已完成的步骤（`--force` 强制）；
- 精修输出若文件名对不上视角标签，会按排序顺序对应并打印警告；
- **搜索范围可调**：默认 ±40°/±20°、20 个表情、预算 16；调大即可覆盖更极端的情形：

  ```bash
  python scripts/enhance_model.py --subject crh_1 --work-name run \
      --yaw-limit 60 --pitch-limit 30 --rolls -15 0 15 \
      --clusters 12 --axes 10 --budget 32 \
      --min-angle-sep 8 --per-base-max 2
  ```

  `--min-angle-sep` 限制已选视角之间的最小 (yaw, pitch) 距离，`--per-base-max`
  限制同一基帧最多选几个——不加这两个，名额容易被"同一角度的不同表情"占满。
  改了搜索参数后第 1 步会自动重跑（旧结果失效）。
- `--dry-run` 只打印将执行的命令；
- SyncFix 有独立环境时用 `--syncfix-env syncfix`（内部走 `conda run --no-capture-output`），
  或 `--syncfix-python <解释器路径>`；不指定时会自动检测名为 `syncfix` 的环境，
  并在执行前探测 `imageio`，缺依赖会直接提示。

## 精修图微调

`find_defects.py` 挑出待修视角并渲染，外部精修器（如 SyncFix）输出到
`<缺陷目录>/refined/` 之后：

```bash
python scripts/refine_from_images.py --subject crh_1 --work-name run \
    --steps 600 --promote --promote-name run_refined --force
```

与上一版相比收紧了三处：

| 项 | 做法 | 原因 |
|---|---|---|
| 精修图体检 | 前景 IoU < 0.5 或色差 > 60 的视角直接跳过 | 避免把坏样本当目标 |
| 高频保持 | 可选 SSIM 项（默认 0） | 上一版 +0.9 dB 但高频能量 −13%；SSIM 的 backward 在个别构建上会触发 CUDA illegal memory access，默认关闭 |
| 检查点选择 | 每 100 步算留出 PSNR 与锐度比，`score = PSNR − 20·max(0, 0.85 − 锐度比)` | 锐度掉太多会被扣分，自动拒绝"用模糊换 PSNR" |

参数范围是颜色 + 不透明度 + 尺度（尺度是补洞的必要项，前一轮已对照验证）。
每批混入 4 帧真实数据锚定；产物写到 `<缺陷目录>/finetune/`，`report.json` 记录
留出指标与逐次评估历史。

## 缺陷评估口径

`find_defects.py` 选出问题视角，`compare_defects.py` 对比两个模型在这些视角上的
结构性指标：

```bash
python scripts/find_defects.py --subject crh_1 --work-name run --dataset crh_1
python scripts/compare_defects.py --subject crh_1 --work-name run2 \
    --base-work run --dataset crh_1
```

| 指标 | 定义 | 说明 |
|---|---|---|
| `hole` | 面 id 图说有表面、但 `alpha < 0.5` 的比例 | 结构断裂/空洞 |
| `spike` | 轮廓内 alpha 的梯度能量 | 表面碎裂/尖刺 |
| `云外溢` | 渲染为实、但网格没有面的像素比例 | 点云与网格的错位程度 |
| `psnr_refined` | 与该视角精修图的 PSNR | 拟合程度，**不是质量判据** |

结论与路线分析见 [`REDESIGN.md`](REDESIGN.md)。

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
