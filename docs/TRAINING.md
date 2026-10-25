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
| 背景监督 | 默认 `mask_weight=1`：落在人像之外的高斯也要被压住，否则会出现"头周围发散" |
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

按**比例**切分：`train.pipeline.test_ratio`（默认 `0.1`）表示末尾 10% 作验证集，
其余训练。例如 crh_1 有 2863 帧 → 训练 2577 / 验证 286。

用比例而不是固定帧数，是为了换数据集时不用重新算参数；也便于让
**增强集按同一比例**切分。

想按固定帧数也可以：`--test-frames 350`（此时忽略比例）。

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

### 在留出集上评估（一次渲染全部模型）

`render_test.py` 会自动扫描模型目录下的所有 `.ply`，逐个渲染：

```bash
python scripts/render_test.py --subject crh_1 --work-name run4 \
    --frame-offset 2577 --frames 286
```

产物（每个模型互不覆盖）：

```
output/render_test/crh_1/run4/
├── model/                 ← 常规模型
│   ├── images/
│   └── report.json
├── model_aug/             ← 增强模型
│   ├── images/
│   └── report.json
└── compare/               ← 统一对比：原图 | model | model_aug …
    ├── 02577.png …
    └── report.json        （各模型中位 PSNR）
```

只看一个模型时用 `--ply` 指定路径即可；`--no-diff` 可关掉对比图。



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

## 8. 速度：GPU 利用率上不去时先查取数

训练时 GPU 空转，绝大多数是**取数在主线程串行做**：解一张 512² PNG 要几毫秒，
一个 batch 就是几十毫秒，而一步计算只有十几毫秒。三档手段，按见效顺序：

**① 预取（默认开）**

```yaml
train.pipeline:
  prefetch_workers: 2      # 后台线程数；0 关闭
```

本机实测（batch 4，CPU 解码）：**85.8 ms/batch → 42.2 ms/batch（2.0×）**。
`close()` 由 Trainer 在结束时调用，线程不会泄漏。

**② 图像缓存（最有效）**

```bash
python scripts/cache_images.py --subject crh_1     # 生成 datasets/crh_1/images.npy
```

把 PNG 打包成 `uint8 [N,H,W,C]`，训练时走 `np.load(mmap_mode="r")`，跳过解码。
`ManifestSource` 自动检测并优先用它。注意：**图像更新后要重新生成**。

**③ 加大 batch**

GPU 利用率低时，把 `train.pipeline.batch_size` 从 10 提到 16–24（显存够的话）。
批越大，每步的固定开销摊得越薄。

三者可以叠加。判断瓶颈在哪：看日志里的 `it/s` —— 如果加大 batch 后
`it/s × batch_size` 明显上升，瓶颈在固定开销；如果不变，瓶颈在数据。

## 9. 早停与"最佳模型"

```yaml
train.pipeline:
  eval_every: 1000    # 每 1000 步验证一次
  patience: 3         # 连续 3 次无提升就停（3 × 1000 = 3000 步）
  min_delta: 0.05     # 提升小于 0.05 dB 不算提升
```

- 验证刷新最好成绩时，会另存一份 `model_best.ply`
- `--promote` 默认装 **`model_best.ply`**（没有就装最终的 `model.ply`）
- `--promote-from` 可以指定任意存档点

调早停的三条经验：

1. `min_delta` 太小 → 噪声被当成提升，迟迟不停；0.05 dB 一般合适
2. `patience × eval_every` 才是"容忍多少步没有进步"；想让总时长更短，
   先减 `patience`，再考虑减 `eval_every`
3. 验证集和训练集同分布时早停很容易触发；跨采集（crh_2）才是真考验

## 10. 增强数据（可选）

把外部生成的图（例如 CAP4D 合成的人像）当作**补充训练样本**，看能否提升。

### ① 准备数据

增强数据放在 `<数据集>/augmented/`，格式与主数据集一致：

```
datasets/crh_1/augmented/
├── images/00000.png …        增强图
└── render_inputs.npz         mesh(5083) / blend_weight(129) / 相机 / names
```

生成这个目录（两步，跨环境）：

```bash
# 让 RDF 追踪增强图（realdenseface 环境）
python scripts/rdf_dump.py --images <增强图目录> --out <增强图目录>/rdf.npz

# 打包成训练格式（live3dgs 环境）
python scripts/build_augmented.py --subject crh_1 \
    --rdf <增强图目录>/rdf.npz --images <增强图目录>
```

### ② 训练时启用

```bash
python scripts/train.py --subject crh_1 --work-name run4 --augment --promote
```

- `--augment` 才启用；不加就完全走真实数据
- `train.pipeline.augment_weight`（默认 1.0）：增强样本的**损失权重**，
  0 表示完全不考虑（此时等于没开）
- 采样按"帧数占比"自然混入：crh_1 2577 帧 + 增强 200 帧 ≈ 7% 的 batch 来自增强

### ③ 产物命名（便于对比）

增强训练的模型文件名带 `_aug` 后缀：

```
output/train/crh_1/run4/model_aug.ply       最终
output/train/crh_1/run4/model_best_aug.ply  验证最好的
models/crh_1/run4/model_aug.ply             装进 models/ 后
```

同一目录下可以和常规 `model.ply` **并存**，直接用 `--ply` 指定对比：

```bash
python scripts/render_test.py --subject crh_1 --work-name run4 \
    --ply models/crh_1/run4/model_aug.ply --frame-offset 2513 --frames 350
```

### ④ 判断有没有用

两条都要看：

1. **留出集（真实数据）不下降** —— 这是底线，增强不能污染真实分布
2. **留出集上升** —— 才是真的增益

只满足 1 说明"没坏但也没用"，此时应把 `augment_weight` 调小或直接放弃。

⚠️ **对比必须用同一个切分**。`test_ratio` 变了，验证集就变了，数字不可比：

```bash
# A（无增强）与 B（有增强）都要跑
python scripts/train.py --subject crh_1 --work-name ab --promote
python scripts/train.py --subject crh_1 --work-name ab --augment --promote

# 两边都用同一段留出集评估
python scripts/render_test.py --subject crh_1 --work-name ab \
    --frame-offset 2577 --frames 286
python scripts/render_test.py --subject crh_1 --work-name ab \
    --ply models/crh_1/ab/model_aug.ply --frame-offset 2577 --frames 286
```

`--frame-offset` 要按 `test_ratio=0.1` 算：crh_1 共 2863 帧 → 验证集从 2577 起。

