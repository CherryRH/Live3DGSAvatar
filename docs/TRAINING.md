# training

`src/live3dgsavatar/training/` 是训练循环与参数计划；基础训练入口是
`scripts/train.py`，在线更新入口是 `scripts/online_update.py`。

## 1. Trainer

`Trainer` 只负责循环、计时、日志和 hook，不关心具体任务：

```python
Trainer(
    source=...,          # 任意 iter_batches(batch_size)
    plan=ParameterPlan,
    objective=Objective,
    render_fn=...,       # (mesh, blend_weight, batch, iteration) -> RenderOutput
    params=...,
    iterations=...,
    batch_size=...,
    blend_start=...,
    hooks=[...],
)
report = trainer.run()
```

- `Hook.on_start/on_step/on_eval/on_end`；
- `ValidationHook` 每隔 `every` 步调用评估回调；`start_after` 在形变预热前不累积早停；
- `TrainReport` 记录 iterations、seconds、iters/s；
- 训练不直接依赖 `data/` 或 `app/`；输入是 batch 协议。

## 2. 损失

`losses.py`：

| 函数 | 作用 |
|---|---|
| `masked_l1` / `weighted_l1` | 前景/像素加权 L1 |
| `masked_mse` | 前景 MSE |
| `local_ssim_loss` | 局部 SSIM 损失 |
| `alpha_hole_loss` | 新视角孔洞惩罚 |
| `surface_spill_loss` | 表面外扩半透明外溢惩罚 |
| `masked_psnr` | 训练期监控 |
| `PhotometricLoss` | 组合损失 |

`objectives.Objective` 把 L1/SSIM/LPIPS 与 mask 权重组合成单次前向损失：

```python
Objective.rgba_avatar_defaults()   # 基础训练
Objective.appearance_only(...)     # 在线更新/外观微调
```

LPIPS 可延迟开启；`ssim_weight`/`lpips_weight` 相对 L1。

## 3. 参数计划

`ParameterPlan` 为每个参数组配置：

```text
初始学习率 / 最终学习率 / 延迟启动步 / 是否启用
```

- `rgba_avatar_defaults(scene_extent)`：基础训练默认；
- `appearance_only(...)`：只更新颜色/不透明度；
- `GroupPlan.at(iter)` / `lr_at(...)`：按指数计划插值；
- `opt_groups(params)`：生成优化器参数组；
- `scaled(factor)`：整体缩放。

`schedule.py` 提供 `expon_lr`、`blend_active`、`EarlyStopping`。

## 4. 观测

`observations.py`：

- `FrameObservation`：单帧的 mesh、blend_weight、相机、目标 RGB、mask；
- `BatchObservation`：多帧组合；
- `.to(device)` 统一搬运；图像尺寸从张量推导。

`updater.py`：

- `UpdateConfig`：颜色/不透明度开关、预算、学习率、观测帧数；
- `UpdateState`：颜色校正 + 可训练 delta（颜色/不透明度）；
- `OnlineUpdater.fit(observations)`：闭式颜色校正 + 梯度微调；
- `apply(avatar)`：把更新写回参数；
- `UpdateReport`：更新前后 PSNR、耗时。

在线更新只改外观，不改几何。

## 5. 基础训练

入口：

```bash
bash train.sh --subject <对象> --work-name <版本>
```

流程：

1. `load_scene` + `ManifestSource` 取训练/留出帧；
2. `ParameterPlan.rgba_avatar_defaults` + `Objective` 构建；
3. `Trainer` 训练；每 `eval_every` 步验证；
4. `ValidationHook` 用 `PSNR − λ·缺口` 的 robust 指标早停；
5. 保存 checkpoint / `model_best.ply`，结束后 `promote_model` 装入
   `models/<对象>/<版本>/`。

默认配置：

```text
batch 10，50000 步，blend_start 3000，test_ratio 0.3
L1 + SSIM 0.05；LPIPS 默认关
新视角 spill：每 4 步，±10..50°，margin 12，weight 0.05
eval_every 500，patience 2，min_delta 0.05
```

## 6. 在线更新

```bash
python scripts/online_update.py --subject <对象> --work-name <版本> \
    --frames 8 --steps 300 --time-budget 10
```

- 用少量真实观测做闭式颜色校正，再梯度微调颜色/不透明度；
- 有时间预算和门控；默认只影响外观；
- 产物在 `output/online_update/`。

## 7. 与后训练的关系

基础训练产出 `model.ply`；后训练是独立入口，读取该模型后继续微调。
后训练不属于 `training/` 包。

## 8. 统一集合加密管线（`densify_train.sh`）

硬表面层**只负责初始化**，之后全部高斯作为**一个集合**漂移并在高残差处分裂。
两段式（默认值见 `configs/train.yaml` 的 `train.pipeline.densify`）：

| 阶段 | 命令 | 训练什么 | 形变基 | 默认 |
|---|---|---|---|---|
| 1 hard | `--stage hard_surface`（UV/FLAME 零初始化） | xyz 5e-5 / opacity / scaling / rotation / f_dc | **关闭** | 1500 步，batch 8，repair 2 |
| 2 densify | `--stage base --densify`（tex 512 统一集合） | 上面全部 + `xyz_b/rotation_b/f_dc_b/weight_module` | 第 0 步开启 | 5000 步上限，batch 4，repair 1，每 300 步 +600 |

stage1 的定位是"**初始化**"而不是"拟合"：

- 损失只覆盖 FLAME 表面内（`mask_weight=0`），几何项最强（贴面/法线/尺度上限）。
- **盘状初始化**（`configs/model.yaml` 的 `init_scale_tangent/normal` = 2.2/0.4 mm）：
  高斯一开始就是一层面内小圆片。这是关键 —— 各向同性球（0.8 mm）要练到 5000 步
  才把切向撑到 2 mm，之前 1500 步时切向只有 1.0 mm，而 256² 的 texel 间距约
  2.3 mm，圆片盖不满就在 UV 网格上留下条纹/空洞。盘状初始化后 1000 步就铺满，
  约 35 秒，大角度溢出从 7.8% 降到 4.4%。
- **形变基关闭**（`init_blend_start` 默认 100000）。表情残差留给 stage2 ——
  init 的表面 PSNR 只有 ~20 不是"没练好"，它本来就只做壳（头发区域是灰的，
  交给 stage2 补）。
- **切空间偏移上限 3 mm**（`hard_surface.max_offset`，每步投影）：允许补偿
  FLAME/RDF 误差的微小形变，但不让少数高斯在长训练里慢慢漂到表面外。
  实测 p99 偏移从不设限的 7~8 mm 压到 3.00 mm，同时表面 PSNR 与孔洞率**都变好**
  （20.10→20.33 dB，10.1%→9.5%）。

### stage1 早停：为什么不用 PSNR 平台

硬表面必然无法兼顾多视角监督，PSNR 会一直缓慢上升（每 100 步 +0.05~0.1 dB），
用它做平台判据永远不会触发。实测（每 100 步验证，120 帧抽样）：

| step | 表面 PSNR | 孔洞% | 每 100 步位移 | 平均偏移 | p99 偏移 |
|---|---|---|---|---|---|
| 499 | 19.55 | 11.0 | 0.298 mm | 1.01 mm | 5.96 mm |
| 799 | 20.07 | 10.7 | 0.233 mm | 1.04 mm | 6.92 mm |
| 999 | 20.32 | 10.1 | 0.214 mm | 1.05 mm | 7.34 mm |
| 1599 | 20.79 | 7.8 | 0.171 mm | 1.06 mm | 8.19 mm |

**平均偏移在 700 步后饱和在 ~1.05 mm**（可解释为"壳已经贴住"），之后还在涨的
只是 1% 的尾巴；PSNR 的边际收益也已经降到 0.05~0.1 dB/100 步。
所以早停判据用"边际收益"而不是"平台"：`min_delta 0.15 dB / 100 步`，连续 2 次
不达标就停，实测落在 900~1200 步；下游最终模型与固定 1000/1500 步的差异
< 0.1 dB。

stage2 是唯一一次"完整训练"：先合并到 tex 512（256² 只有 65536 个槽位，
split 几次就满），全部高斯一起漂移，残差驱动 split。

实测（bala，holdout 1198 帧，mask 内 PSNR 中位 / 全图 / SSIM）：

```text
两段式 1000 + 4500 步（盘状初始化）  28.93 / 32.59 / 0.847   65983 高斯
两段式 1500 + 4000 步（球状初始化）  28.92 / 32.65 / 0.850   64965 高斯
三段式 5000 + 2000 + 2000 步         29.05 / 32.73 / 0.856   61233 高斯
```

同分，步数从 9000 降到 5500；盘状初始化再把 init 从 50 秒压到 35 秒。早期三段式的"硬表面精修"段已删除：它会让壳
自由漂移到头发/背景，既不再是硬表面，又让加密段无事可做。

加密段的**步数与分裂量要匹配**：init 1500 + 每 300 步只加 300 时，会在第
2000 步早停（28.24）；加到 600 才能跑满并继续涨（28.98）。`patience` 默认
放宽到 3（新高斯会让 PSNR 抖动）。

分裂的两个约束（缺一不可）：

1. split/prune 会替换 `nn.Parameter`，所以 `Trainer` 必须拿**取参函数**
   （`params=avatar.parameter_groups`），重建优化器时重新取参；
2. 副本必须沿自身尺度抖动（`split_jitter`），原地复制等于没加密。

stage2 调参（同 init、同评测）：

| 配置 | mask PSNR | 全图 | SSIM |
|---|---|---|---|
| 旧默认：batch4 / ssim0 / count600 / xyz5e-5 | 28.97 | 32.60 | 0.847 |
| + SSIM 0.03、batch 8 | 29.23 | 32.45 | 0.847 |
| + xyz lr 2e-4 | 29.22 | 32.75 | **0.868** |
| + 分裂量 300 步 +900 | **29.26~29.50** | 32.5~32.6 | 0.854 |

结论：SSIM 与 batch 是主要收益（+0.26 dB），xyz 2e-4 明显改善 SSIM（+0.02），
分裂量 600→900 再 +0.1~0.3 dB。**同配置两次跑的差异有 ~0.2 dB**，
小差距不要当信号。现在距 `base`（29.62）在噪声内，距 `base_post` 还差 ~2 dB
（后者的差距主要在 SyncFix 多轮修复数据）。

试过但没用的：

- 面残差滑动平均 + 按已分裂次数做公平惩罚 + 平局打散噪声 → 28.93，
  把分裂摊薄到了低误差面，已回滚。
- 覆盖缺口选点（渲染随机侧视角，按 `alpha<0.5` 找没盖住的面）+ quad 方向
  （切平面 4 方向轮转细分 texel）：同步骤对比（1499 步）PSNR 29.45→29.36、
  大角度孔洞 14.4%→14.4%、面外溢出 4.3%→4.6%，**无收益**。说明大角度退化
  不是"没盖住"，而是"没监督/被平均掉"。两个开关保留（`split_hole_weight`
  默认 0、`split_pattern` 默认 random），换数据集可以再试。

- 真细分（`split_mode: subdivide`：父换成两个 0.7 倍尺度、同透明度的子高斯，
  不再追加半透明副本）：4000 步同条件对比，mask PSNR 29.22→28.95、
  SSIM 0.8647→0.8651、全图 32.91→32.80。**也没用**，默认仍是 `copy`。
  说明"画面偏软"不是副本叠加造成的。

由此判断大角度/纹理的下一个抓手在**数据与采样**，而不是分裂几何：
修复图加密到 5°（用户提议）或按 |yaw| 加权采样，给大角度更多监督。

已知不足：锯齿主要来自分裂方向（随机抖动）而不是选点；下一步可试把像素残差
反投影成面内方向、让副本沿残差方向偏移。

## 9. stage3：加密模型上的短程后训练（`stage3_train.sh`）

在 stage2 产物上做"逐轮角度外扩"的短程微调：渲染色→SyncFix 修复→加入监督→
微调 50 步。锚定帧**只用训练段**（`refine_from_images --anchor-scope train`，
留出集不参与）。

实测（bala，holdout 1198 帧）：

| 模型 | mask PSNR | 全图 | SSIM | 锐度比 |
|---|---|---|---|---|
| `base` | 29.62 | 33.25 | 0.877 | — |
| `densify`（stage2） | 29.19 | 32.78 | 0.867 | 0.80 |
| + 1 轮 50 步，真实帧 | 29.59 | 32.92 | 0.870 | 0.81 |
| **+ 3 轮 50 步，真实帧** | **29.92** | 32.93 | 0.871 | 0.80 |
| + 3 轮，混入 ±40/45/50 修复图（1/8） | 29.56 | 32.91 | 0.869 | 0.80 |

两条结论：

1. **短程真实帧微调确实有效**：3×50 步 +0.73 dB（29.19→29.92），已经超过 base。
2. **SyncFix 修复图在这个阶段没有额外信息**：混入 1/8 后反而低 0.02~0.36 dB，
   并把锐度拉低。和 11-16 的 `none vs syncfix` 消融一致——但现在是在干净切分
   下测的，结论可信。默认因此走 `--real-only`；`--with-repairs` 保留给"数据
   确实缺到大角度之外"的场景（需要角度分层留出集才能验证）。

## 10. stage3 现状（并入 `densify_train.sh --with-s3`）

旧的"逐轮外扩"链路（`post_train.sh` / `iterative_repair.py` /
`refine_from_images.py` / `find_defects.py`）已删除：它慢、且修复图在干净切分下
没有额外信息。stage3 现在就是 `train.py --stage base` 的一段短程微调：
冻结几何与驱动基（`--freeze-geometry`），按 |yaw| 加权采样（`--angle-weight`），
可选混入提前生成好的修复图（`--repair-dir`）。

实测（bala，holdout 1198 帧）：

| stage3 配置 | mask PSNR | 全图 | SSIM | 锐度比 |
|---|---|---|---|---|
| 不做（stage2 直接交付） | 29.19 | 32.78 | 0.867 | 0.80 |
| 冻结几何 + 角度加权 20，400 步 | 26.92 | 31.06 | 0.792 | 0.85 |
| 同上 + 一次性修复图（±40…90，2/8） | 27.02 | 31.14 | 0.792 | 0.84 |
| 同上 + `--lr-scale 0.1` | 27.66 | 31.68 | 0.824 | 0.81 |

**两版都明显掉点**（−1.5~2.3 dB）。原因不是"修复图没用"，而是 stage3 的
损失/LR 配方和短程微调不匹配：`train.py` 默认 opacity lr 0.25、f_dc 0.0125、
mask 权重 1.0 且带新视角 spill；旧的 refine 配方是 opacity 0.01、f_dc 2e-3、
mask 权重 0、无新视角项。用旧配方（已删脚本）做过 3×50 步真实帧微调能到
29.92 dB（+0.73），说明"短程微调"本身有效，缺的是把配方搬进 `train.py`。

所以 stage3 默认关闭（`--with-s3` 才跑）。下一步要么给 train.py 加"微调档"
的 LR/损失预设（对齐旧 refine），要么放弃 stage3、直接用 stage2 产物。

> **阅读提示**：§9–§13 是**按日期记录的历史实验**，数字与当时的配置/口径绑定。
> 现行默认与最新产物见 §14（消融）与 §15（现行状态）；跨节数字不要直接比。

## 11. 2026-11-20 重跑：5° 修复图 + alpha 饱和 + SSIM/LPIPS（历史）

改动：
1. **初始化修复图加密到 5°**：`make_repairs.py` 从 `base` 渲染 ±40…90°、步长 5°
   （22 张），写回 `datasets/bala/repair/repairs`（旧的 10° 一套已删除）。
2. **加密阶段新增 alpha 饱和项**（`train.pipeline.alpha_saturation`，权重 0.03、
   目标 0.98）：FLAME 表面内 `(target-alpha)_+` 的均值。硬表面阶段本来就有
   `surface_hole`，但 stage2 把它关了，导致表面内 ~20% 像素 alpha<0.9，
   大角度能透出背面。
3. **SSIM 0.08 作为主感知项**，LPIPS 0.01 从第 3500 步启用（低权重、后段）。

实测（bala，holdout 1198 帧，同一套评测）：

| 模型 | mask PSNR | 全图 | SSIM | LPIPS | 锐度比 |
|---|---|---|---|---|---|
| `base`（6501 步，干净基线） | 29.62 | 33.25 | 0.877 | 0.0670 | 0.83 |
| 旧加密（10° 修复、无 alpha 项） | 29.19 | 32.78 | 0.867 | 0.0829 | 0.80 |
| **新加密（5° + alpha 0.03 + SSIM/LPIPS）** | **29.52** | 33.04 | 0.870 | **0.0680** | 0.83 |

alpha 覆盖（FLAME 表面内，yaw 0/40/50）：均值 0.986 / 0.993 / 0.996，
`alpha<0.9` 占比 2.7% / 1.8% / 1.0%（改前是 20.9% / 19.6% / 20.3%）。
⚠️ 这是**未腐蚀** `alpha_sat` 的效果；后来为避免轮廓黑片改成 `erode: 3`，
该项在 §14 的消融中被证明已失效并删除，现行模型表面 alpha 回到 ~0.86。

代价：robust 评测里的"面外溢出"从 ~5% 升到 ~18.8% —— alpha 抬升让轮廓外
（头发/边缘）的高斯也变得实心，视觉上是"头发更厚"。这个口径是否算退化，
需要按目视判断，不能只看数字。

## 12. 修复图质量 + 轮廓外黑雾（2026-11-20 第二轮）

**SyncFix 批处理的坑**：`inference.py` 先 `ref_images = ref_images[:ref_size]`，
再按 `refs[i:i+n]` 分配给每个 batch —— 只有第一个 batch 有参考图，后面的都是
**无参考修复**。22 张 × `n_per_pass=4` = 6 个 batch，5/6 没有参考，质量自然差。
现在 `make_repairs.py` 按 `--syncfix-batch`（默认 4）分组、每组单独调用一次，
保证每组都带 3 张参考；修复完再用 BiRefNet 抠图输出 RGBA（alpha 外 RGB 置 0），
与旧数据集约定一致。

**轮廓外黑雾**：stage2 只监督 RGB、从不监督 alpha，黑底修复图/背景让模型把
轮廓外画成半透明黑。新增 `alpha_weight: 0.2`（alpha 直接回归 GT 前景 mask），
并把 `alpha_saturation` 限制在轮廓内部（`erode: 3`，避免把边缘高斯压成黑片）。

白底渲染上"半透明暗像素（alpha 0.05~0.9 且 gray<0.9）"占比：

| 模型 | 占比 |
|---|---|
| `base` | 4.85% |
| 上一版加密（相机绕转修复图，无 alpha 监督） | 6.53% |
| 当天的加密（alpha 0.2 + 内缩 alpha_sat） | 1.47% |

留出集指标（holdout 1198 帧）：

| 模型 | mask PSNR | 全图 | SSIM | LPIPS | 锐度比 | 口径 |
|---|---|---|---|---|---|---|
| `base` | 29.62 | 33.25 | 0.877 | 0.0670 | 0.83 | 相机绕转/头转同值 |
| 当天加密（相机绕转修复图） | 29.84 | 32.99 | 0.871 | 0.0714 | 0.79 | 旧口径 |
| **现行加密（头转修复图，见 §15）** | **29.80** | 33.05 | **0.8786** | **0.0568** | **0.84** | 头转口径 |

代价：旧配置的锐度 0.79（base 0.83）、LPIPS 高 0.004；换成头转修复图后这些
反而全部追平/超过 base（§15），说明当时的"变软"主要来自**语义错位的修复图**，
不是 alpha 监督本身。

### 修复目录结构与"大角度黑色区域"

`make_repairs.py --repair-root datasets/<obj>/repair` 会恢复旧结构：
`repairs/`（RGBA）、`views/`（渲染原图）、`masks/`（BiRefNet alpha）、`refs/`
（身份参考）+ `manifest.json`。

大角度下"脖子下方大片黑色"排查结论：**不是这轮改动引入的**。
在 yaw≈40° 的真实帧上统计"GT 前景里渲染近黑(<20) 的比例"：
`base` 14.5%、当前 `densify` 14.0% —— 两者一致。原因有两层：

1. GT 前景本身就包含深色 T 恤/肩部，模型渲染成黑色是**正确内容**；
2. 高斯只绑定到 FLAME 面（头/颈），脖子以下没有几何，旋转到 ±60° 以上时
   颈部末端到达画面边界，下面自然露出背景（GUI 黑底 → 看起来是黑色空洞）。

## 13. 用 FLAME 头姿合成"转头"视角（`render_head_turn.py`）

数据集的"大角度"是**头转、肩/胸不动**；`orbit_w2c`（相机绕头心转）会把胸口
一起转走。同一帧（yaw≈40）对比：真实头姿渲染 vs GT **27.16 dB**，相机绕转
vs GT 只有 **14.63 dB**。

正确做法（2026-11-20 改版，**不重算 FLAME**）：对**烘焙 mesh 直接做蒙皮旋转**

    v' = v + w_h · (D·(v − J) + J − v)

- `w_h`：每个顶点随头转的权重（`1 − root`，来自 FLAME 的 LBS；下颌/眼球是颈的
  子节点，一起转），牙齿的权重在 `FlameModel` 里已补好；
- `J`：颈关节在数据集坐标中的位置（优先用 `rdf_tracking.npz` 的 `identity` 算
  FLAME 关节再刚体映射；缺 RDF 时用"头/胸权重过渡带质心"兜底——对 yaw 转而言，
  J 沿旋转轴的分量不影响结果）；
- `D`：目标相对当前头姿的世界系增量旋转；驱动参数只改头姿段 `[100:106]`。

**闭环**：`D = I` 时 `v' = v` 严格成立，`turn(idx, 0)` 与烘焙 mesh **逐位一致**
（旧版"重算 FLAME + Kabsch"是 3 mm / p99 2 cm，已删）。60° 时头骨平均位移 91 mm、
颈根 11 mm，胸不动。测试见 `tests/unit/test_pose_drive.py`。

`--yaws 40,60,80` 渲染出的都是"头转、胸口不动"，颈根不再出现黑洞。

已知瑕疵：大角度下眼睛发白（眼睑/眼球基没驱动到位）、耳侧偶有飞点、颈根皮肤拉伸；
±80° 仍可用，>±90° 受限于 FLAME（没有肩/胸几何）。

⚠️ **该原型依赖 `checkpoint/`（旧 tracker 的 GT 参数），不应作为正式方案**。
正确做法是用我们自己的 RDF 跟踪输出驱动：

- RDF dump 里有 `x (T,118) = [exp100][global3][neck3][jaw3][eye_l3][eye_r3][trans3]`
  和 `identity (300)`（见 `tracking/param_map.py` 的布局说明）；
- 头姿 `R_head = R_global @ R_neck`（轴角→矩阵），把 global/neck 转到目标角度，
  再用 FLAME 前向 + `TeethBuilder` 补牙即可得到"头转"网格；
- 已有 RDF dump 的对象：`biden`、`guided_capture_01`；`bala` 是旧 checkpoint
  烘焙格式，要用这条路径需先跑一次 RDF。

用法（**基于 checkpoint，仅作原型**）：
```bash
python scripts/render_head_turn.py --subject bala --work-name densify \
    --yaws "40,60,80" --out output/diag/headturn
```

## 14. 空洞监督消融（2026-11-20）

5 个变体，除被消融项外完全相同（stage2 各 3000 步）：

| 变体 | 最终 mask PSNR | SSIM | LPIPS | 锐度 | stage1 覆盖 |
|---|---|---|---|---|---|
| stage1 无 hole/spill | 29.62 | 0.873 | 0.0695 | 0.80 | alpha 0.995，洞 0% |
| stage1 只有 spill | 29.40 | 0.873 | 0.0700 | 0.80 | alpha 0.872，洞 12.7% |
| stage1 hole+spill（旧默认） | 29.55 | 0.871 | 0.0699 | 0.80 | alpha 0.872，洞 12.7% |
| stage2 无 alpha_sat | 29.27 | 0.872 | 0.0597 | 0.83 | — |
| stage2 无 alpha_sat / alpha_weight | 29.66 | 0.874 | 0.0632 | 0.83 | — |

结论：
- **`surface_hole` 无效**：开与不开，stage1 覆盖与最终指标逐位相同（真正起作用的是 spill）。
- **`surface_spill` 只在 stage1 有效**（把轮廓 alpha 0.995→0.872），stage2 会自己重新修剪，
  最终指标无可测差异。
- **`alpha_saturation`（腐蚀 3px、0.03）已失效**：对覆盖无影响（0.858 vs 0.858）。
- **`alpha_weight`（0.2）** 还能把白底暗雾从 1.48% 压到 0.89%，代价噪声级。

据此删除 `hard_surface.hole_weight` 与整个 `alpha_saturation`，保留 `spill_weight`
与 `alpha_weight`。


## 15. alpha 两项的去留：两轴表（评审 A5）

同一批帧（bala 留出集 1198 帧，3000 步，batch 8，tex 512），只差 alpha 开关；
"白底暗雾"= 半透明（0.05<α<0.9）且叠到白底后仍偏暗的像素占比（现行报告的
`quality.haze` 就是这个口径）。

| 变体 | mask PSNR | SSIM | LPIPS↓ | 锐度比 | 白底暗雾 | 表面 alpha |
|---|---:|---:|---:|---:|---:|---:|
| ctrl（alpha 0.2 + sat 0.03） | 29.55 | 0.8713 | 0.0699 | 0.801 | 0.89% | 0.858 |
| nosat（只留 alpha 0.2） | 29.27 | 0.8721 | 0.0597 | 0.835 | 0.84% | 0.858 |
| noalpha（两项都关） | **29.66** | **0.8740** | 0.0632 | 0.832 | 1.48% | 0.856 |

读法：

- **PSNR 三者差异在噪声内**（±0.2 dB），不构成"alpha 项更好"的证据；
- 关掉后 SSIM/LPIPS/锐度**一致略好**；暗雾只在**同时关掉 `alpha_weight`** 时
  明显变差（1.48% vs 0.84–0.89%）；
- `alpha_sat`（腐蚀 3px）对覆盖没有影响（0.858 vs 0.858），是**无效项**。

据此的默认值：

- 删除 `alpha_saturation`（以及 `hard_surface.hole_weight`，见 §14）；
- **保留** `alpha_weight: 0.2`：它买到的是"暗雾减半"，代价是噪声级的 PSNR；
  文档里只写成"用微量 PSNR 换轮廓干净"，不再写成"默认更好"。


## 16. 现行状态与容量对照（2026-11-20）

**容量对照**（同为 tex 512、同 5000 步、同损失、同 22 张头转修复图；唯一差别是
`--densify-max`，一组允许 split、一组禁止）。为防止"选模型"混淆，两组都取
**第 5000 步的 checkpoint**，而不是 `model_best`：

| 组 | 高斯数 | mask PSNR | 全图 | SSIM | LPIPS↓ | 锐度比 | 白底暗雾 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 允许 split | 74,503 | **29.80** | 33.05 | **0.8786** | 0.0568 | 0.84 | 0.84% |
| 禁止 split（同容量） | 60,353 | 29.48 | 33.02 | 0.8745 | **0.0563** | **0.86** | 0.74% |

结论：**densify 的 split 换到的是 PSNR/SSIM 的小幅提升（+0.32 dB / +0.004），
感知项（LPIPS/锐度）反而略差**。所以"加密让一切更好"的说法不成立，只能说
"加密主要补 PSNR"。

**与 `base` 的关系**：`base` 是 tex 256 / 60,353 高斯 / 6501 步，`densify` 是
tex 512 / 74,503 / 5000 步 —— 容量、纹理分辨率、步数都不同，**不能归因**给
修复图或 alpha 项。主表里 `base` 只作为"同数据、不同管线"的参考行。
