# training

`src/live3dgsavatar/training/` 是训练循环与参数计划。三个入口：

| 入口 | 用途 |
|---|---|
| `train.sh` / `scripts/train.py` | 基础训练（单段，从 FLAME 初始化） |
| `train_avatar.sh` | 单阶段头像重建（UV 初始化 → 统一集合），可选挂加密 |
| `scripts/online_update.py` | 通话时的在线外观更新 |

数值默认值只在 `configs/train.yaml`（与 `configs/model.yaml`）里，本文件不重复
列数字；`python scripts/show_config.py --shell-avatar` 会打印训练脚本实际用的值。

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

- `Hook.on_start/on_step/on_end`；
- `ValidationHook` 每隔 `every` 步评估；`start_after` 之前不累积"无提升"；
  base 阶段的指标是**组合分** `PSNR + 50·SSIM + 50·(1−LPIPS)`（SSIM/LPIPS 在
  240 帧抽样上算），hard_surface 用 `psnr_surface`；
  指标只认**落盘那一刻**的分数（`best_metric` + `best_saved`），不再用历史最大值；
- `TrainReport.iterations` 是**实际执行**的步数（早停不会多算一步）；
- 训练不依赖 `data/` 或 `app/`，输入是 batch 协议。

## 2. 前向：唯一入口

训练、验证、评测都走 `training/forward.py`：

```python
opts = ForwardOptions(build_graph=True, frontface=True, clamp_max_scale=0.002)
res  = forward_avatar(avatar, rasterizer, mesh, weight, camera, bg,
                      options=opts, target_image=...)
```

`ForwardOptions` 表达"要不要构图、要不要剔背面、要不要裁尺度"。**验证必须用和
训练同一份 options** —— 否则早停选出来的模型不是训练时的那个。

## 3. 损失

| 函数 | 作用 |
|---|---|
| `masked_l1` / `weighted_l1` | 前景/像素加权 L1 |
| `masked_mse` | 前景 MSE |
| `local_ssim_loss` | 局部 SSIM 损失 |
| `alpha_hole_loss` | 新视角孔洞惩罚 |
| `surface_spill_loss` | 表面外扩半透明外溢惩罚 |
| `masked_psnr` | 训练期监控 |

`objectives.Objective` 把 L1/SSIM/LPIPS 与 mask 权重组合成单次前向损失。
LPIPS 可延迟开启；`ssim_weight` / `lpips_weight` 相对 L1。

## 4. 参数计划

`ParameterPlan` 为每个参数组配置"初始/最终学习率、延迟启动步、是否启用"：

- `rgba_avatar_defaults(scene_extent)`：基础训练默认；
- `appearance_only(...)`：只更新颜色/不透明度；
- `scaled(factor)`：整体缩放。

`schedule.py` 提供 `expon_lr`、`blend_active`、`EarlyStopping`。

## 5. 观测与在线更新

`observations.py` 的 `FrameObservation` / `BatchObservation` 描述单帧/多帧观测；
`updater.py` 的 `OnlineUpdater.fit()` 做闭式颜色校正 + 梯度微调，只改外观。

```bash
python scripts/online_update.py --subject <对象> --work-name <版本> \
    --frames 8 --steps 300 --time-budget 10
```

## 6. 基础训练

```bash
bash train.sh --subject <对象> --work-name <版本>
```

1. `load_scene` + `ManifestSource` 取训练/留出帧（留出集统一为末尾 30%）；
2. `ParameterPlan.rgba_avatar_defaults` + `Objective`；
3. `Trainer` 训练，每 `eval_every` 步验证；
4. `ValidationHook` 按指标早停并保存 `model_best.ply`，结束时装入
   `models/<对象>/<版本>/`。

## 7. 单阶段头像重建（`train_avatar.sh`）

一次训练完成外观/形变拟合，加密是**可开关的挂载项**（默认关）。只读
`configs/train.yaml` 的 `train.pipeline.avatar`：

1. **初始化**：`scripts/init_uv.py` 生成未训练的 UV 绑定高斯（各向同性球或
   盘状由 `configs/model.yaml` 的 `init_scale_*` 决定）；`scripts/merge_layers.py`
   合并成 tex 512 的统一集合（形变基 K=20 从第 0 步可用）。
2. **训练**（`--stage base`）：真实帧 + 修复图按 `--repair-per-batch` 混批；
   L1 + SSIM +（默认）第 `lpips_from` 步起的 LPIPS；`nv_spill` 抑制轮廓外
   半透明薄雾，`alpha` 向 GT 前景 mask 回归。
3. **加密（可选）**：`--densify` 打开残差驱动 split/prune（每 `every` 步，
   每次最多 `count` 个），默认关。

```bash
bash train_avatar.sh --subject <对象>            # 默认不加密
bash train_avatar.sh --subject <对象> --densify  # 挂上加密
bash train_avatar.sh --subject <对象> --no-lpips # 快速实验
```

依赖：修复图由 **base** 模型生成（缺省自动生成）；缺 `surface_masks.npz`
时脚本会自动跑 `scripts/cache_surface_masks.py`。

## 8. 修复图（`scripts/make_repairs.py`）

`train_avatar.sh` 需要 `datasets/<对象>/repair/repairs/`。修复图由 **base 模型
按目标头姿渲染 → SyncFix 修复 → BiRefNet 抠图** 生成，只生成一次、训练时按
batch 比例混入：

```bash
# 按训练段 yaw 两个方向的 p90 自动排角度（只做 yaw 扇，不做 pitch）
python scripts/make_repairs.py --subject <对象> --work-name base \
    --auto --auto-yaw-only \
    --out output/repairs --out-dir datasets/<对象>/repair/repairs \
    --repair-root datasets/<对象>/repair
```

`--repair-root` 会一并写出 `views/ refs/ masks/ repairs/ manifest.json`。
SyncFix 路径从 `configs/local.yaml` 或 `--syncfix-*` 参数给。

## 9. 视角语义：头转，不是转相机

数据集的"大角度"是**头转、肩/胸不动**；绕头心转相机（`orbit_w2c`）会把胸口
一起转走，与 GT 差约 12 dB。因此修复图、showcase、robust 验证、前端视角
统一走 `tracking/pose_drive.py`：

```
v' = v + w_head · (D·(v − J) + J − v)
```

- `w_head` 是每个顶点随头转的权重（FLAME 的 `1 − root`，牙齿已覆盖）；
- `J` 是颈关节（有 RDF 时由 `identity` 算，缺 RDF 时用过渡带质心）；
- `D` 是相对当前头姿的增量旋转。

`turn(idx, 0)` 与数据集网格**逐位一致**（`tests/unit/test_pose_drive.py`）；
网格拓扑对不上时才退回相机绕转。

## 10. 评测

指标口径、留出集用法、报告字段见 `docs/EVALUATION.md`。
