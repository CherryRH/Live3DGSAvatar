# 配置说明

## 1. 优先级

```text
命令行 > 环境变量 > configs/local.yaml > configs/*.yaml 默认值
```

查看最终值：

```bash
python scripts/show_config.py
python scripts/show_config.py --shell-post
```

## 2. 配置文件

| 文件 | 内容 |
|---|---|
| `configs/system.yaml` | 路径、设备、数据/模型目录、GUI、tracking |
| `configs/render.yaml` | 渲染、模型结构、在线更新默认开关 |
| `configs/train.yaml` | 训练、后训练、在线更新参数 |
| `configs/model.yaml` | 模型结构参考值 |
| `configs/local.yaml` | 本机覆盖，不提交 |

## 3. system.yaml

| 字段 | 默认 | 说明 |
|---|---|---|
| `paths.data_root` | `datasets` | 数据集根目录 |
| `paths.output_dir` | `output` | 产物根目录 |
| `paths.models_dir` | `models` | 模型根目录 |
| `paths.flame_dir` | `data/FLAME2020` | FLAME 资产 |
| `paths.image_subdir` | `images` | 训练/评测使用的人像子目录 |
| `runtime.device` | `cuda` | `cuda` / `cpu` |
| `app.host` / `app.port` | `localhost` / `8000` | GUI 监听 |
| `tracking.enabled` | `true` | 实时驱动开关 |
| `tracking.host` / `port` | `127.0.0.1` / `8980` | RDF 追踪服务 |
| `tracking.frame_subdir` | `auto` | `auto` / `input` / `images` |

路径支持绝对路径、`~`、相对仓库根。环境变量覆盖：

```text
LIVE3DGS_DATA_ROOT
LIVE3DGS_OUTPUT_DIR
```

## 4. render.yaml

| 字段 | 默认 | 说明 |
|---|---|---|
| `render.background` | `[0,0,0]` | 背景色 |
| `render.scale` | `1.0` | 输出分辨率比例 |
| `render.target_fps` | `60` | 服务端推帧上限，`0` 不限 |
| `render.encode` | `jpeg` | `jpeg` / `raw` |
| `render.batch_size` | `4` | core 批量渲染；`render_test` 默认单独用 `1` |
| `render.frames` | `20` | render_test 默认帧数，`-1` 全部 |
| `render.video_fps` | `30` | 对比/展示视频帧率 |
| `model.network.tex_size` | `256` | UV 纹理边长 |
| `model.network.num_basis_in` | `129` | 驱动参数维度 |
| `model.network.num_basis_blend` | `20` | 混合基数量 |
| `model.network.mlp_hidden` | `[128,128]` | 权重 MLP |
| `model.network.use_weight_proj` | `true` | 是否使用权重投影 |

## 5. train.yaml

### 基础训练 `train.pipeline`

| 字段 | 默认 | 说明 |
|---|---|---|
| `iterations` | `50000` | 总步数 |
| `batch_size` | `10` | 每步真实帧数 |
| `blend_start` | `3000` | 开始启用形变基 |
| `scene_extent` | `1.0` | 位置类学习率尺度 |
| `test_ratio` | `0.3` | 末尾留出比例 |
| `mask_weight` | `1.0` | 背景/前景权重 |
| `ssim_weight` | `0.05` | SSIM 相对 L1 权重 |
| `lpips_weight` | `0.0` | LPIPS 权重 |
| `lpips_from` | `20000` | 从第几步启用 LPIPS |
| `eval_every` | `500` | 验证间隔 |
| `patience` | `2` | 连续无提升次数 |
| `min_delta` | `0.05` | 视为提升的最小 dB |
| `robust_eval` | `true` | 大角度验证 + 缺口惩罚 |
| `robust_eval_frames` | `24` | 大角度验证帧数 |
| `robust_eval_weight` | `10.0` | 缺口折算 dB 权重 |

### 新视角监督 `train.pipeline.novel_view`

| 字段 | 默认 | 说明 |
|---|---|---|
| `every` | `4` | 每几步采一个视角 |
| `spill_weight` | `0.05` | 表面外溢（薄雾）惩罚 |
| `hole_weight` | `0.0` | 孔洞惩罚，默认关 |
| `spill_margin` | `12` | 表面外扩像素 |
| `alpha_min` / `alpha_max` | `0.05` / `0.6` | 半透明外溢范围 |
| `frames` | `80` | 支持统计帧数 |
| `yaws` | `±10…±50` | 候选角度 |
| `robust_spill_weight` | `0.0` | 验证指标中的外溢权重 |

### 后训练 `train.pipeline.post_train`

| 字段 | 默认 | 说明 |
|---|---|---|
| `top_n` | `2` | 每圈选几个缺陷视角；SyncFix 会取 `max(top_n, n_per_pass)` |
| `rings` | `auto` | 角度圈；`auto` 按训练段 p90 外扩，也可写 `"20,10;..."` |
| `steps` | `100` | 每轮微调步数 |
| `real_per_batch` | `6` | 训练 batch 真实帧 |
| `refined_per_batch` | `2` | 训练 batch 修复帧 |
| `quality_tol` | `0.01` | 最后一轮相对最佳允许下降 |
| `psnr_weight` | `0.002` | 质量分中的 PSNR 权重 |
| `lr_xyz` | `1e-5` | 位置学习率，0 固定 |
| `syncfix_repo` | 空 | SyncFix 仓库（建议写在 `local.yaml`）|
| `syncfix_python` | `syncfix` | 运行环境名 |
| `syncfix_model` | 空 | 权重目录（建议写在 `local.yaml`）|
| `syncfix_n_per_pass` | `4` | 一次联合修复视图数 |
| `syncfix_ref_size` | `4` | 参考图数量 |

### 在线更新 `train.online_update`

| 字段 | 默认 | 说明 |
|---|---|---|
| `enabled` | `true` | 总开关 |
| `update_color` / `update_opacity` | `true` | 更新内容 |
| `time_budget_s` | `10.0` | 墙钟预算 |
| `steps` | `300` | 最大步数 |
| `lr_color` / `lr_opacity` | `0.002` / `0.02` | 学习率 |
| `mask_weight` | `0.0` | 背景权重 |
| `frames` / `frame_chunk` | `8` / `8` | 观测帧数与每步帧数 |

## 6. 环境变量

| 变量 | 用途 |
|---|---|
| `MAIN_ENV` | 主 conda 环境名，默认 `live3dgs` |
| `LIVE3DGS_DATA_ROOT` | 覆盖数据根目录 |
| `LIVE3DGS_OUTPUT_DIR` | 覆盖产物根目录 |
| `LIVE3DGS_MODELS_DIR` | 覆盖模型根目录 |
| `LIVE3DGS_SUBJECT` / `LIVE3DGS_WORK_NAME` | 覆盖当前对象/版本 |
| `LIVE3DGS_DEVICE` | 覆盖运行设备 |
| `POST_*` | 临时覆盖 `post_train` 参数 |
