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
| `iterations` | `10000` | 总步数上限（三个训练入口共用） |
| `batch_size` | `10` | 每步真实帧数 |
| `blend_start` | `3000` | 开始启用形变基 |
| `scene_extent` | `1.0` | 位置类学习率尺度 |
| `test_ratio` | `0.3` | 末尾留出比例 |
| `mask_weight` | `1.0` | 背景/前景权重 |
| `ssim_weight` | `0.1` | SSIM 相对 L1 权重 |
| `lpips_weight` | `0.0` | LPIPS 权重 |
| `lpips_from` | `20000` | 从第几步启用 LPIPS |
| `eval_every` | `500` | 验证间隔 |
| `patience` | `4` | 连续无提升次数（三个训练入口共用） |
| `min_delta` | `0.05` | 视为提升的最小 dB |
| `robust_eval` | `true` | 大角度验证 + 缺口惩罚 |
| `robust_eval_frames` | `24` | 大角度验证帧数 |
| `robust_eval_weight` | `10.0` | 缺口折算 dB 权重 |
| `alpha_weight` | `0.2` | alpha 向 GT 前景 mask 回归的权重 |

### 选模型/早停 `train.pipeline.selection`

| 字段 | 默认 | 说明 |
|---|---|---|
| `ssim` | `50.0` | 组合分里 SSIM 的权重 |
| `lpips` | `50.0` | 组合分里 (1−LPIPS) 的权重 |
| `perceptual_frames` | `240` | SSIM/LPIPS 的抽样帧数（全量太慢） |

组合分 `score = PSNR + ssim·SSIM + lpips·(1−LPIPS)` 只在 base 阶段作为
`--early-metric` 的默认值（遗留的 `--stage hard_surface` 用 `psnr_surface`）。

### 单阶段训练 `train.pipeline.avatar`（`train_avatar.sh` 读取）

| 字段 | 默认 | 说明 |
|---|---|---|
| `steps` / `batch` | `10000` / `8` | 步数上限 / 批大小 |
| `repair` | `2` | 每批混入的修复图张数（0 = 关） |
| `repair_schedule` | `ratio` | 退火：`none`/`linear`/`cosine`（衰减权重）/`ratio`（2/8→1/8） |
| `repair_final_weight` | `0.2` | `linear`/`cosine` 结束时的修复图权重 |
| `blend_start` | `0` | 形变基启用步（0 = 立即） |
| `ssim_weight` / `lpips_weight` / `lpips_from` | `0.1` / `0.01` / `5000` | 感知项；LPIPS 只在 `--lpips` 时启用 |
| `tex_size` | `512` | 统一集合的 UV 纹理边长 |
| `patience` | `4` | 早停耐心 |
| `densify.*` | 见下 | 加密挂载项（默认关） |

加密 `train.pipeline.avatar.densify`（`--densify` 打开）：

| 字段 | 默认 | 说明 |
|---|---|---|
| `every` / `count` / `max_gaussians` | `300` / `900` / `200000` | split 间隔/每次数量/上限 |
| `prune_opacity` | `0.01` | 剪枝阈值 |
| `split_jitter` | `0.8` | 分裂副本位移（1σ） |
| `split_hole_weight` | `0.0` | 选点权重（缺口 vs 颜色残差） |
| `split_pattern` / `split_mode` / `split_scale` | `random` / `copy` / `0.7` | 分裂方向/方式/尺度 |

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
