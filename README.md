# Live3DGSAvatar

面向视频通话的 3DGS 头部重建：从单目视频 / RDF 跟踪结果训练可驱动高斯头像，
支持加密、留出集评测、浏览器预览和实时驱动。

| 模块 | 职责 |
|---|---|
| `core/` | 高斯表示、FLAME 形变与绑定、相机、光栅化；不依赖上层 |
| `data/` | 数据集读取、场景、样本源、姿态/头姿 |
| `training/` | 训练循环、损失、参数计划、在线更新 |
| `evaluation/` | 缺陷检测、候选视图、PSNR/SSIM/LPIPS |
| `app/` | GUI 后端、协议、会话、设置 |
| `tracking/` | RDF 实时驱动桥 |
| `scripts/` | 各步骤实现和独立工具 |
| `configs/` | 系统、渲染、模型、训练配置 |

## 训练方法

**表示**：每个模型绑定 FLAME 网格与 UV 拓扑，高斯存在 UV texel 上（数量随
tex 分辨率变化，默认几万级）。每帧由 129 维驱动参数生成切空间变换，再绑定回
世界空间；高斯属性包括位置、旋转、尺度、不透明度和 SH DC 颜色。

**基础训练**：真实帧重建 L1，可加 SSIM/LPIPS；每 4 步采一个 ±10–50° 新视角，
对 FLAME 表面外扩 12 px 的半透明外溢（薄雾）做监督，减少侧面和边缘漂浮。
验证用 PSNR 与覆盖缺口合成的鲁棒指标，平台期早停。

**统一训练**：从 UV 初始化合并成 tex 512 的统一集合，一次训练同时拟合外观与形变。
可混入 SyncFix 生成的**修复图**（由 `base` 模型渲染、按 batch 比例监督），只在没有
真实数据的大角度上补一次；**加密**（残差驱动 split/prune）是可开关的挂载项，默认关。

**大角度语义**：数据集里的"转头"是头动、肩胸不动。修复图、showcase、验证和
前端视角都按**头转**渲染（驱动 FLAME 头姿），而不是绕头心转相机。

## 环境

```bash
bash setup_env.sh
conda activate live3dgs
```

## 快速开始

```bash
bash build_dataset.sh <图像目录> --name <对象>
bash build_dataset.sh --video videos/<对象> --name <对象>   # 从视频直接构建
bash train.sh --subject <对象> --work-name base          # 修复图的来源
# 修复图（缺省时 train_avatar.sh 会用它自动生成；SyncFix 路径见 configs/local.yaml）
bash train_avatar.sh --subject <对象>                    # 默认不加密
bash train_avatar.sh --subject <对象> --densify          # 挂上加密
bash eval.sh --subject <对象> --work-name avatar
# 多版本对比：原图 + N 个渲染并排，showcase 为 N 图并列
bash eval.sh --subject <对象> --work-name base -- \
  --work-names base <版本2> <版本3> <版本4>
bash run_gui.sh --subject <对象> --work-name base
```

## 脚本

| 脚本 | 用途 | 主要产物 |
|---|---|---|
| `setup_env.sh` | 创建 conda 环境、编译本地 CUDA 扩展 | `live3dgs` 环境 |
| `build_dataset.sh` | 跟踪、对齐并烘焙训练数据 | `datasets/<对象>/` |
| `train.sh` | 基础训练 | `models/<对象>/<版本>/` |
| `train_avatar.sh` | 单阶段头像重建，可选挂加密 | `models/<对象>/<版本>/` |
| `rgb_train.sh` | RGBAvatar 风格基线（各向同性球、纯 L1、随机背景、无 spill/alpha/修复图/加密） | `models/<对象>/rgb/` |
| `eval.sh` | 留出集渲染评测 + ±45° 大角度指标（支持 `--work-names` 多版本） | `output/render_test/.../report.json`、`compare.mp4`、`showcase.png` |
| `run_gui.sh` | 浏览器预览与实时驱动 | 本地 Web 页面 |
| `run_tracking.sh` | 启动 RDF 实时追踪服务 | 追踪 sidecar |

常用参数：

```bash
# 训练
bash train.sh --subject <对象> --work-name <版本> --no-spill
bash train.sh --subject <对象> --work-name <版本> -- --iterations 8000

# 单阶段重建
bash train_avatar.sh --subject <对象>            # 默认：不加密
bash train_avatar.sh --subject <对象> --densify  # 挂上加密
bash train_avatar.sh --subject <对象> --no-lpips # 快速实验

# 评测（默认末尾留出集全部；--frames 可只测前 N 帧）
bash eval.sh --subject <对象> --work-name <版本> --frames 200
bash eval.sh --subject <对象> --work-name <版本> --base-work <基线版本>
```

## 产物

```text
output/
  train/<对象>/<版本>/          训练 checkpoint、报告、结构快照
  render_test/<对象>/<版本>/      评测报告/对比视频/展示图
  compare_defects/<对象>/<版本>/  与基线的缺陷对比
  analyze/<对象>/<版本>/          解剖与统计
  online_update/<实验名>/        在线更新状态与对比图
  repairs_base_5deg/              修复图生成的中间产物（可删）
  cache/                          可重建缓存（FLAME 拓扑等）
models/<对象>/<版本>/            最终 model.ply + config.yaml + report.json
```

评测其它数据集时，上述路径末尾追加 `/<数据集>`，避免不同数据集覆盖；多版本
评测默认落在 `render_test/<对象>/<主版本>_multi/`。最终模型只保留在 `models/`，
`output/train/` 只留 checkpoint。同名目录/版本默认拒绝覆盖，确认重跑需显式
`--force`。

## 配置

- 路径/设备：`configs/system.yaml`
- 渲染/模型结构：`configs/render.yaml`
- 训练/后训练/在线更新：`configs/train.yaml`
- 本地覆盖：`configs/local.yaml`

## 目录

```text
src/live3dgsavatar/
  core/       高斯、形变、绑定、相机、光栅化
  data/       数据集、场景、样本源、姿态
  training/   训练循环、损失、参数计划
  evaluation/ 缺陷检测、视图候选、指标
  app/        GUI 后端、协议、会话
  tracking/   RDF 实时驱动桥
web/          前端（原生 ES modules）
scripts/      各步骤实现与独立工具
configs/      配置
tests/        后端单测
submodules/   本地 CUDA 扩展
models/       模型（不入库）
datasets/     数据（不入库）
output/       训练/评测产物（不入库）
```

## 测试

```bash
python tests/run_tests.py
node web/tests/all.mjs
```

## 致谢

渲染管线参考 [RGBAvatar](https://github.com/gapszju/RGBAvatar)，并使用
[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)、
[fused-ssim](https://github.com/rahul-goel/fused-ssim)、
[nvdiffrast](https://github.com/NVlabs/nvdiffrast) 等开源工作。

本项目采用 MIT 许可证；`submodules/` 下的第三方源码受各自许可约束，
详见 [`submodules/ATTRIBUTION.md`](submodules/ATTRIBUTION.md)。
