# Live3DGSAvatar

面向视频通话的 3DGS 头部重建：从单目视频 / RDF 跟踪结果训练可驱动高斯头像，
支持后训练修复、留出集评测、浏览器预览和实时驱动。

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

**表示**：每个模型绑定 FLAME 网格与 UV 拓扑，高斯存在 UV texel 上，固定
约 6 万个点。每帧由 129 维驱动参数生成切空间变换，再绑定回世界空间；
高斯属性包括位置、旋转、尺度、不透明度和 SH DC 颜色。

**基础训练**：真实帧重建 L1，可加 SSIM/LPIPS；每 4 步采一个 ±10–50° 新视角，
对 FLAME 表面外扩 12 px 的半透明外溢（薄雾）做监督，减少侧面和边缘漂浮。
验证使用正脸 PSNR 与大角度缺口的加权指标，平台期早停。

**后训练**：先在真实数据中选缺陷视角，再生成修复标签：

- `syncfix`（默认）：SyncFix 对多视图整图联合修复；
- `mask_diffusion`：掩码 + SD1.5/epiC，只重绘缺失区域；
- `none`：只用真实帧，作为消融基线。

修复帧与真实帧混合微调，真实帧占主导；默认保留容差内的最后一轮，避免
小角度留出集误判大角度效果。

## 环境

```bash
bash setup_env.sh
conda activate live3dgs
```

## 快速开始

```bash
bash build_dataset.sh <图像目录> --name <对象>
bash train.sh --subject <对象> --work-name base
bash post_train.sh --subject <对象> --work-name base
bash eval.sh --subject <对象> --work-name base
bash run_gui.sh --subject <对象> --work-name base
```

## 脚本

| 脚本 | 用途 | 主要产物 |
|---|---|---|
| `setup_env.sh` | 创建 conda 环境、编译本地 CUDA 扩展 | `live3dgs` 环境 |
| `build_dataset.sh` | 跟踪、对齐并烘焙训练数据 | `datasets/<对象>/` |
| `train.sh` | 基础训练 | `models/<对象>/<版本>/` |
| `post_train.sh` | 后训练：`syncfix`（默认）/ `mask_diffusion` / `none` | `models/<对象>/<版本>_post_rN/` |
| `eval.sh` | 留出集渲染评测 | `output/render_test/.../report.json`、`compare.mp4`、`showcase.mp4` |
| `run_gui.sh` | 浏览器预览与实时驱动 | `http://localhost:8000` |
| `run_tracking.sh` | 启动 RDF 实时追踪服务 | 追踪 sidecar |
| `scripts/online_update.py` | 用少量观测做在线外观更新 | `output/online_update/` |

常用参数：

```bash
# 训练
bash train.sh --subject <对象> --work-name <版本> --no-spill
bash train.sh --subject <对象> --work-name <版本> -- --iterations 8000

# 后训练
bash post_train.sh --subject <对象> --work-name <版本> --plan
bash post_train.sh --subject <对象> --work-name <版本> --in-place
bash post_train.sh --subject <对象> --work-name <版本> --repair-mode syncfix

# 评测（默认末尾留出集全部；--frames 可只测前 N 帧）
bash eval.sh --subject <对象> --work-name <版本> --frames 200
bash eval.sh --subject <对象> --work-name <版本> --base-work <基线版本>
```

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
