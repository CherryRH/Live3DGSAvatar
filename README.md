# Live3DGSAvatar

**面向视频通话的 3DGS 头部重建：训练、增强、实时渲染。**

给定一段单目视频（RDF 跟踪出逐帧网格与驱动参数），训练一个可驱动的高斯头像；
需要更高质量时，用"找缺陷 → 外部精修 → 微调"再刷一遍外观；最后在浏览器里实时驱动、
拖动视角查看。

## 现在能做什么

| 能力 | 入口 | 状态 |
|---|---|---|
| 训练 | `scripts/train.py` | 可用。默认 50000 步，时长随 GPU 与取数瓶颈变化 |
| 增强微调 | `scripts/enhance_model.py` | 可用。留出集 **+0.4 ~ +0.7 dB**，细节不降 |
| 实时渲染 / 前端 | `bash run_gui.sh` | 可用。视角可实时拖到 ±50°/±30° |
| 数据构建 | `./build_dataset.sh` | 可用（自动跑 RDF、烘焙训练格式） |

**已知边界（重要）**：以 `crh_1` 为例（真实 |yaw| 多在 25° 内，40% 表面积从未朝外），
模型在 ±20° 内可用；更大角度是外推，结构（断裂、空洞）修不好。增强微调只改外观。
下一步的决定性实验是**采一份覆盖 ±35–45° 的数据**，而不是继续调模型。

## 快速开始

```bash
bash setup_env.sh                  # 建环境 + 编译 CUDA 扩展（一次）
conda activate live3dgs

./build_dataset.sh <图像目录> --name <对象>    # ① 数据集
python scripts/train.py --subject <对象> --work-name base --promote   # ② 训练
bash run_gui.sh --subject <对象> --work-name base                     # ③ 预览
```

## 一、训练

```bash
python scripts/train.py --subject crh_1 --work-name run            # 训练并自动挑最佳
python scripts/train.py --subject crh_1 --work-name run --promote  # 装进 models/
```

- 数据：`datasets/<对象>/`（`images/` + `render_inputs.npz`）；
- 切分：末尾 10% 作验证（`test_ratio`），验证集永远只含真实数据；
- 早停：正脸 PSNR 与大角度缺口的加权指标（`--no-robust-eval` 可关）；
- 产物：`output/train/<对象>/<版本>/`，`model_best.ply` 为最佳检查点。

完整参数与取舍见 [`docs/TRAINING.md`](docs/TRAINING.md)。

## 二、增强微调

不动模型格式、不动数据，用外部精修器把模型的薄弱视角修好，再拿修好的图微调：

```bash
python scripts/enhance_model.py --subject crh_1 --work-name run \
    --syncfix-dir ~/Projects/SyncFix \
    --syncfix-model ~/Projects/SyncFix/model/official/ \
    --syncfix-env syncfix \
    --promote-name run_refined
```

四个阶段一条命令：**找缺陷 → 外部精修 → 微调 → 落地 + 评测**。
每步可单独重跑（`--skip-find` / `--skip-syncfix` / `--skip-train`），
状态记在 `<缺陷目录>/enhance_pipeline.json`。

实测（`crh_1` 留出集 286 帧）：

| 版本 | PSNR（mask 内中位） | 锐度比 |
|---|---|---|
| 基线 `run` | 27.42 | 0.834 |
| 增强后 | **28.16（+0.74）** | **0.87** |

规格、参数、验收标准与更多对照见 [`docs/ENHANCE.md`](docs/ENHANCE.md)。

## 三、渲染与前端

```bash
bash run_gui.sh                       # 默认 http://localhost:8000
bash run_gui.sh --port 8080           # 换端口
```

界面要点：

- **模型**：右侧选对象 / 版本，切换即重载；
- **播放条**：播放、单步、跳帧、对比原图（含 PSNR）；
- **视角**：水平 ±50°、俯仰 ±30° 实时拖动，与缺陷检测用的是同一套角度坐标；
  有视角偏移时不计算 PSNR（原图对不上该视角）；
- **驱动**：可用数据集逐帧驱动，也可接 RDF 实时追踪（见下）。

前端无构建步骤（原生 ES modules）；协议见 [`docs/GUI_PROTOCOL.md`](docs/GUI_PROTOCOL.md)。

### 实时驱动（可选）

```bash
conda activate realdenseface && python scripts/rdf_bridge.py --port 8980   # 终端 A
conda activate live3dgs     && bash run_gui.sh                            # 终端 B
```

界面右侧选「RDF 实时追踪」。追踪服务未连上时停在数据集首帧，不会黑屏。
细节见 [`docs/LIVE_DRIVING.md`](docs/LIVE_DRIVING.md)。

## 目录

```
src/live3dgsavatar/
  core/       渲染内核（高斯、形变、光栅化、相机）
  data/       数据层（数据集、样本源、场景、姿态统计）
  training/   训练循环、参数计划、损失、钩子
  evaluation/ 缺陷检测、视角候选、指标
  app/        GUI 后端（会话、协议、设置）
  tracking/   实时驱动的 RDF 桥
  streaming/  推流（预留，未实现）
web/          前端（无构建步骤）
scripts/      训练 / 增强 / 评测 / 数据构建 / 实时驱动
tests/        后端单测（无需 GPU）；web/tests/ 为前端测试
configs/      系统 / 渲染 / 训练 / 模型配置
models/       模型（不入库）
datasets/     数据（不入库）
submodules/   本地 CUDA 扩展（vendored，见 ATTRIBUTION.md）
```

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/TRAINING.md`](docs/TRAINING.md) | 训练：切分、早停、增广、速度 |
| [`docs/ENHANCE.md`](docs/ENHANCE.md) | 增强管线：流程、参数、验收、实测与边界 |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | 分层、模块边界、设计取舍 |
| [`docs/CORE_GUIDE.md`](docs/CORE_GUIDE.md) | 渲染内核导读（形状流转、易错点） |
| [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md) | 坐标 / 矩阵 / 精度约定 |
| [`docs/CONFIG.md`](docs/CONFIG.md) | 配置项说明 |
| [`docs/GUI_PROTOCOL.md`](docs/GUI_PROTOCOL.md) | 前后端协议（HTTP + WebSocket） |
| [`docs/LIVE_DRIVING.md`](docs/LIVE_DRIVING.md) | 实时驱动链路与标定 |
| [`docs/ONLINE_UPDATE.md`](docs/ONLINE_UPDATE.md) | 通话开始时的一段观测做外观微调 |
| [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) | 环境搭建与排错 |
| [`docs/REDESIGN.md`](docs/REDESIGN.md) | 表现力边界的实测与已试方案（记录，代码已回退）|
| [`docs/CAP4D.md`](docs/CAP4D.md) | 生成式数据增广的评估结论（未采纳）|

脚本用法写在各自文件开头（`scripts/*.py`）。

## 测试

```bash
python tests/run_tests.py     # 后端：零依赖、无需 GPU
node web/tests/all.mjs        # 前端：无需浏览器
```

## 致谢与许可

渲染管线参考 [RGBAvatar](https://github.com/gapszju/RGBAvatar)，并使用
[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)、
[fused-ssim](https://github.com/rahul-goel/fused-ssim)、
[nvdiffrast](https://github.com/NVlabs/nvdiffrast) 等开源工作。

本项目为学术研究用途；`submodules/` 下的第三方源码受各自许可约束，
详见 [`submodules/ATTRIBUTION.md`](submodules/ATTRIBUTION.md)。
