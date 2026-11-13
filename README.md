# Live3DGSAvatar

单目视频 / RDF 跟踪结果 → 可驱动的 3D 高斯头像，包含训练、后训练、评测和浏览器预览。

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
| `post_train.sh` | 后训练，支持 `mask_diffusion` / `joint_syncfix` / `none` | `models/<对象>/<版本>_post_rN/` |
| `eval.sh` | 在留出集上渲染评测 | `output/render_test/.../report.json`、`compare.mp4`、`showcase.mp4` |
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
bash post_train.sh --subject <对象> --work-name <版本> --repair-mode joint_syncfix

# 评测
bash eval.sh --subject <对象> --work-name <版本> --frame-mode tail --frames 200
bash eval.sh --subject <对象> --work-name <版本> --base-work <基线版本>
```

## 配置

- 路径/设备：`configs/system.yaml`
- 渲染/模型结构：`configs/render.yaml`
- 训练/后训练/在线更新：`configs/train.yaml`
- 本地覆盖：`configs/local.yaml`

字段和脚本参数见 [docs/PARAMETERS.md](docs/PARAMETERS.md)。设计说明见
[docs/DESIGN.md](docs/DESIGN.md)，开发问题日志见 [docs/ISSUES.md](docs/ISSUES.md)。

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
