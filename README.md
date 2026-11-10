# Live3DGSAvatar

**面向视频通话的 3DGS 头部重建：训练、增强、实时渲染。**

给定一段单目视频（RDF 跟踪出逐帧网格与驱动参数），训练一个可驱动的高斯头像；
需要进一步提升观感时，用"找缺陷 → 外部精修 → 微调"再修一遍外观；最后在浏览器里
实时驱动、调整视角查看。

## 能力

| 能力 | 入口 | 说明 |
|---|---|---|
| 数据构建 | `./build_dataset.sh` | 自动跑 RDF 跟踪，烘焙成训练格式 |
| 训练 | `scripts/train.py` | 单目视频 → 可驱动高斯头像 |
| 增强微调 | `scripts/enhance_model.py` | 找缺陷 → 外部精修 → 微调 → 落地评测 |
| 实时渲染 / 前端 | `bash run_gui.sh` | 浏览器预览，可切模型、调整视角、接实时驱动 |

## 已知边界

- 模型能表示的形变来自驱动参数与预定义的基，**几何本身不可通过微调重建**；
  增强微调只改善外观（颜色、不透明度、尺度）；
- 大角度下的表现取决于采集数据覆盖了多少表面。**本阶段不评估数据集本身的
  质量，因此可用角度范围没有结论**——界面上可拖动到的视角范围是控件范围，
  不代表可用范围；
- 对没有真实观测的区域，任何微调都只能给出推测结果。补齐覆盖要靠采集，
  不是训练技巧。

## 快速开始

```bash
bash setup_env.sh                       # 建环境 + 编译 CUDA 扩展（一次）
conda activate live3dgs

./build_dataset.sh <图像目录> --name <对象>          # ① 数据集
python scripts/train.py --subject <对象> --work-name base --promote   # ② 训练
bash run_gui.sh --subject <对象> --work-name base                     # ③ 预览
```

## 一、训练

```bash
python scripts/train.py --subject <对象> --work-name <版本>
python scripts/train.py --subject <对象> --work-name <版本> --promote   # 装进 models/
```

- 数据：`datasets/<对象>/`（`images/` + `render_inputs.npz`）；
- 切分：末尾一部分作验证（`test_ratio`），验证集只含真实数据；
- 早停：正脸 PSNR 与大角度缺口的加权指标（`--no-robust-eval` 可关）；
- 产物：`output/train/<对象>/<版本>/`，`model_best.ply` 为最佳检查点。

参数与取舍写在训练脚本与 `training/` 的注释里。

## 二、增强微调

不动模型格式、不动数据：先用缺陷检测挑出模型的薄弱视角并渲染，交给外部精修器
（当前用 SyncFix）修好，再以缺陷加权的方式微调，最后按验证指标挑检查点。

```bash
python scripts/enhance_model.py --subject <对象> --work-name <版本> \
    --syncfix-dir <SyncFix 目录> \
    --syncfix-model <权重路径> \
    --syncfix-env syncfix \
    --promote-name <版本>_refined
```

四个阶段由一条命令串起来：**找缺陷 → 外部精修 → 微调 → 落地 + 评测**。
每步都能单独重跑（`--skip-find` / `--skip-syncfix` / `--skip-train`），
状态记在 `<缺陷目录>/enhance_pipeline.json`。

默认参数与验收标准写在 `scripts/enhance_model.py` 的注释里。

## 三、渲染与前端

```bash
bash run_gui.sh                  # 默认 http://localhost:8000
bash run_gui.sh --port 8080      # 换端口
bash run_gui.sh --subject <对象> --work-name <版本>   # 启动即加载
```

界面要点：

- **模型**：右侧选对象 / 版本，切换即重载；
- **播放条**：播放、单步、跳帧、对比原图（含 PSNR）；
- **视角**：水平与俯仰可实时拖动，角度语义与缺陷检测一致；
  有视角偏移时不计算 PSNR（数据集原图对不上该视角）；
- **驱动**：可用数据集逐帧驱动，也可接 RDF 实时追踪（见下）。

前端无构建步骤（原生 ES modules）；协议见后端 `app/protocol.py`。

### 实时驱动（可选）

```bash
conda activate realdenseface && python scripts/rdf_bridge.py --port 8980   # 终端 A
conda activate live3dgs     && bash run_gui.sh                            # 终端 B
```

界面右侧选「RDF 实时追踪」。追踪服务未连上时停在数据集首帧，不会黑屏。
细节见 `scripts/rdf_bridge.py` 与 `tracking/`。

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
