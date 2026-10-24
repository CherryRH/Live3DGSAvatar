# Live3DGSAvatar

**面向视频通话场景的 3DGS 头部重建 —— 实时、可驱动、可持续更新。**

加载预训练模型 → 服务端实时渲染 → 推流到浏览器预览，并支持通话开始时的外观微调。

## 快速开始

```bash
git clone https://github.com/CherryRH/Live3DGSAvatar
cd Live3DGSAvatar

bash setup_env.sh          # 建环境 + 编译本地 CUDA 扩展（只需一次）
conda activate live3dgs
bash run_gui.sh            # 启动服务，浏览器打开 http://localhost:8000
```

> 用 `localhost` 或 `127.0.0.1` 都可以。
> 只想看界面、不连后端：`cd web && python3 -m http.server 8912`（本地预览画面）。

## 实时驱动（可选）

实时驱动需要一个**追踪服务**进程（跑在独立环境里，与本项目不共享 Python）：

```bash
# 终端 A：追踪服务
conda activate realdenseface
python scripts/rdf_bridge.py --port 8980

# 终端 B：本项目
conda activate live3dgs
bash run_gui.sh
```

在界面右侧「驱动」区选 **RDF 实时追踪** 即可。追踪服务未连上时画面停在数据集首帧，
不会黑屏。

## 界面

- **画面区**：勾选「对比原图」后，数据集原图并排显示在左侧，同时给出 PSNR。
- **右侧面板**：模型（对象 / 版本）、驱动（方式 / 数据集）、操作。
- **播放条**：播放、单步、跳帧、对比原图、调试扰动。
- **右上角 ⚙**：渲染与在线更新参数。可改项保存后立即生效；结构性参数只读并说明原因。

## 数据与模型放置

```
datasets/<名字>/                 # 数据集：render_inputs.npz 或 checkpoint/，以及 images/
models/<对象>/<版本>/             # 模型：model.ply + config.yaml
data/FLAME2020/                 # FLAME 资产（启动前必须就位）
```

**只有 FLAME 资产是启动必需的**：模型与数据集可以先空着，起服务后在界面的
「设置 → 数据集根目录」里指到实际位置，再在「模型 / 驱动」里选。

路径都在 `configs/system.yaml`，界面里改的项会存进 `configs/local.yaml`。

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | 分层、模块边界、设计取舍 |
| [`docs/CORE_GUIDE.md`](docs/CORE_GUIDE.md) | 核心渲染层导读（形状流转、陷阱） |
| [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md) | 坐标 / 矩阵 / 精度约定 |
| [`docs/CONFIG.md`](docs/CONFIG.md) | 配置项说明 |
| [`docs/GUI_PROTOCOL.md`](docs/GUI_PROTOCOL.md) | 前后端协议（HTTP + WebSocket） |
| [`docs/LIVE_DRIVING.md`](docs/LIVE_DRIVING.md) | 实时驱动链路与标定 |
| [`docs/ONLINE_UPDATE.md`](docs/ONLINE_UPDATE.md) | 在线更新：方法、实验与判据 |
| [`docs/CAP4D.md`](docs/CAP4D.md) | 数据增广：用 CAP4D 生成缺失角度 |
| [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) | 环境搭建与排错 |

各脚本的用法写在文件开头的注释里（`scripts/*.py`）。测试入口：

```bash
python tests/run_tests.py       # 后端（零依赖，无需 GPU）
node web/tests/all.mjs          # 前端（无需浏览器）
```

## 目录

```
src/          项目源码（core/ 渲染内核、data/ 数据层、tracking/ 实时驱动、app/ GUI 后端）
web/          GUI 前端（无构建步骤，原生 ES modules）
scripts/      环境自检、渲染测试、在线更新、追踪服务
tests/        单元测试
configs/      系统 / 渲染 / 训练配置
models/       预训练模型（不入库）
submodules/   本地 CUDA 扩展与第三方源码（vendored，见其 ATTRIBUTION.md）
docs/         架构 / 导览 / 约定 / 配置 / 协议 / 环境
data/         输入资产（不入库）
output/       渲染与实验产物（不入库）
```

## 致谢

渲染管线参考 [RGBAvatar](https://github.com/gapszju/RGBAvatar)，并使用了
[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)、
[fused-ssim](https://github.com/rahul-goel/fused-ssim)、
[nvdiffrast](https://github.com/NVlabs/nvdiffrast) 等开源工作。

## 许可

本项目为学术研究用途。`submodules/` 下的第三方源码受各自许可约束，
详见 [`submodules/ATTRIBUTION.md`](submodules/ATTRIBUTION.md)。
