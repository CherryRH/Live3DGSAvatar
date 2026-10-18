# Live3DGSAvatar

**面向视频通话场景的 3DGS 人体重建 —— 实时、可驱动、可持续更新的头部数字人**

加载预训练模型 → 服务端实时渲染 → 推流到浏览器预览，并支持通话开始时的外观微调。

## 快速开始

```bash
git clone https://github.com/CherryRH/Live-3DGS-Avatar && cd Live3DGSAvatar

bash setup_env.sh                 # 建环境 + 编译本地 CUDA 扩展（只需一次）
conda activate live3dgs
python scripts/env_check.py       # 环境自检

bash run_gui.sh                   # 启动服务，浏览器打开 http://localhost:8000
```

> **用 `localhost` 或 `127.0.0.1` 都可以** —— 服务默认以 dual-stack 监听，
> 两种写法都通（见 `docs/CONFIG.md`「监听地址」）。
>
> 只想看界面、不连后端：`cd web && python3 -m http.server 8912`（mock 模式，前端本地生成画面）。

## 状态

| 能力 | 说明 | 状态 |
|---|---|---|
| **实时渲染** | 给定驱动参数出图；与数据集原图 PSNR（掩码内）中位 **30.71 dB** @ 175 FPS | ✅ |
| **服务端渲染 + 推流** | 渲染在服务端，客户端只接收视频流；队列容量 1，永不积压 | ✅ |
| **图形界面** | 浏览器实时预览；播放/单步/跳帧、渲染参数实时可调 | ✅ |
| **模型接入** | 读取预训练模型（`models/<subject>/<work_name>/model.ply`） | ✅ |
| **在线更新（热更新）** | 通话开始时用一段观测微调外观（颜色/不透明度），~10 s | ✅ 基础能力（见 [`docs/ONLINE_UPDATE.md`](docs/ONLINE_UPDATE.md)） |
| **训练** | 由合作方预训练提供，本项目**不自研**，仅预留接入点 | 待接入 |

**动态更新**是唯一在册的后续目标。协议已留钩子（数据源 `live`，前端置灰），
但触发策略、抗遗忘采样、与预训练的接口边界都**尚未设计**，不宜现在排期。
其余已评估但暂不做的方向见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §11。

## 使用

```bash
# 查看当前生效的配置（路径是否指对，先看这个）
python scripts/show_config.py

# 渲染测试：出图与数据集原图对比（PSNR / FPS / 显存）
python scripts/render_test.py --frames 20

# 全量 + 并排对比图
python scripts/render_test.py --frames -1 --batch-size 4 --dump-diff

# 无 GPU 的形状/约定自检
python scripts/render_test.py --dry-run

# 实时驱动：桥接进程（realdenseface 环境）+ GUI 里切「实时驱动」
python scripts/rdf_bridge.py --images ~/Datasets/INSTA/crh_2/input --port 8980 --fps 30 --loop

# 在线更新（热更新）：先测闭式颜色校正（无需 GPU）
python scripts/online_update.py --dry-run
python scripts/online_update.py --frames 8 --time-budget 10

# 跨对象更新 + 对比图（crh_2 的画面 → crh_1 的模型）
python scripts/online_update.py --subject crh_1 --data-subject crh_2 \
    --frames 300 --compare-frames 8
```

### GUI 常用参数

```bash
python -m live3dgsavatar.app --help          # 全部选项
python -m live3dgsavatar.app --port 8100     # 换端口
python -m live3dgsavatar.app --target-fps 30 # 帧数上限（同时决定播放推进速度）
```

后端每 `app.stats_log_interval_s`（默认 10 s）向 stdout 打印一行统计，
含真实渲染吞吐与耗时分解：

```
[stats] 10.0s  渲染 598 帧  平均 59.8 FPS  平均 9.04 ms/帧
        （deform 4.18 + raster 3.80 + 其它 1.06）峰值显存 110 MiB  丢帧 0
```

## 安装细节

一键脚本创建独立 conda 环境（`live3dgs`）并编译本地 CUDA 扩展。要求：

- Linux（WSL2 可用）· NVIDIA GPU（CUDA 13.0）· Python 3.10 · PyTorch 2.14.0+cu130

本地 CUDA 扩展（高斯光栅化 / nvdiffrast / fused-ssim）均已随仓库提供，
**无需额外 clone 或联网获取源码**。详见 [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md)。

### 数据与模型

```
data/FLAME2020/generic_model.pkl     # FLAME 模板（需自行从官网获取并遵守其许可）
<DATA_DIR>/<SUBJECT>/checkpoint/     # 每帧 3DMM 参数与相机
<DATA_DIR>/<SUBJECT>/images/         # 每帧图像
```

模型按**人物名 / 工作名**两级目录存放：

```
models/<subject>/<work_name>/model.ply     ← 约 62 MB，不入库
models/<subject>/<work_name>/config.yaml
```

数据集支持两种格式，按文件名自动识别：

| 格式 | 判据 | 说明 |
|---|---|---|
支持两种格式，按目录内容自动识别：

| 格式 | 判据 | 说明 |
|---|---|---|
| **manifest** | 有 `render_inputs.npz` | 已算好逐帧网格/驱动参数/相机，**不需要 3DMM 参数** |
| **checkpoint** | 有 `checkpoint/` + `images/` | 每帧 3DMM 参数，加载时跑一次 FLAME 前向 |

切换对象只改 `configs/system.yaml` 的 `subject` / `work_name`；数据集目录名不同时设 `paths.data_dir`。

模型文件不入库（单个约 62 MB）。把模型与其 `config.yaml` 放到
`models/<subject>/<work_name>/` 下即可；也可以在 `configs/system.yaml`
用 `paths.model_ply` 显式指向别处。

## 测试

```bash
python tests/run_tests.py     # 后端：119 项，零依赖，无需 GPU
node web/tests/all.mjs        # 前端：36 项，无需浏览器
```

## 技术要点

- **表示**：约简高斯 blendshape —— 轻量 MLP 把 3DMM 参数映射为 20 维权重，
  线性混合一组可学习高斯基，在保持表情细节的同时把模型压到约 6 万高斯。
- **绑定**：高斯参数存于模板网格的切空间，通过三角面 TBN 与重心插值跟随网格变形，
  因此任意表情与姿态都可直接驱动，无需重新训练。
- **实时性**：`core/` 走**纯 PyTorch**，batch=1 单帧约 5.3 ms（deform 2.9 + rasterize 2.5），
  约 190 FPS —— 远超 30–60 FPS 的目标。这是**有意识的取舍**：
  CUDA 版 `linear_blending` 在设备不可用时静默返回全零，故默认走纯 PyTorch。
- **训练加速**（合作方）：批并行高斯光栅化 + 颜色初始化，降低 GPU-CPU 同步开销。
- **在线重建**：local-global 双采样池平衡"快速适应新帧"与"不遗忘历史帧"。

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | 架构设计：分层、核心接口、验收证据、后续方向 |
| [`docs/CORE_GUIDE.md`](docs/CORE_GUIDE.md) | **代码导览**：形状流转、设计原因、陷阱清单 |
| [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md) | 坐标 / 矩阵 / 空间 / 精度约定（数值问题的权威来源） |
| [`docs/CONFIG.md`](docs/CONFIG.md) | **配置说明**：路径、渲染参数、优先级、如何改 |
| [`docs/GUI_PROTOCOL.md`](docs/GUI_PROTOCOL.md) | GUI 前后端协议（HTTP + WebSocket） |
| [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) | 环境搭建、版本矩阵、排错 |
| [`docs/ONLINE_UPDATE.md`](docs/ONLINE_UPDATE.md) | 在线更新（热更新）：边界、组件、判据 |

## 目录

```
src/           项目源码（core/ 渲染内核、data/ 数据层、app/ GUI 后端）
web/           GUI 前端（无构建步骤，原生 ES modules）
scripts/       环境自检、渲染测试、等价门、配置查看
tests/         单元测试与数值等价门
configs/       系统配置与渲染配置（见 docs/CONFIG.md）
models/        本项目使用的模型（不入库）
submodules/    本地 CUDA 扩展与第三方源码（vendored，见其 ATTRIBUTION.md）
docs/          架构 / 导览 / 约定 / 配置 / 协议 / 环境 / 差异
data/          输入资产（不入库）
output/        渲染产物（不入库）
```

`src/` 内部分层严格单向（`core/` 不依赖其他任何层），详见
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §2.3 —— 该约束有可执行测试守护。

## 致谢

本项目的渲染管线参考了 [RGBAvatar](https://github.com/gapszju/RGBAvatar)
（CVPR 2025 Highlight）的工作。

## 许可

本项目为学术研究用途。`submodules/` 下的第三方源码受各自许可约束，
详见 `submodules/ATTRIBUTION.md`。
