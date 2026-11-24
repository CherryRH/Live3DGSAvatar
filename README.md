# Live3DGSAvatar

面向视频通话的 3DGS 头部重建：从单目视频 / RDF 跟踪结果训练可驱动高斯头像，
支持留出集评测、大角度头转指标、可选加密、浏览器预览和实时驱动。

| 模块 | 职责 |
|---|---|
| `core/` | 高斯表示、FLAME 形变与绑定、相机、光栅化；不依赖上层 |
| `data/` | 数据集读取、场景、样本源、姿态/头姿 |
| `evaluation/` | 缺陷检测、候选视图、PSNR/SSIM/LPIPS |
| `app/` | GUI 后端、协议、会话、设置 |
| `tracking/` | RDF 实时驱动桥 |
| `scripts/` | 各步骤实现和独立工具 |
| `configs/` | 系统、渲染、模型、训练配置 |

## 训练方法

**表示**：每个模型绑定 FLAME 网格与 UV 拓扑，高斯存在 UV texel 上（数量随
tex 分辨率变化，默认几万级）。每帧由 129 维驱动参数生成切空间变换，再绑定回
世界空间；高斯属性包括位置、旋转、尺度、不透明度和 SH DC 颜色。

**两组配方**（组内只差一个变量，便于归因）：

| | `rgb`（参照） | `base` | `avatar` |
|---|---|---|---|
| 初始化 | 各向同性球（0.8 mm） | 各向同性球（0.8 mm） | 盘状（2.2 / 0.4 mm） |
| 集合 | UV tex 256 直接绑定 | UV tex 256 直接绑定 | 合并 tex 512 统一集合 |
| xyz 学习率 | 8e-4 | 8e-4 | 2e-4 |
| 损失 | L1 + SSIM 0.1 | L1 + SSIM 0.1 | L1 + SSIM 0.1 |
| 额外监督 | — | `nv_spill` 0.05 + `alpha` 0.2 | `nv_spill` 0.05 + `alpha` 0.2 |
| 训练数据 | 真实帧 | 真实帧 | 真实帧 + **修复图 2 张/批**（默认退火：前 50% 步 2 张，之后 1 张） |
| 背景 | 黑底 | 黑底 | 黑底 |
| 加密 | — | — | 可选（默认关） |
| 步数/早停/选模 | 10k / patience 4 / 组合分 | 同左 | 同左 |

训练统一为 L1 + SSIM 0.1、黑底、10k 步上限、每 500 步验证、组合分
（`PSNR + 50·SSIM + 50·(1−LPIPS)`）选模、patience 4 早停；LPIPS 默认不进损失
（`--lpips` 可选）。`avatar_norep` 是 `avatar` 关掉修复图的对照。

**修复图**：由 `base` 模型按目标头姿渲染 → SyncFix 修复 → 背景色键控抠图
（另有两道有效性检查：FLAME 头部区最大连通缺口、渲染轮廓外新增内容）。

**大角度语义**：数据集里的"转头"是头动、肩胸不动。修复图、showcase、验证和
前端视角都按**头转**渲染（驱动 FLAME 头姿），而不是绕头心转相机。

## 环境

```bash
bash setup_env.sh
conda activate live3dgs
```

## SyncFix 配置

修复图由仓库外的 [SyncFix](https://syncfix.github.io/) 生成：它需要**单独的
conda 环境**和官方权重，不在本仓库里。三处路径在 `configs/local.yaml` 配置：

```yaml
syncfix:
  repo: /path/to/SyncFix          # SyncFix 仓库根目录（含 inference.py）
  python: syncfix                 # 跑 SyncFix 的 conda 环境名
  model: /path/to/SyncFix/models/paper   # 官方权重目录
```

`train_avatar.sh` 在缺修复图时会自动调用 `scripts/make_repairs.py`
（内部通过 `scripts/syncfix_label.py` 启动 `syncfix` 环境的 `inference.py`）。
`make_repairs` 的角度/参考图/破洞阈值等参数都在 `configs/train.yaml` 的
`train.pipeline.repair`，命令行只留对象与输出路径：

```bash
python scripts/make_repairs.py --subject <对象> --work-name base --auto \
  --out output/repair/<对象> --out-dir datasets/<对象>/repair/repairs \
  --repair-root datasets/<对象>/repair
```

## 快速开始

```bash
bash build_dataset.sh <图像目录> --name <对象>
bash build_dataset.sh --video videos/<对象> --name <对象>   # 从视频直接构建
bash train_base.sh --subject <对象>                    # 修复图的来源（work-name 默认 base）
bash train_rgb.sh --subject <对象>                     # RGBAvatar 风格参照（work-name 默认 rgb）
# 修复图（缺省时 train_avatar.sh 会用 base 自动生成；SyncFix 路径见 configs/local.yaml）
bash train_avatar.sh --subject <对象>                    # 盘状 + 修复图 + 退火，默认不加密
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
| `train_base.sh` | 基础模型（各向同性初始化 + 高 lr + spill/alpha，修复图来源） | `models/<对象>/base/` |
| `train_avatar.sh` | 盘状初始化 + 低 lr + 修复图（默认退火 2/8→1/8），可选挂加密 | `models/<对象>/<版本>/` |
| `train_rgb.sh` | RGBAvatar 风格参照（各向同性球、L1+SSIM 0.1、黑底、无 spill/alpha/修复图） | `models/<对象>/rgb/` |
| `eval.sh` | 留出集渲染评测 + ±45° 大角度指标（支持 `--work-names` 多版本） | `output/render_test/.../report.json`、`compare.mp4`、`showcase.png` |
| `run_gui.sh` | 浏览器预览与实时驱动 | 本地 Web 页面 |
| `run_tracking.sh` | 启动 RDF 实时追踪服务 | 追踪 sidecar |

常用参数：

```bash
# 训练（三个入口共用 10k 上限 / patience 4 / 组合分选模）
bash train_base.sh --subject <对象> -- --iterations 8000
bash train_base.sh --subject <对象> --work-name <版本> -- --no-spill

# 头像重建
bash train_avatar.sh --subject <对象>            # 默认：修复图 + 退火，不加密
bash train_avatar.sh --subject <对象> --densify  # 挂上加密
bash train_avatar.sh --subject <对象> --lpips    # 额外开 LPIPS（更慢）
bash train_avatar.sh --subject <对象> --no-repairs   # 无修复图对照

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
  repair/<对象>/                  修复图生成的中间产物（views/syncfix/matte/final，可删）
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

渲染管线参考 [RGBAvatar](https://github.com/gapszju/RGBAvatar)，
修复图使用 [SyncFix](https://syncfix.github.io/)
（*Fixing 3D Reconstructions via Multi-View Synchronization*），并使用
[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)、
[fused-ssim](https://github.com/rahul-goel/fused-ssim)、
[nvdiffrast](https://github.com/NVlabs/nvdiffrast) 等开源工作。

本项目采用 MIT 许可证；`submodules/` 下的第三方源码受各自许可约束，
详见 [`submodules/ATTRIBUTION.md`](submodules/ATTRIBUTION.md)。
