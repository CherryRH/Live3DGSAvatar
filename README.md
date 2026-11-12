# Live3DGSAvatar

**面向视频通话的 3DGS 头部重建：训练、后训练、实时渲染。**

给定一段单目视频（RDF 跟踪出逐帧网格与驱动参数），训练一个可驱动的高斯头像；
基础训练默认带新视角 spill 几何监督。需要继续修大角度时，开启**后训练**：
用生成伪标签做数据驱动的迭代修复。最后在浏览器里实时驱动、调整视角查看。

## 能力

| 能力 | 入口 | 说明 |
|---|---|---|
| 数据构建 | `bash build_dataset.sh` | 自动跑 RDF 跟踪，烘焙成训练格式 |
| 训练 | `bash train.sh` | 单目视频 → 可驱动高斯头像（默认 spill 监督） |
| 后训练（可选） | `bash train.sh ... --post-train` | 生成伪标签 + 迭代修复，按 SSIM/LPIPS 选最佳 |
| 评测 | `bash eval.sh` | 对比视频 + PSNR/SSIM/LPIPS |
| 实时渲染 / 前端 | `bash run_gui.sh` | 浏览器预览，可切模型、调整视角、接实时驱动 |

## 已知边界

- 模型固定 N 与 UV texel 绑定；后训练**不新增点**，只优化现有高斯的颜色、
  不透明度、尺度与位置。未观测区域依赖生成伪标签，质量受数据覆盖限制；
- biden 的采集角度不对称（yaw 约 [−3°, +13°]、pitch 约 [−14°, +2°]）。
  后训练用数据驱动选角；外扩到 ±50° 时可能出现脖子/胸口填充，默认按质量分
  （`SSIM − LPIPS + 0.002·PSNR`）早停并保留最佳轮次；
- 对没有真实观测的区域，生成伪标签只能给出**推测**；补齐覆盖要靠采集或更强的
  多视角生成器，不是训练技巧；
- 后训练不做语义分割；默认 `head_cut: true` 把掩码裁到最低观测像素附近，
  避免补出脖子/胸口/衣服（代价是这些区域可能留空）。提示词已加
  clothing / jewelry / makeup 负向词，但无法保证身份完全不变。

## 快速开始

```bash
bash setup_env.sh                       # 建环境 + 编译 CUDA 扩展（一次）
conda activate live3dgs

bash build_dataset.sh <图像目录> --name <对象>            # ① 数据集
bash train.sh --subject <对象> --work-name base             # ② 训练（默认 spill）
bash train.sh --subject <对象> --work-name base --post-train  # ②' 可选后训练
bash run_gui.sh --subject <对象> --work-name base          # ③ 预览
```

后训练需要额外下载 `emilianJR/epiCRealism`（diffusers）；不开启时基础训练不受影响。

## 一、训练

```bash
bash train.sh --subject <对象> --work-name <版本>
bash train.sh --subject <对象> --work-name <版本> --no-spill        # 原版无几何监督基线
```

- 数据：`datasets/<对象>/`（`images/` + `render_inputs.npz`）；
- 切分：末尾 30% 作验证（`test_ratio`），验证集只含真实数据；
- 损失：默认 L1 + 新视角 spill；可选 SSIM/LPIPS（`ssim_weight` /
  `lpips_weight`，默认 0，权重相对 L1，先从小值 A/B）；
- **默认几何监督**：每 4 步采一个 ±10..50° 新视角，罚 FLAME 表面外
  12px 之外的半透明 spill（`weight 0.05`；实色头发不罚）。`--no-spill` 可回退；
- 早停：正脸 PSNR 与大角度缺口的加权指标（`--no-robust-eval` 可关）；
  每 500 步验证，连续 2 次提升 < 0.05 dB 就停（平台期不空转）；
- 计时：每次存盘打印累计耗时（`[save]` / `[best]`），`report.json` 记录
  `best_seconds`、`checkpoints` 与逐次验证的 `eval_history`；
- 产物：`output/train/<对象>/<版本>/`；训练结束**自动**把最佳检查点装进
  `models/<对象>/<版本>/`（同名覆盖）。

## 二、后训练（生成伪标签 + 迭代修复）

一条命令串起：**选角 → epiC 伪标签 → 缺陷加权微调 → 质量门 → 早停/最佳**。

```bash
bash train.sh --subject <对象> --work-name <版本> --post-train
```

基础模型已存在、只想跑后训练时用 `--post-only`；只看这条路线解析出的参数用
`--plan`（不跑 GPU）：

```bash
bash train.sh --subject <对象> --work-name <版本> --post-only
bash train.sh --subject <对象> --work-name <版本> --plan
```

默认每轮保留 `models/<对象>/<版本>_post_rN`，报告里选最佳轮。如果希望后训练
始终写回当前模型名（例如 `best`），加 `--in-place`：

```bash
bash train.sh --subject biden --work-name best --post-train --in-place
# 只有质量更好时才覆盖 models/biden/best；不生成 best_post_rN
```

`best` 是基础训练装入的验证集最佳检查点（`train.py` 用 `model_best.ply`），
后训练直接从它继续。基座原始 checkpoint 仍在
`output/train/<对象>/<版本>/model_best.ply`，需要回滚时用它。

等价于基础训练落地后运行 `scripts/iterative_repair.py`：

```bash
python scripts/iterative_repair.py --subject <对象> --base-work <版本> \
    --prefix <版本>_post --top-n 2 --rings "20,10;30,20;40,30;50,30"
```

- 视角：`find_defects.py` 在当前角度圈里选**最差的 top-N**（不是固定对称角）；
  `yaw/pitch` 是相对头部的绝对角度，渲染时会精确抵消基帧头姿；
  角度圈默认 `rings: auto`：用训练段 |yaw|/|pitch| 的 p90 作为
  覆盖边缘再向外扩，避免在已有真实数据的角度上让扩散重绘；
  基帧 |yaw|>30° 或 |pitch|>20° 会被过滤，避免参考图和补偿旋转过大；
- 标签：`pseudo_label.py` 用 epiC + normal ControlNet 只重绘未观测/缺陷区域，
  过填充、可信区、一致性门后才接受；提示词为 bare face/neck + 负向
  clothing/jewelry/makeup，`head_cut` 默认排除未观测的脖子/胸口；
- 掩码：向缺陷区外扩 12px（`mask_dilate`），给扩散模型足够上下文；
- 监督：累计标签（旧视角继续监督）+ 真实帧 8 : 伪标签 2；
- 选最佳：**`SSIM − LPIPS + 0.002·PSNR`**（PSNR 权重很小，避免选到
  「PSNR 高但脖子起雾」的轮）；`refine_from_images` 的 `model_best` 默认也按
  这个 quality 选（`--select-by quality`）。质量分比最佳低 0.01 就早停，
  且更差的轮次不再作为下一轮起点；
- 产物：模型在 `models/<对象>/<版本>_post_r*/`；一次后训练的全部中间产物都在
  一个目录里，不再散落：

  ```text
  output/post/<对象>/<版本>_post/
    report.json              汇总（每轮耗时、最佳轮）
    r1/defects/              选角结果（渲染视角 + 真实参考帧）
    r1/pseudo/               扩散生成的伪标签
    r2/ ...                  下一圈
    cum/                     累计伪标签 + finetune 产物
  ```

  `rN/defects` 是**输入**（选出来的缺陷视角），`rN/pseudo` 是**输出**（扩散伪标签），
  不是重复；以前它们分别叫 `<版本>_post_rN` 和 `gen_<版本>_post_rN`。

默认参数见 `configs/train.yaml` 的 `post_train` 段；详细说明见
[`docs/POSTTRAIN.md`](docs/POSTTRAIN.md)。旧的外部 SyncFix 链路已移除。

## 三、评测

```bash
bash eval.sh --subject <对象> --work-name <版本>                     # 默认前 64 帧
bash eval.sh --subject <对象> --work-name <版本> --frame-mode tail --frames 120
bash eval.sh --subject <对象> --work-name <版本> --frame-mode random --frames 64
bash eval.sh --subject <对象> --work-name <版本> --base-work <基线版本>
```

- 取帧：`--frames` 定数量，`--frame-mode head|tail|random` 定从哪里取；
- 产物：`report.json`（PSNR / SSIM / LPIPS / 锐度 / 性能）+ `compare.mp4`
  （原图与渲染并排，逐帧叠加指标，30 FPS）；给了 `--base-work` 时额外做同视角缺陷对比。

`scripts/render_test.py` 还额外输出大角度展示视频 `showcase.mp4`（默认开）：

```bash
python scripts/render_test.py --subject <对象> --work-name <版本> --frames 20
# showcase.mp4：yaw 0→40→-40→0，再 pitch 20→-20→0；120 帧 / 30 FPS（约 4s），每段缓动
```

## 四、渲染与前端

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
bash run_tracking.sh    # 终端 A：追踪服务
bash run_gui.sh         # 终端 B：本项目的界面
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
*.sh          一键入口：环境 / 数据集 / 训练 / 评测 / 预览 / 追踪服务
web/          前端（无构建步骤）
scripts/      各步骤的实现（也可单独调用）：train.py、iterative_repair.py 等
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
