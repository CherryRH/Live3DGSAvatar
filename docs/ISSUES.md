# 开发问题日志

记录已遇到、值得保留的问题与处理结论。新增问题追加到表尾，不再写进设计文档。

| 日期 | 问题 | 原因 | 处理 | 状态 |
|---|---|---|---|---|
| 2026-11 | 训练结束找不到 `models/<对象>/<版本>/config.yaml` | 训练中配置目录被移动/删除，结束时按兄弟路径解析 | 训练前把结构配置快照到 `output/train/.../config.yaml` | 已修复 |
| 2026-11 | 后训练 LPIPS 逐轮变差 | 单轮按 PSNR 选 checkpoint、跨轮按分数选 best，和“最后一轮”不一致 | 默认 quality 评分；容差内接受最后一轮/最后检查点；超容差才回退 | 已修复 |
| 2026-11 | 伪标签把锐度 0.913/0.957 的视角判为糊 | mean-gradient 把渲染噪点算进去，大掩码下阈值过严 | 放宽默认阈值并暴露配置；先看 accepted 再收紧 | 已修复 |
| 2026-11 | normal 图上下颠倒、mask/observed 跟着错 | `pseudo_label` 逐视角按 alpha IoU 重判 y 翻转，近似对称轮廓下 0.768 vs 0.765 | `find_defects` 统一判定 `raster_flip` 写入 `selected.json`，下游复用；旧结果回退 | 已修复 |
| 2026-11 | 基帧仰头 36° 时 `pitch+10` 渲染成 roll | 只减 Euler yaw/pitch，基帧 pitch 较大时合成残留 roll | 改用 `R_target @ R_head.T` 精确补偿；过滤极端基帧 | 已修复 |
| 2026-11 | 参考图角度过大、补偿旋转过大 | 缺陷簇 medoid 可能选到极端头姿帧 | 基帧默认限制 `|yaw|≤30°`、`|pitch|≤20°` | 已修复 |
| 2026-11 | syncfix 报“没有可用伪标签” | 合并路径固定找 `gendir/accepted`，SyncFix 输出直接在 `gendir` | 按模式选择 `refined_src` | 已修复 |
| 2026-11 | `--frame-mode tail` 仍显示“起始 0” | 配置行打印的是 `--frame-offset`，tail 不使用 | 配置行标注 frame-mode，载入后打印实际帧范围 | 已修复 |
| 2026-11 | real-only 结果看起来远好于预训练 | refine 用 `split=all`（含基础训练留出段）、角度加权采样、指标不含缺口 | 只用于“修复 vs 不修复”的同数据对照；不要直接和 base 平台期比 | 已记录 |
| 2026-11 | 在线更新 `--compare-frames -1` OOM | 一次把所有对比帧留在显存 | 限制对比帧数，必要时分批 | 已规避 |
| 2026-11 | FPS 看起来过高 | 原来按 batch=4 的批量吞吐统计 | `render_test` 默认 batch=1，按单帧口径报告 | 已修复 |
| 2026-11 | `chumpy`/numpy 导入或 dtype 报错 | chumpy 旧接口、导入顺序、FLAME float64 约定 | 兼容补丁在导入 chumpy 前生效；不要覆盖 FLAME dtype | 已记录 |
| 2026-11 | `nvdiffrast` 无法在 CPU 运行 | 只有 CUDA 后端 | 环境自检/无 GPU 测试绕开渲染；GPU 冒烟单独跑 | 已记录 |
| 2026-11 | 身份条件、颜色对齐反而伤质量 | FaceID/IP-Adapter/颜色匹配在实测中换脸或压锐度 | 默认关闭，保留为可选实验 | 已记录 |
| 2026-11 | 新视角孔洞监督导致碎裂 | 孔洞惩罚和现有高斯容量/覆盖冲突，spike 增加 | `hole_weight` 默认 0，仅保留 spill | 已记录 |
| 2026-11 | 文档整篇重生成覆盖手工修改 | 用脚本从数据重建 Markdown，未保留人工段落 | 文档只做定点更新；实验数据与设计文档分离 | 已修复 |
| 2026-11 | 评测默认取训练段开头，却按留出集口径解读 | `eval.sh` 默认 `head`，训练验证用末尾 30% | `eval.sh` 默认 `heldout`，比例共用 `test_ratio` | 已修复 |
| 2026-11 | GUI 打不开结构不同的模型 | GUI 只读 `render.yaml`，忽略 PLY/config.yaml | 结构按 PLY → config.yaml → render.yaml 解析 | 已修复 |
| 2026-11 | `data/` 反向依赖 `tracking/` | RDF 转化放在 `data/tracked.py`，与 `tracking/mesh_build.py` 成环 | 转化移入 `tracking/tracked.py`；arch 测试覆盖 data | 已修复 |
| 2026-11 | `BatchRasterizer` 无法用于多相机验证 | 只支持固定单相机，且无调用点/无 GPU 测试 | 删除，统一 `SimpleRasterizer` + `render_frame` | 已记录 |
| 2026-11 | SyncFix 就绪开关与实际路径重复 | `syncfix_ready` 与仓库/权重路径检测表达同一件事 | 去掉开关，运行时检测路径与权重 | 已修复 |
| 2026-11 | 评测报告无法证明取帧方式 | `report.json` 只有帧数，没有模式/帧号/种子 | 增加完整 `params` 与 `frame_range` | 已修复 |
| 2026-11 | 许可证文件与项目元数据不一致 | LICENSE 为 MIT，pyproject/README 写学术用途 | 统一为 MIT | 已修复 |
| 2026-11 | output 中同一模型/标签多份并存 | 训练把 best/final 留在 output 又复制到 models；后训练 cum 与 rN 各存一份标签/模型 | output 只留 checkpoint/报告/累计标签，最终模型只在 models；合并后清理本轮原始副本 | 已修复 |
| 2026-11 | 同名 run 可能静默覆盖旧产物 | train/post 直接写回同名目录/版本 | 默认拒绝已存在目录/版本，需显式 `--force`；`--force` 时清理旧版本 | 已修复 |
| 2026-11 | 评测目录多一层同名 dataset | 默认 `subject/work/dataset` 在 dataset==subject 时重复 | 统一 `output_dir_for`：同名省略 dataset 层，不同名才追加 | 已修复 |
| 2026-11 | refine 训练报 `does not require grad` | 统一后的 `render_frame` 默认 `no_grad`，训练态调用未传 `build_graph=True` | 训练/后训练/novel-view 显式保留计算图 | 已修复 |
| 2026-11 | marcel showcase 首帧已是侧脸，随后继续转 40° | 基帧自身头姿较大，showcase 只在数据集相机上叠加 yaw/pitch，未抵消头姿 | 用 `R_target @ R_head.T`；默认基帧改选头姿最正的一帧 | 已修复 |
| 2026-11 | 后训练保留 mask/none 分支导致流程和依赖膨胀 | mask 标签生成依赖 diffusers/SD1.5/ControlNet，none 只作消融，长期不进入主线 | 收敛为 SyncFix 单一路径，移除 pseudo_label、diffusers 依赖和 mask/none 参数；保留 find_defects/compare_defects/analyze 等分析工具 | 已修复 |
| 2026-11 | HF 中残留扩散模型缓存占用空间 | 后训练不再使用 SD1.5/epiC/ControlNet/SDXL | 用 huggingface_hub 缓存 API 只删除这 4 个 repo，保留 Qwen/BGE 等其它缓存 | 已清理 |
| 2026-11-19 | 加密训练 split 之后模型完全不再学习（PSNR 从第一次 split 起一动不动） | `Trainer` 把 `parameter_groups()` 的返回值存成 dict；`split_unified/prune_unified` 用新的 `nn.Parameter` 整体替换张量，`rebuild_optimizer` 却仍按旧 dict 建优化器 → 梯度全落在没人用的旧张量上 | `Trainer` 支持传**取参函数**（`params=avatar.parameter_groups`），重建优化器前重新取参；train.py/refine_from_images.py 改传函数；补单测（分裂后优化器必须指向新张量） | 已修复 |
| 2026-11-19 | 加密分裂原地复制，副本与父高斯完全重合 | `split_unified` 直接 `cat(xyz[idx])`，位置零抖动；即使修好优化器也只得到两个重合高斯，残差位置不会被覆盖 | 副本沿自身尺度随机抖动（`split_jitter`，默认 0.8σ，可配）；补单测锁住"副本必须偏离父高斯" | 已修复 |
| 2026-11-19 | 训练/验证随机崩 `CUDA error: an illegal memory access`（多轮训练在第 1 步或存档时炸，重启有时又正常） | 改造版 rasterizer 的 kernel 固定按 **RGBA** 读 `target_image`（`target_image[3*H*W+pix_id]`，C++ 也按 `4*H*W` 偏移），Python 侧只传 3 通道 RGB → 越界读过整张平面，是否触发缺页取决于显存分配位置 | `SimpleRasterizer.render` 统一把 RGB 补成 RGBA（第 4 通道=写回掩码），并校验 batch/分辨率；补单测 | 已修复 |
| 2026-11-19 | 硬表面阶段不启用形变基时 4000 步后停滞 | `blend_start=3000` 是给 50000 步老流程设的；硬表面管线每段只有几千步，`blend_start >= 步数` 等于整段只拟合静态外观 | 硬表面只做初始化（1500 步，形变基关闭），表情残差交给加密段；两段式实测 28.98，与"init 5000+基 3000 + 精修 + 加密"的三段式 29.05 同分，步数 9000→5500 | 已修复 |
| 2026-11-19 | `base_post` 的留出集 PSNR 显著高于基础模型/加密模型（31.46 vs 29.2） | 后训练（`refine_from_images.py`）的锚定帧用 `ManifestSource(split="all")`，只排除 64 个"选点帧"；实测 bala 留出集 1198 帧里 **1195 帧（99.7%）参与了后训练**。`eval.sh` 又用这 1198 帧评测 → 数字基本是 in-sample | 结论：`base_post` 不能当泛化基线；横向比较要么把 refine 的 anchors 限制在训练段，要么用真正未见的帧。训练器新增 `--train-split all`（诊断用）复现该口径：同 stage2 代码，split=all 得 32.51 dB（高于 base_post），split=train 得 29.2 dB，证明差距来自数据而非监督方法 | 已记录 |
