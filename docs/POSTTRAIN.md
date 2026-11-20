# 后训练

后训练固定使用 SyncFix：从基础模型出发，逐角度圈选缺陷视角、做多视图整图
修复，再用真实帧与修复标签混合微调。流程保持简单、可复现，不再保留其它
标签生成分支。

## 1. 入口

```bash
bash post_train.sh --subject <对象> --work-name <版本>
bash post_train.sh --subject <对象> --work-name <版本> --plan
bash post_train.sh --subject <对象> --work-name <版本> --in-place
bash post_train.sh --subject <对象> --work-name <版本> --force
```

`post_train.sh` 独立读取 `configs/train.yaml` 的 `post_train`，不依赖
`train.sh`。`--plan` 只打印解析后的参数；`--in-place` 直接写回
`models/<对象>/<版本>/`，需要同时给 `--force`。

## 2. 每轮流程

```text
find_defects 选角
  → syncfix_label 多视图整图修复
  → 标签并入 cum/
  → refine_from_images 真实帧 + 修复标签微调
  → 质量门、决定是否继续下一圈
```

- 角度圈：`rings: auto` 用训练段 `|yaw|/|pitch|` 的 p90 作为覆盖边缘，再按
  `ring_step` 向外扩；也可以写死 `"30,20;40,30;50,30"`。
- 选角：在当前圈内按缺陷质量选 top-N；SyncFix 一次要 4 张，实际选
  `max(top_n, syncfix_n_per_pass)` 个视角，并限制每帧最多一个视角。
- `yaw/pitch` 是绝对头姿角度；渲染时用 `R_target @ R_head.T` 抵消基帧头姿。
- 光栅 y 翻转由 `find_defects` 统一判定并写入 `selected.json`，下游复用。

## 3. SyncFix 标签

`syncfix_label` 调用独立 SyncFix 仓库/环境，对选中的多个视角和参考图做整图
联合修复，输出直接作为修复标签，不经过伪标签门控。

要求：

- `syncfix_repo` 指向 SyncFix 仓库；
- `syncfix_model` 指向权重目录，且包含 `model.safetensors`；
- 一次默认 4 视角 / 4 参考图，512×512。

## 4. 微调与选模型

- batch：`real_per_batch` 真实帧 + `refined_per_batch` 修复帧；
- 损失：L1 + SSIM + LPIPS；`lr_xyz` 控制切空间位置偏移，0 表示固定位置；
- 质量分：`SSIM − LPIPS + psnr_weight·PSNR`；
- 默认保留最后一轮/最后一个检查点，只要比当前最佳低不超过 `quality_tol`；
  超容差才回退到最佳并停止；
- 累计标签继续出现在后续轮次；更差但仍可接受的轮次继续作为下一轮起点；
- 被拒绝的末轮模型会回滚，`cum/finetune/` 只保留报告，模型只写 `models/`。

## 5. 产物

```text
output/post/<对象>/<前缀>/
  report.json
  r1/defects/selected.json   本轮选角记录
  r2/ ...
  cum/
    selected.json
    mass.npz
    views/ refs/ refined/    累计视图、参考图、修复标签
    finetune/report.json     微调报告（模型文件不在这里）
```

每轮合并后，`rN/` 只保留 `selected.json`；原始 views/refs/标签已并入 `cum/`，
不再重复保存。模型在 `models/<对象>/<前缀>_rN/`；默认目录已存在时拒绝覆盖，
确认重跑需 `--force`。

## 6. 配置

```yaml
post_train:
  top_n: 2
  rings: auto
  steps: 100
  real_per_batch: 6
  refined_per_batch: 2
  lr_xyz: 1e-5
  quality_tol: 0.01
  psnr_weight: 0.002
  syncfix_repo: ""
  syncfix_python: syncfix
  syncfix_model: ""
  syncfix_n_per_pass: 4
  syncfix_ref_size: 4
```

SyncFix 仓库/权重是本机路径，写在 `configs/local.yaml`。临时覆盖用
`POST_*` 环境变量。

## 7. 分析工具

这些工具不参与训练，但保留用于定位问题和对比版本：

- `find_defects.py`：缺陷视角、覆盖、孔洞、飞点；
- `compare_defects.py`：微调前后同一缺陷视角的结构缺口/锐度/拟合；
- `analyze_gaussians.py`：解剖、UV 密度与消融；
- `render_test.py --work-names ...`：多版本同一留出集渲染对比。
