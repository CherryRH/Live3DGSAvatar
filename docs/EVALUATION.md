# evaluation

`src/live3dgsavatar/evaluation/` 把"哪里渲染坏了"变成可排序、可选视角的量。

## 1. 缺陷

`defects.py` 的缺陷信号最终落到 per-face / per-texel：

| 信号 | 来源 | 抓什么 |
|---|---|---|
| `coverage` | 面 id 图跨帧可见性 | 从没被看到过的区域 |
| `floater` | 高斯到绑定面的距离 | 飞点/离面漂移 |
| `alpha_hole` | alpha vs 面 id 图 | 该有前景却是空的 |
| `sharpness` | 图像梯度 | 糊/涂抹 |
| `delta` | 精修前后差分 | 增强器认为需要改的地方 |

关键函数：

- `attribute_to_faces(values, face_id, n_faces, reduce)`：像素量 → 面；
- `per_texel(per_face, texel_face_id, ...)`：面 → texel；
- `coverage_counts(face_id_maps, n_faces)`：累计面可见次数；
- `floater_offsets(xyz, mesh_verts, ...)`：高斯到绑定面的距离；
- `alpha_hole_ratio(alpha, face_id, ...)`：alpha 孔洞比例；
- `sharpness_map(image)` / `refine_delta(render, refined)`；
- `coverage_defect` / `hole_rate` / `refine_weight_map`；
- `build_defect_map(...)` → `DefectMap(per_face, per_texel, score)`。

`DefectMap.top_faces` 按信号取最差面；`summary()` 输出均/最小/最大。

## 2. 视角

`views.py`：

- `ViewSpec(yaw, pitch, roll)`；标签 `yaw+30_pitch-10_roll+0`；
- `candidate_views(yaw_limit=40, pitch_limit=20, step=10, dense_from=30, dense_step=5)`；
- 侧脸加密：`|yaw|` 超过 `dense_from` 后步长变小；
- 默认 55 个候选（yaw ±40、pitch ±20）；
- `parse_view_label` 严格解析标签，异常输入报错。

## 3. 表情

`expressions.py`：

- `ExpressionSpec(name, weights)`：一个表情/姿态组合；
- `cluster_frames(X, k)`：对驱动参数做 k-means，返回代表帧；
- `axis_extremes(X, k)`：沿主方向取极值；
- `merge_specs(...)`：按余弦相似度去重；
- `load_expressions` / `load_named`：从数据集/配置读候选。

选角时用表情候选 × 视角候选，再由 `selection` 选。

## 4. 选择器

`selection.py`：

- `Candidate(label, covers, payload, position, key)`：一个候选能看到的 texel；
- `greedy_cover(candidates, mass, budget, min_gain_ratio, ...)`：贪心集合覆盖；
- `position` 控制视角多样性，`key` 限制同一基帧数量；
- 返回 `CoverageResult(chosen, gains, total_mass, covered_ratio)`。

选角不按固定对称角，而是按缺陷质量最大化覆盖。

## 5. 鲁棒性

`robustness.py`：

- `RobustConfig`：验证视角（默认 ±20/30/40°）、帧数、缺口权重；
- `robust_views(cfg)`：构造验证视角；
- `alpha_deficit` / `alpha_gradient`：孔洞/边缘缺口；
- `robust_score(psnr_frontal, hole, weight)`：`PSNR − λ·缺口`；
- `summarise_holes` / `summarise_coverage`。

## 6. 评测协议

`protocol.py`：

- `frame_metrics(image, ref_rgb, ref_mask)`：单帧 PSNR/SSIM/LPIPS/锐度；
- `summarize_quality`：汇总 PSNR（全图/mask）、SSIM、LPIPS、锐度；
- `lpips_score`：可选依赖；不可用时返回 None；
- `EvaluationReport`：`has_reference` / `to_dict`；
- `summarize_no_reference`：无真值时的无参考统计。

`render_test.py` 和 GUI 的对比图都调用同一套函数，避免指标口径分叉。

`showcase` 的 yaw/pitch 是**绝对头姿**：渲染时用 `R_target @ R_head.T`
抵消基帧自身头姿。默认基帧从本次评测帧里取头姿最接近正脸的一帧，
也可用 `--showcase-frame` 指定；`--showcase-frame` 会写进报告的
`params.render.showcase_frame`。

多版本对比用 `--work-names`：

```bash
python scripts/render_test.py --subject crh_1 --dataset crh_1 \
    --work-names base base_post_r2 wk1116_mask_r2 wk1116_none_none \
    --frame-mode heldout --frames -1
```

单模型时产物仍是 `compare.mp4`（原图 | 渲染）和 `showcase.mp4`（单模型）。
多模型时默认落在 `<主版本>_multi/`，`compare.mp4` 为“原图 + N 个渲染”横排，
`showcase.mp4` 为 N 个模型同角度横排；`report.json` 的 `models` 记录每个版本
各自的 PSNR/SSIM/LPIPS/性能，顶层 `quality`/`perf` 仍保留主版本，兼容旧读取方。

`eval.sh` 默认用 `--frame-mode heldout --frames -1`：取训练末尾
`train.pipeline.test_ratio`（默认 0.3）对应的留出集，而不是训练段开头。
`render_test.py` 本身默认仍是 `head`，便于快速冒烟和逐帧排查。

`report.json` 的 `params` 记录完整复现信息：命令、模型路径/结构、数据集路径、
取帧模式与实际帧号、背景/批大小/视频参数、完整解析后配置、Python/torch/CUDA
与 GPU。`quality` 仍只放指标，便于对比脚本只读指标。

## 7. 复杂度/解剖

`anatomy.py` 是离线诊断工具：

- `summarize` / `histogram_counts`；
- `dead_mask` / `faces_without_gaussians`；
- `allocate_targets`：按面分配目标点数；
- `rgb_from_sh_dc` / `sh_dc_from_rgb`。

用于分析容量、死高斯和面覆盖，不进入训练/推理主路径。
