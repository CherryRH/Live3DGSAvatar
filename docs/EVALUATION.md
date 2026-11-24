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

- `HOLE_THRESHOLD`：判定"洞"的 alpha 阈值（训练、离线缺陷评估、报告共用）；
- `RobustConfig`：验证视角（默认 ±20/30/40°，头转）、帧数、缺口权重；
- `robust_views(cfg)`：构造验证视角；
- `alpha_deficit` / `alpha_gradient`：孔洞/边缘缺口；
- `robust_score(psnr_frontal, hole, weight)`：`PSNR − λ·缺口`；
- `summarise_holes` / `summarise_coverage`。

## 6. 评测协议

`protocol.py`：

- `frame_metrics(image, ref_rgb, ref_mask, with_lpips=...)`：单帧 PSNR/SSIM/LPIPS/锐度；
- 口径：PSNR 在 mask 内；SSIM 把 mask 外**置 0** 后再滤波（与背景色无关）；
  LPIPS 是 full-image；锐度统一用 `core.metrics.sharpness`（mask 内平均梯度）；
- `summarize_quality`：汇总 PSNR（全图/mask）、SSIM、LPIPS、锐度；
- `lpips_score`：可选依赖；不可用时返回 None；
- `EvaluationReport`：`has_reference` / `to_dict`；
- `summarize_no_reference`：无真值时的无参考统计。

`render_test.py` 和 GUI 的对比图都调用同一套函数，避免指标口径分叉。

对比视频默认**不写**（省时间）；需要时加 `--save-video`。`showcase.png` 是
2x5 拼图（GT + ±15/±30/±45/±60）。

`showcase` 的 yaw/pitch 是**头转**（不是绕头心转相机）：有 RDF/烘焙数据时
由 `tracking/pose_drive.py` 把基帧的头绕颈关节转过去，相机不动；只有网格拓扑
对不上时才退回相机绕转。默认基帧从本次评测帧里取头姿最接近正脸的一帧，
也可用 `--showcase-frame` 指定；会写进报告的 `params.render.showcase_frame`。

多版本对比用 `--work-names`：

```bash
python scripts/render_test.py --subject bala --dataset bala \
    --work-names base avatar \
    --frame-mode heldout --frames -1
```

单模型时产物是 `compare.mp4`（原图 | 渲染）和 `showcase.png`（固定视角拼图）。
多模型时默认落在 `<主版本>_multi/`，`compare.mp4` 为“原图 + N 个渲染”横排，
`showcase.png` 为固定视角拼图；`report.json` 的 `models` 记录每个版本
各自的 PSNR/SSIM/LPIPS/性能，顶层 `quality`/`perf` 仍保留主版本，兼容旧读取方。

`eval.sh` 默认用 `--frame-mode heldout --frames -1`：取训练末尾
`train.pipeline.test_ratio`（默认 0.3）对应的留出集，而不是训练段开头。
`render_test.py` 本身默认仍是 `head`，便于快速冒烟和逐帧排查。

### 大角度指标（`perf.large_angle`，默认 ±45°）

留出集几乎全是正脸（bala p50 yaw 0.9°），PSNR 看不出大角度退化。`render_test.py`
默认再按 **±45° 头转**（相机不动、由 `pose_drive` 转基帧）渲染两张，记录：

| 字段 | 含义 |
|---|---|
| `coverage` / `hole` | FLAME 表面内 alpha>0.5 / <0.5 的比例（几何覆盖与空洞） |
| `outside` | 实心像素落在 FLAME 表面外的比例（头发/肩/漂浮，需配合目视） |
| `haze` | 白底上半透明暗像素占比（轮廓薄雾） |
| `sharp` / `sharp_rel` | 表面内平均梯度，及相对正脸的比值（细节保留率） |
| `area` | 剪影面积（跨角度稳定性） |

关闭用 `--no-large-angle`，换角度用 `--large-angle-yaws "45,-45"`。
报告里同时有 `"+0"` 作为参照。需要**身份一致性**时用独立脚本
`scripts/large_angle_metrics.py`（ArcFace 余弦 vs 正脸，只在同角度跨模型比较）。

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

## 评测口径与严谨性（2026-11-20 评审 A1）

**统一留出集**：末尾 30%（`train.pipeline.test_ratio`）既是验证集也是最终报告集，
不另切第三份。这是本项目的固定约定，换来的是所有配置都在**同一批帧**上比较。

**必须知道的代价**：`model_best` 是在这个 30% 上按**组合分**挑出来的
（`score = PSNR + 50·SSIM + 50·(1−LPIPS)`，SSIM/LPIPS 抽 240 帧），所以报告是
"best-of-N on test"，绝对值偏乐观。选中的 checkpoint 常在中段（如 5499 步），
而末段 checkpoint 的 PSNR 可能更低、LPIPS 更高——跨行比较请统一用同一套
选择规则与同一 `metric_version`。

**报告字段**（`report.json`）：

- `params.model_structure` / `model_structure_per_model`：**实际加载**的结构与
  高斯数（不再写 CLI 默认值）；
- `params.evaluation`：`test_split`、`selection_split`、`selection_metric`，
  明确"选模型与报数共用同一留出集"；
- `params.provenance`：git commit + dirty、模型文件 sha256、`metric_version`、
  mask 定义、孔洞阈值、LPIPS backbone/版本。

**可比性规则**：

1. 只有 `metric_version` 相同的数字才能并排比较（2026-11-20 起 robust/展示图从
   "相机绕转"改为"FLAME 头转"，口径变了）；
2. 同一张表里所有行必须用同一选择规则（同一 robust 定义、同一 `min_delta`）与
   同一训练步数/LPIPS 启用点；消融尤其要注意（3000 步 + LPIPS@2000 与 5000 步 +
   LPIPS@3500 不可混排）；
3. `--no-lpips` 现在**真的**跳过 LPIPS（此前只改 warning，报告里 `lpips: false`
   却仍有数值）。
