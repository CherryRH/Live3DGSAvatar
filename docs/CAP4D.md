# CAP4D：作为数据增广来源

> 定位：**训练数据的补充手段**，不是运行时依赖。
> 渲染链路、驱动链路都不需要它；只有"某个角度/表情没被采到"时才用它造数据。

---

## 1. 它能提供什么，不能提供什么

CAP4D（CVPR 2025）是一个**多视角扩散生成器**：给几张参考图，生成同一身份的
多角度、多表情图，**每张图自带相机与 FLAME 参数**。

| 能 | 不能 |
|---|---|
| 生成 ±55° 偏航 / ±20° 俯仰的**新视角** | 不能"修复"某一帧（不是超分/去噪/inpaint）|
| 附带逐帧 `expr` / `eye_rot` / `rot` / `tra` / `extr` | 条件里 **`jaw_rot` 恒为中性**（闭颌），张嘴靠表情基，幅度有限 |
| 支持最多 4 张参考图融合 | 眼睑无法表达（RDF 也不输出）|
| 512×512，SD2.1 级别画质 | 身份细节（皮肤纹理、发丝）会损失 |

**它没有可训练的部分**：仓库里只有推理代码，MMDM 权重（约 3.6 GB）是预训练的，
我们既不能用它训练，也不能把我们的 `.ply` 塞进去。

## 2. 一条命令导出 CAP4D 的输入

在 RDF 的环境里跑（`flame_solver` 是 cpython-311 编译产物）：

```bash
conda activate realdenseface
cd ~/Projects/Live3DGSAvatar

python scripts/export_cap4d.py --check-deps          # 先确认环境

python scripts/export_cap4d.py \
    --data ~/Datasets/INSTA/<数据集> \
    --out output/cap4d/<数据集> \
    --limit 600
```

脚本做三件事：

1. **跑 RDF**：逐帧跟踪，`identity` 靠关键帧缓冲在整段视频上收敛（不是取首帧）
2. **求位姿**：逐帧用 **Kabsch** 把"规范网格"刚体对齐到 RDF 的输出网格
3. **自检 + 落盘**：重投影误差、残差、参考帧选择、目录布局

### 产物

```
output/cap4d/<数据集>/
├── fit.npz                  shape(150) / expr(T,65) / rot / tra / eye_rot
│                            / fx,fy,cx,cy / extr / resolutions / camera_order
│                            + jaw_rot / x(118) / verts / 残差（CAP4D 忽略这些）
├── reference_images.json    [[ "cam0", t ], ...] 最多 4 张
├── images/cam0/00000.png    参考图（RGB）
├── bg/cam0/00000.png        前景权重（alpha），CAP4D 据此合成白底
├── gen_data.npz             目标表情池（**我们自己的分布**，见 §5b）
└── summary.json             残差分布、内参、参考帧的 yaw
```

参考帧按 **yaw 分位**挑选，让 4 张图覆盖尽量宽的转角（挑连续的几帧几乎没有增益）；
在每个 yaw 区间里**优先挑表情最小的帧** —— 参考图决定 MMDM 学到的"这个人长什么样"，
参考帧自己表情夸张的话，身份先验会被污染。

## 3. 约定：为什么 `extr` 是单位阵

RDF 的相机是 `fov_y=30°`、位于 `(0,0,1)`、`rot=diag(1,-1,-1)`；
CAP4D 在投影前会把顶点做一次 `diag(1,-1,-1)` 翻转，再用 `extr` 变换。

两者合起来，取

```
p_cap4d = flip(R·v_canonical + t),  extr = I
```

时，`p_cap4d` 恰好等于 RDF 的**视图空间**坐标，于是同一个内参投影出的像素坐标
与 RDF 自己完全一致（RDF 的 `project_points(in_viewport=True)` 也是 `u = fx·x/z + cx`）。

`R`/`t` 不手推，而是**逐帧 Kabsch** 求：源是"identity + expr、姿态归零"的规范网格，
目标是 RDF 实际输出的网格（`z` 减去相机距离，即上式的逆翻转）。
这样连 RDF 的 neck 关节效应也被最小二乘吸收，并留下一个可检验的残差。

## 4. 怎么判断这份数据能不能用

脚本会打印四行数字：

```
逐顶点重投影误差：中位 1.82 px  p95 2.31 px  （每帧内先取中位/p95）
系统偏移 (du, dv) = (+0.50, +0.50) px  —— 常数偏差只说明约定差了常量
Kabsch 残差（逐帧均值）：中位 2.29 mm  p95 2.70 mm
Kabsch 残差（逐顶点 p95）：中位 3.90 mm  —— 尾部主要来自下颌/口内
```

怎么读：

| 现象 | 含义 | 处理 |
|---|---|---|
| 逐顶点中位 **> 10 px** | 坐标约定错了 | 脚本报错退出（返回码 2），**不要**用这份数据 |
| 逐顶点中位 **3–10 px** | 多半是下颌残差（CAP4D 条件是闭颌的） | 只警告，继续导出；用 `--max-residual-mm` 筛帧 |
| **系统偏移**明显不为 0（几十 px） | 少了一个常量变换 | 贴给维护者，按偏移反推 |
| 系统偏移 ≈ (0.5, 0.5) px | RDF 的 NDC→像素映射本身少半像素 | 无害 |

⚠️ 早期版本在这里取的是**逐顶点最大值**，而 Kabsch 残差报的是**均值** ——
两者口径不同，2 mm 的均值会被几个口内顶点放大成 14 px，误判成"约定错了"。
现在统一用**逐顶点分布**（中位 / p95），并单独报系统偏移。

## 5. 跑 CAP4D 生成（它自己的环境）

```bash
cd ~/Projects/cap4d
conda activate cap4d_env        # pytorch3d 只在跑它自己的头像训练时才需要

python cap4d/inference/generate_images.py \
    --config_path configs/generation/low_quality.yaml \
    --reference_data_path <上一步的 out 目录> \
    --output_path examples/output/<数据集>/
```

质量档位：`debug`(28 张) / `low`(210) / `medium`(420) / `default`(840) / `high`(840)。
建议先用 `low_quality` 试 210 张，确认大角度侧脸可信再跑全量。

生成结果：

```
<输出>/generated_images/images/*.png     合成图
<输出>/generated_images/flame/*.npz      逐帧的参数（expr/eye_rot/rot/tra/extr/crop_box）
```

**不需要**跑它的 `gaussianavatars/train.py`（那是它自己那套高斯模型，与我们的
reduced-blendshape 参数化不同），跳过即可。

## 6. 已知限制与风险

| 项 | 说明 |
|---|---|
| 许可证 | CAP4D 是 **CC BY-NC 4.0**（非商业），其头像部分另叠 Inria/MPII 的 research-only 许可。**研究可用，产品化不行** |
| 画质上限 | SD2.1 + 512²，身份细节会软化。用合成图训练的风险是**把模型往合成域拽** |
| 闭颌 | 条件几何不含下颌，张嘴幅度基本补不了 |
| **维度截断** | 身份 300→**150**、表情 100→**64**。⚠️ CAP4D 的 65 维 expr 里**最后一维是眨眼**（`flowface` 把基的最后一列换成 `blink_blendshape`），RDF 不估眼睑，末位填 0 |
| 大角度 | ±55° 是生成分布边缘，也正是幻觉最重的地方 —— 而它恰是我们最缺的角度 |

所以：**先用真实数据量一下收益上界，再决定要不要用它**。
如果"同一人、第二段采集"带来的提升本身很小，合成数据不可能更好。

## 7. 常见报错

| 报错 | 原因 | 处理 |
|---|---|---|
| `AssertionError`（`flowface/flame/flame.py:142`，`shape_params.shape[1] == self.n_shape_params`） | 传了 300 维 shape，CAP4D 只要 **150** | 用本脚本导出的 `fit.npz`（已截断），或手工 `shape = shape[:150]` |
| 同一个 `assert` 但来自 `expr` | 传了 100 维，CAP4D 只要 **65** | `expr = expr[:, :65]` |
| `WARNING: ... 210 incompatible with R=4, adjusting ... 212` | 4 张参考图时 `V-R=4`，210 不是 4 的倍数 | **无害**，会自动调成 212 |
| `FileNotFoundError: .../cap4d_mmdm_100k.ckpt` | 没下权重（3.6 GB）| `bash scripts/download_mmdm_weights.sh` |
| `FileNotFoundError: .../flame2023_no_jaw.pkl` | 没下 FLAME 资产 | `bash scripts/download_flame.sh`（需 FLAME 账号）|

## 8. 生成结果表情夸张怎么办

**原因**：CAP4D 自带的 `gen_data.npz`（目标表情池）来自
**MEAD / VFHQ / AVA / Nersemble**，其中 MEAD 是**表演性**情感数据。
实测逐帧 `||expr||`：

| 来源 | 中位 | p95 | 最大 |
|---|---|---|---|
| CAP4D 自带（MEAD 等）| **9.80** | 14.11 | 19.64 |
| 我们 crh_1（真实说话）| **7.38** | 9.95 | 12.60 |

比值约 **1.3×**，尾部差距更大。用它当驱动，生成出来的人自然"做不出这种表情"。

**解法**：换掉目标表情池。导出脚本现在会在输出目录里写 `gen_data.npz`
（用我们自己的表情分布，格式与官方一致：`expr (N,65)` 末位眨眼=0、`eye_rot (N,3)`）。

复制一份生成配置，改两行：

```bash
cd ~/Projects/cap4d
cp configs/generation/low_quality.yaml configs/generation/ours.yaml
```

```yaml
generation_data:
  data_path: /home/test/Projects/Live3DGSAvatar/output/cap4d/crh_1/gen_data.npz   # ← 改这里
  yaw_range: 55
  pitch_range: 20
  expr_factor: 0.8        # ← 再压一档，保险
  n_samples: 210
```

然后用 `--config_path configs/generation/ours.yaml` 重新生成。

`--gen-expr-factor 0.9` 也可以直接在导出时把整池压小，省得改配置。

## 9. 少数图像扭曲/弥散

这是扩散模型在分布边缘的典型失败，按影响从大到小：

| 措施 | 做法 |
|---|---|
| 避开极端角度 | `yaw_range` 从 55 降到 40–45；我们缺的主要是 30–45°，不需要 55 |
| 更多采样步数 | `n_ddim_steps` 50 → 100（`medium/default` 档就是 100）|
| 降低引导强度 | `cfg_scale` 2.0 → 1.5，过强的 CFG 会放大伪影 |
| 换种子 | `seed` 固定为 124，换几个种子看稳定性 |
| 参考图更中性 | 导出脚本已改为"每区间挑表情最小" |

另外：**参考图上限确实是 4**。这不是配置随意，而是模型的训练口径 ——
`data/weights/mmdm/config_dump.yaml` 里 `n_ref: 4`（`max_n_references: ${n_ref}`），
sampler 里 `R = min(n_ref, R_max)`，且 `e_t = model_output[:, R:]` 按 R 切分批次。
想用更多参考就得重训 MMDM。

## 10. 在 CAP4D 的图上渲染（诊断）

用途：不训练，先看**现有 crh_1 模型**能不能重现这些生成视角。
它直接决定要不要投入微调。

### 做法：让 RDF 拟合生成图

```bash
# ① realdenseface 环境：RDF 拟合每张生成图 → x + 5023 顶点
conda activate realdenseface
python scripts/rdf_dump.py \
    --images ~/output/crh_1_200/generated_images/images \
    --out ~/output/crh_1_200/cap4d_rdf.npz

# ② live3dgs 环境：补牙 + 映射 129 维 + 用数据集相机渲染对比
conda activate live3dgs
python scripts/render_test.py --subject <对象> --work-name <版本> \
    --cap4d ~/output/crh_1_200 --dump-diff
```

### 为什么不是"读 CAP4D 的参数自己算网格"

试过，PSNR 只有 **2.4 dB** —— 头和身子都不在同一区域。那条路要同时对四件事：

| 要对齐 | 出错后果 |
|---|---|
| CAP4D 的 `crop_box` → 裁剪后内参 | 整体缩放/平移错 |
| 它那套 FLAME2023 的 150 维 identity | 脸型比例不符 |
| 姿态约定（我们放 neck，它放 root） | 位置整体偏 |
| 我们模型的世界空间 | 无法定位 |

而 **RDF 的虚拟相机与本数据集相机完全一致**（`f=955.4` 正好是
`0.5×512/tan(15°)`，`w2v=diag(1,-1,-1)`、z 平移 1），所以让 RDF 去拟合图之后，
网格与相机天然自洽，不需要任何换算。这也是实时链路本来的形态。

### 怎么读结果

| 现象 | 含义 |
|---|---|
| 并排图里头的位置/大小对上了，但细节差 | 有意义的对比，PSNR 可用来判断"要不要微调" |
| 位置/大小明显不同 | RDF 没拟合好（边缘角度、白底干扰），先看 RDF 自己的诊断 |
| PSNR 高（>30）| 模型本来就能覆盖这些视角，微调没必要 |
| PSNR 低但并排图看着合理 | 模型确实没覆盖到 → 值得微调 |
