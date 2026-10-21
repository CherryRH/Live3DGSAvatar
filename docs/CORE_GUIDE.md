# core/ 代码导览

> 面向**要深入理解代码、准备定位问题**的读者。
> 阅读方式：先看 §1 建立形状感，再按 §2 的顺序通读一遍代码，遇到疑问查 §3（设计原因）
> 与 §4（易错点）。§5 给出"我想验证 X，该跑哪个脚本"的对照表。
>
> 配套：`docs/ARCHITECTURE.md`（为什么这样分层）、`docs/CONVENTIONS.md`（数值约定，权威）。

---

## 1. 先建立形状感

**核心变量只有 5 个**，记住它们的含义和形状，整条链路就通了一半：

| 符号 | 含义 | 典型值（duda） |
|---|---|---|
| `B` | batch（帧数） | 1 或 3 或 10 |
| `V` | 模板网格顶点数 | 5083（FLAME + teeth） |
| `F` | 三角面数 | 10032 |
| `N` | **高斯数量** | 60353 |
| `K` | 约简基数量 | 20 |
| `D` | 驱动参数维度 | 129 |

**`N` 是怎么来的？** 这是理解整个表示的关键：

```
UV 图 tex_size × tex_size = 256 × 256 = 65536 个 texel
          ↓  UV 域光栅化，落在三角形内的 texel 才有效
      有效的 texel 数 = 60353  ← 即 N
```

**一个有效 texel = 一个高斯。** 高斯参数与 UV 图逐像素对齐，所以
`N` 由 `tex_size` 决定，而不是可任意设定。这带来两个后果：

1. 想增加高斯密度只能增大 `tex_size`（且 PLY 里 `weight_module` 压进单列，
   要求 `N ≥ MLP 参数量`，所以 `tex_size` 有下界）；
2. 高斯天然带有 UV 坐标，可以铺回纹理图查看（`GaussianAvatar.extract_texture`）。

### 1.1 三个空间，别混

| 空间 | 谁在哪 | 判据（代码里有断言） |
|---|---|---|
| **切空间** `tangent` | `BlendField` 的输出 | `GaussianSet.space == "tangent"` |
| **世界空间** `world` | `MeshBinder` 的输出、`Rasterizer` 的输入 | `space == "world"` |
| **UV 空间** | 只用于构建绑定关系 | `[0,1]²` |

代码强制这条边界：把世界空间高斯再绑一次、或把切空间高斯直接光栅化，都会 `ValueError`。
**这是故意的**——靠调用顺序隐式区分时，出错只会得到错误图像而不报错。

---

## 2. 一帧的完整数据流

按这个顺序读代码，每一步我都标了张量形状。

### 步骤 0：参数在 PLY 里

```
xyz      [N,3]        基态位置（切空间，初始为 0）
opacity  [N,1]        logit 空间
scaling  [N,3]        log 空间
rotation [N,4]        四元数（WXYZ）
color    [N,1,3]      SH 0 阶系数
xyz_b      [K,N,3]    位置基
rotation_b [K,N,4]    旋转基
color_b    [K,N,1,3]  颜色基
binding_face_id   [N]     每个高斯绑在哪个三角面
binding_face_bary [N,3]   该面内的重心坐标
```

> **为什么是 `[K,N,...]` 而不是 `[N,K,...]`？**
> 因为基与批次无关，`[K,N,...]` 让"取第 k 个基"是连续的切片。
> CUDA 内核（`linear_blending.cu`）也是这个布局。

读代码：`core/io/ply.py::load_ply` → `core/avatar.py::GaussianAvatar`

### 步骤 1：驱动参数 → 混合权重

```
blend_weight [B,D]  ──weight_module(MLP)──▶  weights [B,K]
```

- `D = 129`（100 表情 + 6 neck + 6 jaw + 12 眼 + 1 眼睑 + 3 位移 + …）
- MLP 结构 `D → 128 → 128 → K`，**这是训练侧的固定结构**，
  所以 PLY 里的 `weight_module` 参数量是固定的 35732（D=129, K=20）

读代码：`GaussianBlendField.project_weight`（`core/deform/blend.py`）

### 步骤 2：线性混合 → 切空间高斯

```
weights [B,K]  ×  base [N,·] + Σ_k weights[:,k] · basis[k] [K,N,·]
                              ▼
              xyz [B,N,3], rotation [B,N,4], color [B,N,1,3]
                              ▼ 施加激活
   opacity = sigmoid(base)  ← ⚠️ 不参与混合
   scaling = exp(base)      ← ⚠️ 不参与混合
   rotation = normalize(...)
```

**两条容易读错的点**：

- `opacity` 与 `scaling` **不参与 blendshape 混合**，只有基态值。
  这是约简高斯 blendshape 的关键设计（论文卖点），也是迁移时最容易改错的地方；
- `color` 在 PLY 里存的是 **SH DC 系数**，不是 `[0,1]` 颜色。
  两者相差 `(x-0.5)/0.28209479177387814`，`extract_texture` 会转回颜色。

读代码：`GaussianBlendField.__call__`

### 步骤 3：绑定 → 世界空间

这是最容易出错的一步，也是当前有未解决偏差的一步。

```
输入：mesh.verts [B,V,3]（已变形的网格）
      binding.face_id [N]、binding.face_bary [N,3]
      xyz_tan [B,N,3]、rotation_tan [B,N,4]

① 逐面 TBN（切空间正交基）：
   tri_verts = verts[:, faces]              [B,F,3,3]
   face_tbn  = compute_face_tbn(tri, uvs[uv_faces])   [B,F,3,3]，**第 j 列是第 j 个基**

② 按绑定关系取每个高斯的：
   R = face_tbn[:, face_id]                 [B,N,3,3]
   bv = tri_verts[:, face_id]               [B,N,3,3]

③ 位置：局部坐标经 Rᵀ 旋转 + 重心插值平移
   offset = Σᵢ baryᵢ · bv[:,:,i]            [B,N,3]      ← 重心插值
   xyz_world = Rᵀ · xyz_tan + offset        [B,N,3]

④ 旋转：R 的四元数左乘局部四元数
   rot_world = q(R) ⊗ rotation_tan          [B,N,4]
```

**几何直觉**：高斯"贴"在三角面上。位置由重心坐标决定它贴在哪，
朝向跟随该面的切空间旋转。所以网格一变，高斯跟着变，不需要重新训练。

读代码：`core/deform/tbn.py::compute_face_tbn` → `core/deform/bind.py::MeshBinder.bind`

### 步骤 4：光栅化

```
输入：xyz [B,N,3] 世界空间
      rotation [B,N,4]、scaling [B,N,3]、opacity [B,N,1]、color [B,N,1,3]
      camera（w2c [B,4,4] + K [B,3,3]）
输出：color [B,3,H,W]、alpha [B,1,H,W]
      est_color [B,N,3]、est_weight [B,N]、radii [B,N]（仅训练态）
```

⚠️ **矩阵转置只允许发生在 `core/render/camera_utils.py::to_kernel_matrix`**。
内核按列主序读取，而 PyTorch 是行主序，所以传入前必须转置一次。
把转置散落在各个调用点、再被复制多遍，是典型的隐患来源。

读代码：`core/render/rasterizer.py::SimpleRasterizer`（推理）/ `BatchRasterizer`（训练）

---

## 3. 关键设计原因（"为什么这么写"）

| 设计 | 原因 |
|---|---|
| `GaussianSet` 带 `space` 字段 | 切空间/世界空间混用不会报错、只会给出错误图像，必须显式标注 |
| batch-first，单帧 `B=1` | 混用 `[N,·]` 与 `[B,N,·]` 两套形状会导致重复实现 |
| `BlendField` / `Binder` 拆成两个协议 | 这是为"未来上半身"预留的唯一扩展点：头部是"20 基 + TBN"，身体换成"LBS + 混合拓扑"时渲染层不动 |
| `to_kernel_matrix` 唯一转置点 | 分散的转置是"相机镜像/全黑"这类难查故障的温床 |
| `linear_blending` 默认走纯 PyTorch | CUDA 版在设备不可用时**静默返回全零**，默认走它等于把静默失败放在主链路上 |
| 激活在构造侧完成 | `GaussianSet` 内永远是可直接渲染的物理量，避免"这个张量到底是 log 还是 exp"的反复确认 |
| `AvatarConfig` 是 dataclass | `Struct(**dict)` 式传参无类型检查；而 PLY 还原 MLP 需要知道网络结构 |

---

## 4. 高危陷阱清单

按"踩了会怎样"排序：

| # | 陷阱 | 症状 | 在哪防住 |
|---|---|---|---|
| 1 | 忘记 `to_kernel_matrix` 转置 | 渲染全黑或镜像 | `camera_utils.py` 单点 |
| 2 | 混合 `opacity`/`scaling` | 表情驱动时透明度/尺度异常 | `blend.py` 注释 + 测试 |
| 3 | 把 `color` 当 RGB 而非 SH DC | 整体亮度/对比度偏移 | `extract_texture` 里换算 |
| 4 | `blend_weight` 多 `unsqueeze` 一次 | 会得到"奇怪的渲染结果" | `blend.py` 形状断言 |
| 5 | 切空间高斯直接光栅化 | 高斯挤在一团 | `Rasterizer` 空间断言 |
| 6 | TBN 列/行主序弄反 | 高斯朝向怪异 | `tbn.py` 文档 + 布局测试 |
| 7 | FLAME dtype 是 float64 | 与内核 float32 不匹配 | `conventions §6.3` |
| 8 | `face_id == 0` 当成有效面 | 绑定整体错位一个面 | `binding.py` 减 1 + 掩码 |

---

## 5. "我想验证 X" 对照表

| 想验证 | 命令 / 文件 | 需要 GPU |
|---|---|---|
| 相机矩阵约定 | `tests/unit/test_camera.py` | 否 |
| TBN / 绑定 / 混合的数学正确 | `tests/unit/test_deform.py` | 否 |
| `build_binding` 的面号与重心约定 | `tests/unit/test_binding_construction.py` | 否 |
| PLY 往返与 schema 互操作 | `tests/unit/test_ply.py` | 否 |
| `deform` 全链路形状/单位性/MLP 生效 | `tests/unit/test_core_e2e.py` | 否 |
| 分层依赖规则没被破坏 | `tests/unit/test_architecture.py` | 否 |

| **绑定层的原理性不变量**（刚性等变等） | `tests/unit/test_deform.py` | 否 |
| core 渲染 vs 数据集原图 | `scripts/render_test.py` | **是** |
| 批大小对耗时的实际影响 | `scripts/bench.py --parts render` | **是** |
| 渲染链路的形状/约定 + 指标函数 + 参数校验 + 摘要打印 | `scripts/render_test.py --dry-run` | 否 |
| 全项目无未定义名字（含 GPU-only 分支） | `tests/unit/test_architecture.py` | 否 |
| 配置是否指对路径 | `scripts/show_config.py` | 否 |
| 渲染 / 跟踪 / 全链路性能 | `scripts/bench.py`（可指定数据集、模型与帧数）| **是** |
| GUI 前端（协议/绘制/控件） | `node web/tests/all.mjs` | 否 |


全部不需要 GPU 的测试：

```bash
python tests/run_tests.py            # 零依赖，未装 pytest 也能跑
```

---

## 6. 易错点与已知风险



- **绑定位置项用 `R·x`**（`R` 的列为基向量）。
  ⚠️ 这里**极易搞错**：CUDA `face_tbn.cu` 返回的是「行为基」的版本（内部有 `transpose`），
  而本项目的 Python 侧是「列为基」。**跨来源比对布局必错。**
  这一层还无法自证（`mesh_binding` 是逐元素操作，`R` 与 `Rᵀ` 都数学自洽），
  判据只能是**实测渲染结果**。详见 `CONVENTIONS.md` §3.2.1。
- **TBN 的 tangent 与 bitangent 一般不正交**，这是 UV 参数化的正常现象，
  不是缺陷（`normalize(t)·normalize(b) = −cos(∠A')`）。**不要试图"修"它。**
- **退化面**（UV 面积趋零）会使 `f = 1/det` 爆炸，理论上可产生 `nan`。
  duda 模型暂无此现象，但 781/10032 个面的 UV 面积 < 1e-6，属潜在风险。
- **`face_id` 越界/负值** 已加校验：负索引会被 PyTorch 当作"从末尾数"静默取错面。
- **FLAME 的 dtype 不得覆盖**（float64 是刻意的），否则 `Double/Float` 混算报错
  （见 `ENVIRONMENT.md` §3.11）。
- **需要 pickle/chumpy 的模块必须在模块层 `import live3dgsavatar`**，否则 chumpy 会因 numpy
  别名缺失而 `ImportError`（见 `ENVIRONMENT.md` §3.10）。

## 7. 建议的阅读顺序

```
docs/CONVENTIONS.md §1–§3        ← 先建立空间与矩阵约定
  ↓
core/types.py                    ← 5 个数据结构，含形状断言
  ↓
core/avatar.py                   ← 参数怎么组织、Config 为什么存在
  ↓
core/deform/tbn.py               ← 最短，先看懂切空间正交基
  ↓
core/deform/bind.py              ← 绑定公式（对应 §2 步骤 3）
  ↓
core/deform/blend.py             ← 混合与激活
  ↓
core/deform/binding.py           ← UV 光栅化怎么定出 N
  ↓
core/render/camera_utils.py      ← 唯一转置点
  ↓
core/render/rasterizer.py        ← CUDA 内核的唯一封装
  ↓
tests/unit/test_core_e2e.py      ← 看它怎么用 API，最快理解调用方式
```

每读完一个模块，可以在 `tests/unit/` 里找对应测试，它通常比文档更精确。
