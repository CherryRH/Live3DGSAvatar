# 坐标系统与数值约定

> 本文件是 Live3DGSAvatar 中**所有涉及坐标、矩阵、空间归属与数值精度问题的唯一权威来源**。
> 任何与此处描述不符的实现都视为 bug。修改本文件必须同步检查 `core/` 与相关测试。
>
> 这些约定来自**实测行为**，已在
> `tests/unit/` 中被验证。

---

## 1. 空间一览

| 空间 | 记法 | 含义 | 谁定义 |
|---|---|---|---|
| 模板空间 | `template` | FLAME 模板网格的规范姿态顶点 | FLAME |
| 切空间 | `tangent` | 以绑定三角面的 TBN 为基的局部坐标系 | `Binder` 的输入 |
| 世界空间 | `world` | 变形后的全局坐标系 | `Binder` 的输出 |
| 相机空间 | `camera` | 视点坐标系，`+x` 右、`+y` 下、`+z` 前 | `Camera.w2c` |
| 裁剪空间 | `clip` | 齐次坐标，`w = z_camera` | `Camera.full_proj` |
| NDC | `ndc` | 透视除法后，`x,y ∈ [-1,1]`，**`z ∈ [0,1]`** | 光栅化器 |
| 像素空间 | `pixel` | `[0,W] × [0,H]`，左上原点 | 光栅化器 |
| UV 空间 | `uv` | `[0,1]²`，用于高斯与网格的绑定 | FLAME 的 `uvs`/`uv_faces` |

**规则**：`GaussianSet.space` 字段必须显式标注 `"tangent"` 或 `"world"`。
跨空间传递高斯是最高频的错误来源，禁止靠调用顺序隐含约定。

---

## 2. 矩阵约定

### 2.1 存储与乘法

- 数学约定：**右乘列向量**，`p_clip = P_full @ p_world`。
- PyTorch 中的 4×4 矩阵按**行主序**存放，即 `M[i, j]` 为第 i 行第 j 列。
- CUDA 内核按 **列主序**读取（`glm::mat3(m[0], m[4], m[8], …)` 形式），
  因此**传入内核前必须转置一次**。

### 2.2 唯一的转置边界

```
core/render/camera_utils.py :: to_kernel_matrix(M)
```

**整个代码库只允许在这一处发生「行主序 → 列主序」的转置。**
其余任何地方不得出现对 `w2c` / `full_proj` 的 `transpose`。

早期实现把该转置散落在调用点（
`camera.get_full_proj.transpose(0, 1)` 各出现一次，且被三份重复的
`render_gs_batch` 各复制一遍），这是需要消除的隐患。

### 2.3 相机矩阵的构造

`Camera` 内部**只存** `K` 与 `w2c`，其余全部派生：

```
# w2c（world → camera）
R = rot                      # 3×3，view→world 的旋转
pos = t                      # 3，相机在世界中的位置
w2c[:3, :3] = R.T
w2c[:3,  3] = -R.T @ pos

# 投影矩阵
fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
n, f, W, H = znear, zfar, width, height
P = zeros(4, 4)
P[0, 0] = 2 * fx / W
P[1, 1] = 2 * fy / H
P[0, 2] = -1 + 2 * (cx / W)
P[1, 2] = -1 + 2 * (cy / H)
P[3, 2] = 1.0
P[2, 2] = f / (f - n)
P[2, 3] = -(f * n) / (f - n)

full_proj = P @ w2c          # 一次性预乘，避免内核侧重复
```

### 2.4 深度范围是 `[0, 1]`（不是 `[-1, 1]`）

`P[2,2] = f/(f-n)`, `P[2,3] = -f·n/(f-n)`, `P[3,2] = 1` 使

```
z_ndc = (f·z_c - f·n) / ((f - n)·z_c)   →   z_c = n 时 0，z_c = f 时 1
```

这是 **OpenGL 的 `[0,1]` 约定**，不是 `[-1,1]`。光栅化器内部的可见性判断与深度排序都依赖它。

### 2.5 视场角与焦距

光栅化器收到的是 `tan_fovx = tan(fov_x / 2)`，内核内部再算
`focal_x = W / (2 · tan_fovx)`，与投影矩阵的 `2·fx/W` 自洽：

```
fov_x = 2·atan(W / (2·fx))      fx = W / (2·tan(fov_x/2))
```

**注意**：内核用 `focal` 做 EWA 协方差投影，用 `full_proj` 做中心投影。两者必须来自同一
`fov`，否则高斯的形状与位置会不一致。**只允许 `Camera` 派生 `fov`，不得在光栅化封装里另算。**

### 2.6 无 GPU 时的矩阵自检

`tests/unit/test_camera.py` 在 CPU 上验证上式，
因此矩阵约定的回归**不需要 GPU**。

---

## 3. 切空间绑定约定

### 3.1 TBN 的构造

对每个三角面 `(v0, v1, v2)` 与其 UV `(uv0, uv1, uv2)`：

```
edge1 = v1 - v0            edge2 = v2 - v0
duv1  = uv1 - uv0          duv2  = uv2 - uv0
f     = 1 / (duv1.x·duv2.y - duv2.x·duv1.y)
tangent   = (edge1·duv2.y - edge2·duv1.y) · f
bitangent = (edge2·duv1.x - edge1·duv2.x) · f
normal    = cross(edge1, edge2)
TBN = normalize_stack([tangent, bitangent, normal])   # 按列堆叠
```

### 3.2 高斯的绑定变换

```
bary     = binding_face_bary[n]            # [N, 3]，重心坐标
face_id  = binding_face_id[n]              # [N]，绑定的三角面下标
offset   = Σ_i bary_i · verts[face_id, i]
R        = TBN[face_id]

xyz_world  = R @ xyz_tangent + offset
rot_world  = q(R) ⊗ rot_tangent            # 四元数左乘，WXYZ
```

**位置用重心插值跟随表面，朝向跟随面切空间旋转。**

### 3.2.1 `R` 的布局约定

`R` 的**列**是基向量 `(tangent, bitangent, normal)`。但"基向量排成行还是列"在
不同来源里是**不同**的，跨来源比对必错：

| 来源 | 布局 |
|---|---|
| CUDA `cuda_utils/face_tbn.cu` | `TBNs[idx] = transpose(mat3(t,b,n))` → **行**为基 |
| 早期 Python 实现 | `stack([t,b,n], dim=-1)` → **列**为基 |
| 本项目 `core/deform/tbn.py` | 同上，**列**为基 |

本项目采用**列为基**，因此位置项用 **`R @ xyz`**（与 `binding_rotations @ gs.xyz` 同语义）。

> ⚠️ **不要**依据 CUDA 内核内部的 `transpose(binding_rotation) * gs_xyz`
> 推断应该用 `Rᵀ` —— 那个 `transpose` 是针对**传入布局**的修正，
> 不能脱离"传入的是什么"来判断。
>
> 这一层**没有"哪个数学上更对"可自证**：`mesh_binding` 是逐元素操作，
> 转置与否都自洽，连"刚性等变"这类不变量也对两者同等成立。
> **判据只能是实测渲染结果**（`tests/unit/test_deform.py` 覆盖关键约定）。

**回归测试**：`tests/unit/test_deform.py::test_bind_matches_reference_translation`
+ `test_mesh_binder_uses_R_not_transpose`（直接打在真实代码路径上）。

### 3.3 深度四元数顺序

全部为 **WXYZ**（`q[0]` 是实部）。`matrix_to_quaternion` 与 `quaternion_multiply`
均要求该顺序；`normalize` 施加在四元数上。

---

## 4. 高斯属性的激活约定

`GaussianSet` 内部**永远存放已激活的物理量**，激活发生在属性构造侧：

| 属性 | 存储形态 | 激活 | 逆变换 |
|---|---|---|---|
| `opacity` | `[0, 1]` | `sigmoid` | `logit` |
| `scaling` | 正尺度 | `exp` | `log` |
| `rotation` | 单位四元数 `[B,N,4]` | `normalize(dim=-1)` | — |
| `color` | SH 0 阶系数 `[B,N,1,3]` | — | `rgb2sh0(rgb) = (rgb - 0.5)/0.28209479177387814` |
| `xyz` | 米 | — | — |

- **sh_degree 恒为 0**，只用 DC 项。写 PLY 时补 45 个零列 `f_rest` 作为占位，
  本项目保留该占位以兼容 3DGS 工具链，但**其值恒为 0**，不得据此推断支持高阶 SH。
- **opacity 与 scaling 不参与 blendshape 混合**（只有 xyz / rotation / color 有基）。
  这是约简高斯 blendshape 的关键设计，迁移时最容易改错。

---

## 5. 批量约定

- **Batch-first**：所有张量第一维恒为 batch，单帧时为 1。
- 历史上存在 `[N,·]` 与 `[B,N,·]` 两套形状并存的情况（
  / `mv_reconstruction.render_gs_batch` 三份实现），`core/` 统一为 batch-first。
- 光栅化器需要 `[N,·]`（无 batch 维），该降维**只允许**在 `Rasterizer` 实现内部发生。

### 5.1 背景色

| 提供方 | 形状 | 处理 |
|---|---|---|
| `SimpleRasterizer` | `[3]` 或 `[B,3]` | 广播到 `[B,3]`，逐帧取一行 |
| `BatchRasterizer` | `[B,3,H,W]` 或 `[B,3]` | 训练时用于随机背景色 |

光栅化器内部做 `bg_color.reshape(-1, 3)` 并
`repeat` 到 batch —— 新实现必须复现这一语义（背景色按 batch 逐行传入内核）。

---

### 5.2 设备一致性（`Scene` 的契约）

`load_scene(device=...)` 返回的 `Scene` 里**所有张量都在该设备上**：
逐帧 `mesh`/`blend_weight`、拓扑 `faces`/`uvs`/`uv_faces`、相机 `K`/`R`/`T`。

调用方直接组装 `Mesh`/`Camera` 即可，**不要再逐个 `.to()`**。

> ⚠️ 这条曾是真实的 bug：`manifest` 路径把帧搬到了 GPU，却直接返回缓存的
> FLAME 拓扑（CPU），于是 `mesh.verts[:, mesh.faces[used]]` 抛
> `RuntimeError: indices should be either on cpu or on the same device`。
> 报错发生在 `core/deform/bind.py`，与根因（`data/` 少搬一个张量）相距很远。
>
> 同理，`_flame_topology()` 的缓存**以 CPU 为准**、每次调用再搬设备 ——
> 若缓存持有 GPU 张量，"先 cuda 后 cpu"的调用会拿到错误设备。
>
> 回归测试：`test_scene_tensors_share_one_device`、
> `test_flame_topology_is_device_independent`。

## 6. 数据目录约定

**任何"可渲染目录"都是同一套结构**：

```
<目录>/
├── images/             图像（文件名按字典序对应帧号）
└── render_inputs.npz   帧数据
```

`render_inputs.npz` 的**内容**有两种 schema，读取时按键自动判别
（`data/cap4d.py::frames_schema`）：

| schema | 必需键 | 来源 | 是否需转换 |
|---|---|---|---|
| `baked` | `mesh (N,V,3)`、`blend_weight (N,D)` | 数据集、增强集 | 否，直接渲染 |
| `rdf` | `x (N,118)`、`verts (N,5023,3)` | `rdf_dump.py`（CAP4D 生成图等）| 补牙 + 映射为 129 维 |

同一约定适用于：`datasets/<对象>/`、`datasets/<对象>/augmented/`、
CAP4D 输出目录（`<任务>/generated_images/`）。

旧文件名（`rdf.npz`、`cap4d_rdf.npz`）仍可读，但会打警告提示改名。

### 构建数据集

**唯一入口**是根目录的 `build_dataset.sh`：它按需跑 RDF 跟踪，再整理成上面的结构。

```bash
./build_dataset.sh <图像目录> --name crh_3     # 新数据集 → datasets/crh_3/
./build_dataset.sh <图像目录> --into crh_1     # 增强集 → datasets/crh_1/augmented/
```

目录里若已有 `render_inputs.npz`（例如 CAP4D 生成图已跟踪过），会跳过 RDF。

图像一律命名为 `00000.png …`（5 位、从 0 连续），与 `render_inputs.npz`
的行序严格对应；构建脚本会重命名并校验。

### 评测产物

`render_test.py` 只接受**模型 + 数据集**两个输入，产物固定为三层：

```
output/render_test/<对象>/<版本>/<数据集>/
├── report.json     指标
├── images/         渲染图
└── diff/           对比图（原图 | 渲染）
```

帧数较多时整轮要几十秒（渲染 ~0.1 s/帧、对比图 ~0.08 s/帧），过程会打印
`[render] 40/212` 这样的进度；只想要指标可加 `--no-diff`，只测速度可加
`--no-save`。

---

## 7. 数值精度与容差

### 7.1 验收门

数值容差的取值依据：

| 比较对象 | 判据 |
|---|---|
| 中间属性（`xyz` / `rotation` / `opacity` / `scaling` / `sh`） | `max\|Δ\| < 1e-5` |
| 最终渲染图 `color` / `alpha` | **`PSNR > 60 dB` 或 `max\|Δ\| < 1e-3`** |

渲染图放宽到 1e-3 是因为光栅化在 tile 内的累加顺序在浮点下不严格可复现。

### 7.2 已知的非确定性来源

| 来源 | 影响 | 处置 |
|---|---|---|
| UV 光栅化精度（`binding()` 的 `compute_rast_info`） | `binding_face_id` / `binding_face_bary` 在极端 texel 上可能差 1 个面 | 绑定结果随模型文件缓存；加载时校验 `N` 与 `binding_face_id` 长度一致 |
| tile 内 α 合成累加顺序 | 末位浮点差异 | 容差 1e-3 |
| `torch.use_deterministic_algorithms` | 未启用 | 等价测试固定随机种子且不涉及随机算子 |

### 6.3 单一 dtype 策略

- 计算链统一 `float32`（内核只接受 float32）。
- **FLAME 模板的 `dtype` 默认是 `float64`**。
  网格顶点是 `float64`，进入绑定前必须显式降到 `float32`，否则会与内核精度不匹配。
  这是最容易出现精度问题的点之一。

---

