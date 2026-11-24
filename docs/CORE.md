# core

`src/live3dgsavatar/core/` 是渲染内核，不依赖数据、训练或 GUI。

## 1. 数据类型

### GaussianSet

```text
xyz      [B, N, 3]
rotation [B, N, 4]   单位四元数，WXYZ
scaling  [B, N, 3]   正尺度（已 exp）
opacity  [B, N, 1]   [0, 1]（已 sigmoid）
color    [B, N, 1, 3] SH DC
space    "tangent" | "world"
```

- 所有张量 batch-first；
- 存放**激活后的物理量**，不是 logit；
- `space` 显式标注空间，`MeshBinder` 输入必须是 `tangent`，输出是 `world`；
- `to_space()` 只改标注，不做几何变换，用于断言流转。

### Mesh

```text
verts    [B, V, 3]   世界空间顶点
faces    [F, 3]      三角面
uvs      [Vuv, 2]    模板 UV
uv_faces [F, 3]      UV 三角面
```

拓扑与 UV 对所有帧恒定，不带 batch 维。

### Camera

只存 `K` 与 `w2c`（world → camera）：

```text
K    [B, 3, 3] 或 [3, 3]
w2c  [B, 4, 4] 或 [4, 4]
width / height
znear / zfar
```

- 相机沿 **+z** 看；
- 投影深度映射到 `[0, 1]`；
- `proj` 不含 view，完整变换是 `proj @ w2c`；
- 行主序存储；唯一转置发生在 `core/render/camera_utils.to_kernel_matrix`。

## 2. Avatar

### AvatarConfig

```text
tex_size           UV 图边长，决定高斯容量
num_basis_in       驱动参数维度 D
num_basis_blend    约简基数量 K
mlp_hidden         权重投影隐藏层
use_weight_proj    是否启用 D → K 投影
init_opacity       初始化不透明度
init_scaling       初始化尺度
sh_degree          固定 0（只存 DC）
```

`build_weight_module()` 构造权重投影；不使用投影时直接取前 `K` 维驱动参数。

### GaussianAvatar

参数：

```text
xyz      [N, 3]              opacity [N, 1]（logit）
scaling  [N, 3]（log）        rotation [N, 4]（WXYZ）
color    [N, 1, 3]（SH DC）
xyz_b      [K, N, 3]
rotation_b [K, N, 4]
color_b    [K, N, 1, 3]
weight_module（可选）
```

- 只对 **xyz / rotation / color** 做 blendshape 混合；
- opacity 与 scaling 不参与混合；
- `parameter_groups()` 按语义分组，供优化器设不同学习率；
- `deform(mesh, blend_weight)` 组合 `GaussianBlendField` + `MeshBinder`；
- `extract_texture()` 只在构建绑定（`valid_mask` 为 `tex_size²`）时可用；
  从 PLY 载入后 texel 位置已丢失，会显式报错。

## 3. 形变

### 权重投影 / 线性混合

`GaussianBlendField`：

1. `blend_weight [B, D]` →（可选 MLP）→ `[B, K]`；
2. 对 `xyz_b/rotation_b/color_b` 做线性混合；
3. 加上基态，得到切空间 `GaussianSet`。

CUDA `linear_blending` 可加速；默认走 PyTorch einsum。

### TBN

`compute_face_tbn(tri_verts, tri_uvs)` 对每个三角面构造切空间基。
本项目返回的 `TBN` **列**为基向量；CUDA `face_tbn.cu` 内部是行基，两者互为转置。

### 绑定

```text
xyz_world = R @ xyz_tangent + Σ bary_i · v_i
rot_world = q(R) ⊗ rot_tangent
```

- `R = TBN[face_id]`，用 **R** 而不是 `Rᵀ`，与参照实现逐位一致；
- 四元数乘法 WXYZ；
- 矩阵转四元数使用四分支 Shepperd，分支判据与参照一致，避免 GPU 同步；
- 只对高斯实际引用的面计算 TBN；绑定面索引可缓存，TBN 必须逐帧重算。

## 4. 渲染

`render_frame()` 是唯一的“网格 + 驱动参数 → 图像”组合路径：

```text
mesh + blend_weight
  → avatar.deform
  → rasterizer.render
  → RenderOutput(color, alpha, est_color, est_weight, radii)
```

- 推理默认 `torch.no_grad()`；训练用 `build_graph=True`；
- 需要复用同一份形变结果渲染多个相机时用 `render_gaussians()`；
- 只有 `SimpleRasterizer` 一种实现，训练/推理/热更新共用；
- `timings` / `sync` 可选，供 GUI 统计 deform / rasterize 耗时；
- 背景色支持 `[3]` 或 `[B,3]`；
- `RenderOutput` 的颜色/alpha 为 batch-first。

辅助渲染：

- `face_id_map`：面 id 图，用于覆盖统计与孔洞；
- `normal_map` / `vertex_normals`：法线图，支持 OpenGL/其他约定；
- `detect_flip`：判断光栅化 y 方向；`make_repairs.py` 用同一套光栅化生成修复图。

## 5. PLY

顶点列顺序：

```text
x y z  nx ny nz  opacity
scale_0..2  rot_0..3  f_dc_0..2
f_rest_0..44
xyz_b_0..3K-1
rot_b_0..4K-1
f_dc_b_0..3K-1
weight_module
[face_id] [face_bary_0..2]
```

- `f_rest` 固定 45 列零占位；
- `weight_module` 展平进单列，前 P 行有值、其余补零；
- PLY 头写 `comment gaussian_config {...}`，加载时与列数交叉校验；
- `load_ply(config=None)` 时从注释还原；无注释必须显式给 `AvatarConfig`；
- 绑定信息（`face_id` / `face_bary_*`）在加载时必需；缺列的模型无法构造 `GaussianAvatar`。

## 6. 指标

`core/metrics.py`：

- `psnr` / `psnr_masked`；
- `ssim`（可选 mask）；
- `psnr_pair`；
- `max_abs_diff`；
- `Accum`：逐帧累积并输出中位/最小/最大/均值。

评估脚本的 PSNR/SSIM/LPIPS 口径由 `evaluation/protocol.py` 统一。
