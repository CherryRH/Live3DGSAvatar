# submodules/ 归属说明

本目录存放**外部源码的 vendored 副本**，不是本项目原创代码。
三者均由 `setup_env.sh` 直接构建，**不从 `~/Libraries/` 构建**。

## 清单

| 目录 | 上游 | 取得途径 | 许可 | 本项目改动 |
|---|---|---|---|---|
| `diff-gaussian-rasterization/` | [graphdeco-inria/diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)（含 RGBAvatar 扩展） | RGBAvatar `submodules/diff-gaussian-rasterization`，commit `903653b689a8b860df30f165e242ccd6ed3f5583` | 见 `LICENSE.md`（非商业研究） | 无源码改动。删 `build/`、`*.egg-info/`、`third_party/glm/{doc,test}/` |
| `nvdiffrast/` | [NVlabs/nvdiffrast](https://github.com/NVlabs/nvdiffrast) v0.4.0 | 本机 `~/Libraries/nvdiffrast`（上游官方 clone） | NVIDIA Source Code License (NSCL) | 无。删 `build/`、`*.egg-info/`、`.git/`、`samples/data/` |
| `fused-ssim/` | [rahul-goel/fused-ssim](https://github.com/rahul-goel/fused-ssim) v1.0.0 | 本机 `~/Libraries/fused-ssim`（上游官方 clone） | 见 `LICENSE` | 无。删 `build/`、`*.egg-info/`、`.git/` |
| `flame/` | FLAME PyTorch 实现（MPG） | RGBAvatar `submodules/flame` | 非商业研究（MPG 协议） | 无 |
| `fuhead/` | FaceWareHouse（清华） | RGBAvatar `submodules/fuhead` | 见上游 | 无 |

> `flame/` 与 `fuhead/` 属于 P1 的 vendor 计划，当前尚未拷入。

## `diff-gaussian-rasterization` 相对原始 3DGS 的差异

这些差异来自 RGBAvatar，**已包含在本副本中**：

| 文件 | 差异 |
|---|---|
| `cuda_rasterizer/fast_forward.cu` | 新增。按累计权重阈值一次性写入高斯颜色 |
| `cuda_rasterizer/forward.cu` | 新增 `est_color` / `est_weight` / `target_image` 输出 |
| `cuda_utils/linear_blending.cu` | 新增。blendshape 线性混合（前向 + 反向） |
| `cuda_utils/mesh_binding.cu` | 新增。网格绑定（TBN 旋转 + 重心插值平移） |
| `cuda_utils/face_tbn.cu` | 新增。逐三角面 TBN |
| `cuda_utils/rotation.cu` | 新增。矩阵↔四元数、四元数乘法 |
| `rasterize_points.cu` | 新增 `batch_rasterize_gaussians1/2` 两阶段批并行入口 |
| `diff_gaussian_rasterization/__init__.py` | 新增 `BatchGaussianRasterizer` 与 `BatchRasterizationTensors` |

## 为什么全部 vendor，而不引用 `~/Libraries/`

`~/Libraries/` 下的源码树残留了此前手工构建的 `build/` 目录，导致
`pip install ~/Libraries/<pkg>` 在源码树内删除/重建时失败：

```
error: could not delete 'build/lib.linux-x86_64-cpython-310/nvdiffrast/__init__.py': Permission denied
```

Vendoring 到本仓库后：构建可复现、随仓库固定版本、不依赖工作区外的路径。

构建与环境见 `setup_env.sh` 与 `docs/ARCHITECTURE.md`。

## 修改约定

对本目录任何文件的修改，必须在本文件的清单/差异表中登记。
未登记的修改视为违规。
