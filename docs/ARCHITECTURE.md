# 架构

## 1. 数据流

```text
单目视频 / 外部帧
  → build_dataset：RDF 跟踪 → FLAME 网格、blend_weight、相机
  → datasets/<对象>/render_inputs.npz + images/ + input/
  → train：GaussianAvatar 参数（model.ply）
  → train_avatar：单阶段统一集合训练（可选加密、混入修复图）
  → eval：留出集指标 + compare/showcase
  → app：浏览器预览、实时驱动
```

## 2. 分层

```text
core/        高斯、形变、绑定、相机、光栅化、PLY、指标
data/        数据集、Scene、样本源、FLAME、姿态、模型目录
training/    Trainer、损失、参数计划、观测
evaluation/  缺陷、视角、表情、选择器、鲁棒性、评测协议
app/         GUI 后端、WebSocket 协议、会话、设置
tracking/    RDF 实时驱动桥
streaming/   推流编码
config/      配置加载与覆盖
compat/      chumpy / numpy 兼容补丁
```

依赖方向严格单向：

```text
app / tracking / scripts → evaluation / training / data → core
config / compat 作为基础设施，可被任意层使用
```

- `core/` 不导入 `data/training/app/tracking/evaluation/streaming`；
- `core/` 只接受显式传入的张量与类型，不做文件 I/O、参数解析
  （`core/io/` 的 PLY 序列化是唯一例外）；
- `data/` 不反向导入 `tracking/`；RDF 目录/参数转化归 `tracking/tracked.py`；
- CUDA 扩展只出现在 `core/render/rasterizer.py`、`core/deform/blend.py`、
  `core/deform/binding.py`、`core/render/faceid.py`、`core/render/normals.py`。

## 3. 目录

```text
src/live3dgsavatar/
  core/
    types.py           GaussianSet / Mesh / Camera / RenderOutput
    avatar.py          AvatarConfig / GaussianAvatar
    camera.py          orbit / roll / project
    color.py           SH DC ↔ RGB
    metrics.py         PSNR / SSIM / 累积器
    deform/
      tbn.py           面 TBN
      blend.py         驱动参数 → 切空间高斯
      bind.py          切空间 → 世界空间
      binding.py       UV 光栅化 → Binding
    io/ply.py          PLY 读写 + 内嵌结构
    render/
      pipeline.py      render_frame / render_gaussians（deform → render 的唯一组合）
      rasterizer.py    SimpleRasterizer
      faceid.py        面 id 图 / y 翻转
      normals.py       法线图
      camera_utils.py  行主序 → 列主序
  data/                datasets / scene / sources / flame_* / pose / models / checkpoint_loader
  training/            trainer / losses / objectives / plans / schedule / observations / updater / color_fit
  evaluation/          defects / views / expressions / selection / robustness / protocol / anatomy
  app/                 protocol / server / session / settings
  tracking/            rdf_wire / rdf_client / tracked / param_map / align / frame_source / live_source / mesh_build
  streaming/codec.py   JPEG 编码
  config/__init__.py   Config + 合并/校验
  compat/__init__.py   chumpy 兼容
```

## 4. 入口

| 入口 | 作用 |
|---|---|
| `build_dataset.sh` | 跟踪、烘焙数据集 |
| `train_base.sh` / `scripts/train.py` | 基础训练 |
| `train_avatar.sh` | 单阶段头像重建，可选挂加密 |
| `eval.sh` / `scripts/render_test.py` | 留出集评测 |
| `run_gui.sh` / `app/server.py` | GUI 与推流 |
| `run_tracking.sh` / `scripts/rdf_bridge.py` | 实时追踪服务 |
| `scripts/make_repairs.py` | 用 SyncFix 生成修复图（由 base 渲染；`train_avatar.sh` 的前置） |

## 5. 核心接口

```text
GaussianAvatar.deform(mesh, blend_weight) -> GaussianSet
render_frame(avatar, rasterizer, mesh, blend_weight, camera, bg) -> RenderOutput
render_gaussians(gaussians, rasterizer, camera, bg) -> RenderOutput
load_ply(path, config=None) -> GaussianAvatar
save_ply(avatar, path)
load_scene(cfg, frames=...) -> Scene
ManifestSource(...).batch(indices) -> Batch
Trainer(source, plan, objective, render_fn).run() -> TrainReport
forward_avatar(avatar, rasterizer, mesh, weight, camera, bg, options=...) -> ForwardResult
```

训练与推理的唯一模型接口是 `model.ply` 和 `[B, D]` 驱动参数。
