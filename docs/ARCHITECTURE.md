# Live3DGSAvatar 架构设计文档

> 视频通话场景 3DGS 人体重建 —— 第一阶段（头部）架构设计
> 状态：**待审核**　版本：v0.1　最后更新：待定

---

## 0. 本文档的边界

配套文档：`docs/CONFIG.md`（配置项）、`docs/CONVENTIONS.md`（数值约定）、
`docs/CORE_GUIDE.md`（代码导览）、`docs/ENVIRONMENT.md`（环境）、`docs/MIGRATION.md`（与参照的差异）。

**目标**：定义一个**足够简单、可落地、可扩展**的工程架构，把 RGBAvatar 的算法能力内化成本项目自己的代码，并支撑后续的视频通话场景扩展。

**明确不做**：
- 不追求架构完备性。层级只到"能隔离变化"为止，不做插件框架、不做通用中间表示、不做多后端抽象。
- 不考虑非 NVIDIA 平台。CUDA 13.0 为唯一目标工具链。
- 不引入 VoluMe 等前馈路线。见 §11（后续方向）。

**建议阅读顺序**：§2（分层）→ §3（接口契约）→ §4（目录结构）→ §9（当前状态与验收）。其余为实现细节。

---

## 1. 目标与硬约束

### 1.1 产品目标

第一阶段的理想产物：**一个支持静态训练、实时渲染、动态更新的图形化程序**。

| 能力 | 含义 | 对应后端 |
|---|---|---|
| 静态训练 | 给定单目视频序列，重建一个可驱动头部 avatar | `training/` |
| 实时渲染 | 给定驱动参数，实时出图（目标 ≥ 60 FPS @ 512²） | `core/render/` |
| 动态更新 | 流式到达的帧可持续优化模型（在线重建） | `training/online.py` |

### 1.2 已确定的硬约束

| 编号 | 约束 | 来源 |
|---|---|---|
| C1 | 目标平台为 NVIDIA GPU，CUDA **13.0** | 用户决策。实测 11.8 反而不兼容，官方文档不准确，统一 13.0 |
| C2 | 训练与推理技术栈：PyTorch + 自研 CUDA 光栅化扩展 | 复用 RGBAvatar。已实测可用：torch `2.14.0+cu130` |
| C3 | 渲染部署形态：**服务端渲染，推流给用户端** | 用户决策（实验室算力充足） |
| C4 | 覆盖范围：**仅头部** | RGBAvatar 拓扑边界所限；上半身为未来工作 |
| C5 | 用途：现阶段纯学术/自用 | 商用未定；架构上不封死替换 3DMM/tracker 的路径即可 |
| C6 | `RGBAvatar/` 为**只读参照**，所有实现落在 `Live3DGSAvatar/` | 约束 |
| C7 | 数据集**只用 `duda`** | 这是唯一由项目作者亲手跑通 tracker → INSTA → RGBAvatar 全链路的数据；其余数据集已删除 |
| C8 | 独立 conda 环境 `live3dgs` | 与参照仓库的 `rgbavatar` 环境隔离，避免互相污染 |

### 1.3 本机开发环境

| 项 | 值 |
|---|---|
| Conda 环境 | **`live3dgs`**（新建，python 3.10.21）—— 参照环境为 `rgbavatar` |
| PyTorch | 2.14.0+cu130 |
| CUDA Toolkit | 13.0（conda 环境内 nvcc 13.0.48 / 系统 13.0.88，二者皆可用） |
| GPU | RTX 3060 Laptop, 6 GB |
| 系统 | WSL2, 16 核, 7 GB RAM |

搭建脚本、版本矩阵、排错表见 **`docs/ENVIRONMENT.md`**。

> **说明**：6 GB 显存是本机训练的限制项，但不是架构约束——训练可放实验室服务器。本机足以完成复现与内核重写验证。

---

## 2. 系统总览

### 2.1 一句话架构

**一个只读的 `GaussianSet`（单帧高斯集合）+ 三条纯函数管线（blend / bind / rasterize）+ 两个门面（训练、运行时）。**

### 2.2 分层图

```
┌─────────────────────────────────────────────────────────────┐
│  app/          图形化程序 · CLI · 服务端入口                  │
│                训练监控面板 / 实时预览 / 会话管理              │
├─────────────────────────────────────────────────────────────┤
│  runtime/      门面层：把"模型+数据集+渲染器"组装成可用对象     │
│                AvatarRuntime: train() / render() / update()  │
│                Session: 服务端渲染会话（预留）                 │
├─────────────────────────────────────────────────────────────┤
│  training/     训练循环 · 损失 · 优化器 · 采样池               │
│                offline.py（离线）  online.py（在线/流式）      │
├─────────────────────────────────────────────────────────────┤
│  core/         算法核心（无训练逻辑、无 I/O、无 GUI）          │
│  ├─ types.py        GaussianSet / Mesh / Camera / Frame      │
│  ├─ template/       FLAME / FuHead 模板（封装，不改数学）      │
│  ├─ deform/         blend: 权重→高斯    bind: 法空间→世界     │
│  ├─ render/         rasterize 封装（训练态/推理态两条路径）    │
│  └─ avatar.py       GaussianAvatar（参数容器 + 序列化）       │
├─────────────────────────────────────────────────────────────┤
│  submodules/   唯一允许出现 CUDA 的地方（vendored 源码）        │
│                diff_gaussian_rasterization/（vendored）       │
├─────────────────────────────────────────────────────────────┤
│  data/         数据集读取（FLAME 格式 / NeRSemble 格式）       │
│  tracking/     驱动参数求解（接口 + 离线实现 + 实时占位）      │
│  streaming/    服务端渲染推流（预留）                          │
├─────────────────────────────────────────────────────────────┤
│  compat/       与 RGBAvatar 的对照实现（仅用于回归测试）       │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 依赖方向（严格单向）

```
app → runtime → training → core → ext
                  ↓         ↑
                data ───────┘
                tracking ───┘
```

规则：
- `core/` **不得** import `training/`、`data/`、`app/`、`tracking/`。
- `src/` **不得** import `tests/` —— `tests/` 不是包（无 `__init__.py`），
  只能靠 `sys.path` 找到：
  参照实现适配层原本放在 `tests/` 下，被 `data/scene.py` 引用。现已移入 `data/reference.py`。
- `core/` **不得**做文件 I/O 与参数解析（`avatar.py` 的 `save/load` 除外，见 §9.3）。
- `submodules/` 只被 `core/render/` 引用，其他任何地方不得直接 import。

---

## 3. 核心接口契约（概要）

完整签名、形状约定与逐步骤数据流见 **`docs/CORE_GUIDE.md`**；本节目的是让读者先掌握边界。

**一个数据契约 + 三条纯函数管线 + 两个门面**：

| 组件 | 位置 | 职责 | 关键约束 |
|---|---|---|---|
| `GaussianSet` | `core/types.py` | 跨层唯一数据契约 | batch-first；存**已激活**物理量；带 `space` 标注 |
| `Mesh` / `Camera` / `Frame` | `core/types.py` | 几何、相机、单帧输入 | `Camera` 只存 `K` 与 `w2c`，其余派生 |
| `BlendField` | `core/deform/blend.py` | `[B,D]` 驱动参数 → 切空间高斯 | `opacity`/`scaling` **不参与混合** |
| `Binder` | `core/deform/bind.py` | 切空间 → 世界空间 | 位置用 **`R·x`**（`R` 列为基；布局易错，见 `docs/CONVENTIONS.md` §3.2.1） |
| `Rasterizer` | `core/render/rasterizer.py` | 世界空间高斯 → 图像 | 唯一接触 CUDA 扩展的位置 |
| `GaussianAvatar` | `core/avatar.py` | 参数容器 + PLY 序列化 | 网络结构由 `AvatarConfig` 显式给出 |
| `AvatarRuntime` | `runtime/`（待实现） | 组装门面：`setup/train/render/update` | — |

**约定摘要**（详见 `docs/CONVENTIONS.md`）：

1. 所有张量 batch-first，单帧 `B=1`；
2. 激活在构造侧完成，`GaussianSet` 内永远是可直接渲染的物理量；
3. 行主序 → 列主序的转置**只允许**发生在 `camera_utils.to_kernel_matrix`；
4. 深度映射到 **[0, 1]**（OpenGL 约定），不是 `[-1, 1]`；
5. `sh_degree = 0`，只用 SH 的 DC 项。

## 4. 目录结构

```
Live3DGSAvatar/
├── docs/
│   ├── ARCHITECTURE.md          ← 本文档
│   ├── ENVIRONMENT.md           ← 环境搭建、版本矩阵、排错
│   ├── CONVENTIONS.md           ← 坐标系与数值约定
│   └── MIGRATION.md             ← 与 RGBAvatar 的行为差异清单
├── configs/                     ← system.yaml / render.yaml（见 docs/CONFIG.md）
├── src/live3dgsavatar/
│   ├── core/
│   │   ├── types.py             # GaussianSet / Mesh / Camera / Frame / RenderOutput
│   │   ├── avatar.py            # GaussianAvatar / AvatarConfig
│   │   ├── template/
│   │   │   ├── flame.py         # 封装 submodules/flame（不改数学）
│   │   │   └── fuhead.py
│   │   ├── deform/
│   │   │   ├── blend.py         # BlendField：权重→切空间高斯（linear_blending + MLP）
│   │   │   ├── bind.py          # Binder：切空间→世界（TBN + 重心）
│   │   │   └── tbn.py           # compute_face_tbn 的参考实现
│   │   └── render/
│   │       ├── rasterizer.py    # Rasterizer 协议 + 两个实现
│   │       └── camera_utils.py  # 矩阵转置/列主序转换的唯一入口
│   ├── data/                    # 数据层（无 GPU 计算，只做加载与适配）
│   │   ├── scene.py             #   模板几何 + 相机 + 逐帧 mesh/驱动参数
│   │   │                        #   支持 manifest（自包含）与 checkpoint 两种格式
│   │   └── reference.py         #   参照实现适配（等价门与渲染测试用）
│   ├── training/                # 待接入（不自研，见 ARCHITECTURE §6）
│   ├── tracking/                # 待实现（Tracker 协议 + 离线 metrical-tracker 适配）
│   ├── runtime/                 # 待实现（AvatarRuntime 门面、服务端会话）
│   ├── streaming/               # 待实现（编码/传输）
│   ├── app/                     # GUI 后端：protocol / session / server
│   └── compat/
│       └── __init__.py          # 第三方兼容补丁（numpy 2.x 别名等）
├── submodules/                  ← 全部 vendored，统一从此处构建
│   ├── ATTRIBUTION.md           #   来源 / 许可 / 差异登记
│   ├── diff-gaussian-rasterization/   # 唯一的 CUDA 内核
│   ├── nvdiffrast/                    # UV 域光栅化
│   ├── fused-ssim/                    # 多视角 SSIM（可选）
│   ├── flame/                         # 待拷入
│   └── fuhead/                        # 待拷入
├── tests/
│   ├── run_tests.py             # 零依赖测试运行器（未装 pytest 也能跑）
│   ├── support.py               # 参照加载与比较工具
│   ├── unit/                    # 单测（119 项，全部无需 GPU）
│   └── equivalence/             # 数值等价门（stages.py + GPU 测试）
├── scripts/
│   ├── env_check.py             # 环境自检（无 GPU 可跑）
│   ├── render_test.py           # 统一渲染测试：core vs 参照 vs 数据集原图
│   ├── equivalence_check.py     # 数值等价验收门（与参照逐层比对）
│   └── _compat_shim.py          # 独立脚本用的精简兼容补丁
├── pyproject.toml               # 打包配置（src 布局 + console script）
├── setup_env.sh                 # 一键建环境（顶层，用户直接跑）
├── run_gui.sh                   # 一键启动 GUI 服务（顶层）
├── web/                         # GUI 前端（无构建步骤，原生 ES modules）—— 已完成
├── models/                      # 本项目模型（<模型名>/<工作名>/model.ply，gitignore）
├── data/                        # 输入资产（gitignore）
│   └── FLAME2020/               # generic_model.pkl / flame_uv.npz / eyelid
├── output/                      # 产物（gitignore）
│   ├── equivalence/             #   equivalence_check.py 的输出
│   └── render_test/             #   render_test.py 的输出 + report.json
├── requirements.txt
└── README.md
```

### 4.1 vendored 源码的统一处置

**所有外部源码一律放在 `submodules/`，并从 `submodules/` 构建。**
不再区分 `ext/` 与 `third_party/`，也不再从 `~/Libraries/` 构建。

| 来源 | 处置 | 理由 |
|---|---|---|
| `submodules/diff-gaussian-rasterization` | vendor | 本项目唯一的 CUDA 内核，必须能独立编译、固定版本、按需修改 |
| `~/Libraries/nvdiffrast` | vendor 到 `submodules/nvdiffrast/` | 见 §4.2 |
| `~/Libraries/fused-ssim` | vendor 到 `submodules/fused-ssim/` | 见 §4.2 |
| `submodules/flame`, `submodules/fuhead` | vendor（待拷入） | 纯 Python，LBS 数学正确，不重写；需明确归属与许可 |
| `model/`, `diff_renderer/`, `camera/`, `dataset/` | **重写为 `src/`** | 这是本项目的主要工作量 |
| `train_*.py`, `render*.py`, `utils.py` | **重写为 `app/` + `training/`** | RGBAvatar 的脚本层与本项目形态差异最大 |

`submodules/ATTRIBUTION.md` 登记来源 commit、原许可、本项目的删减与修改点。

### 4.2 为什么 `nvdiffrast` / `fused-ssim` 也要 vendor

这两个库原本放在 `~/Libraries/`（源码 + 手工构建）。**直接把 `~/Libraries/...` 交给 `pip install` 会失败**：

```
error: could not delete 'build/lib.linux-x86_64-cpython-310/nvdiffrast/__init__.py': Permission denied
```

原因不是权限属主（文件属主就是当前用户），而是这两个源码目录里残留了此前手工构建的 `build/` 目录，而构建工具需要在源码树内删除/重建它。

**结论**：`~/Libraries/` 只作为「上游源码的获取来源」，**不作为构建源**。全部 vendor 到 `submodules/` 后：构建可复现（源码随仓库固定）、不依赖工作区外路径的可写性、与 CUDA/torch 版本解耦（换环境重新编译即可）。

### 4.3 数据集布局（当前仅 `duda`）

```
/home/crh/Datasets/INSTA/duda/          # 外部路径，不入库
├── images/       254 张 PNG             ← RGBAvatar 读取
├── checkpoint/   254 个 .frame          ← RGBAvatar 读取（FLAME 参数 + 相机）
├── flame/exp/                           ← INSTA 中间产物（当前不用）
├── meshes/ depth/ matted/ seg_mask/     ← INSTA/tracker 中间产物（当前不用）
└── transforms*.json  canonical.obj
```

数据集只需 `images/` 与 `checkpoint/` 两项即可驱动 RGBAvatar 的 `FLAMEDataset`；其余为 tracker/INSTA 的中间产物，保留但本项目不读取。

> **数据集范围限制**：当前**只以 `duda` 为基准**。所有验收均以 duda 为准。新增数据集前需先确认其 tracker 链路可复现。

---

## 5. 与 RGBAvatar 的对应关系

参照实现是**只读**的算法蓝本。本项目的重写范围与所有有意偏离，
**完整登记在 `docs/MIGRATION.md`**（缺陷修复 K、有意偏离 H、持续关注 O、已排除的错误推理 D）。

简述：`model/` `diff_renderer/` `camera/` `dataset/` 与脚本层**全部重写**；
`submodules/` 下的 CUDA 扩展与 FLAME/FuHead 实现**vendor 后原样使用**。

---

## 6. 训练管线 —— **待接入**

> **本项目的训练部分不自研、也不复刻 RGBAvatar。**
> 预训练由另一位同学的工作提供，未来以清晰接口接入。

因此本节只定义**接入点**，不描述内部实现：

| 接入点 | 契约 | 状态 |
|---|---|---|
| 模型产物 | `.ply`（3DGS 属性 + `xyz_b_*` / `rot_b_*` / `f_dc_b_*` 基 + `face_id` / `face_bary_*` 绑定） | ✅ 本项目写出的文件另含 `comment gaussian_config` 自描述；**参照写出的没有**，结构需由 `configs/render.yaml` 提供 |
| 模型读取 | `core/io/ply.py::load_ply` → `GaussianAvatar` | ✅ 可用（当前消费参照仓库产出的模型） |
| 驱动参数 | `[B, D]` 张量，`D = model.network.num_basis_in` | ✅ 已定义 |
| 渲染接口 | `AvatarRuntime`（见 §7.1） | ⏳ 待实现 |
| 训练产物落盘 | 同上 `.ply` 约定 | ⏳ 待对方确认 |

**当前阶段**：直接使用数据集与**已有模型**即可，不需要训练。
`src/live3dgsavatar/training/` 目录保留为空占位，接入时再填充。

---

## 7. 渲染与实时性

### 7.1 推理路径（本阶段重点）

```
Frame(mesh, blend_weight, camera)
  └─ blend → bind → SimpleRasterizer → RenderOutput
```

推理时 `blend_weight` 来自 `tracking/`。本阶段 `tracking/offline_flame.py` 复用数据集中已拟合的 FLAME 参数（与训练同源），**不引入新的 tracker 依赖**——这保证第一阶段可以完全复现 RGBAvatar 的渲染结果。

### 7.2 实时性的三个来源

1. **不做梯度**：推理路径只走 forward，不分配 `grad_*` 缓冲。
2. **不做逐帧重建**：模型参数固定，每帧只做 `blend + bind + rasterize`。RGBAvatar 实测约 400 FPS（RTX 3090，含动画），这是本架构的性能天花板参考。
3. **零拷贝**：`GaussianSet` 的 5 个张量直接从 `GaussianAvatar` 的参数视图构造，避免 `expand/cat` 产生的隐式拷贝。

### 7.3 服务端渲染（预留接口）

`runtime/session.py` 定义：

```python
class Session:
    def __init__(self, avatar: GaussianAvatar, runtime: AvatarRuntime): ...
    def push_drive_params(self, params: torch.Tensor) -> None: ...  # 接收驱动流
    def render_frame(self, view: Camera) -> RenderOutput: ...       # 按观看者视角渲染
    def push_observation(self, frame: Frame) -> None: ...           # 供在线更新
```

`streaming/` 只留空模块与 TODO，**不实现**。具体传输方案（WebRTC / 编码器选型）留待后续决策。

---

## 8. 环境与复现

> **权威来源**：`docs/ENVIRONMENT.md`。本节只保留架构相关的要点——完整的搭建步骤、
> 双 CUDA 工具链说明、排错表、验证记录都在那里。

### 8.1 固定版本矩阵

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | 3.10.21 | CUDA 扩展编译产物为 `cpython-310` |
| PyTorch | **2.14.0+cu130** | 实测可用，不再改动 |
| CUDA Toolkit | **13.0** | 环境内 nvcc 13.0.48 / 系统 13.0.88，二者皆可用 |
| torchvision | 0.29.0+cu130 | |
| nvdiffrast | 0.4.0 | vendored 到 `submodules/nvdiffrast/` |
| fused-ssim | 1.0.0 | vendored 到 `submodules/fused-ssim/` |
| GPU 架构 | `sm_86` | `TORCH_CUDA_ARCH_LIST=8.6` |
| 环境名 | **`live3dgs`** | 与参照仓库的 `rgbavatar` 隔离（约束 C8） |

> **重要**：CUDA 13.0 是本项目基线。RGBAvatar 官方 README 建议 11.8，但在本环境下 11.8 反而不可用。

### 8.2 三条构建规则

1. **一切从 `submodules/` 构建，不从 `~/Libraries/` 构建。** 后者残留手工构建的 `build/` 目录，
   会导致 `pip install` 在源码树内删除失败；vendor 到工作区后正常。理由见 §4.2。
2. **所有本地扩展用 `--no-build-isolation`**，且**必须先把 `setuptools/wheel/ninja` 装进环境**——
   否则 pip 不会自动准备构建依赖，报 `Failed to build installable wheels for ...`。
3. **扩展用 `pip install -e`（editable）**：`.so` 落在源码目录内，便于用 `git status` 发现产物与源码不同步。

一键脚本 `setup_env.sh` 已把上述规则固化，幂等可重跑。

### 8.3 数据与模型资产（当前仅 `duda`）

| 资产 | 路径 | 状态 |
|---|---|---|
| FLAME2020 | `data/FLAME2020/generic_model.pkl` | ✅ 已有（gitignore） |
| flame_uv | `data/FLAME2020/flame_uv.npz` | ✅ 已有 |
| eyelid | `data/FLAME2020/{l,r}_eyelid.npy` | ✅ 已有 |
| 预训练模型（参照） | `RGBAvatar/output/duda/test/model.ply` | ✅ 只读参照 |
| 数据集 | `/home/crh/Datasets/INSTA/duda`（254 帧） | ✅ 已有 |
| 渲染测试产物 | `output/render_test/`（由 `scripts/render_test.py` 生成） | ✅ |

> **`data/checkpoints` 不需要**：RGBAvatar 的数据集契约只有"多个 `<模型名>/<工作名>` 下的 `checkpoint/` + `images/`"，
> 而这两者都在数据集目录内（`<DATA_DIR>/<SUBJECT>/{checkpoint,images}`）。本项目训练产出的
> `model.ply` 归属于**训练输出**，参照放在 `output/<模型名>/<工作名>/`；
> 本项目照此约定放在 `models/<模型名>/<工作名>/`，不新建 `data/checkpoints/`。

> **数据集范围限制**：本阶段只以 `duda` 为基准，所有质量验收均以 duda 为准（约束 C7）。
> `NeRSemble` 相关代码路径本阶段不引入。

---

## 9. 当前状态与验收

### 9.1 已完成

| 模块 | 验收标准 | 证据 |
|---|---|---|
| **环境与基线** | 单条命令建环境；`scripts/env_check.py` 全绿 | `setup_env.sh` 幂等可重跑 |
| **`core/` 渲染内核** | **数值等价门**：分层比对参照实现，中间属性 `max\|Δ\| < 1e-5`，渲染图 **`PSNR > 60 dB` 或 `max\|Δ\| < 1e-3`** | **31 项全部通过**（见 9.2） |
| **GUI（前后端）** | 读取已有模型并实时预览 | 协议见 `docs/GUI_PROTOCOL.md` |

### 9.2 等价门结果

`python scripts/equivalence_check.py`，**31 项全部通过**：

| 层 | 结果 |
|---|---|
| 参数（10 项） | 逐位一致（`max\|Δ\| = 0`）：8 组参数 + `weight_module` + 通道数 |
| 绑定构建（5 项） | 逐位一致：`valid_mask` / `face_id` / `face_bary` / 高斯数 / 与参照缓存一致 |
| 混合（6 项） | `xyz` 1.5e-08、`rotation` 7.9e-07、`color` 1.4e-06，均远低于 1e-5 |
| 绑定（5 项） | `xyz` **2.98e-08**、`rotation` 6.6e-07，其余逐位一致 |
| 渲染（4 项） | `color` **PSNR 133.47 dB**、`alpha` 135.48 dB |

### 9.2.1 逐帧渲染复现

`python scripts/render_test.py --frames -1`：

| 指标 | 结果 |
|---|---|
| 可比帧数 | **254 / 254** |
| 渲染图 PSNR | **中位 101.07 dB**，最小 86.63 dB |
| `max\|Δ\|` | 中位 **1 / 255**（uint8 最后一位） |
| 峰值显存 | 106 MiB（参照 batch=1 为 145 MiB） |

### 9.2.2 测试

| 命令 | 覆盖 |
|---|---|
| `python tests/run_tests.py` | **119 通过 / 0 失败 / 1 跳过**（跳过项为需 GPU 的等价测试） |
| `node web/tests/all.mjs` | **36 通过**（协议、帧解析、控件装配、id 一致性） |

### 9.2.3 性能

`core/` 走**纯 PyTorch**（`linear_blending` 默认不用 CUDA 内核），
batch=1 单帧约 **5.3 ms**（deform 2.9 + rasterize 2.5），**约 190 FPS**；
参照走 CUDA 内核约 2.4 ms。

这是**有意识的取舍**：CUDA 版 `linear_blending` 在设备不可用时**静默返回全零**（见 `docs/MIGRATION.md` H4）。
GUI 的目标帧率是 30–60 FPS，**余量充足**；需要再提速时见 §11.2。

### 9.3 分层规则的可执行化

`docs/ARCHITECTURE.md` §2.3 声明的依赖规则由 `tests/unit/test_architecture.py` 强制检查：

| 检查 | 内容 |
|---|---|
| 分层单向 | `core/` 不得 import `data/` / `training/` / `app/` / `tracking/` / `runtime/` / `streaming/` |
| CUDA 隔离 | `diff_gaussian_rasterization` 只允许在 `core/render/rasterizer.py` 与 `core/deform/blend.py` 出现 |
| 无 I/O | `core/` 不得 import `argparse` / `sys`；`plyfile` / `json` 仅限 `core/io/ply.py` |
| 导入期无设备访问 | `core/` 模块作用域不得调用 `torch.cuda.*`（否则无 GPU 环境下导入即失败） |
| 配置来源 | 脚本参数无业务默认量、代码无硬编码路径（`tests/unit/test_config.py`） |
| 无未定义名字 | 全项目静态检查（AST 模块绑定 + `symtable` 词法/闭包解析），覆盖只在 GPU 上跑的分支 |
| 兼容补丁顺序 | 加载参照实现的模块必须在模块层 `import live3dgsavatar`（否则 chumpy ImportError，见 `ENVIRONMENT.md` §3.10） |
| FLAME dtype | 不得覆盖 `FlameConfig.dtype`（float64 是刻意的，见 `ENVIRONMENT.md` §3.11） |

**「导入 `core` 不需要 GPU」** 这一性质尤其重要：CPU 侧的 119 项测试才得以成立。

### 9.4 等价门的执行结构

等价门分**六步**执行，其中比对分五层，任一层失败都能直接定位，
而不是只看到「图不一样」：

| 步骤 | 内容 | 判据 |
|---|---|---|
| 0 前置检查 | 路径、数据集子目录、FLAME 模型是否就位（**不需 GPU**） | 全部存在 |
| 1 参数 | PLY 读入的 8 组参数 + `weight_module` | 逐位一致（`max\|Δ\| = 0`） |
| 2 绑定构建 | UV 域光栅化：`valid_mask` / `face_id` / `face_bary` | 精确 / `1e-5` |
| 3 混合 | `project_weight` + blendshape 线性混合（激活前后） | `max\|Δ\| < 1e-5` |
| 4 绑定 | `mesh_binding` 输出（xyz / rotation / 颜色 / 不透明度 / 尺度） | `max\|Δ\| < 1e-5` |
| 5 渲染 | 光栅化输出 `color` / `alpha` | `PSNR > 60 dB` 或 `max\|Δ\| < 1e-3` |

**第 2 层（绑定构建）单独成层很重要**：UV 光栅化决定了「哪些 texel 变成高斯、每个高斯绑到哪个面」，
若它不一致，后面所有几何比对都必然失败。缺少这一层就只能看到「xyz 不一样」，
无法区分是光栅化错了还是 `mesh_binding` 错了。

比对逻辑集中在 `tests/equivalence/stages.py`，**脚本与测试共用同一份**，避免判据漂移。

---

## 10. 避免继承 RGBAvatar 的缺陷

完整登记见 **`docs/MIGRATION.md`**（A 节：4 项缺陷修复 K1–K4；B 节：6 项有意偏离 H1–H6；
C 节：5 项持续关注 O1–O5）。核心几条：

| 参照缺陷 | 本项目处置 |
|---|---|
| `save_ply()` 引用不存在的属性 → 保存必失败 | 绑定信息归属 `GaussianAvatar`（`register_buffer`） |
| `load_ply()` 从配置读基数量而非文件列数 → 静默读错 | 加载时与**文件列数**交叉校验（K / D / `tex_size²` 容量），不一致即报错 |
| `clone()` 签名不匹配 → 死代码 | 不移植；快照用 `state_dict()` |
| 三份重复的 `render_gs_batch`（含一份 `# legacy`） | 合并为 `Rasterizer` 协议 + 两个后端 |
| 单帧/批量两套张量形状并存 | 固定 batch-first，`GaussianSet.space` 显式标注空间 |
| `linear_blending` 在设备不可用时**静默返回全零** | 默认走纯 PyTorch，CUDA 路径显式 opt-in + 两道防护 |
| CUDA/torch 版本强耦合 | `env_check.py` + `docs/ENVIRONMENT.md` 版本矩阵 |

---

## 11. 后续方向

当前**链路已打通**：加载已有模型 → 渲染 → 推流 → 浏览器实时预览。

### 11.1 动态更新（唯一在册的目标）

让模型在通话过程中**持续适应**新到达的帧，而不是只消费训练好的静态模型。

- **现状**：驱动参数来自数据集里已有的采样（`source.kind = "checkpoint"`）
- **目标**：接入实时采集的驱动参数，并让模型随之在线微调
- **协议已留钩子**：`source.kind = "live"`（前端置灰，后端忽略该分支的字段）
- **未定项**：在线更新的触发策略、抗遗忘采样、与预训练（合作方提供）的接口边界

> 这一部分**路线完全未定**，需要先做深入构思与设计，不宜现在排期。

### 11.2 已评估、暂不做

| 项 | 何时再考虑 |
|---|---|
| 接入 `mesh_binding` CUDA 内核（deform 3.4 → ~0.3 ms） | 需要更高分辨率、多路流，或在线更新需要梯度路径时。内核已 vendored，且**自带 backward** |
| 用官方 3DGS 光栅化器替换改造版 | 纯依赖整洁性收益，无性能收益；且验证链条仍需改造版（见 `docs/MIGRATION.md`） |
| 上半身/肩颈扩展 | 需引入混合拓扑模板与蒙皮权重迁移 |
| 光照解耦 | 当前颜色为 SH DC（烘焙），通话中光照变化无法适应 |
| 服务端并发（单卡多路） | 多会话调度与资源隔离 |

## 附录 B：术语表

| 术语 | 含义 |
|---|---|
| 切空间 (tangent space) | 以绑定三角面的 TBN 为基的局部坐标系；高斯参数默认存于此 |
| 世界空间 (world space) | 变形后的全局坐标系；光栅化器在此空间工作 |
| 约简基 (reduced blendshapes) | MLP 把 D 维 3DMM 参数压成 K=20 维权重，混合 K 组高斯基 |
| fast forward | 用渲染统计量（`est_color/est_weight`）一次性写入高斯颜色的初始化策略 |
| local-global 池 | 在线训练中"新帧池 + 历史帧池"的双池采样策略，用于抗遗忘 |
| TBN | tangent / bitangent / normal，逐三角面的切空间正交基 |
