# Live3DGSAvatar 架构设计文档

> 视频通话场景 3DGS 人体重建 —— 第一阶段（头部）架构设计
> 状态：**待审核**　版本：v0.1　最后更新：待定

---

## 0. 本文档的边界

配套文档：`docs/CONFIG.md`（配置项）、`docs/CONVENTIONS.md`（数值约定）、
`docs/CORE_GUIDE.md`（代码导览）、`docs/CONFIG.md`（配置）、
`docs/ENVIRONMENT.md`（环境）、`docs/ONLINE_UPDATE.md`（热更新）。

**目标**：定义一个**足够简单、可落地、可扩展**的工程架构，支撑"训练好的头像 → 实时渲染 → 推流预览 → 在线上更新"这条链路。

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
| C2 | 训练与推理技术栈：PyTorch + CUDA 光栅化扩展 | 已实测可用：torch `2.14.0+cu130` |
| C3 | 渲染部署形态：**服务端渲染，推流给用户端** | 用户决策（实验室算力充足） |
| C4 | 覆盖范围：**仅头部** | 当前网格拓扑所限；上半身为未来工作 |
| C5 | 用途：现阶段纯学术/自用 | 商用未定；架构上不封死替换 3DMM/tracker 的路径即可 |
| C6 | 本项目**自包含**：数据、模型、FLAME 资产都在仓库内 | 不再依赖任何外部仓库 |
| C7 | 数据集采用**自包含清单**（`render_inputs.npz`） | 逐帧网格/驱动参数/相机都已算好，渲染不需要 3DMM 参数 |
| C8 | 独立 conda 环境 `live3dgs` | 不与其他项目共用，避免依赖冲突 |

### 1.3 本机开发环境

| 项 | 值 |
|---|---|
| Conda 环境 | **`live3dgs`**（python 3.10） |
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
│  compat/       第三方库的兼容补丁（numpy 别名等）              │
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
  `src/` **不得** import `tests/`（`tests/` 不是包，只能靠 `sys.path` 找到）。
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
│   └── ONLINE_UPDATE.md         ← 在线更新（热更新）设计
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
│   │   ├── flame_topology.py    #   FLAME 网格拓扑（只读资产，带缓存）
│   │   ├── flame_model.py       #   FLAME 前向（checkpoint 格式需要）
│   │   └── checkpoint_loader.py  #   参照格式数据集（每帧 .frame）
│   ├── training/                # 在线更新（observations/losses/color_fit/updater）
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
├── tests/
│   ├── run_tests.py             # 零依赖测试运行器（未装 pytest 也能跑）
│   ├── support.py               # 测试共用小工具
│   └── unit/                    # 单测（133 项，全部无需 GPU）
├── scripts/
│   ├── env_check.py             # 环境自检（无 GPU 可跑）
│   ├── render_test.py           # 渲染测试：core vs 数据集原图（质量 + 性能）
│   ├── online_update.py         # 在线更新（闭式颜色校正 + 梯度微调）
│   ├── profile_deform.py        # deform 阶段耗时分解
│   ├── show_config.py           # 查看生效配置
│   └── _compat_shim.py          # 独立脚本用的精简兼容补丁
├── pyproject.toml               # 打包配置（src 布局 + console script）
├── setup_env.sh                 # 一键建环境（顶层，用户直接跑）
├── run_gui.sh                   # 一键启动 GUI 服务（顶层）
├── web/                         # GUI 前端（无构建步骤，原生 ES modules）—— 已完成
├── models/                      # 本项目模型（<模型名>/<工作名>/model.ply，gitignore）
├── data/                        # 输入资产（gitignore）
│   └── FLAME2020/               # generic_model.pkl / flame_uv.npz / eyelid
├── output/                      # 产物（gitignore）
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
| `train_*.py`, `render*.py`, `utils.py` | 对应到 `app/` + `training/` | 脚本层与库层分离 |

`submodules/ATTRIBUTION.md` 登记来源 commit、原许可、本项目的删减与修改点。

### 4.2 为什么 `nvdiffrast` / `fused-ssim` 也要 vendor

这两个库原本放在 `~/Libraries/`（源码 + 手工构建）。**直接把 `~/Libraries/...` 交给 `pip install` 会失败**：

```
error: could not delete 'build/lib.linux-x86_64-cpython-310/nvdiffrast/__init__.py': Permission denied
```

原因不是权限属主（文件属主就是当前用户），而是这两个源码目录里残留了此前手工构建的 `build/` 目录，而构建工具需要在源码树内删除/重建它。

**结论**：`~/Libraries/` 只作为「上游源码的获取来源」，**不作为构建源**。全部 vendor 到 `submodules/` 后：构建可复现（源码随仓库固定）、不依赖工作区外路径的可写性、与 CUDA/torch 版本解耦（换环境重新编译即可）。

### 4.3 数据集布局

本项目使用**自包含清单**格式，一个目录即可驱动全部功能：

```
<DATA_DIR>/<subject>/
├── render_inputs.npz    mesh [N,5083,3] / blend_weight [N,129] / K / w2v /
│                        width / height / names
├── images/              512×512 RGBA（PSNR 的 GT）
└── input/               未抠图的原始帧（可选）
```

**为什么优先这种格式**：逐帧的网格顶点与驱动参数在**预处理阶段**就算好了，
所以渲染时不需要 3DMM 参数、不需要 FLAME 前向、也不需要 tracker。

也支持**参照格式**（`images/` + `checkpoint/`，每帧 3DMM 参数）：
此时由本项目自带的 `data/flame_model.py`（约 250 行）跑一次 FLAME 前向，
把参数变成网格顶点。两种格式产出**完全相同**的 `Scene`，下游无差别。

渲染只需要**面表**（拓扑），它从本项目的 FLAME 资产里读一次并缓存。

---

## 5. 与起点项目的关系

本项目的渲染管线**起步于** [RGBAvatar](https://github.com/gapszju/RGBAvatar)
（CVPR 2025 Highlight）的约简高斯 blendshape 方案，但从"能跑"之后就**独立演进**：

| 阶段 | 关系 |
|---|---|
| 起步 | 参考其渲染管线的数学（约简 blendshape + 网格绑定） |
| 之后 | **自包含**：数据格式、FLAME 拓扑读取、光栅化调用、训练/更新都归本项目 |

具体地，本项目**不再**读取任何外部仓库：

| 曾经依赖 | 现状 |
|---|---|
| 早期的 `FLAMEDataset` 式加载（读 `checkpoint/`） | 改为自包含清单 `render_inputs.npz` |
| 早期的 `FLAME` 封装（925 行 + chumpy） | 改为直接读 pickle（`data/flame_topology.py`，约 240 行） |
| 曾从外部仓库回退找模型 | 只从 `models/<subject>/<work_name>/` 取 |
| 数值等价门（逐层比对旧实现） | **已移除**：渲染已稳定，不再需要逐层对照 |

> `submodules/` 下的 CUDA 扩展仍是 vendored 副本（保证版本可复现），
> 它们的归属与许可见 `submodules/ATTRIBUTION.md`。

## 6. 训练管线 —— **待接入**

> **本项目的训练部分不自研、也不复刻任何外部实现。**
> 预训练由另一位同学的工作提供，未来以清晰接口接入。

因此本节只定义**接入点**，不描述内部实现：

| 接入点 | 契约 | 状态 |
|---|---|---|
| 模型产物 | `.ply`（3DGS 属性 + `xyz_b_*` / `rot_b_*` / `f_dc_b_*` 基 + `face_id` / `face_bary_*` 绑定） | ✅ 文件另含 `comment gaussian_config` 自描述 |
| 模型读取 | `core/io/ply.py::load_ply` → `GaussianAvatar` | ✅ 可用 |
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

推理时 `blend_weight` 来自数据集清单（与训练同源）。**不引入新的 tracker 依赖**。

### 7.2 实时性的三个来源

1. **不做梯度**：推理路径只走 forward，不分配 `grad_*` 缓冲。
2. **不做逐帧重建**：模型参数固定，每帧只做 `blend + bind + rasterize`。
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
| 环境名 | **`live3dgs`** | 独立环境（约束 C8） |

> **重要**：CUDA 13.0 是本项目基线，与当前 PyTorch 构建匹配。

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
| 预训练模型 | `models/<subject>/<work_name>/model.ply` | ✅ 本项目 models/ |
| 数据集 | `<DATA_DIR>/<subject>/{render_inputs.npz, images/}` | ✅ 已有 |
| 渲染测试产物 | `output/render_test/`（由 `scripts/render_test.py` 生成） | ✅ |

> **资产归属**：数据集在仓库外（`<DATA_DIR>/<subject>/`），模型在本项目
> `models/<subject>/<work_name>/`，FLAME 资产在 `data/FLAME2020/`。
> 三者各自独立，没有隐式的目录耦合。

---

## 9. 当前状态与验收

### 9.1 已完成

| 模块 | 验收标准 | 证据 |
|---|---|---|
| **环境与基线** | 单条命令建环境；`scripts/env_check.py` 全绿 | `setup_env.sh` 幂等可重跑 |
| **`core/` 渲染内核** | 与数据集原图的 **PSNR（前景掩码内）** | 见 9.2 |
| **GUI（前后端）** | 读取模型并实时预览 | 见 `docs/GUI_PROTOCOL.md` |
| **在线更新** | 修复前后 PSNR 提升 | `crh_1` 8 帧：**31.37 → 32.52 dB（+1.15）**，182 步 / 10.2 s；约 50 步即收敛（见 `docs/ONLINE_UPDATE.md` §4） |

### 9.2 渲染质量与性能

`python scripts/render_test.py`（`crh_1`，512²）：

| 指标 | 数值 |
|---|---|
| PSNR（全图） | 中位 **25.02 dB** |
| **PSNR（前景掩码内）** | 中位 **30.71 dB** |
| 单帧耗时 | 中位 **4.60 ms** |
| 吞吐 | **175 FPS** |
| 峰值显存 | 153 MiB |

**性能取舍**：`core/` 走纯 PyTorch（`linear_blending` 不用 CUDA 内核），
约 190 FPS；目标是 30–60 FPS，**余量充足**。需要再提速时的选项见 §11.2。

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
| 兼容补丁顺序 | 需要 pickle/chumpy 的脚本应在模块层 `import live3dgsavatar`（补丁在导入期施加） |
| FLAME dtype | 不得覆盖 `FlameConfig.dtype`（float64 是刻意的，见 `ENVIRONMENT.md` §3.11） |

**「导入 `core` 不需要 GPU」** 这一性质尤其重要：CPU 侧的 119 项测试才得以成立。

## 10. 早期踩过并已内化的几点

起步阶段对照参照实现时发现的问题，现已成为本项目代码/测试的一部分（不需要再对照）：

| 问题 | 本项目处置 |
|---|---|
| `save_ply` 引用不存在的属性 → 保存必失败 | 写出/读取由 `core/io/ply.py` 统一，有往返测试 |
| `load_ply` 从配置读基数量而非文件列数 → 静默读错 | 加载时与**文件列数**交叉校验，不一致即报错 |
| 设备不可用时 CUDA 内核**静默返回全零** | 默认走纯 PyTorch；CUDA 路径显式 opt-in 并加设备检查 |
| 三份重复的渲染函数（含一份 `# legacy`） | 收敛到 `core/render/rasterizer.py` 单一入口 |
| 转置散落在各调用点 | 收敛到 `camera_utils.to_kernel_matrix` 单点 |
| 切空间 / 世界空间靠调用顺序隐含区分 | `GaussianSet.space` 显式标注 + 断言 |

## 11. 后续方向

当前**链路已打通**：加载已有模型 → 渲染 → 推流 → 浏览器实时预览。

### 11.1 在线更新（已实现基础能力）

让模型在通话过程中**持续适应**新到达的帧，而不是只消费训练好的静态模型。

- **现状**：驱动参数来自数据集里已有的采样（`source.kind = "checkpoint"`）
- **目标**：接入实时采集的驱动参数，并让模型随之在线微调
- **协议已留钩子**：`source.kind = "live"`（前端置灰，后端忽略该分支的字段）
- **已实现**：`training/` 提供外观微调（闭式颜色校正 + 梯度微调），
  详见 [`docs/ONLINE_UPDATE.md`](ONLINE_UPDATE.md)
- **边界**：只改颜色/不透明度，不碰几何、融合基与 MLP
- **未定项**：触发策略、抗遗忘采样、与预训练（合作方提供）的接口边界

### 11.2 已评估、暂不做

| 项 | 何时再考虑 |
|---|---|
| 接入 `mesh_binding` CUDA 内核（deform 3.4 → ~0.3 ms） | 需要更高分辨率、多路流，或在线更新需要梯度路径时。内核已 vendored，且**自带 backward** |
| 用官方 3DGS 光栅化器替换改造版 | 纯依赖整洁性收益，无性能收益 |
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
