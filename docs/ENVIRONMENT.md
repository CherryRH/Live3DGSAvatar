# 环境搭建与复现

> 目标：一条命令得到一个可复现的开发环境。
> **环境配置成功与否，以你执行 `setup_env.sh` 的结果为准。**

---

## 1. 版本矩阵（本项目唯一权威来源）

| 组件 | 版本 | 说明 |
|---|---|---|
| 操作系统 | WSL2 (kernel 6.6.87.2-microsoft-standard) | GPU 直通依赖 `/dev/dxg` + `/usr/lib/wsl/lib/libcuda.so` |
| Python | 3.10 | CUDA 扩展编译产物为 `cpython-310` |
| CUDA Toolkit | **13.0** | 见 §3.4 的双工具链说明 |
| PyTorch | **2.14.0+cu130** | 实测可用，不再改动 |
| numpy | **2.2.6**（不 pin） | 由其他依赖自然解析；chumpy 兼容性由 `src/live3dgsavatar/compat/` 修补，见 §3.6 |
| torchvision | 0.29.0+cu130 | |
| nvdiffrast | 0.4.0 | 从 `submodules/nvdiffrast/` 构建 |
| fused-ssim | 1.0.0 | 从 `submodules/fused-ssim/` 构建；仅多视角训练需要 |
| diff-gaussian-rasterization | 0.0.0 | 从 `submodules/diff-gaussian-rasterization/` 构建 |
| GPU 架构 | `sm_86`（RTX 3060 Laptop） | `TORCH_CUDA_ARCH_LIST=8.6` |
| 环境名 | `live3dgs` | |

> **重要**：CUDA 13.0 是本项目的基线（与当前 PyTorch 构建匹配）。以本表为准。

---

## 2. 快速开始

```bash
cd ~/Projects/Live3DGSAvatar

# 1) 建环境（推荐：建在 conda 默认位置）
bash setup_env.sh
conda activate live3dgs

#    或：建在工作区内（无 ~/miniconda3/envs 写权限时）
ENV_PREFIX=$PWD/.conda/envs/live3dgs bash setup_env.sh
conda activate $PWD/.conda/envs/live3dgs

# 2) 环境自检（可单独重跑，无需 GPU）
python scripts/env_check.py

# 3) GPU 冒烟测试 + 基线性能采集
python scripts/render_test.py --frames 20
```

`setup_env.sh` 是幂等的：重复执行会跳过已完成步骤；`FORCE=1` 可强制重建。

可调变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `ENV_NAME` / `ENV_PREFIX` | `live3dgs` / 空 | 环境位置 |
| `TORCH_CUDA_ARCH_LIST` | `8.6` | 编译目标架构 |
| `MAX_JOBS` / `NVCC_THREADS` | `2` / `2` | 编译并发；WSL 内存不足时下调 |
| `CUDA_CHANNEL` | `nvidia/label/cuda-13.0.0` | CUDA Toolkit 来源 |

> **不需要联网 clone 任何源码**：`nvdiffrast`、`fused-ssim`、光栅化扩展全部已 vendor 到 `submodules/`。

---

## 3. 手工步骤（`setup_env.sh` 的等价展开）

**顺序不可调换。**

### 3.1 创建环境

```bash
conda create -n live3dgs python=3.10 -y
conda activate live3dgs
```

### 3.2 安装 CUDA Toolkit

```bash
conda install -c nvidia/label/cuda-13.0.0 cuda-toolkit -y
```

### 3.3 安装 PyTorch

```bash
pip install torch torchvision
```

> **不要**加 `--index-url`。pip 默认源已能解析出 `+cu130` 构建（实测得到 `2.14.0+cu130`）。指定旧版本 cu 源反而会引入不匹配。

### 3.4 关于双 CUDA 工具链

本环境下同时存在两套 CUDA 13.0：

| 位置 | 版本 | 来源 | 何时被用到 |
|---|---|---|---|
| `$CONDA_PREFIX/bin/nvcc` | 13.0.**48** | §3.2 的 conda 安装 | **优先**：conda 环境激活后 PATH 首位即此 |
| `/usr/local/cuda-13.0/bin/nvcc` | 13.0.**88** | 系统安装 | 回退：未装 conda cuda-toolkit 时 |

两者都是完整的 CUDA 13.0（含 CCCL/Thrust 头文件），**都可以用**。差异仅在补丁级，对 ABI 无影响。
`scripts/env_check.py` 会校验「nvcc ↔ torch 头文件版本一致」，只要都是 13.0 即通过。

### 3.5 补全构建工具与 Python 依赖

```bash
pip install --upgrade pip setuptools wheel ninja
pip install chumpy --no-build-isolation      # ← 必须单独装，见 §3.8
pip install -r requirements.txt
```

> ⚠️ **两个顺序陷阱，都会导致 `pip install -r` 失败**：
>
> **(a) 构建工具必须先装。** 下面三个本地扩展与 `chumpy` 都用 `--no-build-isolation` 安装，此时 pip **不会**自动准备构建依赖。缺了会报
> `Failed to build installable wheels for some pyproject.toml based projects`。
>
> **(b) `chumpy` 不能列进 `requirements.txt`。** 它缺少 PEP 517 元数据，在 `-r` 的依赖解析阶段会报
> `ERROR: Failed to build 'chumpy' when getting requirements to build wheel`。
> 用 `-r` 安装 `chumpy` 时 pip 会走 `get_requires_for_build_wheel` 路径，而直接指定包名时只需构建 wheel 本身，前者在当前 setuptools 下必然失败。

### 3.6 关于 numpy：不要 pin 旧版本

**`requirements.txt` 中刻意不含 `numpy`。** 这是实测得出的结论，不是疏漏。

**表面现象**：`pip install -r requirements.txt` 报
`ERROR: Failed to build 'chumpy' when getting requirements to build wheel`。

**真实原因**（逐层定位）：

| # | 发现 | 证据 |
|---|---|---|
| 1 | 报错并不是 chumpy 造成的 | 失败点在解析 `numpy==1.23.5`：`ERROR: Failed to build 'numpy' when getting requirements to build wheel` |
| 2 | numpy 1.23.5 的 wheel 拿不到了 | `pip download "numpy==1.23.5" --only-binary=:all:` → `No matching distribution found`。官方与镜像的 simple 索引里该文件仍在（56 个文件、wheel 标签 `cp310-cp310-manylinux_2_17_x86_64` 看着正常），但 pip 不再选用，于是退回源码包 |
| 3 | 源码编译必然失败 | `ModuleNotFoundError: No module named 'distutils.msvccompiler'`（numpy 1.23 的 `numpy.distutils` 在新 setuptools 下已不可用） |
| 4 | 而 numpy 又不能随便升 | chumpy 0.70 在 numpy ≥ 1.24 下**导入即崩**：`ImportError: cannot import name 'int' from 'numpy'` |
| 5 | 于是形成死结 | chumpy 要 numpy < 1.24；`scipy>=1.14` 要 numpy ≥ 1.23.5 → 唯一解恰好是 1.23.5，而它已拿不到 |

**采取的方案**：放开 numpy（由其他依赖自然解析为 2.x），改在 `src/live3dgsavatar/compat/` 里
补齐 numpy 2.x 移除的 7 个别名（`bool/int/float/complex/object/unicode/str`）。
补丁仅在别名缺失时生效，numpy 1.x 下自动无操作。
`live3dgsavatar/__init__.py` 保证它在任何 chumpy 相关导入之前执行。

**实测结果**（Python 3.10.21 + numpy 2.2.6）：

```
[1] 补丁生效: ('int', 'float', 'complex', 'object', 'unicode', 'str')
[2] chumpy 导入 OK | numpy 2.2.6
[3] Ch 运算: 6.0
[4] FLAME 构造+前向 OK -> verts (1, 5083, 3) | faces (10032, 3) | uvs (5178, 2)
```

**为什么选这条方案**：另一条路是把 numpy 钉在 1.23.5（需要绕过镜像直接指定 wheel URL，或改用 conda 装 numpy）。
本项目选择修 chumpy 侧，因为它是**唯一不依赖外部索引状态**的做法，也不会因镜像同步策略变化而再次失效。

**适用范围**：本项目加载 FLAME pickle 的地方已不再 import chumpy
（`data/flame_topology.py` 只读 pickle），因此补丁主要用于兼容其它间接依赖。

**可移除条件**：chumpy 上游发布兼容 numpy 2.x 的版本后，删除该补丁并恢复 numpy pin。

### 3.7 三个本地扩展（全部从 `submodules/` 构建）

```bash
pip install submodules/nvdiffrast --no-build-isolation
pip install submodules/fused-ssim --no-build-isolation

export TORCH_CUDA_ARCH_LIST="8.6"
MAX_JOBS=2 NVCC_THREADS=2 pip install -e submodules/diff-gaussian-rasterization --no-build-isolation
```

说明：

- **不使用 `~/Libraries/`**。那里的源码树残留手工构建的 `build/`，会让 `pip install` 在源码树内删除失败。详见 §7。
- `--no-build-isolation` 是必需的：隔离构建环境看不到当前环境里的 torch。
- 光栅化扩展用 `-e`（editable）：`.so` 落在源码目录内，便于用 `git status` 观察产物与源码是否同步。
- `nvdiffrast` / `fused-ssim` 也可直接 `pip install submodules/<pkg>`（非 editable）。

### 3.8 `chumpy`（FLAME 依赖）

`chumpy` 提供 FLAME `generic_model.pkl` 的反序列化能力。它**必须单独安装**：

```bash
pip install chumpy --no-build-isolation
```

原因见 §3.5 (b)：`chumpy` 缺少 PEP 517 元数据，列进 `requirements.txt` 会让 `pip install -r`
在其依赖解析阶段失败。单独指定包名时 pip 只需构建 wheel 本身，因此可以通过。

它同时需要 numpy 兼容性补丁，见 §3.6。

---

### 3.9 nvdiffrast 没有 CPU 后端

`nvdiffrast.torch.RasterizeGLContext` 在当前版本**只是 `RasterizeCudaContext` 的兼容包装**
（构造时会发出 `DeprecationWarning`）。也就是说：

- UV 域光栅化（`core/deform/binding.py::build_binding`）**必须**有可用 CUDA 设备；
- 不存在「纯 CPU 跑通完整前向」的路径。

因此渲染相关的检查无法在无 GPU 的环境下完成。
可在 CPU 上验证的部分（矩阵、混合、绑定数学、PLY、输入校验）已由
`tests/unit/` 的 30 项测试覆盖，无需 GPU。

---

### 3.10 兼容补丁必须在导入 chumpy **之前**生效

chumpy 0.70 依赖 numpy 已移除的别名（`np.int` / `np.float` / `np.object` …），
故 `src/live3dgsavatar/compat/__init__.py` 在**导入期**补齐这些别名。
补丁由 `live3dgsavatar/__init__.py` 触发（它 `import compat`）。

**因此凡是要 `pickle.load` 的脚本/模块，都必须在「模块层」先
`import live3dgsavatar`。** 一旦顺序反了就会报：

```
ImportError: cannot import name 'int' from 'numpy'
```

`scripts/render_test.py` 与 `src/live3dgsavatar/data/scene.py` 都已在模块层显式触发；
由 `tests/unit/test_architecture.py::test_scripts_trigger_compat_before_chumpy`
静态兜住（该测试已验证：移除那行 import 会立即失败）。

> 另一个坑：`import live3dgsavatar` 必须放在 `sys.path.insert(..., src)` **之后**。

---

### 3.11 FLAME 的 dtype 不得覆盖（float64 是刻意的）

`FLAMEDataset` 把姿态/形状参数存为 **float64**，而 `FLAME` 的 `v_template` / `shapedirs`
跟随 `FlameConfig.dtype`。若把 dtype 改成 float32，会在
`template_vertices + blend_shapes(betas, shapedirs)` 处抛：

```
RuntimeError: expected scalar type Double but found Float
```

**本项目一律不做覆盖**，并由
`tests/unit/test_architecture.py::test_flame_dtype_is_not_overridden` 静态兜住。

---

## 4. 验证

### 4.1 环境自检（无需 GPU）

```bash
conda activate live3dgs
python scripts/env_check.py            # 有警告也返回 0
python scripts/env_check.py --strict   # 有警告即返回 1（CI 用）
```

覆盖 26 项：Python/conda 环境、numpy 兼容补丁、torch/torchvision、nvcc 与 CUDA 头文件
版本一致性、`TORCH_CUDA_ARCH_LIST`、13 个 Python 依赖、chumpy（补丁后验证）、nvdiffrast、
光栅化扩展算子、GPU（非阻塞）。

光栅化扩展的算子清单**直接从 `ext.cpp` 的 pybind 注册表解析**，不手工维护，
因此新增算子后自检会自动覆盖，不会与实现漂移。

`setup_env.sh` 会以 `pip install --no-deps -e .` **可编辑安装本项目自身**，
让 `import live3dgsavatar` 与 `python -m live3dgsavatar.app` 可用。
必须加 `--no-deps`：torch 与三个 CUDA 扩展有自己的安装顺序。

`setup_env.sh` 在第 6 步会先做一次同等的扩展校验，失败即中止，
不会把问题留到第 7 步。

### 4.2 GPU 渲染测试

```bash
conda activate live3dgs
python scripts/show_config.py          # 先确认路径指对了
python scripts/render_test.py --frames 20
```

产出：`output/render_test/`（渲染图 + `report.json`）。
报告含与数据集原图的 PSNR（全图 / 前景掩码内）与性能指标。
路径与参数全部来自 `configs/`，见 `docs/CONFIG.md`。

只想过一遍流程时用 `--frames 3`；没有 GPU 时用 `--dry-run`。

---

## 5. 相对官方/历史流程的简化

| 历史流程 | 本项目流程 | 理由 |
|---|---|---|
| `pip install git+https://gh-proxy.com/.../nvdiffrast.git` | `pip install submodules/nvdiffrast` | 免联网、版本固定 |
| `pip install ~/Libraries/fused-ssim` | `pip install submodules/fused-ssim` | 摆脱工作区外路径依赖 |
| 上游 `requirements.txt` | 本仓库 `requirements.txt`（已补全） | 上游清单缺 `Pillow`、`chumpy`、`scipy` 等实测必需项 |
| `pip install submodules/diff-gaussian-rasterization` | `pip install -e submodules/diff-gaussian-rasterization` | editable 便于追踪编译产物 |
| 无 | `setup_env.sh` + `scripts/env_check.py` + `scripts/render_test.py` | 一条命令可复现 + 可自检 + 可出渲染报告 |
| 环境名 `rgbavatar` | 环境名 `live3dgs` | 独立环境，避免与其它项目互相污染 |

**保留原样的部分**：`TORCH_CUDA_ARCH_LIST=8.6`、`MAX_JOBS/NVCC_THREADS` 限流、`--no-build-isolation`。这些是实测有效的关键设置。

---

## 6. 验证记录

`setup_env.sh` 在第 6、7 步会自行校验扩展算子与运行完整自检，无需手工记录。
渲染与性能报告由 `scripts/render_test.py` 写入 `output/render_test/report.json`。

---

## 7. 已知坑与排错

| 现象 | 根因 | 处置 |
|---|---|---|
| `_CONDA_PYTHON_SYSCONFIGDATA_NAME_USED: unbound variable`（在 `deactivate-gcc_linux-64.sh` 中） | `cuda-toolkit` → `cuda-compiler` → `cuda-nvcc` → `gcc_linux-64`/`gxx_linux-64` 的传递依赖。该 deactivate 钩子在第 130 行直接引用 `_CONDA_PYTHON_SYSCONFIGDATA_NAME_USED`（**该变量只在 activate 时赋值**），一旦调用方开了 `set -u` 就致命退出 | **不要对 `setup_env.sh` 使用 `set -u`**（脚本内部已刻意只用 `set -eo pipefail`）。这是 conda 生态的通行问题，不是本项目代码缺陷 |
| `PermissionError: .../build/.../nvdiffrast/__init__.py: Permission denied` | 源码树内残留此前手工构建的 `build/`，构建工具需要删除/重建它 | 从 `submodules/` 构建，不要用 `~/Libraries/` |
| `Failed to build installable wheels for ... pyproject.toml` | `--no-build-isolation` 下缺 setuptools/wheel | 先 `pip install setuptools wheel ninja` |
| `Failed to build 'chumpy' when getting requirements to build wheel` | **真实原因通常是 numpy**：`chumpy` 缺失 PEP 517 元数据（单独装可绕过），但它只是同批里第一个报出来的；若日志里还有 `Failed to build 'numpy'` / `ModuleNotFoundError: No module named 'distutils.msvccompiler'`，说明是 **numpy pin 到了 1.23.5 而 wheel 已拿不到**，pip 退回源码编译 | ① `chumpy` 单独装、不要进 `requirements.txt`；② **不要 pin numpy**，见 §3.6 |
| `ImportError: cannot import name 'int' from 'numpy'` | chumpy 0.70 与 numpy ≥ 1.24 不兼容（`chumpy/__init__.py:11` 导入已移除的标量别名） | 导入 `live3dgsavatar` 即自动修补（`src/live3dgsavatar/compat/`） |
| `ERROR: Could not find a version that satisfies the requirement numpy==1.23.5` | 官方与镜像 simple 索引中该文件仍存在，但 pip 当前解析策略不再选用（wheel 标签兼容性问题） | 不要 pin numpy，见 §3.6 |
| `ModuleNotFoundError: No module named 'distutils.msvccompiler'` | 旧版 numpy 源码包在新 setuptools 下无法编译 | 不要 pin 旧 numpy |
| `Error 304: OS call failed` / `cuda.is_available()==False` | 当前会话无法访问 GPU 设备（WSL2 下为无法打开 `/dev/dxg`）；容器、远程会话或无 GPU 权限时常见 | 在可访问 GPU 的终端中运行；这不是环境损坏 |
| `PermissionError: ~/.cache/torch_extensions` | 编译缓存目录不可写 | `export TORCH_EXTENSIONS_DIR=<可写路径>` |
| `NoWritableEnvsDirError` | conda 默认 envs 目录不可写 | 用 `ENV_PREFIX=...` 走 `-p`；或改用 `conda create --clone` |
| 编译时内存不足 / 进程被 kill | WSL 内存有限（本机 7 GB），nvcc 并发过高 | `export MAX_JOBS=1 NVCC_THREADS=1` |
| `No CUDA runtime is found, using CUDA_HOME=...` | torch 探测不到 GPU（同 Error 304 根因） | 无害警告；编译仍会正确使用 `CUDA_HOME` |
| `Unable to detect CUDA version due to permission error` | conda 的 virtual package 探测失败 | 无害；本项目不依赖 conda 的 cuda 元包 |

---

## 8. 相关文件

| 文件 | 用途 |
|---|---|
| `setup_env.sh` | 一键建环境（幂等） |
| `scripts/env_check.py` | 环境自检（无 GPU 可跑） |
| `scripts/render_test.py` | 渲染测试（本项目出图 vs 数据集原图） |
| `requirements.txt` | Python 依赖（版本来自实测环境） |
| `submodules/ATTRIBUTION.md` | vendored 源码的来源、许可与差异 |
| `src/live3dgsavatar/compat/__init__.py` | 第三方兼容性补丁（numpy 2.x 别名，见 §3.6） |
| `scripts/_compat_shim.py` | 上者的精简副本，供独立脚本在未安装本项目时使用 |
| `docs/ARCHITECTURE.md` §4、§8 | 目录结构与环境在整体设计中的位置 |
