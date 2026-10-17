# 配置说明

本项目的**所有可调参数**都集中在 `configs/`，代码中不再有硬编码的路径与业务默认量。

| 文件 | 内容 |
|---|---|
| `configs/system.yaml` | 路径、设备、数据集主体名 |
| `configs/render.yaml` | 渲染参数、模型结构、数据集读取选项 |
| `configs/local.yaml` | **可选**，本机私有覆盖（已 gitignore） |

## 1. 优先级

```
命令行参数  >  环境变量  >  configs/*.yaml  >  代码内置兜底
```

代码内置兜底（`src/live3dgsavatar/config/__init__.py::_FALLBACK`）**刻意保持最小**，
只为保证字段缺失时不 `KeyError`；真正的默认量都在 YAML 里，便于查看与修改。

`configs/local.yaml` 在 `system.yaml` / `render.yaml` 之后**深合并**，用于放不想入库的
本机覆盖（例如另一个数据集路径）。

## 2. 查看最终生效值

```bash
python scripts/show_config.py                 # 全部
python scripts/show_config.py --section paths # 只看路径
python scripts/show_config.py --sources       # 配置文件是否被读到 + 环境变量
python scripts/show_config.py --json          # 机器可读
```

`show_config.py` 还会提示哪些关键路径当前不存在（换机器/未下载时正常）。

## 3. 环境变量

| 变量 | 覆盖字段 |
|---|---|
| `LIVE3DGS_DATA_ROOT` | `paths.data_root` |
| `LIVE3DGS_OUTPUT_DIR` | `paths.output_dir` |
| `LIVE3DGS_REFERENCE_ROOT` | `paths.reference_root` |
| `LIVE3DGS_MODELS_DIR` | `paths.models_dir` |
| `LIVE3DGS_SUBJECT` | `subject` |
| `LIVE3DGS_WORK_NAME` | `work_name` |
| `LIVE3DGS_DEVICE` | `runtime.device` |

环境变量只需填**路径字符串**，加载时会解析为绝对路径。

## 4. `system.yaml` 字段

| 字段 | 说明 |
|---|---|
| `subject` | **人物名**，与 RGBAvatar 的 `--subject` 同名同义。既用于数据集（`<data_root>/<subject>/`），也用于模型的一级目录 |
| `work_name` | 工作名（一次训练/导出的名字），参照里常用 `test` |
| `paths.data_root` | 数据集根目录；下需含 `<subject>/images/` 与 `<subject>/checkpoint/` |
| `paths.models_dir` | **本项目自己的模型根目录**（默认 `models`）。模型放 `<models_dir>/<subject>/<work_name>/` |
| `paths.output_dir` | 产物根目录。各脚本写到 `<output_dir>/<脚本名>/` |
| `paths.reference_root` | RGBAvatar 参照仓库根（**只读**）。取 FLAME 模板、模板 UV、数据集读取代码，以及**它自己的**同名模型 |
| `paths.model_ply` | 显式指定模型文件。`null` 表示按下面的顺序自动查找 |
| `paths.models_config` | 显式指定模型随附的 `config.yaml`。`null` 表示从模型目录里读 |
| `paths.image_subdir` | 数据集原图子目录（默认 `images`），用作 PSNR 的 GT |
| `runtime.device` | `cuda` 或 `cpu`。`cpu` 仅供不依赖 CUDA 内核的自检 |
| `runtime.split` | 数据集切分：`all` / `train` / `test` |
| `smoke.max_frames` | 采样帧数上限；`null` 表示不限制 |
| `paths.data_dir` | 显式指定数据集目录（绝对路径或相对仓库根）。`null` 表示用 `<paths.data_root>/<subject>`。**数据集目录名与 `subject` 不同时用它** |
| `app.host` / `app.port` | GUI 服务监听地址与端口。默认 `localhost:8000`。**本机回环地址会用 dual-stack socket 监听**，使 `localhost` / `127.0.0.1` / `::1` 三种写法都能访问（见下） |
| `app.target_fps` | **帧数上限**：每秒最多渲染几帧。同时决定播放推进速度（30 时动作比 60 慢一倍）。`0` 表示不限，此时帧率受渲染耗时限制 |
| `app.status_interval_s` | `status` 消息推送间隔（秒）。前端据此更新帧号条；太小占满通道，太大帧号一跳一跳 |
| `app.stats_log_interval_s` | 后端统计日志间隔（秒）。日志走 stdout，与前端 FPS 解耦 |
| `app.autostart` | 打开页面是否自动推帧 |
| `app.preload_frames` | GUI 启动时预载帧数。**`-1`（默认）表示全部**。设小会让「数据集帧」滑块被夹在已载入范围内 |

### 模型目录约定（与 RGBAvatar 的 `--subject` / `--work_name` 保持一致）

```
models/<subject>/<work_name>/model.ply          ← 本项目渲染统一用这里
models/<subject>/<work_name>/config.yaml        ← 模型随附配置（从 RGBAvatar 复制）
```

参照实现渲染时用它**自己的**同名模型：`<reference_root>/output/<subject>/<work_name>/`。
这与 RGBAvatar CLI 完全一致（`--subject` + `--work_name` 派生 `output_path` 与 `data_path`）：

```python
# RGBAvatar/render.py
output_path = os.path.join(args.output_dir, args.subject, args.work_name)  # 模型
data_path   = os.path.join(config['data_dir'], args.subject)               # 数据集
```
两边目录结构同名，便于对照与切换。

**查找顺序**（`resolve_model_ply`，见 `src/live3dgsavatar/config/__init__.py`）：

1. `paths.model_ply` —— 若显式配置，直接用它（文件不存在则报错，不静默回退）
2. `<models_dir>/<subject>/<work_name>/model.ply` —— **本项目优先**
3. `<reference_root>/output/<subject>/<work_name>/model.ply` —— 回退到参照

> 渲染测试里，**core 用 `models/`，参照用 `output/`**，刻意分开取 ——
> 否则"对照"会变成"自己跟自己比"。

切换到别的模型：

```bash
# 命令行
python scripts/render_test.py --subject duda --work-name test

# 配置文件：改 subject / work_name 两处

# 环境变量
LIVE3DGS_SUBJECT=duda LIVE3DGS_WORK_NAME=test python scripts/render_test.py
```

### 监听地址：为什么要 dual-stack

单 host 只能覆盖一个地址族。实测本机（WSL2，`networkingMode=mirrored`）：

| `host` | `127.0.0.1` | `::1` |
|---|---|---|
| `127.0.0.1` | ✓ | ✗ |
| `localhost` | ✓ | ✗ |
| `::` | ✗（本机 `v6only=1`） | ✓ |

而 `/etc/hosts` 里 `localhost` **优先解析为 `::1`**，浏览器打开
`http://localhost:8000` 会先试 `::1` —— 只绑 IPv4 时 Windows 浏览器报
**「无法连接」**，而 VS Code 内置浏览器（走 `127.0.0.1`）却正常。

因此回环地址一律改用 **dual-stack socket**（绑 `::` 并关闭 `IPV6_V6ONLY`），
两种写法都通。启动日志会打印实际监听情况：

```
监听：IPv4 127.0.0.1 + IPv6 ::1（dual-stack，端口 8000）
```

### 数据集的两种格式

`load_scene` 按**文件名**自动识别，不需要额外配置：

| 格式 | 判据 | 需要什么 | 适用 |
|---|---|---|---|
| **manifest**（自包含） | 目录下有 `render_inputs.npz` | 只要 FLAME 资产（取拓扑） | 新采集的数据 |
| **checkpoint**（参照格式） | 目录下有 `checkpoint/` + `images/` | 需 FLAME 前向 + chumpy + GPU | 兼容 duda 等旧数据 |

manifest 里已算好逐帧 `mesh` / `blend_weight` / 相机，**不需要 3DMM 参数**，
因此不需要 `checkpoint/`，也不需要 GPU 就能加载（渲染仍需 GPU）。

```text
<dataset>/
├── render_inputs.npz      # mesh [N,5083,3], blend_weight [N,129],
│                          # K [3,3], w2v [4,4], width, height, names [N]
├── images/                # 512×512 原图（PSNR 的 GT）
└── input/                 # 未抠图的原帧（可选）
```

加载时会校验三件事，任何一项不符都立即报错（而不是等到训练/渲染中途才炸）：

1. `blend_weight` 维度 == `model.network.num_basis_in`
2. `mesh` 顶点数足以覆盖 FLAME 面表引用的顶点
3. `mesh` / `blend_weight` 是 float32

### 切换对象

```yaml
# configs/system.yaml
subject: crh_1          # 模型目录 <models_dir>/<subject>/<work_name>/
work_name: test
paths:
  data_dir: null        # null → <data_root>/<subject>
```

若数据集目录名与 `subject` 不一致，设 `paths.data_dir` 即可，不必改 `subject`。

### 路径写法

三种都支持：

```yaml
data_root: /home/you/Datasets/INSTA    # 绝对路径
data_root: ~/Datasets/INSTA            # ~ 展开
reference_root: ../RGBAvatar           # 相对**仓库根**
output_dir: output                     # output 特殊：相对**当前工作目录**
```

> `output_dir` 是唯一按 cwd 解析的字段 —— `--output-dir out` 通常指用户当前目录。

## 5. `render.yaml` 字段

### `render` —— 渲染

| 字段 | 默认 | 说明 |
|---|---|---|
| `background` | `[0,0,0]` | 背景色。数据集原图为黑底；白底用 `[1,1,1]` |
| `scaling_modifier` | `1.0` | 高斯缩放系数 |
| `sh_degree` | `0` | 球谐阶数。本项目只用 DC 项 |
| `batch_size` | `4` | 批大小。**core 与参照实现共用**，保证计时口径一致。6GB 显存建议 ≤ 4 |
| `frames` | `20` | 默认渲染帧数；`-1` 为全部 |
| `max_batch_size` / `max_gaussian_size` / `num_streams` | `10` / `60353` / `3` | 训练用的预分配上限（训练待接入，先保留） |

> 输出**分辨率**不在此配置 —— 它由数据集相机内参决定。

### `model.network` —— 模型结构

⚠️ **这一节必须与 `.ply` 文件匹配。** 但要注意**校验强度并不一致**：

| 字段 | 能否从文件校验 | 写错的后果 |
|---|---|---|
| `num_basis_blend` (K) | ✅ 按 `xyz_b_*` 列数校验 | **明确报错** |
| `num_basis_in` (D) | ✅ 按 `weight_module` 列数校验 | **明确报错** |
| `tex_size` | ⚠️ 仅按容量上界 `tex_size² ≥ N` 校验 | 上界够大时不报错（见下） |
| `mlp_hidden` / `use_weight_proj` | ❌ 不参与存储 | 写错会加载出**结构不同的网络** |

原因：**参照实现写出的 `.ply` 没有 `comment gaussian_config` 注释**
（本项目自己写出的文件才有，见 `docs/CONVENTIONS.md` §4）。
因此消费外部模型时，结构信息只能来自本配置节。

`tex_size` 尤其要注意：它**不参与存储**（只决定"一个 UV texel = 一个高斯"的容量上界），
所以无法从文件反推。加载时会校验 `N ≤ tex_size²`，上界够大时**不会报错**，
配错会静默带进后续的保存/重建。

| 字段 | 默认 | 说明 |
|---|---|---|
| `tex_size` | `256` | UV 纹理边长；高斯数上限为 `tex_size²` |
| `num_basis_in` | `129` | 驱动参数维度（FLAME 姿态/表情等） |
| `num_basis_blend` | `20` | blendshape 基数量 K |
| `mlp_hidden` | `[128,128]` | 权重映射 MLP 隐藏层宽度；`null` 表示不用 MLP |
| `use_weight_proj` | `true` | 是否使用 MLP 权重投影 |

### `dataset` / `test`

| 字段 | 说明 |
|---|---|
| `dataset.use_shape_weight` / `use_pose_weight` | 参照侧 `FLAMEDataset` 的读取选项 |
| `dataset.pin_memory` | 是否锁页内存 |
| `test.dump_diff` | 渲染测试是否输出并排对比图 |

## 6. 脚本参数：只保留关键可调项

命令行是**纯覆盖层**。不传就用配置值，因此日常使用不必带任何参数：

```bash
python scripts/render_test.py                     # 全部取配置
python scripts/render_test.py --frames 5 --batch-size 2   # 临时覆盖
python scripts/equivalence_check.py --skip-render
```

`test_scripts_have_no_business_defaults` 会静态检查脚本里不再出现
`default=256` 这类业务默认量。

常用参数：

| 参数 | 作用 |
|---|---|
| `--subject` / `--work-name` | 选对象与工作名（对应 `models/<subject>/<work_name>/`）。**同时接受 RGBAvatar 风格的下划线写法** `--work_name`，两边命令可直接互相复制 |
| `--ply` | 直接指定模型文件，覆盖上面的推导 |
| `--frames` / `--batch-size` | 采样规模与显存占用 |
| `--skip-reference` | 不加载参照实现（只测本项目） |
| `--dry-run` | CPU 自检，不访问数据集/GPU |

## 7. 写自己的脚本时怎么用

```python
from live3dgsavatar.config import load_config

cfg = load_config()                       # 读仓库根的 configs/
data_dir = cfg.paths.data_root / cfg.subject
batch = cfg.get("render.batch_size", 1)   # 点路径 + 默认值
cfg.set("render.batch_size", 8)           # 就地覆盖
print(cfg.resolved())                     # 便于写日志/报告
```

要点：

- 路径字段已是**绝对 `Path`**，直接用；
- `cfg.get("a.b.c", default)` 支持点路径，缺失时返回 `default`；
- 把 `cfg.resolved()` 写进结果文件，便于事后复现；
- **`core/` 不允许依赖配置**（见 `docs/ARCHITECTURE.md` §2.3）；
  配置属于应用层，`core/` 只接受显式传入的参数。

## 8. 相关测试

| 测试 | 覆盖 |
|---|---|
| `test_config.py::test_loads_defaults_from_yaml_files` | 默认量确实来自 YAML |
| `test_config.py::test_env_overrides_yaml` | 环境变量优先于 YAML |
| `test_config.py::test_local_yaml_is_merged_last` | `local.yaml` 深合并 |
| `test_config.py::test_wrong_type_is_rejected` | 类型错误明确报错 |
| `test_config.py::test_no_hardcoded_absolute_paths_in_source` | **代码中无硬编码路径** |
| `test_config.py::test_scripts_have_no_business_defaults` | **脚本无业务默认量** |
