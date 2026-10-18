# GUI 前后端协议

> **状态**：前后端均已实现。启动：`bash run_gui.sh`（默认 <http://localhost:8000>，
> 服务以 dual-stack 监听，`127.0.0.1` 同样可用）。
> 前端可脱离后端独立运行（mock 模式，见 §4）。
>
> **代码位置**：前端 `web/`；后端 `src/live3dgsavatar/app/`
> （`protocol.py` 编解码 / `session.py` 渲染 / `server.py` FastAPI）。

设计目标：**简单好用**。本地单用户工具，不做鉴权、不做多路复用、不做视频编解码。

---

## 0. 为什么这些设计

| 决策 | 理由（有实测依据） |
|---|---|
| 帧走 **WebSocket 原始 RGB8**，不压缩 | 回环实测 **855 MiB/s**；512×512 一帧 768 KiB，在 90 FPS 下只需 68 MiB/s，**余量 13×**。而 PNG 编码 27 ms/帧、WebP 11 ms/帧都会成为新瓶颈（实测数据见 `docs/ARCHITECTURE.md` §9.2） |
| **前端算 FPS、后端写日志** | 两端关心的事不同：前端算"对方实际看到的画面到达率"，后端记"服务端真实渲染吞吐"。互相塞数字只会像之前那样——后端发 `fps: 0.0` 并注明"由前端计算"，而前端从未实现，于是顶栏恒为 0 |
| 每帧**只发 4 字节**帧号，不发几何 | 分辨率与背景色在一次会话内不变，用 JSON `config` 发一次即可 |
| **丢帧而不是排队** | 渲染永不阻塞；GUI 永远显示最新帧。浏览器端 `bufferedAmount` 超阈值也丢，避免延迟累积 |
| 渲染在**独立线程**且独占 CUDA | CUDA 上下文有线程亲和性；控制消息只改状态、不碰 GPU |
| 控制走 JSON 文本帧 | 频率低（人手操作），无需二进制协议 |

---

## 1. 会话流程

```
浏览器                          服务端
  │  GET /                     │  返回静态页面
  │  GET /api/state            │  当前模型/参数（用于初始化控件）
  ├────────────────────────────>│
  │  POST /api/model            │  切换 subject / work_name → 重载模型
  ├────────────────────────────>│
  │  WS  /ws                    │  建立双向通道
  ├────────────────────────────>│
  │  ← {"type":"config", ...}   │  分辨率、背景、可用参数范围
  │  ← {"type":"status", ...}   │  FPS、耗时分解、高斯数、丢帧数（周期推送）
  │  ← <binary 4 字节>           │  一帧 RGB8（紧跟 config.width*height*3 字节）
  │  ──{"type":"params", ...}──>│  改参数
  │  ──{"type":"stream", ...}──>│  暂停/继续
```

**握手顺序固定**：服务端连上后**先**发一条 `config`，再开始推帧。客户端在收到 `config` 之前不建立画布。

### 1.1 渲染循环的语义（**帧数上限 + 等待**）

后端**没有把渲染结果缓存起来再播放**。它是一条持续工作的流：

```
loop:
    t0 = now
    render()                      # 用最新参数，绝不读缓存的帧
    推进播放（若在播放）           # 渲染一帧 = 播放一帧
    打包推入队列（容量 1，满则丢旧）
    sleep(interval - elapsed)     # 上一轮没超时就等满间隔
```

由此得到两条刻意的性质：

| 性质 | 含义 |
|---|---|
| **实际帧率 = `min(帧数上限, 1/渲染耗时)`** | 渲染慢于间隔时自动降级为"渲染完立刻继续"，**绝不补帧、绝不排队**。这就是"上一帧没渲染完就等待"的真实含义——不堆积待渲染的帧 |
| **渲染一帧 = 播放一帧** | 推进与渲染在同一处，二者不可能漂移。推论：**播放速度由帧数上限决定**——`30` 时动作比 `60` 慢一倍 |

**静态帧也全速推送。** 同一输入产生**逐位相同**的输出（实测 `max|Δ| = 0`），
所以静态时传的是重复字节 —— 这是有意为之：链路始终活着，
丢帧/背压/统计都在真实工作，接上实时采集时才不会暴露出新问题。

**`status` 按 `render.status_interval_s`（默认 0.25 s，即 4 Hz）周期推送**，
前端据此更新帧号条、单帧耗时与显存。

> 早先只在控制消息后才回 `status`，结果是**播放时帧号条不动、暂停时才突然跳**
> —— 因为暂停恰好触发了那条消息。周期推送是必需的。
>
> 但 `status` **刻意不含 `fps`**：前端画面帧率由前端自己按帧到达间隔计算
> （那反映"对方实际看到的"），后端吞吐走 stdout 日志。两端解耦。

---

## 2. HTTP 接口

### `GET /api/state`

初始化控件用。返回当前生效的模型与参数（默认值来自 `configs/*.yaml`）。

```json
{
  "subject": "duda",
  "work_name": "test",
  "model": {
    "path": "models/duda/test/model.ply",
    "num_gaussians": 60353,
    "num_basis": 20,
    "num_basis_in": 129,
    "tex_size": 256
  },
  "render": {
    "background": [0.0, 0.0, 0.0],
    "scaling_modifier": 1.0,
    "scale": 1.0,
    "target_fps": 60
  },
  "dataset": {
    "num_frames": 254,
    "image_subdir": "images"
  }
}
```

### `POST /api/model`

```json
{"subject": "duda", "work_name": "test"}
```

成功返回新的 `/api/state` 同构对象；失败返回 `400` + `{"error": "..."}`。
**切换模型会重建渲染器**，期间帧流暂停。

### `GET /api/settings`

返回**全部**配置项（含锁定项）+ 分组元数据。字段元数据的唯一来源是
`src/live3dgsavatar/app/settings.py`，前端只负责按它画控件。

```json
{
  "groups": [{"key": "render", "title": "渲染", "help": "…", "order": 10}],
  "fields": [{
    "key": "render.background", "label": "背景色", "group": "render",
    "order": 10, "kind": "color", "editable": true, "reason": "",
    "help": "…", "unit": "", "minimum": null, "maximum": null,
    "step": 0.05, "choices": [], "requires": null, "value": [0, 0, 1]
  }],
  "overrides_path": "/…/configs/local.yaml",
  "overrides": {"render.target_fps": 30}
}
```

`fields` 已按「分组 → 组内 order」排好序，前端不需要再排。

### `POST /api/settings`

```json
{"patch": {"render.target_fps": 30, "train.online_update.color_l1": true}}
```

三步：**校验 → 即时应用 → 写入 `configs/local.yaml`**。

| 情况 | 响应 |
|---|---|
| 成功 | `200` + `{ok, applied, saved_to, warning, groups, fields, render, status}` |
| 未知键 / 锁定键 / 越界 / 空 patch | `400` + `{detail: "..."}`，**整批拒绝** |

**整批拒绝**是刻意的：一批里有一项非法时，不能出现"前两项改了、第三项失败"的
中间状态 —— 那种状态没人能推理清楚。

**存盘失败不回滚**：若 `local.yaml` 写不进去（只读文件系统），修改仍然生效，
但响应里 `saved_to: null` 且 `warning` 说明"重启后会丢"。

`applied` 只列出**立即生效**的项（渲染参数与两个间隔）；其余项（如
`train.online_update.*`）已写入配置，在下次被读取时生效。

`render` 是渲染参数的**权威值**（与 `/api/state` 的 `render` 同构），
供前端同步镜像控件。它**不**放进 `status`：`status` 每 4 Hz 推一次，
而这些值很少变，塞进去既是带宽浪费、也混淆了"状态"与"配置"两种语义。

### `GET /healthz`

返回 `{"ok": true, "gpu": "NVIDIA GeForce RTX 3060 Laptop GPU"}`。

---

## 3. WebSocket

端点 `/ws`。

### 3.1 服务端 → 客户端

#### 文本帧：`config`（连接后第一条）

```json
{
  "type": "config",
  "width": 512,
  "height": 512,
  "background": [0.0, 0.0, 0.0],
  "num_frames": 254
}
```

#### 文本帧：`status`（周期推送，默认 2 Hz）

```json
{
  "type": "status",
  "seq": 1024,
  "fps": 167.7,
  "ms": {
    "frame": 5.96,
    "deform": 3.40,
    "rasterize": 2.10
  },
  "dropped": 12,
  "peak_memory_mib": 95.0,
  "paused": false,
  "frame": 42,
  "playing": false,
  "total_frames": 254
}
```

`frame` / `playing` / `total_frames` 由**后端**权威给出；前端只回填控件，
不自己维护一份状态（否则两端会漂移）。

**刻意不含 `fps`** —— 前端 FPS 由前端按帧到达间隔自行计算（滑动窗口 1 s），
后端只把周期统计写到 stdout：

```
[stats] 10.0s  渲染 598 帧  平均 59.8 FPS  平均 5.10 ms/帧
        (deform 3.20 / raster 1.90)  峰值显存 95 MiB  丢帧 0
```

#### 二进制帧：一帧图像

| 偏移 | 长度 | 内容 |
|---|---|---|
| 0 | 4 | `seq`，**uint32 小端** |
| 4 | `width*height*3` | RGB8，行优先，无 padding |

- 几何（宽高）由 `config` 提供，**不在二进制帧里重复**
- 客户端若发现 `seq` 跳号，累加本地 `dropped` 计数（服务端丢帧时不通知）

### 3.2 客户端 → 服务端

#### `params`

只发改动的字段；服务端按字段合并。

```json
{
  "type": "params",
  "render": {"background": [1.0, 1.0, 1.0], "scale": 0.75},
  "drive": {"mode": "dataset", "frame": 42}
}
```

`params` 分两层：**数据源**（驱动参数从哪来）与**驱动方式**（同一数据源下怎么用）。

```json
{"type":"params",
 "source": {"kind": "checkpoint"},
 "drive":  {"mode": "play", "frame": 42, "amplitude": 0.1, "playing": true},
 "render": {"background": [0,0,0], "scale": 1.0, "target_fps": 60}}
```

| 层级 | 取值 | 可用性 |
|---|---|---|
| `source.kind` | `checkpoint` | ✅ 数据集里已有的采样 |
| | `live` | ⬜ **空钩子，待接入**（实时 tracking）。前端把控件置灰 |

`drive` 字段：

| 字段 | 含义 |
|---|---|
| `frame` | 跳到的帧号（夹到 `[0, total-1]`） |
| `playing` | 是否推进帧号。到末尾回绕 |
| `perturb` | 调试：以当前帧为基准加随机扰动（正交于 `playing`） |
| `amplitude` | `perturb` 的幅度 |

> **刻意没有 `drive.mode`。** 原本的 `still` / `play` 与播放按钮语义重复
> （"定住"就是"暂停"），而且容易误解为**"暂停就不渲染了"**。
> 现在只有 `playing` 一个开关：
>
> **暂停 ≠ 停止渲染。** 渲染循环仍按帧数上限全速工作，只是帧号停住 ——
> 画面内容不变，但链路一直是活的。这正是"视频流"而非"离线播放"的形态。

**`source.kind = "live"` 时 `drive` 里的字段不适用，后端会忽略它们**（而不是报错）——
这样前端切到 live 时不必先清空控件，切回 checkpoint 时也不必重设。

> **为什么播放由后端推进**：渲染循环每出一帧就 `advance()` 一次，
> 「渲染的帧」与「显示的帧」天然一致。若交给前端定时器发消息，
> 两者会因消息延迟与丢帧而漂移。
> 播放到末尾**回绕**；`frame` 跳转与 `step` 在边界**夹住**（不回绕）。

#### `step` —— 单帧前进/后退

单步审查用。**刻意不传绝对帧号**：前端无需知道当前是第几帧，
也不会因丢帧而算错。

```json
{"type": "step", "delta": -1}
```

`delta` 为 `+1`（下一帧）或 `-1`（上一帧）。边界夹住，不回绕。
服务端处理后回一条 `status`，其 `frame` 字段是新的权威帧号。

#### `stream`

```json
{"type": "stream", "paused": true}
```

#### `capture`

```json
{"type": "capture", "name": "snapshot-001"}
```

服务端把当前帧以无损格式写到 `output/gui_captures/`，并回一条 `status` 附带路径。

---

## 4. 前端结构（已实现）

```
web/
├── index.html          # 单页；canvas + 控件面板
├── css/app.css
├── js/
│   ├── main.js         # 入口：装配 client + view + controls
│   ├── protocol.js     # 帧解析（ArrayBuffer → ImageData）
│   ├── client.js       # WebSocket 客户端（重连、丢帧统计、背压）
│   ├── view.js         # canvas 绘制
│   ├── controls.js     # 参数面板 ↔ 消息
│   └── mock.js         # mock 帧源（后端未就绪时用来跑通界面）
└── README.md
```

**无构建步骤**：原生 ES modules，由服务端静态托管即可。

### mock 模式

后端未就绪时用 `python -m http.server` 打开 `web/`，前端自动回退到 mock：
本地生成帧（含运动图案）+ 假 `status`，用于验证
**画布绘制、丢帧统计、参数控件、重连逻辑**。
URL 加 `?mock=1` 强制 mock，`?mock=0` 强制连真后端。

---

## 5. 与配置的关系

GUI 设置面板的**参数元数据**（可编辑性、范围、分组、排序）集中在
`src/live3dgsavatar/app/settings.py` —— 那是唯一事实来源。
前端 `web/js/settings.js` 只按 schema 画控件、只发**差异**。

为什么只发差异：后端把 patch 写进 `local.yaml` 作为覆盖项。若每次保存都写全部字段，
`local.yaml` 会变成整份配置的副本，之后改默认值会被它永久盖住，而且没人看得出
"哪些是我改的"。


后端的默认值一律来自 `configs/*.yaml`（见 `docs/CONFIG.md`）：

| 配置项 | GUI 用途 |
|---|---|
| `paths.models_dir` / `subject` / `work_name` | 模型选择器初值 |
| `render.background` | 背景色控件初值 |
| `render.scaling_modifier` | 缩放控件初值 |
| `render.batch_size` | 单帧渲染时为 1 |

GUI 只会**临时覆盖**这些值，不回写 YAML。用户想持久化就改配置文件。

---

## 6. 未来可做的优化（已登记，暂不做）

| 项 | 预期 | 依据 |
|---|---|---|
| 接入 `mesh_binding` CUDA 内核 | deform 3.4 → ~0.3 ms，整帧 5.96 → ~2.7 ms（**约 370 FPS**） | 内核已 vendored，只是从"PyTorch 复现其数学"改成"直接调用" |
| 前端提速 | 帧率超过显示器刷新率后收益递减 | 167 FPS 已远超 60-144 Hz 面板 |
| 分辨率缩放 | `render.scale` 是显存与帧率最有效的旋钮 | 6 GB 卡上尤其明显 |
