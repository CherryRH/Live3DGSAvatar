# web/ —— 实时预览界面

图形化前端。**无构建步骤**：原生 ES modules，直接由后端静态托管。

> **状态**：前后端均已实现。启动：`bash run_gui.sh`（见根 `README.md`）。
> 协议见 [`docs/GUI_PROTOCOL.md`](../docs/GUI_PROTOCOL.md)。

## 为什么不用 Vite / TypeScript / 框架

界面就是「一个 canvas + 一组控件」。引入构建链的代价（`node_modules` 100–200 MB、
构建步骤、版本漂移）换不来对应的收益。这个项目的既有取向是**依赖最少、每层显式**。

以后界面变大（多标签、复杂状态）再引入 Vite 也不迟 —— 原生 ESM 可以平滑迁移。

## 目录

```
web/
├── index.html          单页：顶栏 + canvas 舞台 + 参数面板
├── css/app.css
├── js/
│   ├── main.js         入口：装配数据源 / view / controls
│   ├── protocol.js     帧与消息解析（含长度校验、版本校验）
│   ├── client.js       WebSocket 客户端（重连、丢帧统计、背压）
│   ├── view.js         canvas 绘制（RGB→RGBA，复用缓冲）
│   ├── controls.js     参数面板与播放条 ↔ 消息装配
│   ├── fps.js          前端画面帧率（滑动窗口）
│   └── mock.js         mock 帧源（复刻真实协议）
└── tests/
    ├── all.mjs         汇总入口
    ├── ids.mjs         静态检查：HTML id ↔ JS 引用 ↔ 测试桩
    ├── run.mjs         protocol.js / fps.js / mock.js
    └── dom.mjs         view.js / controls.js（用最小桩）
```

## 运行

### 正常使用（连后端）

```bash
bash run_gui.sh          # 浏览器打开 http://localhost:8000
```

由后端在同一端口同时提供静态文件与 `/api/*`、`/ws`（见 `docs/GUI_PROTOCOL.md`）。

### 只看界面（mock 模式）

```bash
cd web && python3 -m http.server 8912   # 浏览器打开 http://127.0.0.1:8912
```

前端会先尝试连后端；连续两次失败后**自动回退到 mock**，用本地生成的图案
验证画布、丢帧统计、参数控件与重连逻辑。强制指定：`?mock=1` 用 mock，`?mock=0` 只用后端。

> ⚠️ 用 `file://` 直接打开 `index.html` 时**不能**连后端（浏览器不允许），会自动进入 mock 模式。

## 测试

```bash
node web/tests/all.mjs
```

覆盖 `protocol.js`（帧布局、长度校验、版本校验）与 `mock.js`（协议一致性）、
以及 `view.js` / `controls.js`（用最小 DOM 桩）。

**`client.js` 的真实 WebSocket 行为只能在浏览器里验证** ——
重连、背压、丢帧统计需要真实网络栈。

## 播放条（画面下方，仿视频播放器）

| 控件 | 行为 |
|---|---|
| **▶ / ⏸** | 播放/暂停。状态交给**后端**：渲染循环每出一帧推进一次 |
| **◀\| / \|▶** | 单步前进/后退，边界**夹住不回绕**（逐帧审查用） |
| **← / → / 空格** | 与单步、播放按钮等价（键盘更快） |
| **进度条** | 拖动跳到指定帧；拖动期间**不被服务端回填**（否则和手指打架） |
| **帧数上限** | 每秒最多渲染几帧。**同时决定播放推进速度**：30 比 60 慢一倍 |
| **数据源** | `checkpoint` 可用；`live`（实时采集）为**空钩子，待接入**，选中后驱动控件置灰 |
| **调试·随机扰动** | 以当前帧为基准加扰动，探查训练分布外行为。与播放正交 |

**暂停 ≠ 停止渲染**：暂停只是帧号停住，渲染循环仍按帧数上限全速工作
（画面内容不变，但链路一直是活的）。

帧号与播放状态以**后端**为准（`status.frame` / `status.playing`），
前端只回填控件 —— 避免两端各记一份而漂移。

### FPS 归属

| 指标 | 谁算 | 反映 |
|---|---|---|
| 顶栏「画面 FPS」 | **前端**（1 秒滑动窗口，按帧到达间隔） | 你实际收到的画面帧率 |
| 后端 stdout 日志 | **后端**（每 `app.stats_log_interval_s` 秒一行） | 服务端真实渲染吞吐 |

**刻意不互相塞数字**：后端不提供 `fps` 字段。

`status` 消息由后端按 `app.status_interval_s`（默认 4 Hz）**周期推送**，
前端据此更新帧号条与耗时显示。

### 排查「连不上」/「看不到变化」

| 现象 | 原因 | 处理 |
|---|---|---|
| 浏览器「无法连接」，但 VS Code 内置浏览器正常 | 只绑了 IPv4，而 `localhost` 优先解析为 `::1` | 已用 dual-stack socket 修复；重启后端 |
| 改了前端却看不到变化 | 浏览器缓存 | 顶栏看**构建标记**（`protocol.js::BUILD_TAG`）；硬刷新 `Ctrl+Shift+R` |
| 界面静止、无播放条 | JS 在初始化时抛错（如 id 不匹配） | 页面会显示红字错误；也可跑 `node web/tests/ids.mjs` |

## 实现约束（改代码时请保留）

| 位置 | 约束 |
|---|---|
| `mock.js::start()` | `config` **必须异步派发**。同步派发会让 `start()` 之后才挂监听的地方永久漏掉它（画布不 resize，界面卡在"等待连接…"） |
| `protocol.js::parseFrame` | 长度不符**必须抛错**，不能凑合画 —— 错位的帧看起来像"模型坏了"，极难排查 |
| `view.js::draw` | 复用 `ImageData` 与 RGB→RGBA 缓冲，避免每帧分配；`putImageData` 忽略 CSS 变换 |
| `client.js::_pump` | **只派发最新一帧**，`bufferedAmount` 超限就丢 —— 背面压，避免延迟累积 |
| `controls.js` | 控件只"读值"，消息统一在 `_emitParams` 装配，避免每个控件各写一遍 |
| `client.js` 的 `error` 事件 | **两个来源必须统一形状**：传输层错误与后端 `{type:"error"}` 都以 `detail.error`（Error 实例）派发。后端错误带 `fromServer=true` 以便区分。曾因后端错误原样转发（`detail.error` 不存在）导致**错误处理自己抛 TypeError**，把真正的错误盖掉 |

## 未做

- 驱动参数的**分区直接控制**（下颌/眼睑/眼球/表情）——需要 FLAME 语义切分
- GT 叠加对照
- 参数快照保存/加载
- 数据源 `live`（实时采集）——协议已留钩子，前端置灰
