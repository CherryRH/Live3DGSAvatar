/**
 * 入口：装配 client / view / controls，并维护状态。
 *
 * 三种数据源共用一个接口（`send(msg)` + `config`/`frame`/`status` 事件）：
 *   - `RenderClient`：真实 WebSocket 后端
 *   - `MockSource`  ：本地 mock（后端未就绪）
 * 因此下面的装配逻辑对两者完全一致。
 *
 * 启动参数（URL query）：
 *   ?mock=1  强制 mock      ?mock=0  强制连后端     默认：先试后端，失败则 mock
 *
 * **FPS 由前端自己算**（滑动窗口，按帧到达间隔）。它反映"对方实际看到的画面"。
 * 后端不提供 fps 字段 —— 后者只把周期统计写到 stdout 日志，两端解耦。
 */

import { RenderClient } from "./client.js";
import { MockSource } from "./mock.js";
import { FrameView } from "./view.js";
import { Controls } from "./controls.js";
import { SettingsPanel } from "./settings.js";
import { FpsMeter } from "./fps.js";
import { BUILD_TAG } from "./protocol.js";

const el = {
  build: document.querySelector("#build-tag"),
  canvas: document.querySelector("#canvas"),
  overlay: document.querySelector("#overlay"),
  canvasTarget: document.querySelector("#canvas-target"),
  refreshModels: document.querySelector("#btn-refresh-models"),
  refreshDatasets: document.querySelector("#btn-refresh-datasets"),
  busy: document.querySelector("#busy"),
  toast: document.querySelector("#toast"),
  toastText: document.querySelector("#toast-text"),
  busyText: document.querySelector("#busy-text"),
  overlayText: document.querySelector("#overlay-text"),
  connDot: document.querySelector("#conn-dot"),
  connText: document.querySelector("#conn-text"),
  fps: document.querySelector("#stat-fps"),
  frame: document.querySelector("#stat-frame"),
  deform: document.querySelector("#stat-deform"),
  raster: document.querySelector("#stat-raster"),
  dropped: document.querySelector("#stat-dropped"),
  mem: document.querySelector("#stat-mem"),
  mockNote: document.querySelector("#mock-note"),
};

// 立刻显示构建标记：让人一眼看出浏览器加载的是哪个版本。
// 若这里显示的还是旧值，说明拿到的是缓存 —— 硬刷新即可。
el.build.textContent = BUILD_TAG;
console.info(`[Live3DGSAvatar] 前端构建 ${BUILD_TAG}`);

const view = new FrameView(el.canvas);
// 对比画布：默认隐藏，开启"对比原图"时才显示（布局由 CSS 的 .is-compare 控制）。
// 声明必须在使用之前：`targetView` 若只在事件回调里出现就是 ReferenceError。
const targetView = el.canvasTarget ? new FrameView(el.canvasTarget) : null;

/** mock 源；非 null 表示当前用的是 mock。 */
let mock = null;
/** 真实客户端；非 null 表示正在连后端。 */
let client = null;
/** 当前数据源（client 或 mock），二者接口一致。 */
let source = null;

let paused = false;
let backendTried = false;

const fpsMeter = new FpsMeter(1000);

// ------------------------------------------------------------------ 显示 --

function setConn(state, text) {
  el.connDot.className = "dot " + ({
    open: "dot--on",
    connecting: "dot--wait",
    retrying: "dot--wait",
    closed: "dot--off",
  }[state] ?? "dot--off");
  el.connText.textContent = text;
}

/**
 * 全页加载浮窗。只用于**慢且可能失败**的操作（换模型、换数据集）。
 *
 * 语义是"别动，等结果"：期间覆盖整页并拦住交互。成功/失败都**必须**
 * 调用 `hideBusy()` —— 忘记收起会留下一个永远转圈的页面。
 */
function showBusy(text) {
  if (el.busy) el.busy.hidden = false;
  if (el.busyText) el.busyText.textContent = text;
}

function hideBusy() {
  if (el.busy) el.busy.hidden = true;
}

let toastTimer = null;

/** 短暂提示（成功/失败都走它）。`isError` 只影响配色。 */
function setStatusNote(text, isError = false) {
  if (!el.toast) return;
  el.toastText.textContent = text;
  el.toast.classList.toggle("is-error", Boolean(isError));
  el.toast.hidden = false;
  if (toastTimer) clearTimeout(toastTimer);
  // 失败多留一会儿：用户需要时间读原因
  toastTimer = setTimeout(() => { el.toast.hidden = true; },
                          isError ? 8000 : 4000);
}

function showOverlay(text) {
  if (text === null) {
    el.overlay.hidden = true;
    return;
  }
  el.overlay.hidden = false;
  el.overlayText.textContent = text;
}

function updateStats(status) {
  // ⚠️ 刻意**不读** status.fps —— 前端 FPS 由 fpsMeter 自己算（见上）。
  //    后端也不再提供该字段：那反映服务端吞吐，与"对方看到的"不是一回事。
  const ms = status.ms ?? {};
  if (typeof ms.frame === "number" && ms.frame > 0) {
    el.frame.textContent = ms.frame.toFixed(2);
  }
  if (typeof ms.deform === "number" && ms.deform > 0) {
    el.deform.textContent = `${ms.deform.toFixed(2)}`;
  }
  if (typeof ms.rasterize === "number" && ms.rasterize > 0) {
    el.raster.textContent = `${ms.rasterize.toFixed(2)}`;
  }
  if (typeof status.peak_memory_mib === "number" && status.peak_memory_mib > 0) {
    el.mem.textContent = status.peak_memory_mib.toFixed(0);
  }
  if (typeof status.paused === "boolean") {
    paused = status.paused;
    controls.setPaused(paused);
  }
  // 帧号与播放状态以**后端**为准，避免两端各记一份而漂移。
  // 后端按 render.status_interval_s（默认 4 Hz）周期推 status，
  // 因此播放时帧号条会平滑推进（只在控制消息后回 status 则不会动）。
  // live 状态会变（桥接起停、断流），只在 /api/state 里给一次是不够的
  if (status.live) controls.applyLiveStatus(status.live);

  if (typeof status.frame === "number" || typeof status.playing === "boolean") {
    controls.applyPlayback({ frame: status.frame, playing: status.playing });
  }
}

// -------------------------------------------------------------- 设置浮窗 --

// 面板自己按后端下发的 schema 渲染控件；这里只负责"应用之后同步右侧快捷控件"。
// 两处都能改渲染参数（例如背景色），保存后必须让快捷控件跟上，
// 否则界面会显示旧值 —— 而实际渲染已经用了新值。
const settings = new SettingsPanel(document, {
  onApplied(body) {
    // `render` 是渲染参数的权威值（与 /api/state 同构）。
    // 目前右侧面板不再镜像这些参数，保留这条同步是为了将来加控件时不必再想一遍。
    if (body?.render) controls.applyState({ render: body.render });
    // 路径类设置（尤其数据集根目录）改完后，下拉框里的内容可能已经变了。
    // 这里直接重扫，用户不必再去点「刷新列表」。
    const applied = body?.applied ?? [];
    if (applied.some((k) => k.startsWith("paths."))) {
      refreshModels();
      refreshDatasets().then(() => setStatusNote("路径已更新，数据集列表已重新扫描"));
    }
  },
  onError(err) {
    console.warn("[main] 设置面板不可用：", err.message);
  },
});

// ------------------------------------------------------------------ 控件 --

const controls = new Controls(document, {
  onParams(msg) {
    source?.send(msg);
  },
  onLoadModel({ subject, work_name }) {
    if (mock) {
      controls.showModelError("mock 模式下不能切换模型（请先连后端）");
      return;
    }
    // 换模型同样"慢且可能失败"：浮窗覆盖 → 等结果 → 成功才刷新界面。
    // 失败时**不动任何状态**（旧模型仍在跑，界面继续显示旧的）。
    showBusy(`正在加载模型 ${subject}/${work_name}…`);
    fetch("/api/model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ subject, work_name }),
    })
      .then(async (resp) => {
        const body = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(body.detail ?? body.error ?? `HTTP ${resp.status}`);
        controls.applyState(body);
        refreshDatasets();      // 新模型的"自带数据集"可能不同
        refreshModels();        // 版本列表以磁盘为准
        showOverlay(null);
        setStatusNote(`已加载模型 ${subject}/${work_name}，已回到第 0 帧并暂停`);
      })
      .catch((err) => {
        controls.showModelError(`加载失败：${err.message}`);
        setStatusNote(`加载模型失败：${err.message}`, true);
      })
      .finally(hideBusy);
  },
  /**
   * 切换驱动数据集。
   *
   * 为什么用 HTTP 而不是 WS 的 params：这是个**慢且可能失败**的操作
   * （重载 scene 要磁盘 IO，还可能拓扑不匹配）。前端需要"显示浮窗 → 等结果 →
   * 成功才刷新界面"，这要求一次明确的请求/响应。
   *
   * 成功后：帧条回到 0、切回暂停、状态整体刷新（后端已把 timeline 归零）。
   * 失败后：**不动任何状态**，只把下拉框退回上一次成功的值。
   */
  onDatasetChange(name) {
    if (mock) {
      controls.revertDataset();
      controls.showModelError("mock 模式下不能切换数据集（请先连后端）");
      return;
    }
    showBusy(`正在加载数据集 ${name || "（模型自带）"}…`);
    fetch("/api/dataset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset: name }),
    })
      .then(async (resp) => {
        const body = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(body.detail ?? body.error ?? `HTTP ${resp.status}`);
        // 先刷新下拉框（提交新值），再整体应用状态
        controls.commitDataset();
        controls.applyState(body);
        showOverlay(null);
        setStatusNote(`已切到数据集 ${name || "（模型自带）"}：`
          + `${body.dataset?.num_frames ?? "?"} 帧，已回到第 0 帧并暂停`);
      })
      .catch((err) => {
        controls.revertDataset();
        setStatusNote(`切换数据集失败：${err.message}`, true);
      })
      .finally(hideBusy);
  },

  onPause(next) {
    paused = next;
    controls.setPaused(paused);
    source?.send({ type: "stream", paused });
  },
  onStep(delta) {
    // 单步：mock 直接改本地帧号，后端发独立 op（前端无需知道当前帧号）
    if (mock) {
      mock.send({ type: "step", delta });
      return;
    }
    source?.send({ type: "step", delta });
  },
  onCapture() {
    const name = `snapshot-${new Date().toISOString().replace(/[:.]/g, "-")}`;
    if (mock) {
      // mock 不走后端，直接本地下载
      view.toBlob().then((blob) => {
        if (!blob) return;
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = `${name}.png`;
        a.click();
        URL.revokeObjectURL(a.href);
      });
      return;
    }
    source?.send({ type: "capture", name });
  },
  onReconnect() {
    if (mock) { restartMock(); return; }
    client?.close();
    client = null;
    connectToBackend();
  },
});

// ------------------------------------------------------------- 数据源装配 --

/** 把 `config` / `frame` / `status` 事件接到 view 与控件上。 */
function wireSource(src, { isMock }) {
  src.addEventListener("config", (ev) => {
    const g = ev.detail;
    view.resize(g.width, g.height, g.channels ?? 3);
    if (targetView) targetView.resize(g.width, g.height, g.channels ?? 3);
    if (g.numFrames) controls.setFrameRange(g.numFrames);
    if (Array.isArray(g.background)) {
      controls.applyState({ render: { background: g.background } });
    }
    showOverlay(null);
    el.mockNote.hidden = !isMock;
  });

  src.addEventListener("frame", (ev) => {
    const frame = ev.detail;
    try {
      if (frame.bitmap) {
        // 编码帧：直接画位图，画完立刻释放（不关会一直占显存）
        view.drawBitmap(frame.bitmap, frame.seq);
        frame.bitmap.close();
      } else {
        view.draw(frame);
      }
    } catch (err) {
      frame.bitmap?.close?.();
      console.error("[main] 绘制失败：", err);
      showOverlay(`绘制失败：${err.message}`);
    }
    // 前端自测画面帧率：在**真正画上去之后**打点
    fpsMeter.tick();
    if (src.dropped !== undefined) el.dropped.textContent = String(src.dropped);
  });

  src.addEventListener("target", (ev) => {
    const frame = ev.detail;
    // 对比帧同样可能是编码的（bitmap）或原始的（pixels）
    if (targetView) {
      try {
        if (frame.bitmap) {
          targetView.drawBitmap(frame.bitmap, frame.seq);
          frame.bitmap.close();
        } else if (view.width && view.height) {
          // config 到达前不会有 target；保险起见再挡一次
          targetView.resize(view.width, view.height, 3);
          targetView.draw(frame);
        }
      } catch (err) {
        frame.bitmap?.close?.();
        console.error("[main] 对比帧绘制失败：", err);
      }
    }
    // PSNR 来自帧头，与这幅图同一帧号
    controls.applyPsnr(frame.psnr);
  });

  src.addEventListener("status", (ev) => updateStats(ev.detail));
  src.addEventListener("captured", (ev) => {
    console.info("[main] 已截取：", ev.detail.path);
  });
}

function startMock(reason) {
  mock?.stop();
  client?.close();
  client = null;

  mock = new MockSource();
  source = mock;
  wireSource(mock, { isMock: true });
  mock.start();

  setConn("open", `mock（${reason}）`);
  // mock 没有 /api/state，用 config 里给的默认值
  controls.applyState({
    subject: document.querySelector("#f-subject").value,
    work_name: document.querySelector("#f-work-name").value,
    render: { background: [0, 0, 0], scaling_modifier: 1, scale: 1, target_fps: 60 },
    drive: { mode: "still", frame: 0, amplitude: 0.1, playing: false },
    dataset: { num_frames: 254 },
  });
  controls.showModelError("mock：无真实模型信息");
}

function restartMock() {
  startMock("手动重连");
}

/** 拉一次可用模型列表（失败不影响主流程）。`announce` 为真时给出提示条。 */
function refreshModels(announce = false) {
  return fetch("/api/models")
    .then((r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    })
    .then((payload) => {
      controls.applyModels(payload);
      if (announce) {
        const n = payload.models?.length ?? 0;
        setStatusNote(n ? `已刷新模型列表：${n} 个` : "没找到模型（models/*/*/model.ply）",
                      n === 0);
      }
    })
    .catch((err) => {
      console.warn("[main] /api/models 不可用：", err.message);
      if (announce) setStatusNote(`刷新模型列表失败：${err.message}`, true);
    });
}

/** 拉一次可用数据集列表（失败不影响主流程）。 */
function refreshDatasets(announce = false) {
  return fetch("/api/datasets")
    .then((r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    })
    .then((payload) => {
      controls.applyDatasets(payload);
      if (announce) {
        const n = payload.datasets?.length ?? 0;
        setStatusNote(n ? `已刷新数据集列表：${n} 个` : "没找到数据集", n === 0);
      }
    })
    .catch((err) => {
      console.warn("[main] /api/datasets 不可用：", err.message);
      if (announce) setStatusNote(`刷新数据集列表失败：${err.message}`, true);
    });
}

function connectToBackend() {
  backendTried = true;
  showOverlay("正在连接后端…");

  // 先取 /api/state 初始化控件；失败则回退 mock
  // ⚠️ 顺序要紧：先填数据集下拉框，再 applyState。
  //    否则恢复 `drive_dataset` 时选项还不存在，会被静默忽略。
  // 手动刷新：中途往磁盘里放了新模型/新数据集时不必重启
  el.refreshModels?.addEventListener("click", () => refreshModels(true));
  el.refreshDatasets?.addEventListener("click", () => refreshDatasets(true));

  // 顺序：先填模型与数据集下拉框，再 applyState（否则回填时选项还不存在）
  refreshModels()
    .then(refreshDatasets)
    .then(() => fetch("/api/state"))
    .then((r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    })
    .then((state) => controls.applyState(state))
    .catch((err) => console.warn("[main] /api/state 不可用：", err.message));

  client = new RenderClient();
  source = client;
  wireSource(client, { isMock: false });

  client.addEventListener("state", (ev) => {
    const s = ev.detail.state;
    if (s === "open") setConn("open", "已连接");
    else if (s === "connecting") setConn("connecting", "连接中…");
    else if (s === "retrying") setConn("retrying", "重连中…");
    else setConn("closed", "已断开");
  });

  client.addEventListener("error", (ev) => {
    // 用可选链兜底：错误处理路径**绝不能再抛错**，否则真正的错误会被掩盖。
    const err = ev.detail?.error;
    const msg = err?.message ?? "未知错误";
    showOverlay(err?.fromServer ? `后端错误：${msg}` : `连接失败：${msg}`);
    if (!err?.fromServer) setConn("closed", "连接失败");
  });

  client.addEventListener("retry", (ev) => {
    // 重试两次仍不通 ⟹ 判定后端未启动，切 mock（避免无限刷日志）
    if (ev.detail.attempt >= 2 && !mock) {
      client.close();
      startMock("后端未响应");
    }
  });

  client.connect();
}

// -------------------------------------------------------------------- 启动 --

// 脚本出错时要**显式可见**。否则表现为"界面完全没反应"，极难排查。
window.addEventListener("error", (ev) => {
  showOverlay(`前端脚本错误：${ev.message}（构建 ${BUILD_TAG}）`);
  console.error("[main] 未捕获错误", ev.error ?? ev.message);
});
window.addEventListener("unhandledrejection", (ev) => {
  console.error("[main] 未处理的 Promise 拒绝", ev.reason);
});

const params = new URLSearchParams(location.search);
const forceMock = params.get("mock") === "1";
const forceBackend = params.get("mock") === "0";

// 页面由 file:// 打开时不可能连后端，直接 mock
const isFileProtocol = location.protocol === "file:";

if (forceMock || isFileProtocol) {
  startMock(forceMock ? "?mock=1" : "file:// 打开");
} else if (forceBackend) {
  connectToBackend();
} else {
  // 默认：先试后端，失败自动回退（见 retry 处理）
  connectToBackend();
}

// 前端 FPS 独立刷新：不依赖后端的 status 推送节奏
// （后端刻意不推周期心跳，见 docs/GUI_PROTOCOL.md §1.1）。
setInterval(() => { el.fps.textContent = fpsMeter.display; }, 250);

window.addEventListener("beforeunload", () => {
  client?.close();
  mock?.stop();
});
