/**
 * WebSocket 客户端：连接、重连、丢帧统计、背压。
 *
 * 见 docs/GUI_PROTOCOL.md §3。
 *
 * 三个刻意的设计：
 * 1. **丢帧不排队**：本地只保留最新一帧。渲染永不阻塞，GUI 永远显示最新画面，
 *    代价是画面可能跳帧 —— 这比"延迟越积越大"好得多。
 * 2. **原始二进制**：帧是 ArrayBuffer，不做 base64/JSON 包装。
 * 3. **重连用指数退避**：后端重启时不至于把日志刷满。
 */

import { MSG_TARGET, parseFrame, parseMessage, geometryFromConfig } from "./protocol.js";

const RECONNECT_BASE_MS = 500;
const RECONNECT_MAX_MS = 8000;

export class RenderClient extends EventTarget {
  /**
   * @param {object} [opts]
   * @param {string} [opts.url] WebSocket 地址；默认按当前页面推导 `/ws`
   * @param {number} [opts.maxBufferedBytes] 本地发送缓冲上限，超过则丢新帧
   */
  constructor(opts = {}) {
    super();
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    this.url = opts.url ?? `${proto}//${location.host}/ws`;
    this.maxBufferedBytes = opts.maxBufferedBytes ?? 4 * 1024 * 1024;

    /** @type {WebSocket|null} */
    this.ws = null;
    this.geometry = null;
    this.lastSeq = null;
    /** 本地统计的丢帧数（由 seq 跳号推断；服务端丢帧时不发通知） */
    this.dropped = 0;
    this.retries = 0;
    this.closedByUser = false;
    this._retryTimer = null;

    // 只保留最新一帧：渲染忙时新的帧直接覆盖旧的
    this._pending = null;
    this._pumping = false;
  }

  // ------------------------------------------------------------ 生命周期 --

  connect() {
    this.closedByUser = false;
    this._clearRetry();
    this._setState("connecting");

    let ws;
    try {
      ws = new WebSocket(this.url);
    } catch (err) {
      // 例如 URL 非法。`new WebSocket` 对网络故障不会抛，但对非法 URL 会。
      this._emitError(err);
      this._scheduleRetry();
      return;
    }
    ws.binaryType = "arraybuffer";
    this.ws = ws;

    ws.onopen = () => {
      this.retries = 0;
      this.lastSeq = null;
      this.geometry = null;   // config 会重新发，旧几何作废
      this._setState("open");
    };
    ws.onmessage = (ev) => this._onMessage(ev);
    ws.onerror = () => {
      // 事件里没有可用的错误详情，只能表态 + 等 onclose 触发重连
      this._emitError(new Error("WebSocket 出错（详情见浏览器控制台）"));
    };
    ws.onclose = (ev) => {
      this.ws = null;
      this._setState("closed");
      if (!this.closedByUser) {
        this.dispatchEvent(new CustomEvent("closed", {
          detail: { code: ev.code, reason: ev.reason, willRetry: true },
        }));
        this._scheduleRetry();
      }
    };
  }

  close() {
    this.closedByUser = true;
    this._clearRetry();
    if (this.ws) this.ws.close();
  }

  get state() { return this._state ?? "closed"; }

  // ---------------------------------------------------------------- 发送 --

  /**
   * 发送一条控制消息。未连接时静默丢弃并返回 false。
   * @param {object} msg
   * @returns {boolean}
   */
  send(msg) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return false;
    // 控制消息很小，但积压说明后端卡了，此时丢掉新的更合理
    if (this.ws.bufferedAmount > this.maxBufferedBytes) {
      console.warn("[client] 发送缓冲积压，丢弃控制消息");
      return false;
    }
    this.ws.send(JSON.stringify(msg));
    return true;
  }

  // ---------------------------------------------------------------- 内部 --

  _onMessage(ev) {
    if (typeof ev.data === "string") {
      const msg = parseMessage(ev.data);
      if (!msg) return;
      if (msg.type === "config") {
        try {
          this.geometry = geometryFromConfig(msg);
        } catch (err) {
          this._emitError(err);
          return;
        }
        this.dispatchEvent(new CustomEvent("config", { detail: this.geometry }));
        return;
      }
      // 后端的业务错误（`{type:"error", message}`）也统一成 `{error}` 形状。
      // ⚠️ 必须与 `_emitError` 一致：监听方只读 `detail.error.message`。
      //    曾经这里原样转发 `detail = msg`，而 `msg.error` 不存在 ——
      //    于是**错误处理本身抛 TypeError**，把真正的错误信息盖掉了。
      if (msg.type === "error") {
        const err = new Error(typeof msg.message === "string" ? msg.message : "后端返回未知错误");
        err.fromServer = true;
        this._emitError(err);
        return;
      }
      // status / captured 等直接转发
      this.dispatchEvent(new CustomEvent(msg.type, { detail: msg }));
      return;
    }

    // 二进制：一帧图像
    let frame;
    try {
      frame = parseFrame(ev.data, this.geometry);
    } catch (err) {
      this._emitError(err);
      return;
    }

    // 对比帧独立派发：它不参与"最新一帧"的节流（那是渲染帧的逻辑）。
    // 顺序上后端保证它紧跟同一 seq 的渲染帧，所以直接画即可。
    if (frame.kind === MSG_TARGET) {
      this.dispatchEvent(new CustomEvent("target", { detail: frame }));
      return;
    }

    // seq 跳号 ⟹ 中间有帧没能送到（或本地忙时被丢）
    if (this.lastSeq !== null && frame.seq > this.lastSeq + 1) {
      this.dropped += frame.seq - this.lastSeq - 1;
    }
    this.lastSeq = frame.seq;

    this._pending = frame;
    this._pump();
  }

  /** 一次只派发一帧；渲染忙时新帧覆盖旧帧，不排队。 */
  async _pump() {
    if (this._pumping) return;
    this._pumping = true;
    try {
      while (this._pending) {
        const frame = this._pending;
        this._pending = null;
        this.dispatchEvent(new CustomEvent("frame", { detail: frame }));
        // 让出一轮事件循环，渲染端的 rAF 才有机会跑
        await new Promise((r) => requestAnimationFrame(r));
      }
    } finally {
      this._pumping = false;
    }
  }

  _scheduleRetry() {
    this._clearRetry();
    const delay = Math.min(RECONNECT_BASE_MS * 2 ** this.retries, RECONNECT_MAX_MS);
    this.retries += 1;
    this._setState("retrying");
    this.dispatchEvent(new CustomEvent("retry", { detail: { delay, attempt: this.retries } }));
    this._retryTimer = setTimeout(() => this.connect(), delay);
  }

  _clearRetry() {
    if (this._retryTimer !== null) {
      clearTimeout(this._retryTimer);
      this._retryTimer = null;
    }
  }

  _setState(state) {
    if (this._state === state) return;
    this._state = state;
    this.dispatchEvent(new CustomEvent("state", { detail: { state } }));
  }

  _emitError(err) {
    console.error("[client]", err);
    this.dispatchEvent(new CustomEvent("error", { detail: { error: err } }));
  }
}
