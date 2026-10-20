/**
 * mock 帧源：后端未就绪时用本地生成的帧跑通界面。
 *
 * **刻意复刻真实协议**（同样的消息类型、同样的二进制帧布局），
 * 这样界面与解析路径能被真实验证；等后端就绪时把本模块摘掉即可，
 * 其余代码一行不用改。
 *
 * 见 docs/GUI_PROTOCOL.md §4「mock 模式」。
 */

import { HEADER_BYTES, MSG_RENDER, PROTOCOL_VERSION } from "./protocol.js";

const DEFAULT_GEOM = { width: 512, height: 512, channels: 3, numFrames: 254 };
const STATUS_INTERVAL_MS = 500;
const FRAME_INTERVAL_MS = 16;   // ~60 FPS

export class MockSource extends EventTarget {
  /**
   * @param {{width?:number, height?:number, numFrames?:number, fps?:number}} [opts]
   */
  constructor(opts = {}) {
    super();
    this.geom = { ...DEFAULT_GEOM, ...opts };
    this.totalFrames = this.geom.numFrames;
    this.frameIndex = 0;
    this.playing = false;
    this.frameIntervalMs = opts.fps ? 1000 / opts.fps : FRAME_INTERVAL_MS;
    this.seq = 0;
    this.background = [0, 0, 0];
    this.drive = { mode: "still", frame: 0, amplitude: 0.1 };
    this.sourceKind = "checkpoint";
    this.paused = false;
    this.t0 = performance.now();
    this._timers = { frame: null, status: null };
    this._fps = 0;
    this._framesInWindow = 0;
    this._windowStart = performance.now();
    this._stopped = false;
  }

  start() {
    // ⚠️ `config` 必须在**下一个微任务**派发，不能同步派发。
    //    真实 WebSocket 的 config 也是异步到达的；若这里同步派发，
    //    `start()` 之后才 `addEventListener("config")` 的调用方会永久漏掉它，
    //    画布就不会被 resize（表现为界面一直停在"等待连接…"）。
    queueMicrotask(() => {
      if (this._stopped) return;
      this._emitConfig();
    });
    this._timers.frame = setInterval(() => this._tick(), this.frameIntervalMs);
    this._timers.status = setInterval(() => this._emitStatus(), STATUS_INTERVAL_MS);
  }

  stop() {
    this._stopped = true;
    for (const k of Object.keys(this._timers)) {
      if (this._timers[k] !== null) clearInterval(this._timers[k]);
      this._timers[k] = null;
    }
  }

  /** 与真实后端相同的 `send(msg)` 接口，便于上层无差别替换。 */
  send(msg) {
    if (!msg || typeof msg.type !== "string") return false;
    if (msg.type === "params") {
      const r = msg.render ?? {};
      if (Array.isArray(r.background)) this.background = r.background.map(Number);
      const src = msg.source ?? {};
      if (src.kind === "checkpoint" || src.kind === "live") {
        this.sourceKind = src.kind;
        if (src.kind === "live") this.playing = false;   // live 下无帧号概念
      }
      const d = msg.drive ?? {};
      this.drive = {
        mode: d.mode ?? this.drive.mode,
        frame: d.frame ?? this.drive.frame,
        amplitude: d.amplitude ?? this.drive.amplitude,
      };
      if ("playing" in d) this.playing = Boolean(d.playing);
      if (d.mode === "still") this.playing = false;   // 定住模式不播放
      if ("frame" in d) {
        this.frameIndex = Math.max(0, Math.min(Number(d.frame), this.totalFrames - 1));
        this.drive.frame = this.frameIndex;
      }
      return true;
    }
    if (msg.type === "stream") {
      this.paused = Boolean(msg.paused);
      return true;
    }
    if (msg.type === "step") {
      // 与后端 FrameTimeline.step 同语义：**边界夹住，不回绕**
      const d = Number(msg.delta ?? 1);
      this.frameIndex = Math.max(0, Math.min(this.frameIndex + d, this.totalFrames - 1));
      this._emitStatus();
      return true;
    }
    if (msg.type === "capture") {
      this.dispatchEvent(new CustomEvent("captured", {
        detail: { type: "captured", path: "（mock 模式不落盘）", name: msg.name },
      }));
      return true;
    }
    return false;
  }

  // ---------------------------------------------------------------- 内部 --

  _emitConfig() {
    this.dispatchEvent(new CustomEvent("config", {
      detail: {
        width: this.geom.width,
        height: this.geom.height,
        channels: this.geom.channels,
        background: this.background,
        numFrames: this.geom.numFrames,
        version: PROTOCOL_VERSION,
        mock: true,
      },
    }));
  }

  _tick() {
    if (this.paused) return;

    // 与后端一致：**出帧之后**推进播放（回绕）。
    // 只有 `play` 模式才推进 —— `still` 定住不动，`random` 是扰动不是播放。
    if (this.playing && this.drive.mode === "play" && this.sourceKind !== "live") {
      this.frameIndex = (this.frameIndex + 1) % this.totalFrames;
      this.drive.frame = this.frameIndex;
    }

    const { width: w, height: h } = this.geom;
    const bytes = HEADER_BYTES + w * h * 3;
    const buffer = new ArrayBuffer(bytes);
    const view = new DataView(buffer);
    this.seq += 1;
    // 与后端一致的 9 字节头：类型 / seq / psnr
    view.setUint8(0, MSG_RENDER);
    view.setUint32(1, this.seq, true);
    view.setFloat32(5, Number.NaN, true);

    const px = new Uint8ClampedArray(buffer, HEADER_BYTES);
    this._renderPattern(px, w, h);

    this._framesInWindow += 1;
    const now = performance.now();
    if (now - this._windowStart >= 1000) {
      this._fps = (this._framesInWindow * 1000) / (now - this._windowStart);
      this._framesInWindow = 0;
      this._windowStart = now;
    }

    this.dispatchEvent(new CustomEvent("frame", {
      detail: { seq: this.seq, pixels: px },
    }));
  }

  /**
   * 画一个会动的"头形"图案 —— 不为好看，只为让界面能看出：
   * 帧在更新、背景色变了、帧号滑块有响应。
   */
  _renderPattern(px, w, h) {
    const [bgR, bgG, bgB] = this.background.map((c) => Math.round(c * 255));
    px.fill(0);
    for (let i = 0; i < px.length; i += 3) {
      px[i] = bgR; px[i + 1] = bgG; px[i + 2] = bgB;
    }

    const cx = w / 2;
    const cy = h / 2;
    const rx = w * 0.22;
    const ry = h * 0.30;
    // 帧号与扰动幅度都会体现在图案上，便于确认参数真的传到了"后端"
    const phase = (this.frameIndex % this.totalFrames) / this.totalFrames;
    const amp = this.drive.mode === "random" ? this.drive.amplitude : 0;
    const wobble = Math.sin(phase * Math.PI * 2) * 0.12 + amp * 0.5;

    const skin = [214, 176, 150];
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const nx = (x - cx) / (rx * (1 + wobble));
        const ny = (y - cy) / ry;
        if (nx * nx + ny * ny <= 1) {
          const i = (y * w + x) * 3;
          // 加一点竖直渐变，避免纯色看不出画面在动
          const shade = 1 - 0.35 * (ny * 0.5 + 0.5);
          px[i] = skin[0] * shade;
          px[i + 1] = skin[1] * shade;
          px[i + 2] = skin[2] * shade;
        }
      }
    }

    // 中心十字线：用于确认帧没有错位/错行
    const mid = Math.floor(w / 2);
    for (let y = 0; y < h; y++) {
      const i = (y * w + mid) * 3;
      px[i] = 90; px[i + 1] = 200; px[i + 2] = 255;
    }
    const midY = Math.floor(h / 2);
    for (let x = 0; x < w; x++) {
      const i = (midY * w + x) * 3;
      px[i] = 90; px[i + 1] = 200; px[i + 2] = 255;
    }
  }

  _emitStatus() {
    const fake = this.paused ? 0 : this._fps;
    this.dispatchEvent(new CustomEvent("status", {
      detail: {
        type: "status",
        seq: this.seq,
        fps: fake,
        ms: {
          frame: fake > 0 ? 1000 / fake : 0,
          deform: 0.0,      // mock 不假装知道真实分解
          rasterize: 0.0,
        },
        dropped: 0,
        peak_memory_mib: 0,
        paused: this.paused,
        frame: this.frameIndex,
        playing: this.playing,
        total_frames: this.totalFrames,
        source: this.sourceKind,
        mock: true,
      },
    }));
  }
}
