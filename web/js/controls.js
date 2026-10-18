/**
 * 参数面板与播放条 ↔ 消息装配。
 *
 * 分工原则：
 * - **控件只负责"读值"**；消息统一在 `_emitParams` 里装配，避免每个控件各写一遍。
 * - **帧号与播放状态以服务端为准**（`status.frame` / `status.playing`），
 *   前端只回填，不自己维护一份 —— 否则两端必然漂移。
 * - **这里只管"不持久化的实时操作"**：播放、单步、数据源、扰动、暂停、截图。
 *   渲染参数（背景色/缩放/分辨率/帧数上限）在设置面板里，走 `/api/settings`
 *   并持久化到 `configs/local.yaml`。两边同时管会互相覆盖 —— 曾经就是这样。
 *
 * 见 docs/GUI_PROTOCOL.md §3.2。
 */

/** 面板默认值。 */
export const DEFAULTS = {
  sourceKind: "checkpoint",
  frame: 0,
  playing: false,
  perturb: false,
  amplitude: 0.1,
};

export class Controls {
  /**
   * @param {Document|HTMLElement} root
   * @param {{onParams:Function, onLoadModel:Function, onPause:Function,
   *          onCapture:Function, onReconnect:Function, onStep:Function}} handlers
   */
  constructor(root, handlers) {
    this.root = root;
    this.handlers = handlers;
    this.el = {
      // 模型
      subject: root.querySelector("#f-subject"),
      workName: root.querySelector("#f-work-name"),
      loadModel: root.querySelector("#btn-load"),
      modelInfo: root.querySelector("#model-info"),
      // 播放条
      player: root.querySelector("#player"),
      play: root.querySelector("#btn-play"),
      stepBack: root.querySelector("#btn-step-back"),
      stepFwd: root.querySelector("#btn-step-fwd"),
      frame: root.querySelector("#f-frame"),
      frameOut: root.querySelector("#o-frame"),
      total: root.querySelector("#o-total"),
      source: root.querySelector("#f-source"),
      liveStatus: root.querySelector("#live-status"),
      amp: root.querySelector("#f-amp"),
      ampOut: root.querySelector("#o-amp"),
      perturb: root.querySelector("#f-perturb"),
      // 操作
      pause: root.querySelector("#btn-pause"),
      capture: root.querySelector("#btn-capture"),
      reconnect: root.querySelector("#btn-reconnect"),
    };
    this.numFrames = null;
    this._playing = false;
    this._dragging = false;
    this._wire();
    this._renderOutputs();
  }

  // ------------------------------------------------------------ 状态同步 --

  /** 用 `/api/state` 的返回值初始化控件。 */
  applyState(state) {
    if (!state) return;
    const el = this.el;
    if (state.subject !== undefined) el.subject.value = state.subject;
    if (state.work_name !== undefined) el.workName.value = state.work_name;

    const model = state.model;
    if (model) {
      this.showModelInfo(
        `${model.num_gaussians ?? "?"} 高斯 · K=${model.num_basis ?? "?"} · ` +
        `D=${model.num_basis_in ?? "?"} · tex=${model.tex_size ?? "?"}`);
    }
    // ⚠️ 不再回填 render.* —— 这些参数的唯一 UI 在设置面板，
    //    这里若也写一份，两边会互相覆盖（历史 bug）。
    // 数据源：刷新页面后要恢复成后端正在用的那个
    const kind = state.render?.source_kind;
    if (typeof kind === "string"
        && [...el.source.options].some((o) => o.value === kind)) {
      el.source.value = kind;
    }
    const drive = state.drive;
    if (drive) {
      if (typeof drive.frame === "number") el.frame.value = String(drive.frame);
      if (drive.amplitude !== undefined) el.amp.value = drive.amplitude;
      if (typeof drive.playing === "boolean") this._setPlayButton(drive.playing);
      if (typeof drive.perturb === "boolean") el.perturb.checked = drive.perturb;
    }
    if (state.live) this.applyLiveStatus(state.live);
    if (state.dataset?.num_frames) this.setFrameRange(state.dataset.num_frames);
    this._renderOutputs();
  }

  /** 数据集总帧数确定后设置滑块范围。 */
  setFrameRange(n) {
    this.numFrames = n;
    this.el.frame.max = String(Math.max(0, n - 1));
    this.el.total.textContent = String(n);
    if (Number(this.el.frame.value) > n - 1) this.el.frame.value = String(n - 1);
    this._renderOutputs();
  }

  setPaused(paused) {
    this.paused = paused;
    this.el.pause.textContent = paused ? "继续推流" : "暂停推流";
  }

  showModelInfo(text) { this.el.modelInfo.textContent = text; }
  showModelError(text) { this.el.modelInfo.textContent = text; }

  /**
   * 用**服务端权威**的帧号/播放状态回填控件。
   * `frame` 在拖动过程中不回填，否则会和手指打架。
   */
  applyPlayback({ frame, playing }) {
    if (typeof frame === "number" && !this._dragging) {
      this.el.frame.value = String(frame);
      this._renderOutputs();
    }
    if (typeof playing === "boolean") this._setPlayButton(playing);
  }

  // ---------------------------------------------------------------- 内部 --

  _setPlayButton(playing) {
    this._playing = playing;
    this.el.play.textContent = playing ? "⏸" : "▶";
    this.el.play.classList.toggle("is-on", playing);
  }

  _wire() {
    const el = this.el;

    // ---- 模型 ----
    el.loadModel.addEventListener("click", () => {
      const subject = el.subject.value.trim();
      const workName = el.workName.value.trim();
      if (!subject || !workName) {
        this.showModelError("subject 与 work_name 都不能为空");
        return;
      }
      this.handlers.onLoadModel({ subject, work_name: workName });
    });

    // ---- 播放条 ----
    el.play.addEventListener("click", () => this._setPlaying(!this._playing));

    el.stepBack.addEventListener("click", () => this.handlers.onStep(-1));
    el.stepFwd.addEventListener("click", () => this.handlers.onStep(+1));

    // 拖动帧滑块：拖动期间标记，避免服务端回填与手指打架；松手才发消息
    el.frame.addEventListener("pointerdown", () => { this._dragging = true; });
    const endDrag = () => {
      if (!this._dragging) return;
      this._dragging = false;
      this._emitParams();
    };
    el.frame.addEventListener("pointerup", endDrag);
    el.frame.addEventListener("pointercancel", endDrag);
    el.frame.addEventListener("change", endDrag);
    // 键盘操作滑块（无 pointer 事件）时也要生效
    el.frame.addEventListener("input", () => {
      this._renderOutputs();
      if (!this._dragging) this._emitParams();
    });

    // 数据源：切到 live 时把驱动相关控件置灰
    el.source.addEventListener("change", () => {
      this._renderOutputs();
      this._emitParams();
    });
    el.perturb.addEventListener("change", () => {
      this._renderOutputs();
      this._emitParams();
    });
    el.amp.addEventListener("input", () => { this._renderOutputs(); this._emitParams(); });

    // ---- 操作 ----
    el.pause.addEventListener("click", () => this.handlers.onPause(!this.paused));
    el.capture.addEventListener("click", () => this.handlers.onCapture());
    el.reconnect.addEventListener("click", () => this.handlers.onReconnect());

    // 键盘：← / → 单步，空格播放/暂停（审查时比点按钮快）
    this._onKey = (ev) => {
      const t = ev.target;
      if (t instanceof HTMLInputElement || t instanceof HTMLSelectElement) return;
      if (ev.key === "ArrowLeft") { ev.preventDefault(); this.handlers.onStep(-1); }
      else if (ev.key === "ArrowRight") { ev.preventDefault(); this.handlers.onStep(+1); }
      else if (ev.key === " ") { ev.preventDefault(); this._setPlaying(!this._playing); }
    };
    (eventTarget(this.root)).addEventListener("keydown", this._onKey);
  }

  _setPlaying(playing) {
    this._setPlayButton(playing);
    // 走统一的 `_emitParams`，而不是只发 `{playing}`。
    // 这样能把当前帧号一起带上 —— 否则「拖了帧滑块还没松手就点播放」
    // 会丢失刚选的帧号（拖动期间不发消息，详见 `_dragging` 的处理）。
    // 播放推进本身仍由**服务端**负责：渲染循环每出一帧推进一次。
    this._emitParams();
  }

  /**
   * 显示实时驱动状态。**只做显示，不禁用控件** ——
   * live 不可用时后端会返回明确错误，让用户看到"为什么不行"，
   * 比把下拉框置灰（点不动、也没提示）更有用。
   */
  applyLiveStatus(live) {
    const el = this.el.liveStatus;
    if (!el) return;
    const usingLive = this.el.source.value === "live";
    el.hidden = !usingLive;
    if (!usingLive) return;

    // 回退态优先提示：此时画面上是**数据集首帧**，不是摄像头画面 ——
    // 不明确说明的话，用户会以为摄像头在工作。
    if (live.fallback) {
      el.textContent = `⚠ 显示首帧（${live.reason ?? "摄像头未就绪"}）`;
    } else if (!live.enabled) {
      el.textContent = "实时驱动已在配置里关闭";
    } else if (!live.connected) {
      el.textContent = "未连接桥接进程（请先运行 scripts/rdf_bridge.py）";
    } else if (!live.available) {
      el.textContent = `已连接，但 ${live.max_age_s}s 内没有新帧`;
    } else {
      el.textContent = `RDF ${live.fps ?? 0} FPS · 帧间隔 ${live.age_s ?? 0}s`;
    }
  }

  /** 组装并发送一条 `params` 消息。 */
  _emitParams() {
    const live = this.el.source.value === "live";

    const drive = {};
    if (!live) {
      // **没有** drive.mode：是否推进帧号只由 `playing` 表达。
      // 暂停 ≠ 停止渲染，只是帧号停住（渲染仍按帧数上限全速进行）。
      drive.frame = Number(this.el.frame.value);
      drive.playing = this._playing;
      drive.perturb = this.el.perturb.checked;
      drive.amplitude = Number(this.el.amp.value);
    }

    // **不发 render**：渲染参数由设置面板经 `/api/settings` 修改。
    // 若这里也发，会在用户拖动播放条时把设置面板刚存的值覆盖回旧值。
    this.handlers.onParams({
      type: "params",
      source: { kind: this.el.source.value },
      drive,
    });
  }

  _renderOutputs() {
    const el = this.el;
    el.frameOut.textContent = el.frame.value;
    el.ampOut.textContent = Number(el.amp.value).toFixed(2);

    const live = el.source.value === "live";
    el.player.classList.toggle("is-live", live);

    el.perturb.disabled = live;
    el.amp.disabled = live || !el.perturb.checked;
    // live 下帧号不适用
    el.frame.disabled = live;
    el.stepBack.disabled = live;
    el.stepFwd.disabled = live;
  }

  destroy() {
    if (this._onKey) {
      (eventTarget(this.root)).removeEventListener("keydown", this._onKey);
    }
  }
}

/**
 * 取"挂键盘监听的宿主"。
 *
 * 浏览器里 `root` 就是 `document`；但测试用最小桩时它只是个普通对象，
 * 那种情况下退回全局 `document`（测试会提供）。
 */
function eventTarget(root) {
  if (typeof root.addEventListener === "function") return root;
  if (typeof document !== "undefined" && typeof document.addEventListener === "function") {
    return document;
  }
  throw new Error("找不到可挂载键盘监听的宿主");
}
