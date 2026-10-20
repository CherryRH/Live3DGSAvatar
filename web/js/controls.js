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
      dataset: root.querySelector("#f-dataset"),
      datasetInfo: root.querySelector("#dataset-info"),
      liveStatus: root.querySelector("#live-status"),
      amp: root.querySelector("#f-amp"),
      ampOut: root.querySelector("#o-amp"),
      perturb: root.querySelector("#f-perturb"),
      compare: root.querySelector("#f-compare"),
      psnr: root.querySelector("#psnr-out"),
      // 操作
      pause: root.querySelector("#btn-pause"),
      capture: root.querySelector("#btn-capture"),
      reconnect: root.querySelector("#btn-reconnect"),
    };
    this.numFrames = null;
    //: 最近一次 `applyDatasets()` 拿到的列表（用于提示文案与合法性判断）
    this._datasets = [];
    //: 最近一次 `applyModels()` 拿到的列表（对象/版本两个下拉框共用）
    this._models = [];
    this._playing = false;
    this._dragging = false;
    this._wire();
    this._renderOutputs();
  }

  // ------------------------------------------------------------ 状态同步 --

  /**
   * 填模型下拉框（对象 → 版本）。
   *
   * 用下拉而不是手输：subject/work_name 打错一个字就是一次失败的加载，
   * 而且用户没有理由记住目录名。只列**含 model.ply** 的目录。
   */
  applyModels(payload) {
    if (!payload) return;
    this._models = payload.models ?? [];
    const subjects = [...new Set(this._models.map((m) => m.subject))];
    const el = this.el.subject;
    if (typeof el?.replaceChildren === "function") {
      el.replaceChildren(...subjects.map((s) => new Option(s, s)));
    }
    const want = payload.current?.subject;
    if (want && subjects.includes(want)) el.value = want;
    this._fillWorkNames(payload.current?.work_name);
  }

  _fillWorkNames(want) {
    const el = this.el.subject;
    const names = this._models
      .filter((m) => m.subject === el?.value)
      .map((m) => m.work_name);
    const target = this.el.workName;
    if (typeof target?.replaceChildren === "function") {
      target.replaceChildren(...names.map((n) => new Option(n, n)));
    }
    if (want && names.includes(want)) target.value = want;
    else if (names.length) target.value = names[0];
  }

  /**
   * 填数据集下拉框。
   *
   * 为什么用 `Option` + `replaceChildren` 而不是拼 `innerHTML`：
   * 测试桩里没有 HTML 解析器，拼字符串就只能在真实浏览器里验证；
   * 用 DOM API 后同一段逻辑在桩里也能测。
   */
  applyDatasets(payload) {
    if (!payload) return;
    this._datasets = payload.datasets ?? [];
    const el = this.el.dataset;
    if (!el || typeof el.replaceChildren !== "function") return;

    el.replaceChildren(...this._datasets.map((d) => new Option(
      d.is_model ? `${d.name}（模型自带）` : d.name, d.name)));

    const want = payload.current || payload.model_subject || "";
    if (want && this._datasets.some((d) => d.name === want)) el.value = want;
    else if (this._datasets.length) el.value = this._datasets[0].name;
    // 填完列表时的值是"后端当前在用的" —— 记下来，切换失败时据此回滚
    this._committedDataset = el.value ?? "";
    this._renderDatasetInfo();
  }

  /** 数据集切换成功后调用：记住新的"已提交"值。 */
  commitDataset() {
    this._committedDataset = this.el.dataset?.value ?? "";
  }

  /** 切换失败后调用：下拉框退回上一次成功的值，界面不显示错的选择。 */
  revertDataset() {
    const back = this._committedDataset;
    if (back !== undefined && this.el.dataset) this.el.dataset.value = back;
    this._renderDatasetInfo();
  }

  _renderDatasetInfo() {
    const el = this.el.datasetInfo;
    if (!el) return;
    const item = this._datasets.find((d) => d.name === this.el.dataset?.value);
    if (!item) {
      el.textContent = "无可用数据集";
      return;
    }
    const live = this.el.source.value === "live";
    const how = live ? "实时追踪" : "数据集参数";
    // 两个画面目录都没有才真的不能实时驱动（images/ 是回退来源）
    const missing = live && !item.has_input && !item.has_images
      ? " · ⚠ 无画面" : "";
    el.textContent = `${item.frames} 帧 · ${how}${missing}`;
  }

  /** 用 `/api/state` 的返回值初始化控件。 */
  applyState(state) {
    if (!state) return;
    const el = this.el;
    // 下拉框里没有的选项不能直接赋值（会变成空值），先判断存在。
    // ⚠️ **空字符串一律忽略**：模型未加载时后端返回 subject=""，
    //    若照写会把已经自动选好的第一项清空（曾经因此"对象选了、版本空着"）。
    const subjectOk = Boolean(state.subject)
      && [...(el.subject.options ?? [])].some((o) => o.value === state.subject);
    if (subjectOk) {
      el.subject.value = state.subject;
      this._fillWorkNames(state.work_name);
    } else if (Boolean(state.work_name)
        && [...(el.workName.options ?? [])].some((o) => o.value === state.work_name)) {
      el.workName.value = state.work_name;
    }

    const model = state.model;
    if (model) {
      // 只给一个用户看得懂的规模数字；结构参数（K/D/tex）属于开发细节
      this.showModelInfo(
        model.num_gaussians ? `${model.num_gaussians} 个高斯` : "");
    }
    // ⚠️ 不再回填 render.* —— 这些参数的唯一 UI 在设置面板，
    //    这里若也写一份，两边会互相覆盖（历史 bug）。
    // 数据源：刷新页面后要恢复成后端正在用的那个
    const kind = state.render?.source_kind;
    if (typeof kind === "string"
        && [...el.source.options].some((o) => o.value === kind)) {
      el.source.value = kind;
    }
    // 驱动数据集：空串表示"跟随模型"，此时下拉框保留列表默认项
    const driveDataset = state.render?.drive_dataset;
    if (typeof driveDataset === "string" && driveDataset
        && [...(el.dataset?.options ?? [])].some((o) => o.value === driveDataset)) {
      el.dataset.value = driveDataset;
    }
    // 记下"后端已确认的"数据集：切换失败时据此回滚下拉框
    this._committedDataset = el.dataset?.value ?? "";
    if (typeof state.render?.compare === "boolean" && el.compare) {
      el.compare.checked = state.render.compare;
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
    // 换对象要重填"版本"下拉框（同一对象可能有多个 work_name）
    el.subject?.addEventListener("change", () => this._fillWorkNames());
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
    // 换数据集要重载 scene（几百毫秒到几秒，还可能失败）→ 走独立回调，
    // 由 main 显示加载浮窗并用 HTTP 请求等待结果，而不是塞进 WS 的 params。
    el.dataset?.addEventListener("change", () => {
      const name = el.dataset.value;
      this._committedDataset ??= name;
      this._renderDatasetInfo();
      if (this.handlers.onDatasetChange) this.handlers.onDatasetChange(name);
    });
    // 对比开关：**立即**重发 params（后端据此决定要不要多送一帧）
    el.compare?.addEventListener("change", () => {
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
   * 更新 PSNR 读数（来自帧头，与画面同一帧号）。
   *
   * 只在"对比"开启时显示 —— 关掉对比还留着一个数字会让人以为它还在算。
   */
  applyPsnr(value) {
    const el = this.el.psnr;
    if (!el) return;
    const on = Boolean(this.el.compare?.checked);
    el.hidden = !on;
    if (!on) return;
    el.textContent = Number.isFinite(value) ? `PSNR ${value.toFixed(2)} dB` : "PSNR –";
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
      el.textContent = `⚠ 显示首帧 · ${live.reason ?? "等待追踪服务"}`;
    } else if (!live.enabled) {
      el.textContent = "实时驱动已关闭";
    } else if (!live.connected) {
      el.textContent = "未连接追踪服务";
    } else if (!live.available) {
      el.textContent = "已连接，等待画面";
    } else {
      el.textContent = `${live.fps ?? 0} FPS`;
    }
  }

  /** 组装并发送一条 `params` 消息。 */
  _emitParams() {
    const live = this.el.source.value === "live";

    // **没有** drive.mode：是否推进帧号只由 `playing` 表达。
    // 暂停 ≠ 停止渲染，只是帧号停住（渲染仍按帧数上限全速进行）。
    // live 模式下这些字段同样有效；后端只忽略 perturb/amplitude。
    const drive = {
      frame: Number(this.el.frame.value),
      playing: this._playing,
      perturb: this.el.perturb.checked,
      amplitude: Number(this.el.amp.value),
    };

    // **不发 render**：渲染参数由设置面板经 `/api/settings` 修改。
    // 若这里也发，会在用户拖动播放条时把设置面板刚存的值覆盖回旧值。
    this.handlers.onParams({
      type: "params",
      // 运行期开关：是否并排显示数据集原图（不持久化）
      compare: Boolean(this.el.compare?.checked),
      source: {
        kind: this.el.source.value,
        // 空串 = 跟随模型自带的数据集（列表还没加载时保持这个语义）
        dataset: this.el.dataset?.value || "",
      },
      drive,
    });
  }

  _renderOutputs() {
    const el = this.el;
    // 对比开启时给 stage 加类，由 CSS 改成"目标 | 渲染"并排
    // 防御：测试桩/嵌入场景里 document 可能是部分实现，缺 getElementById
    const stage = typeof document !== "undefined" && document.getElementById
      ? document.getElementById("stage") : null;
    if (stage) stage.classList.toggle("is-compare", Boolean(el.compare?.checked));
    if (el.psnr) el.psnr.hidden = !el.compare?.checked;
    el.frameOut.textContent = el.frame.value;
    el.ampOut.textContent = Number(el.amp.value).toFixed(2);

    const live = el.source.value === "live";
    el.player.classList.toggle("is-live", live);

    // **两种驱动方式都支持播放/暂停/跳帧**：
    // live 模式下取帧线程按帧号送图（暂停重送同一帧、跳帧立刻送目标帧），
    // 所以这里不再禁用帧号控件。
    el.perturb.disabled = live;
    el.amp.disabled = live || !el.perturb.checked;
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
