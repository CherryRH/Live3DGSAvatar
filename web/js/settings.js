/**
 * 设置浮窗：由后端下发的**参数注册表**驱动渲染。
 *
 * 为什么不在前端硬编码控件：
 * 参数可编辑性、范围、分组、排序的唯一事实来源在后端（`app/settings.py`）。
 * 前端如果各写一份，两边必然漂移 —— 前端放开了后端拒绝的值、
 * 或后端加了参数前端看不见。所以这里只做「按 schema 画控件 + 收集差异」。
 *
 * 布局规则：
 * - **全量展示**：锁定项也列出来，只是不可编辑并显示原因。
 *   不展示会让人以为"没有这个参数"。
 * - 分组内按后端给的 `order` 排序（后端已排好，这里保持顺序）。
 * - 可编辑项用输入控件；锁定项用只读文本 + 锁标记 + 原因。
 *
 * 导出分成两层：
 * - **纯函数**（`groupFields` / `coerceFieldValue` / `buildPatch` / `formatValue`）
 *   —— 不碰 DOM，可在 node 里直接测；
 * - `SettingsPanel` —— 只负责 DOM 与网络。
 */

/** 每个分组前的说明（后端也会给，这里作为兜底）。 */
export const GROUP_FALLBACK = {
  render: "影响出图与帧率，改完立即生效",
  train: "外观微调的超参与门控",
  system: "对象、路径与设备（结构性，改动需重启或重载）",
};

/**
 * 把扁平字段列表按分组组织，**保持后端给的顺序**。
 *
 * @param {Array<object>} fields 后端 `fields`
 * @param {Array<{key:string,title:string,help:string,order:number}>} groups
 * @returns {Array<{key:string, title:string, help:string, fields:Array<object>}>}
 */
export function groupFields(fields, groups) {
  const meta = new Map((groups ?? []).map((g) => [g.key, g]));
  const buckets = new Map();

  for (const f of fields ?? []) {
    if (!buckets.has(f.group)) buckets.set(f.group, []);
    buckets.get(f.group).push(f);
  }

  // 分组的先后由后端的 order 决定；没有元数据的分组排到最后。
  const keys = [...buckets.keys()].sort((a, b) => {
    const oa = meta.get(a)?.order ?? Number.MAX_SAFE_INTEGER;
    const ob = meta.get(b)?.order ?? Number.MAX_SAFE_INTEGER;
    return oa - ob;
  });

  return keys.map((key) => ({
    key,
    title: meta.get(key)?.title ?? key,
    // 页签上要显示这个分组对应哪份配置文件
    file: meta.get(key)?.file ?? "",
    help: meta.get(key)?.help ?? GROUP_FALLBACK[key] ?? "",
    fields: buckets.get(key).slice().sort((a, b) => (a.order ?? 0) - (b.order ?? 0)),
  }));
}

/**
 * 把输入控件的原始值转成后端接受的类型。
 *
 * 只做**类型**转换；范围校验交给后端（它才知道真正的约束，
 * 且未来改范围不需要动前端）。但明显非法的输入要在这里就拦住并给出提示，
 * 否则会发出一个必然被拒的请求。
 *
 * @throws {Error} 空值 / 非数字
 */
export function coerceFieldValue(field, raw) {
  switch (field.kind) {
    case "bool":
      return Boolean(raw);

    case "int": {
      // ⚠️ 空字符串必须**拒绝**：`Number("")` 是 0，会把"清空输入框"
      //    静默变成 0（学习率变 0、帧数上限变 0 = 不限速）。
      const n = Number(blankToNaN(raw));
      if (!Number.isFinite(n)) throw new Error(`${field.label}：需要整数`);
      if (!Number.isInteger(n)) throw new Error(`${field.label}：需要整数`);
      return n;
    }

    case "float": {
      const n = Number(blankToNaN(raw));
      if (!Number.isFinite(n)) throw new Error(`${field.label}：需要数字`);
      return n;
    }

    case "color": {
      const arr = Array.isArray(raw) ? raw : [];
      if (arr.length !== 3) throw new Error(`${field.label}：需要 3 个分量`);
      return arr.map((c) => {
        const n = Number(blankToNaN(c));
        if (!Number.isFinite(n)) throw new Error(`${field.label}：分量必须是数字`);
        return n;
      });
    }

    default:
      return String(raw);
  }
}

/**
 * 只挑出**真正改动过**的项，构成 `patch`。
 *
 * 为什么只发差异：后端会把这批值写进 `local.yaml` 作为覆盖项。
 * 若每次保存都把全部字段写进去，`local.yaml` 会变成整份配置的副本，
 * 之后改默认值就被它永久盖住 —— 而且没人看得出「哪些是我改的」。
 */
export function buildPatch(original, current) {
  const patch = {};
  for (const [key, value] of Object.entries(current ?? {})) {
    if (!sameValue(original?.[key], value)) patch[key] = value;
  }
  return patch;
}

/** 值相等判断：数组按元素比（颜色是 3 分量）。 */
export function sameValue(a, b) {
  if (Array.isArray(a) && Array.isArray(b)) {
    return a.length === b.length && a.every((x, i) => Number(x) === Number(b[i]));
  }
  if (a === null || a === undefined || b === null || b === undefined) return a === b;
  if (typeof a === "number" || typeof b === "number") return Number(a) === Number(b);
  return a === b;
}

/** 锁定项显示的文本值（数组用逗号连接，布尔转中文）。 */
export function formatValue(field) {
  const v = field.value;
  if (v === null || v === undefined) return "—";
  if (field.kind === "bool") return v ? "开" : "关";
  if (Array.isArray(v)) return v.map((x) => Number(x).toFixed(2)).join(", ");
  if (field.kind === "float") {
    const n = Number(v);
    return Number.isInteger(n) ? String(n) : n.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
  }
  return String(v);
}

/** 空 / null / undefined → NaN，让 `Number.isFinite` 能正常拦住。 */
function blankToNaN(raw) {
  if (raw === null || raw === undefined) return NaN;
  if (typeof raw === "string" && raw.trim() === "") return NaN;
  return raw;
}

/** 数值输入框的步长：后端没给就用类型的合理默认。 */
export function stepFor(field) {
  if (field.step !== null && field.step !== undefined) return field.step;
  return field.kind === "int" ? 1 : 0.01;
}


// ------------------------------------------------------------- DOM 层 --

export class SettingsPanel {
  /**
   * @param {Document|HTMLElement} root
   * @param {{onApplied?:Function, onError?:Function}} handlers
   */
  constructor(root, handlers = {}) {
    this.root = root;
    this.handlers = handlers;
    this.el = {
      button: root.querySelector("#btn-settings"),
      dialog: root.querySelector("#settings-dialog"),
      body: root.querySelector("#settings-body"),
      tabs: root.querySelector("#settings-tabs"),
      close: root.querySelector("#btn-settings-close"),
      reset: root.querySelector("#btn-settings-reset"),
      save: root.querySelector("#btn-settings-save"),
      status: root.querySelector("#settings-status"),
      source: root.querySelector("#settings-source"),
    };
    /** 后端最近一次下发的字段（用于重置与差异比较）。 */
    this.fields = [];
    this.groups = [];
    /** 当前编辑中的值：key → value。 */
    this.draft = new Map();
    this.loaded = false;
    /** 当前页签（分组键）。默认第一个（渲染）。 */
    this.active = null;
    this._wire();
  }

  // ------------------------------------------------------------ 生命周期 --

  _wire() {
    this.el.button?.addEventListener("click", () => this.toggle());
    this.el.close?.addEventListener("click", () => this.close());
    this.el.reset?.addEventListener("click", () => this.reset());
    this.el.save?.addEventListener("click", () => this.save());
    // 点遮罩关闭；点面板内部不关
    this.el.dialog?.addEventListener("click", (ev) => {
      if (ev.target === this.el.dialog) this.close();
    });
    this.root.addEventListener?.("keydown", (ev) => {
      if (ev.key === "Escape" && this.isOpen) this.close();
    });
  }

  get isOpen() {
    return this.el.dialog && !this.el.dialog.hidden;
  }

  async toggle() {
    if (this.isOpen) return this.close();
    return this.open();
  }

  async open() {
    if (!this.loaded) {
      const ok = await this.reload();
      if (!ok) return;                    // 拉不到 schema 就不要开一个空白窗
    }
    if (this.el.dialog) this.el.dialog.hidden = false;
    if (this.el.button) this.el.button.setAttribute("aria-expanded", "true");
  }

  close() {
    if (this.el.dialog) this.el.dialog.hidden = true;
    if (this.el.button) this.el.button.setAttribute("aria-expanded", "false");
  }

  /** 拉取 schema 与当前值。返回是否成功。 */
  async reload() {
    this._status("加载中…");
    try {
      const resp = await fetch("/api/settings");
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const body = await resp.json();
      this._absorb(body);
      this._status("");
      return true;
    } catch (err) {
      this._status(`无法读取设置：${err.message}`, true);
      this.handlers.onError?.(err);
      return false;
    }
  }

  _absorb(body) {
    this.fields = body.fields ?? [];
    this.groups = body.groups ?? [];
    this.draft.clear();
    for (const f of this.fields) this.draft.set(f.key, f.value);
    if (this.el.source) {
      // 完整路径属于内部细节；这里只说"改动会保存"，路径留到悬停提示
      this.el.source.textContent = body.overrides_path ? "改动会保存" : "";
      this.el.source.title = body.overrides_path ?? "";
    }
    this.loaded = true;
    // 保持用户当前所在的页签；首次或该分组消失时回落到第一个
    const keys = groupFields(this.fields, this.groups).map((g) => g.key);
    if (!this.active || !keys.includes(this.active)) this.active = keys[0] ?? null;
    this.render();
  }

  // -------------------------------------------------------------- 渲染 --

  render() {
    const host = this.el.body;
    if (!host) return;
    host.replaceChildren();

    const grouped = groupFields(this.fields, this.groups);
    this._renderTabs(grouped);

    // 只渲染**当前页签**：三个 config 全展开会很长（50+ 项）。
    for (const group of grouped) {
      if (group.key !== this.active) continue;
      const section = document.createElement("section");
      section.className = "sgroup";
      section.dataset.group = group.key;

      const head = document.createElement("h3");
      head.className = "sgroup__title";
      head.textContent = group.title;
      if (group.help) {
        const hint = document.createElement("span");
        hint.className = "sgroup__hint";
        hint.textContent = group.help;
        head.append(" ", hint);
      }
      section.append(head);

      for (const field of group.fields) section.append(this._row(field));
      host.append(section);
    }
  }

  /** 页签：每个配置文件一个，显示标题 + 文件名 + 可编辑项数。 */
  _renderTabs(grouped) {
    const bar = this.el.tabs;
    if (!bar) return;
    bar.replaceChildren();

    for (const group of grouped) {
      const editable = group.fields.filter((f) => f.editable).length;
      const btn = document.createElement("button");
      btn.className = "tab" + (group.key === this.active ? " tab--active" : "");
      btn.setAttribute("role", "tab");
      btn.setAttribute("aria-selected", group.key === this.active ? "true" : "false");
      btn.dataset.group = group.key;

      const title = document.createElement("span");
      title.className = "tab__title";
      title.textContent = group.title;
      btn.append(title);

      // 配置文件（render.yaml 等）是内部细节：不展示，只留悬停提示
      if (group.file) btn.title = group.file;

      const count = document.createElement("span");
      count.className = "tab__count";
      count.textContent = editable ? `${editable} 项可改` : "只读";
      btn.append(count);

      btn.addEventListener("click", () => {
        if (this.active === group.key) return;
        this.active = group.key;
        this.render();
      });
      bar.append(btn);
    }
  }

  _row(field) {
    const row = document.createElement("div");
    row.className = "srow" + (field.editable ? "" : " srow--locked");
    row.dataset.key = field.key;

    const label = document.createElement("div");
    label.className = "srow__label";
    const name = document.createElement("span");
    name.className = "srow__name";
    name.textContent = field.label;
    label.append(name);
    if (field.help) {
      const help = document.createElement("span");
      help.className = "srow__help";
      help.textContent = field.help;
      label.append(help);
    }
    if (!field.editable && field.reason) {
      const why = document.createElement("span");
      why.className = "srow__reason";
      why.textContent = `🔒 ${field.reason}`;
      label.append(why);
    }
    row.append(label);

    const value = document.createElement("div");
    value.className = "srow__value";
    if (field.editable) {
      value.append(this._control(field));
    } else {
      const text = document.createElement("span");
      text.className = "srow__readonly";
      text.textContent = formatValue(field);
      value.append(text);
    }
    row.append(value);

    // 配置键名属于内部细节：不显示，只留悬停提示（排查时仍可看到）
    row.title = field.key;

    return row;
  }

  _control(field) {
    const current = this.draft.get(field.key);

    if (field.kind === "bool") {
      const input = document.createElement("input");
      input.type = "checkbox";
      input.checked = Boolean(current);
      input.addEventListener("change", () => {
        this.draft.set(field.key, input.checked);
        this._dirty();
      });
      return input;
    }

    if (field.kind === "enum") {
      const select = document.createElement("select");
      for (const choice of field.choices ?? []) {
        const opt = document.createElement("option");
        opt.value = choice;
        opt.textContent = choice;
        if (choice === current) opt.selected = true;
        select.append(opt);
      }
      select.addEventListener("change", () => {
        this.draft.set(field.key, select.value);
        this._dirty();
      });
      return select;
    }

    if (field.kind === "color") {
      const wrap = document.createElement("div");
      wrap.className = "scolor";
      const arr = Array.isArray(current) ? current : [0, 0, 0];
      for (let i = 0; i < 3; i++) {
        const input = document.createElement("input");
        input.type = "number";
        input.min = "0";
        input.max = "1";
        input.step = stepFor(field);
        input.value = arr[i];
        input.addEventListener("input", () => {
          const next = [...(this.draft.get(field.key) ?? [0, 0, 0])];
          next[i] = Number(input.value);
          this.draft.set(field.key, next);
          this._dirty();
        });
        wrap.append(input);
      }
      return wrap;
    }

    if (field.kind === "int" || field.kind === "float") {
      const input = document.createElement("input");
      input.type = "number";
      if (field.minimum !== null && field.minimum !== undefined) {
        input.min = String(field.minimum);
      }
      if (field.maximum !== null && field.maximum !== undefined) {
        input.max = String(field.maximum);
      }
      input.step = String(stepFor(field));
      input.value = String(current ?? "");
      input.addEventListener("input", () => {
        this.draft.set(field.key, input.value === "" ? "" : Number(input.value));
        this._dirty();
      });
      return field.unit ? wrapWithUnit(input, field.unit) : input;
    }

    const input = document.createElement("input");
    input.type = "text";
    input.value = current ?? "";
    input.addEventListener("input", () => {
      this.draft.set(field.key, input.value);
      this._dirty();
    });
    return input;
  }

  // -------------------------------------------------------------- 保存 --

  /** 当前草稿与后端的差异（只有可编辑项会被收集）。 */
  collectPatch() {
    const current = {};
    for (const f of this.fields) {
      if (!f.editable) continue;
      current[f.key] = this.draft.get(f.key);
    }
    return buildPatch(
      Object.fromEntries(this.fields.filter((f) => f.editable).map((f) => [f.key, f.value])),
      current,
    );
  }

  async save() {
    const raw = this.collectPatch();
    if (Object.keys(raw).length === 0) {
      this._status("没有改动");
      return;
    }

    // 先在前端做一次类型转换：明显非法的输入直接拦住，
    // 不必往返一次再收到 400。
    const patch = {};
    try {
      for (const f of this.fields) {
        if (f.key in raw) patch[f.key] = coerceFieldValue(f, raw[f.key]);
      }
    } catch (err) {
      this._status(err.message, true);
      return;
    }

    this._status("保存中…");
    this.el.save.disabled = true;
    try {
      const resp = await fetch("/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ patch }),
      });
      const body = await resp.json().catch(() => ({}));
      if (!resp.ok) throw new Error(body.detail ?? body.error ?? `HTTP ${resp.status}`);

      this._absorb(body);
      if (body.warning) {
        this._status(body.warning, true);
      } else {
        const n = Object.keys(patch).length;
        this._status(`已保存 ${n} 项并应用`);
      }
      this.handlers.onApplied?.(body);
    } catch (err) {
      this._status(`保存失败：${err.message}`, true);
    } finally {
      this.el.save.disabled = false;
    }
  }

  /** 放弃未保存的修改，回到后端最近一次下发的值。 */
  reset() {
    this.draft.clear();
    for (const f of this.fields) this.draft.set(f.key, f.value);
    this.render();
    this._status("已放弃未保存的修改");
  }

  _dirty() {
    const n = Object.keys(this.collectPatch()).length;
    this._status(n ? `${n} 项未保存` : "");
  }

  _status(text, isError = false) {
    if (!this.el.status) return;
    this.el.status.textContent = text ?? "";
    this.el.status.classList.toggle("is-error", Boolean(isError));
  }
}

function wrapWithUnit(input, unit) {
  const wrap = document.createElement("div");
  wrap.className = "sunit";
  const suffix = document.createElement("span");
  suffix.className = "sunit__suffix";
  suffix.textContent = unit;
  wrap.append(input, suffix);
  return wrap;
}
