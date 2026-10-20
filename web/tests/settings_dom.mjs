/**
 * 设置面板的 **DOM 渲染路径**冒烟测试（用最小桩，不需要浏览器）。
 *
 * 纯逻辑在 `settings.mjs` 里测过了。这里测的是"按 schema 画控件"这段：
 * 它最容易出的错是 **静默的** —— 某个字段没画出来、锁定项也给了输入框、
 * 分组顺序反了。这些都不会抛异常，只是面板少了一块，用户以为参数不存在。
 *
 * 桩只实现 SettingsPanel 真正用到的方法；不模拟浏览器行为。
 */

import { SettingsPanel, groupFields } from "../js/settings.js";

let pass = 0;
const fails = [];
const t = (name, fn) => {
  try { fn(); pass++; console.log(`  \x1b[32m✓\x1b[0m ${name}`); }
  catch (e) { fails.push([name, e]); console.log(`  \x1b[31m✗\x1b[0m ${name}\n      ${e.message}`); }
};
const eq = (a, b, m) => {
  if (JSON.stringify(a) !== JSON.stringify(b)) {
    throw new Error(`${m ?? "不相等"}: 期望 ${JSON.stringify(b)} 实际 ${JSON.stringify(a)}`);
  }
};
const ok = (c, m) => { if (!c) throw new Error(m ?? "断言失败"); };

// ------------------------------------------------------------ 最小 DOM 桩 --

function makeEl(tag = "div") {
  return {
    tag,
    className: "",
    dataset: {},
    textContent: "",
    hidden: false,
    disabled: false,
    type: "",
    value: "",
    checked: false,
    min: "",
    max: "",
    step: "",
    children: [],
    listeners: {},
    classList: {
      _set: new Set(),
      toggle(name, on) { on ? this._set.add(name) : this._set.delete(name); },
      contains(name) { return this._set.has(name); },
    },
    append(...kids) {
      for (const k of kids) {
        if (typeof k === "string") this.children.push({ tag: "#text", textContent: k });
        else this.children.push(k);
      }
    },
    appendChild(k) { this.children.push(k); return k; },
    replaceChildren(...kids) { this.children = kids; },
    setAttribute(k, v) { this[k] = v; },
    addEventListener(ev, fn) { (this.listeners[ev] ??= []).push(fn); },
  };
}

globalThis.document = {
  createElement: (tag) => makeEl(tag),
  addEventListener() {},
};

function makeRoot(ids) {
  const map = new Map();
  // `ids` 本身就带 `#` 前缀（与 SettingsPanel 里的 querySelector 写法一致）
  for (const id of ids) map.set(id, makeEl());
  return {
    querySelector: (sel) => map.get(sel) ?? null,
    addEventListener() {},
    _map: map,
  };
}

const IDS = ["#btn-settings", "#settings-dialog", "#settings-body", "#settings-tabs",
  "#btn-settings-close", "#btn-settings-reset", "#btn-settings-save",
  "#settings-status", "#settings-source"];

// ------------------------------------------------------------- 假 schema --

const GROUPS = [
  { key: "render", title: "渲染 / 模型", file: "render.yaml", help: "", order: 10 },
  { key: "train", title: "训练 / 在线更新", file: "train.yaml", help: "", order: 20 },
  { key: "system", title: "系统 / 服务", file: "system.yaml", help: "", order: 30 },
];

function field(key, group, order, kind, editable, extra = {}) {
  return {
    key, group, order, kind, editable, label: key, help: "", reason: "",
    unit: "", minimum: null, maximum: null, step: null, choices: [],
    value: 0, ...extra,
  };
}

const FIELDS = [
  field("render.background", "render", 10, "color", true, { value: [0, 0, 0] }),
  field("render.scaling_modifier", "render", 20, "float", true, { value: 1 }),
  field("render.sh_degree", "render", 30, "int", false, { reason: "与模型绑定" }),
  field("train.online_update.enabled", "train", 10, "bool", true, { value: true }),
  field("train.online_update.steps", "train", 20, "int", true, { value: 300, unit: "步" }),
  field("runtime.device", "system", 10, "enum", false,
    { reason: "需重启", choices: ["cuda", "cpu"], value: "cuda" }),
  field("app.target_fps", "render", 40, "float", true, { value: 60, unit: "FPS" }),
];

function build() {
  const root = makeRoot(IDS);
  const panel = new SettingsPanel(root, {});
  panel._absorb({ fields: FIELDS, groups: GROUPS, overrides_path: "/tmp/local.yaml" });
  return { panel, root, body: root._map.get("#settings-body") };
}

/** 切到某个页签并重渲染（等价于点一下那个 tab）。 */
function activate(panel, key) {
  panel.active = key;
  panel.render();
}

const allEls = (node, out = []) => {
  out.push(node);
  for (const c of node.children ?? []) allEls(c, out);
  return out;
};
/** 按 class **token** 精确匹配：`srow__label` 不应被当成 `srow`。 */
const hasClass = (el, name) =>
  String(el?.className ?? "").split(/\s+/).includes(name);
const rowsOf = (body) => allEls(body).filter((e) => hasClass(e, "srow"));

console.log("\n渲染");

t("页签只显示组名，配置文件退到悬停提示", () => {
  const { root } = build();
  const tabs = root._map.get("#settings-tabs").children;
  eq(tabs.length, GROUPS.length, "页签数应等于组数");
  // 界面上不展示 render.yaml 这类内部文件名，只在 title 里保留（排查用）
  const labels = tabs.map((b) => b.children.map((c) => c.textContent).join(""));
  ok(labels.every((s) => !s.includes(".yaml")), `页签不该出现文件名：${labels}`);
  eq(tabs.map((b) => b.title), ["render.yaml", "train.yaml", "system.yaml"]);
});

t("默认只显示第一个页签的内容", () => {
  const { panel, body } = build();
  eq(panel.active, "render");
  const keys = rowsOf(body).map((r) => r.dataset.key);
  eq(keys, FIELDS.filter((f) => f.group === "render").map((f) => f.key));
});

t("切遍所有页签能看到**全部**字段（全量展示）", () => {
  const { panel, body } = build();
  const seen = new Set();
  for (const g of GROUPS) {
    activate(panel, g.key);
    for (const row of rowsOf(body)) seen.add(row.dataset.key);
  }
  eq(seen.size, FIELDS.length, "有字段在任何页签里都看不到");
});

t("页签顺序与后端一致（渲染 → 训练 → 系统）", () => {
  const { root } = build();
  const order = root._map.get("#settings-tabs").children.map((b) => b.dataset.group);
  eq(order, ["render", "train", "system"]);
});

t("可编辑项有输入控件，锁定项没有", () => {
  const { panel, body } = build();
  for (const g of GROUPS) activate(panel, g.key);
  for (const row of rowsOf(body)) {
    const key = row.dataset.key;
    const spec = FIELDS.find((f) => f.key === key);
    const inputs = allEls(row).filter((e) =>
      ["input", "select", "textarea"].includes(e.tag));
    if (spec.editable) {
      ok(inputs.length > 0, `${key} 可编辑但没画控件`);
    } else {
      eq(inputs.length, 0, `${key} 锁定却画了输入控件`);
      ok(hasClass(row, "srow--locked"), `${key} 缺少锁定样式`);
    }
  }
});

t("锁定项显示原因（用户要知道为什么不能改）", () => {
  const { panel, body } = build();
  const texts = [];
  for (const g of GROUPS) {
    activate(panel, g.key);
    texts.push(...allEls(body).map((e) => e.textContent ?? ""));
  }
  const joined = texts.join(" ");
  ok(joined.includes("与模型绑定"), "sh_degree 的原因没显示");
  ok(joined.includes("需重启"), "device 的原因没显示");
});

t("bool 画成 checkbox 且反映当前值", () => {
  const { panel, body } = build();
  activate(panel, "train");
  const row = rowsOf(body).find((r) => r.dataset.key === "train.online_update.enabled");
  const box = allEls(row).find((e) => e.type === "checkbox");
  ok(box, "没找到 checkbox");
  eq(box.checked, true);
});

t("enum 锁定项显示当前值", () => {
  const { panel, body } = build();
  activate(panel, "system");
  const row = rowsOf(body).find((r) => r.dataset.key === "runtime.device");
  // 锁定项是只读文本，不是 select —— 这里验证显示的是当前值
  const texts = allEls(row).map((e) => e.textContent ?? "").join(" ");
  ok(texts.includes("cuda"), "锁定项应显示当前值 cuda");
});

t("color 画成 3 个数字输入", () => {
  const { body } = build();
  const row = rowsOf(body).find((r) => r.dataset.key === "render.background");
  const nums = allEls(row).filter((e) => e.type === "number");
  eq(nums.length, 3);
});

t("带单位的数值显示单位", () => {
  const { body } = build();
  const row = rowsOf(body).find((r) => r.dataset.key === "app.target_fps");
  const texts = allEls(row).map((e) => e.textContent ?? "").join(" ");
  ok(texts.includes("FPS"), "缺少单位后缀");
});

t("标题栏只说会保存，路径退到悬停提示", () => {
  const { root } = build();
  const src = root._map.get("#settings-source");
  // 界面上不暴露 configs/local.yaml 这种内部路径
  ok(!src.textContent.includes(".yaml"), `不该出现路径：${src.textContent}`);
  ok(src.textContent.length > 0, "应有一句简短说明");
  ok(String(src.title).includes("local.yaml"), `路径应在 title 里：${src.title}`);
});

console.log("\n交互");

t("改动后 collectPatch 只返回改动项", () => {
  const { panel } = build();
  eq(panel.collectPatch(), {}, "初始不应有改动");
  panel.draft.set("app.target_fps", 30);
  eq(panel.collectPatch(), { "app.target_fps": 30 });
});

t("锁定项即使草稿被改也不会进 patch", () => {
  const { panel } = build();
  panel.draft.set("render.sh_degree", 2);
  eq(panel.collectPatch(), {});
});

t("reset 丢弃未保存改动", () => {
  const { panel } = build();
  panel.draft.set("app.target_fps", 30);
  panel.reset();
  eq(panel.collectPatch(), {});
  eq(panel.draft.get("app.target_fps"), 60);
});

t("open/close 切换 hidden", () => {
  const { panel, root } = build();
  const dialog = root._map.get("#settings-dialog");
  panel.loaded = true;
  panel.open().then?.(() => {});
  ok(!dialog.hidden, "open 后应可见");
  panel.close();
  ok(dialog.hidden, "close 后应隐藏");
});

console.log("");
if (fails.length === 0) {
  console.log(`\x1b[32m通过 ${pass}，失败 0\x1b[0m`);
  process.exit(0);
}
console.log(`\x1b[31m通过 ${pass}，失败 ${fails.length}\x1b[0m`);
for (const [name, e] of fails) console.log(`  ${name}: ${e.message}`);
process.exit(1);
