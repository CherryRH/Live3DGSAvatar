/**
 * 前端 DOM 相关模块自检（**不需要浏览器**）：用最小桩验证
 * `view.js` 的 RGB→RGBA 转换与 `controls.js` 的消息装配 / 播放条逻辑。
 *
 * canvas 与 WebSocket 的真实行为仍只能在浏览器里验证；
 * 这里覆盖的是**我们自己写的逻辑**，而不是浏览器的实现。
 *
 * 用法::
 *
 *     node web/tests/dom.mjs
 */

import { FrameView } from "../js/view.js";
import { Controls } from "../js/controls.js";

let pass = 0; const fails = [];
const t = (name, fn) => { try { fn(); pass++; console.log(`  \x1b[32m✓\x1b[0m ${name}`); }
  catch (e) { fails.push([name, e]); console.log(`  \x1b[31m✗\x1b[0m ${name}\n      ${e.message}`); } };
const eq = (a, b, m) => { if (a !== b) throw new Error(`${m ?? "不相等"}: 期望 ${b} 实际 ${a}`); };
const ok = (c, m) => { if (!c) throw new Error(m ?? "断言失败"); };

// ---- 全局桩：Controls 会在宿主上挂键盘监听，并用 instanceof 判断事件目标 ----
const keyboard = { handlers: {} };
globalThis.document = {
  addEventListener: (ev, fn) => { (keyboard.handlers[ev] ??= []).push(fn); },
  removeEventListener: (ev, fn) => {
    keyboard.handlers[ev] = (keyboard.handlers[ev] ?? []).filter((f) => f !== fn);
  },
  querySelector: () => null,
  // Controls 会用 getElementById("stage") 切布局类；桩里返回一个可断言的对象
  getElementById: (id) => (id === "stage" ? globalThis.__stage : null),
};
globalThis.Option = class Option {
  constructor(text, value) { this.text = String(text); this.value = String(value); }
};
globalThis.HTMLInputElement = class HTMLInputElement {};
globalThis.HTMLSelectElement = class HTMLSelectElement {};

// ---- FrameView 桩 ----
let canvas0;
function makeCanvas() {
  return {
    width: 0, height: 0,
    getContext: () => ({
      createImageData: (w, h) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }),
      putImageData: (img) => { canvas0.lastPainted = img; },
      save() {}, restore() {}, setTransform() {}, fillRect() {}, fillStyle: "",
    }),
    toBlob: (cb) => cb(null),
  };
}

console.log("\nview.js");
t("RGB→RGBA 转换正确且 alpha=255", () => {
  canvas0 = makeCanvas();
  const v = new FrameView(canvas0);
  v.resize(2, 1, 3);
  v.draw({ seq: 7, pixels: new Uint8ClampedArray([10, 20, 30, 40, 50, 60]) });
  const px = canvas0.lastPainted.data;
  eq(px[0], 10); eq(px[1], 20); eq(px[2], 30); eq(px[3], 255, "alpha");
  eq(px[4], 40); eq(px[5], 50); eq(px[6], 60); eq(px[7], 255, "alpha[1]");
  eq(v.lastSeq, 7);
  eq(v.framesDrawn, 1);
});
t("resize 会设置 canvas 尺寸", () => {
  canvas0 = makeCanvas();
  const v = new FrameView(canvas0);
  v.resize(8, 4, 3);
  eq(canvas0.width, 8); eq(canvas0.height, 4);
});
t("像素不足时明确报错（不画错位图）", () => {
  canvas0 = makeCanvas();
  const v = new FrameView(canvas0);
  v.resize(4, 4, 3);
  let threw = false;
  try { v.draw({ seq: 1, pixels: new Uint8ClampedArray(5) }); } catch (e) { threw = /像素不足/.test(e.message); }
  ok(threw, "应抛「像素不足」");
});
t("未 resize 就 draw 应报错", () => {
  canvas0 = makeCanvas();
  const v = new FrameView(canvas0);
  let threw = false;
  try { v.draw({ seq: 1, pixels: new Uint8ClampedArray(12) }); } catch (e) { threw = /尚未 resize/.test(e.message); }
  ok(threw);
});

// ---- Controls 桩：元素清单与 index.html 对应 ----
function el(tag = "input", attrs = {}) {
  const listeners = {};
  return Object.assign({
    tagName: tag.toUpperCase(), value: "", disabled: false, max: "", textContent: "",
    checked: false,
    // `options` + `replaceChildren` 模拟 `<select>` 的常见用法：
    // 真实 DOM 用 `new Option(text, value)` 追加子节点。
    options: [],
    replaceChildren(...items) {
      this.options = items;
      this.value = items.length ? items[0].value : "";
    },
    classList: {
      _set: new Set(),
      add(c) { this._set.add(c); },
      remove(c) { this._set.delete(c); },
      contains(c) { return this._set.has(c); },
      toggle(c, on) {
        const want = on === undefined ? !this._set.has(c) : Boolean(on);
        if (want) this._set.add(c); else this._set.delete(c);
        return want;
      },
    },
    addEventListener(ev, fn) { (listeners[ev] ??= []).push(fn); },
    _fire(ev) { (listeners[ev] ?? []).forEach((f) => f()); },
  }, attrs);
}

function makeRoot() {
  globalThis.__stage = el("div");
  const map = {
    // 模型
    "#f-subject": el("input", { value: "duda" }),
    "#f-work-name": el("input", { value: "test" }),
    "#btn-load": el("button"), "#model-info": el("p"),
    // 播放条
    "#player": el("div"),
    "#btn-play": el("button"),
    "#btn-step-back": el("button"), "#btn-step-fwd": el("button"),
    "#f-frame": el("input", { value: "0" }), "#o-frame": el("output"),
    "#o-total": el("span"),
    "#f-source": el("select", { value: "checkpoint",
                                options: [{ value: "checkpoint" }, { value: "live" }] }),
    "#f-dataset": el("select"),
    "#dataset-info": el("p"),
    "#f-amp": el("input", { value: "0.1" }), "#o-amp": el("output"),
    "#f-perturb": el("input", { checked: false }),
    "#f-compare": el("input", { checked: false }),
    "#psnr-out": el("span"),
    "#live-status": el("span"),
    // 操作
    "#btn-pause": el("button"), "#btn-capture": el("button"), "#btn-reconnect": el("button"),
  };
  return {
    querySelector: (s) => map[s] ?? null,
    // 宿主自己也有 addEventListener：Controls 优先用它
    addEventListener: (ev, fn) => { (keyboard.handlers[ev] ??= []).push(fn); },
    removeEventListener: (ev, fn) => {
      keyboard.handlers[ev] = (keyboard.handlers[ev] ?? []).filter((f) => f !== fn);
    },
    _map: map,
  };
}

function makeControls(root, over = {}) {
  return new Controls(root, {
    onParams() {}, onLoadModel() {}, onPause() {}, onCapture() {},
    onReconnect() {}, onStep() {}, ...over,
  });
}

console.log("\ncontrols.js");
t("_emitParams 装配出协议要求的层级（且不再发 render）", () => {
  const root = makeRoot(); let sent = null;
  const c = makeControls(root, { onParams: (m) => { sent = m; } });
  root._map["#f-frame"].value = "42";
  c._emitParams();
  eq(sent.type, "params");
  eq(sent.source.kind, "checkpoint", "数据源层级");
  ok(!("mode" in sent.drive), "不应再有 drive.mode（语义已并入 playing）");
  eq(sent.drive.frame, 42);
  // **渲染参数不再由这里发送**：它们走 /api/settings 并持久化。
  // 若这里也发，会在拖动播放条时把设置面板刚存的值覆盖回旧值。
  ok(!("render" in sent), `不应发送 render，实际：${JSON.stringify(sent.render)}`);
  c.destroy();
});

t("applyModels 填对象/版本下拉，换对象重填版本", () => {
  const root = makeRoot();
  const c = makeControls(root);
  c.applyModels({
    models: [{ subject: "alice", work_name: "run1" },
             { subject: "alice", work_name: "run2" },
             { subject: "bob", work_name: "test" }],
    current: { subject: "alice", work_name: "run2" },
  });
  const subj = root._map["#f-subject"];
  const work = root._map["#f-work-name"];
  eq(subj.options.length, 2, "对象应去重成 2 个");
  eq(subj.value, "alice");
  eq(work.options.length, 2, "alice 下有 2 个版本");
  eq(work.value, "run2", "应选中后端当前的版本");

  // 换对象 → 版本列表跟着换
  subj.value = "bob";
  subj._fire("change");
  eq(work.options.length, 1);
  eq(work.value, "test");
  c.destroy();
});

t("对比原图：开关会发 params.compare 并切布局", () => {
  const root = makeRoot(); let sent = null;
  const c = makeControls(root, { onParams: (m) => { sent = m; } });
  ok(!globalThis.__stage.classList.contains("is-compare"), "默认不并排");

  root._map["#f-compare"].checked = true;
  root._map["#f-compare"]._fire("change");
  eq(sent.compare, true, "应把开关发给后端");
  ok(globalThis.__stage.classList.contains("is-compare"), "应切到并排布局");

  root._map["#f-compare"].checked = false;
  root._map["#f-compare"]._fire("change");
  eq(sent.compare, false);
  ok(!globalThis.__stage.classList.contains("is-compare"), "关闭后恢复单幅");
  c.destroy();
});

t("PSNR 读数只在对比开启时显示", () => {
  const root = makeRoot();
  const c = makeControls(root);
  const out = root._map["#psnr-out"];

  root._map["#f-compare"].checked = false;
  c.applyPsnr(21.5);
  ok(out.hidden, "未开启对比时不该显示 PSNR");

  root._map["#f-compare"].checked = true;
  c.applyPsnr(21.5);
  ok(!out.hidden, "开启后应显示");
  ok(/21\.50/.test(out.textContent), `应显示两位小数：${out.textContent}`);

  c.applyPsnr(Number.NaN);
  ok(out.textContent.includes("–"), `无值时应显示占位：${out.textContent}`);
  c.destroy();
});

t("模型未加载时 applyState 不能用空值清空下拉框", () => {
  const root = makeRoot();
  const c = makeControls(root);
  c.applyModels({
    models: [{ subject: "crh_1", work_name: "test" },
             { subject: "crh_2", work_name: "test" }],
    current: { subject: "", work_name: "" },   // 后端还没加载模型
  });
  const subj = root._map["#f-subject"];
  const work = root._map["#f-work-name"];
  eq(subj.value, "crh_1", "应自动选中第一个对象");
  eq(work.value, "test", "版本也要自动选中");

  // 收到一份"空 subject/work_name"的 state：不得把上面的选择清掉
  c.applyState({ subject: "", work_name: "", dataset: { num_frames: 10 } });
  eq(subj.value, "crh_1", "空值不该清空对象");
  eq(work.value, "test", "空值不该清空版本");
  c.destroy();
});

t("勾选随机扰动 → drive.perturb=true，幅度控件启用", () => {
  const root = makeRoot(); let sent = null;
  const c = makeControls(root, { onParams: (m) => { sent = m; } });
  ok(root._map["#f-amp"].disabled, "未开启扰动时幅度应禁用");
  root._map["#f-perturb"].checked = true;
  // 走真实路径：勾选会触发 change（真实浏览器如此），由它驱动 UI 与消息
  root._map["#f-perturb"]._fire("change");
  eq(sent.drive.perturb, true);
  ok(!root._map["#f-amp"].disabled, "开启扰动后幅度应启用");
  c.destroy();
});

t("applyDatasets 填选项、恢复当前值、给出帧数提示", () => {
  const root = makeRoot();
  const c = makeControls(root);
  c.applyDatasets({
    datasets: [
      { name: "crh_1", frames: 2863, is_model: true, has_input: true },
      { name: "crh_2", frames: 3524, is_model: false, has_input: true },
      { name: "duda", frames: 254, is_model: false, has_input: false },
    ],
    model_subject: "crh_1",
    current: "crh_2",
  });
  const sel = root._map["#f-dataset"];
  eq(sel.options.length, 3, "应有 3 个选项");
  eq(sel.value, "crh_2", "应恢复后端当前的驱动数据集");
  ok(sel.options[0].text.includes("模型自带"), `模型自带项要有标记：${sel.options[0].text}`);
  const info = root._map["#dataset-info"].textContent;
  ok(info.includes("3524"), `提示里应有帧数：${info}`);
  c.destroy();
});

t("换数据集 → 走独立回调（慢操作要能显示加载浮窗）", () => {
  const root = makeRoot();
  let sent = null; const asked = [];
  const c = makeControls(root, {
    onParams: (m) => { sent = m; },
    onDatasetChange: (name) => asked.push(name),
  });
  c.applyDatasets({
    datasets: [{ name: "crh_1", frames: 10, is_model: true },
               { name: "crh_2", frames: 20, is_model: false }],
    model_subject: "crh_1", current: "",
  });
  sent = null;
  root._map["#f-dataset"].value = "crh_2";
  root._map["#f-dataset"]._fire("change");
  eq(asked.length, 1, "应触发一次切换请求");
  eq(asked[0], "crh_2");
  ok(sent === null, "不应把数据集塞进 WS params（慢操作走 HTTP）");
  c.destroy();
});

t("切换失败 → 下拉框回到上一次成功的值", () => {
  const root = makeRoot();
  const c = makeControls(root, { onDatasetChange: () => {} });
  c.applyDatasets({
    datasets: [{ name: "a", frames: 1, is_model: true },
               { name: "b", frames: 2, is_model: false }],
    model_subject: "a", current: "a",
  });
  root._map["#f-dataset"].value = "b";
  c.revertDataset();
  eq(root._map["#f-dataset"].value, "a", "应退回已提交值");
  c.destroy();
});

t("live 模式：说明送给追踪；有 images/ 回退时不该误报", () => {
  const root = makeRoot();
  const c = makeControls(root);
  // 只有 images/ 的数据集（如 duda）—— 可以喂追踪，不该报警
  c.applyDatasets({
    datasets: [{ name: "duda", frames: 254, is_model: false,
                 has_input: false, has_images: true }],
    model_subject: "crh_1", current: "duda",
  });
  root._map["#f-source"].value = "live";
  c._renderDatasetInfo();
  let info = root._map["#dataset-info"].textContent;
  ok(info.includes("追踪"), `live 模式应说明送给追踪：${info}`);
  ok(!info.includes("⚠"), `有 images/ 回退不应警告：${info}`);

  // 两个目录都没有 → 必须警告
  c.applyDatasets({
    datasets: [{ name: "broken", frames: 10, is_model: false,
                 has_input: false, has_images: false }],
    model_subject: "crh_1", current: "broken",
  });
  c._renderDatasetInfo();
  info = root._map["#dataset-info"].textContent;
  ok(info.includes("⚠"), `无画面目录必须警告：${info}`);
  c.destroy();
});

t("数据源 live：播放条仍有效（可暂停、可跳帧）", () => {
  const root = makeRoot(); let sent = null;
  const c = makeControls(root, { onParams: (m) => { sent = m; } });
  root._map["#f-source"].value = "live";
  root._map["#f-frame"].value = "37";
  c._renderOutputs();
  c._emitParams();
  eq(sent.source.kind, "live");
  // 帧号与播放态在 live 下**必须**发：取帧线程按帧号送图
  eq(sent.drive.frame, 37, "live 下也要发帧号");
  eq(sent.drive.playing, false, "live 下也要发播放态");
  ok(!root._map["#f-frame"].disabled, "live 下帧滑块应可用");
  ok(!root._map["#btn-step-fwd"].disabled, "live 下单步应可用");
  // 调试扰动仍不适用
  ok(root._map["#f-perturb"].disabled, "live 下扰动应禁用");
  c.destroy();
});

t("单步按钮触发 onStep(±1)", () => {
  const root = makeRoot();
  const steps = [];
  const c = makeControls(root, { onStep: (d) => steps.push(d) });
  root._map["#btn-step-back"]._fire("click");
  root._map["#btn-step-fwd"]._fire("click");
  eq(steps.length, 2); eq(steps[0], -1); eq(steps[1], +1);
  c.destroy();
});

t("方向键单步、空格播放/暂停", () => {
  keyboard.handlers.keydown = [];       // 清掉其他用例遗留的监听
  const root = makeRoot();
  const steps = [];
  let sent = null;
  const c = makeControls(root, {
    onStep: (d) => steps.push(d), onParams: (m) => { sent = m; },
  });
  const fire = (key) => (keyboard.handlers.keydown ?? []).forEach(
    (f) => f({ key, target: {}, preventDefault() {} }));
  fire("ArrowLeft"); fire("ArrowRight");
  eq(steps[steps.length - 2], -1); eq(steps[steps.length - 1], +1);
  fire(" ");
  ok(sent && sent.drive.playing === true, "空格应触发播放");
  c.destroy();
  eq((keyboard.handlers.keydown ?? []).length, 0, "destroy() 后应摘掉监听");
});

t("播放按钮在 播放/暂停 之间切换并上报", () => {
  const root = makeRoot();
  const sent = [];
  const c = makeControls(root, { onParams: (m) => sent.push(m) });
  root._map["#btn-play"]._fire("click");
  eq(sent[0].drive.playing, true, "第一次点击应播放");
  eq(root._map["#btn-play"].textContent, "⏸", "按钮应变为暂停图标");
  root._map["#btn-play"]._fire("click");
  eq(sent[1].drive.playing, false, "第二次点击应暂停");
  eq(root._map["#btn-play"].textContent, "▶");
  c.destroy();
});

t("播放状态由服务端回填，不自己记", () => {
  const root = makeRoot();
  const c = makeControls(root);
  c.applyPlayback({ frame: 42, playing: true });
  eq(root._map["#f-frame"].value, "42", "帧号应回填");
  eq(root._map["#btn-play"].textContent, "⏸", "播放态应回填");
  c.applyPlayback({ frame: 7, playing: false });
  eq(root._map["#f-frame"].value, "7");
  eq(root._map["#btn-play"].textContent, "▶");
  c.destroy();
});

t("拖动帧滑块期间不被回填（否则和手指打架）", () => {
  const root = makeRoot();
  const c = makeControls(root);
  root._map["#f-frame"].value = "100";
  root._map["#f-frame"]._fire("pointerdown");
  c.applyPlayback({ frame: 3, playing: false });
  eq(root._map["#f-frame"].value, "100", "拖动中不应被服务端回填");
  root._map["#f-frame"]._fire("pointerup");
  c.applyPlayback({ frame: 3, playing: false });
  eq(root._map["#f-frame"].value, "3", "松手后应恢复回填");
  c.destroy();
});

t("applyState 写入模型信息、总帧数与驱动方式", () => {
  const root = makeRoot();
  const c = makeControls(root);
  // 先填下拉框：不在选项里的值不能被"选中"（会变成空值）
  c.applyModels({
    models: [{ subject: "alice", work_name: "run1" },
             { subject: "alice", work_name: "run2" },
             { subject: "bob", work_name: "test" }],
    current: { subject: "alice", work_name: "run1" },
  });
  c.applyState({
    subject: "alice", work_name: "run1",
    model: { num_gaussians: 60000, num_basis: 20, num_basis_in: 129, tex_size: 256 },
    drive: { frame: 5, playing: true, perturb: true, amplitude: 0.2 },
    dataset: { num_frames: 100 },
  });
  eq(root._map["#f-subject"].value, "alice");
  ok(/60000/.test(root._map["#model-info"].textContent), "高斯数应显示");
  eq(root._map["#f-frame"].max, "99", "帧范围上限");
  eq(root._map["#o-total"].textContent, "100", "总帧数应显示");
  eq(root._map["#f-frame"].value, "5");
  ok(root._map["#f-perturb"].checked, "扰动开关应回填");
  c.destroy();
});

t("暂停时仍发送 playing=false（渲染继续，只是帧号停住）", () => {
  const root = makeRoot(); const sent = [];
  const c = makeControls(root, { onParams: (m) => sent.push(m) });
  root._map["#btn-play"]._fire("click");     // 播放
  eq(sent[0].drive.playing, true);
  root._map["#btn-play"]._fire("click");     // 暂停
  eq(sent[1].drive.playing, false);
  // 暂停后仍应带完整驱动参数（后端据此继续渲染同一帧，而不是停掉渲染）
  ok("frame" in sent[1].drive, "暂停时仍应带 frame");
  // 渲染参数不再随本消息发送（见「_emitParams 装配出协议要求的层级」）
  ok(!("render" in sent[1]), "不应发送 render");
  c.destroy();
});

t("subject 为空时拒绝加载", () => {
  const root = makeRoot(); let called = false;
  const c = makeControls(root, { onLoadModel: () => { called = true; } });
  root._map["#f-subject"].value = "  ";
  root._map["#btn-load"]._fire("click");
  ok(!called, "不应触发加载");
  ok(/不能为空/.test(root._map["#model-info"].textContent));
  c.destroy();
});

console.log(`\n${"─".repeat(50)}`);
if (fails.length === 0) {
  console.log(`\x1b[32m通过 ${pass}，失败 0\x1b[0m`);
} else {
  console.log(`通过 ${pass}，\x1b[31m失败 ${fails.length}\x1b[0m`);
  for (const [n, e] of fails) console.log(`\n--- ${n} ---\n${e.stack}`);
  process.exit(1);
}
