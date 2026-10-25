/**
 * `RenderClient._pump` 的回归测试（**不需要浏览器**）。
 *
 * 背景：`_pump` 每次派发一帧后用 `requestAnimationFrame` 让出主线程。
 * 但 rAF 在标签页被遮挡、窗口最小化、主线程被长时间占用时可能**长时间
 * 不触发**。一旦卡住，`_pumping` 永远是 true，后续帧只会覆盖 `_pending`，
 * 渲染画面就停住 —— 而对比帧走的是另一条同步路径，表现为
 * "原图在动、渲染图卡住"，指标却照常更新。
 *
 * 这里把 `requestAnimationFrame` 做成**永不回调**，验证兜底 timeout 能让
 * 派发继续下去。
 *
 * 用法::
 *
 *     node web/tests/pump.mjs
 */

import { RenderClient } from "../js/client.js";

let pass = 0; const fails = [];
const t = async (name, fn) => {
  try { await fn(); pass++; console.log(`  \x1b[32m✓\x1b[0m ${name}`); }
  catch (e) { fails.push([name, e]); console.log(`  \x1b[31m✗\x1b[0m ${name}\n      ${e.message}`); }
};
const eq = (a, b, m) => { if (a !== b) throw new Error(`${m ?? "不相等"}: 期望 ${b} 实际 ${a}`); };

// ---- 浏览器全局桩 ----
globalThis.location = { protocol: "http:", host: "localhost:0" };
globalThis.WebSocket = class { constructor() { this.readyState = 1; } };
// Node 自带 `EventTarget`，它要求派发的是真正的 `Event` —— 所以继承 Event
globalThis.CustomEvent = class CustomEvent extends Event {
  constructor(type, init = {}) { super(type); this.detail = init.detail; }
};
// 关键：rAF 永不回调，模拟"标签页被遮挡"
globalThis.requestAnimationFrame = () => 0;

const frame = (seq) => ({ kind: 1, seq, bitmap: { width: 4, height: 4, close() {} } });

const mkClient = () => {
  const c = new RenderClient({ url: "ws://x/ws" });
  const seen = [];
  c.addEventListener("frame", (e) => seen.push(e.detail.seq));
  return { c, seen };
};

// ---- 用例 ----

await t("rAF 不触发时，帧仍能继续派发（靠 timeout 兜底）", async () => {
  const { c, seen } = mkClient();
  for (let i = 1; i <= 3; i += 1) {
    c._pending = frame(i);
    c._pump();
    await new Promise((r) => setTimeout(r, 260));
  }
  eq(seen.length, 3, "三帧都应派发出去");
  eq(seen.join(","), "1,2,3");
});

await t("派发顺序单调递增，不会回退", async () => {
  const { c, seen } = mkClient();
  c._pending = frame(5);
  c._pump();
  await new Promise((r) => setTimeout(r, 260));
  eq(seen.at(-1), 5);
});

await t("_pumping 最终会复位（不会永久卡住）", async () => {
  const { c } = mkClient();
  c._pending = frame(1);
  c._pump();
  await new Promise((r) => setTimeout(r, 260));
  eq(c._pumping, false);
});

// ---- 汇总 ----
console.log("");
if (fails.length) {
  for (const [name, e] of fails) console.log(`${name}\n${e.stack}`);
  console.log(`\x1b[31m失败 ${fails.length}，通过 ${pass}\x1b[0m`);
  process.exit(1);
}
console.log(`\x1b[32m通过 ${pass}\x1b[0m`);
