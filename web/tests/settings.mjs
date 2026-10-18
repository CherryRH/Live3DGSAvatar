/**
 * 设置面板的**纯逻辑**自检（不需要浏览器 / 后端）。
 *
 * 只测不碰 DOM 的那一层：分组排序、类型转换、差异计算、显示格式。
 * DOM 与网络部分只能在浏览器里验证；这里覆盖的是**容易悄悄错**的判定逻辑：
 *
 *   - 分组顺序错了 → 面板把系统项排在渲染项前面（用户找不到常用参数）
 *   - 差异算错 → 每次保存都把整份配置写进 local.yaml，
 *                之后改默认值会被它永久盖住
 *   - 类型转换漏了 → 发出必然被后端拒绝的请求
 *
 * 用法::
 *
 *     node web/tests/settings.mjs
 */

import {
  GROUP_FALLBACK, buildPatch, coerceFieldValue, formatValue, groupFields,
  sameValue, stepFor,
} from "../js/settings.js";

let pass = 0;
const fails = [];
const t = (name, fn) => {
  try { fn(); pass++; console.log(`  \x1b[32m✓\x1b[0m ${name}`); }
  catch (e) { fails.push([name, e]); console.log(`  \x1b[31m✗\x1b[0m ${name}\n      ${e.message}`); }
};
const eq = (a, b, m) => {
  const sa = JSON.stringify(a), sb = JSON.stringify(b);
  if (sa !== sb) throw new Error(`${m ?? "不相等"}: 期望 ${sb} 实际 ${sa}`);
};
const ok = (c, m) => { if (!c) throw new Error(m ?? "断言失败"); };
const throws = (fn, m) => {
  try { fn(); } catch { return; }
  throw new Error(m ?? "应当抛出异常");
};

// 与后端 app/settings.py 的 GROUPS 对齐
const GROUPS = [
  { key: "render", title: "渲染", help: "", order: 10 },
  { key: "train", title: "训练 / 在线更新", help: "", order: 20 },
  { key: "system", title: "系统", help: "", order: 30 },
];
const F = (key, group, order, kind = "float", editable = true, extra = {}) => ({
  key, group, order, kind, editable, label: key, help: "", reason: "",
  unit: "", minimum: null, maximum: null, step: null, choices: [], value: 0, ...extra,
});

console.log("\n分组与排序");

t("分组按 order 排列（渲染 → 训练 → 系统）", () => {
  const fields = [
    F("paths.data_root", "system", 50),
    F("app.target_fps", "render", 30),
    F("train.online_update.steps", "train", 60),
  ];
  const grouped = groupFields(fields, GROUPS);
  eq(grouped.map((g) => g.key), ["render", "train", "system"]);
});

t("组内按 order 排序（不依赖输入顺序）", () => {
  const fields = [
    F("b", "render", 20),
    F("a", "render", 10),
    F("c", "render", 30),
  ];
  const [render] = groupFields(fields, GROUPS);
  eq(render.fields.map((f) => f.key), ["a", "b", "c"]);
});

t("未知分组排在最后，且用兜底说明", () => {
  const fields = [F("x", "zzz", 1), F("a", "render", 1)];
  const grouped = groupFields(fields, GROUPS);
  eq(grouped[grouped.length - 1].key, "zzz");
  ok(typeof grouped[grouped.length - 1].help === "string");
});

t("空字段列表不炸", () => {
  eq(groupFields([], GROUPS), []);
  eq(groupFields(null, null), []);
});

t("GROUP_FALLBACK 覆盖后端的三个分组", () => {
  for (const { key } of GROUPS) ok(GROUP_FALLBACK[key], `缺少 ${key} 的兜底说明`);
});

console.log("\n类型转换");

t("int 接受整数、拒绝小数与非数字", () => {
  eq(coerceFieldValue(F("k", "train", 1, "int"), "42"), 42);
  throws(() => coerceFieldValue(F("k", "train", 1, "int"), "3.5"), "小数应被拒");
  throws(() => coerceFieldValue(F("k", "train", 1, "int"), "abc"), "非数字应被拒");
});

t("float 接受数字与数字字符串", () => {
  eq(coerceFieldValue(F("k", "train", 1, "float"), "0.25"), 0.25);
  eq(coerceFieldValue(F("k", "train", 1, "float"), 3), 3);
  throws(() => coerceFieldValue(F("k", "train", 1, "float"), ""), "空值应被拒");
});

t("bool 三态都能转", () => {
  eq(coerceFieldValue(F("k", "train", 1, "bool"), true), true);
  eq(coerceFieldValue(F("k", "train", 1, "bool"), false), false);
  eq(coerceFieldValue(F("k", "train", 1, "bool"), 0), false);
});

t("color 要求 3 个数字分量", () => {
  eq(coerceFieldValue(F("k", "render", 1, "color"), [1, 0, 0]), [1, 0, 0]);
  throws(() => coerceFieldValue(F("k", "render", 1, "color"), [1, 0]), "分量不足应被拒");
  throws(() => coerceFieldValue(F("k", "render", 1, "color"), [1, "x", 0]), "非数字应被拒");
});

t("enum / text 转成字符串", () => {
  eq(coerceFieldValue(F("k", "system", 1, "enum", false, { choices: ["a"] }), "a"), "a");
  eq(coerceFieldValue(F("k", "system", 1, "text"), "hi"), "hi");
});

console.log("\n差异计算（决定 local.yaml 写什么）");

t("没改动 → 空 patch", () => {
  eq(buildPatch({ a: 1, b: [0, 0, 0] }, { a: 1, b: [0, 0, 0] }), {});
});

t("只挑改动项", () => {
  eq(buildPatch({ a: 1, b: 2 }, { a: 1, b: 3 }), { b: 3 });
});

t("数组按元素比较（颜色）", () => {
  eq(buildPatch({ c: [0, 0, 0] }, { c: [1, 0, 0] }), { c: [1, 0, 0] });
  eq(buildPatch({ c: [0, 0, 0] }, { c: [0, 0, 0] }), {});
});

t("数字字符串与数字视为相同（输入框给的是字符串）", () => {
  eq(buildPatch({ n: 30 }, { n: "30" }), {});
});

t("null / undefined 相等判断", () => {
  ok(sameValue(null, null));
  ok(sameValue(undefined, undefined));
  ok(!sameValue(null, 0), "null 与 0 不应相等");
});

console.log("\n显示格式");

t("布尔显示为开/关", () => {
  eq(formatValue({ kind: "bool", value: true }), "开");
  eq(formatValue({ kind: "bool", value: false }), "关");
});

t("颜色显示为三位小数逗号分隔", () => {
  eq(formatValue({ kind: "color", value: [1, 0.5, 0] }), "1.00, 0.50, 0.00");
});

t("空值显示为破折号", () => {
  eq(formatValue({ kind: "text", value: null }), "—");
});

t("步长兜底：int 为 1，float 不为 0", () => {
  eq(stepFor({ kind: "int", step: null }), 1);
  ok(stepFor({ kind: "float", step: null }) > 0);
  eq(stepFor({ kind: "float", step: 0.05 }), 0.05);
});

console.log("");
if (fails.length === 0) {
  console.log(`\x1b[32m通过 ${pass}，失败 0\x1b[0m`);
  process.exit(0);
}
console.log(`\x1b[31m通过 ${pass}，失败 ${fails.length}\x1b[0m`);
for (const [name, e] of fails) console.log(`  ${name}: ${e.message}`);
process.exit(1);
