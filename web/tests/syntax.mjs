#!/usr/bin/env node
/**
 * 语法检查：`web/js/**\/*.js` 每个文件都要能被 `node --check` 解析。
 *
 * 为什么单独一个套件：其他套件只 `import` 自己用到的模块，
 * **没人 import 的语法错误就永远不会暴露**。真实事故：改 `main.js` 时替换文本
 * 把 `function xxx() {` 那行一起删掉了 —— 语法直接坏掉，但 4 个套件全绿，
 * 直到浏览器里报 `Illegal return statement` 才发现。
 *
 * `node --check` 只查语法，不查"引用了未声明的变量"（那是运行时错误，
 * 需要执行到那条路径才会暴露），所以它拦不住所有问题 —— 但它是零成本的一层。
 */

import { spawnSync } from "node:child_process";
import { readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const jsDir = join(here, "..", "js");

const files = readdirSync(jsDir).filter((n) => n.endsWith(".js")).sort();
let failed = 0;

for (const name of files) {
  const r = spawnSync(process.execPath, ["--check", join(jsDir, name)],
                      { encoding: "utf8" });
  if (r.status === 0) {
    console.log(`  \x1b[32m✓\x1b[0m ${name}`);
  } else {
    failed += 1;
    console.log(`  \x1b[31m✗\x1b[0m ${name}\n${(r.stderr || "").trim()}`);
  }
}

console.log("");
if (failed === 0) {
  console.log(`\x1b[32m通过 ${files.length}，失败 0\x1b[0m`);
  process.exit(0);
}
console.log(`\x1b[31m通过 ${files.length - failed}，失败 ${failed}\x1b[0m`);
process.exit(1);
