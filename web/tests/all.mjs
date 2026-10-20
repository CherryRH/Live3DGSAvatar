#!/usr/bin/env node
/**
 * 前端测试汇总入口：依次跑本目录下所有套件。
 *
 * 用法::
 *
 *     node web/tests/all.mjs
 *
 * 另有一个 Python 侧的同名概念：`python tests/run_tests.py`（后端/core）。
 * 两边互不依赖，可分别运行。
 */

import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const suites = ["syntax.mjs",  // 先查语法：坏了就没必要跑后面的
                "ids.mjs", "run.mjs", "dom.mjs", "settings.mjs",
                "settings_dom.mjs"];

let failed = 0;
for (const suite of suites) {
  console.log(`\n${"═".repeat(56)}\n${suite}\n${"═".repeat(56)}`);
  const r = spawnSync(process.execPath, [join(here, suite)], { stdio: "inherit" });
  if (r.status !== 0) failed += 1;
}

console.log(`\n${"═".repeat(56)}`);
if (failed === 0) {
  console.log("\x1b[32m全部前端测试套件通过\x1b[0m");
  process.exit(0);
}
console.log(`\x1b[31m${failed} 个套件失败\x1b[0m`);
process.exit(1);
