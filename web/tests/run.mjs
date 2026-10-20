#!/usr/bin/env node
/**
 * 前端模块自检（**不需要浏览器、不需要后端**）。
 *
 * 本文件测不依赖 DOM 的部分（protocol.js / mock.js）；
 * `dom.mjs` 用桩测 view.js / controls.js。
 * `client.js` 的真实 WebSocket 行为只能在浏览器里验证。
 *
 * 用法::
 *
 *     node web/tests/run.mjs          # 只跑本文件
 *     node web/tests/all.mjs          # 跑全部前端测试
 */

import { parseFrame, parseMessage, geometryFromConfig, HEADER_BYTES, PROTOCOL_VERSION }
  from "../js/protocol.js";
import { MockSource } from "../js/mock.js";
import { FpsMeter } from "../js/fps.js";
import { RenderClient } from "../js/client.js";

let passed = 0;
const failures = [];

function test(name, fn) {
  try {
    fn();
    passed += 1;
    console.log(`  \x1b[32m✓\x1b[0m ${name}`);
  } catch (err) {
    failures.push({ name, err });
    console.log(`  \x1b[31m✗\x1b[0m ${name}\n      ${err.message}`);
  }
}

function assert(cond, msg) {
  if (!cond) throw new Error(msg ?? "断言失败");
}

function assertEqual(actual, expected, msg) {
  if (actual !== expected) {
    throw new Error(`${msg ?? "不相等"}：期望 ${expected}，实际 ${actual}`);
  }
}

function assertThrows(fn, re, msg) {
  try {
    fn();
  } catch (err) {
    if (re && !re.test(err.message)) {
      throw new Error(`${msg ?? "错误信息不符"}：${err.message}`);
    }
    return;
  }
  throw new Error(`${msg ?? "应当抛错"}，但没有`);
}

// ------------------------------------------------------------ protocol --

console.log("\nprotocol.js");

test("config 几何解析正常", () => {
  const g = geometryFromConfig({ width: 512, height: 256, background: [1, 1, 1],
                                 num_frames: 254, version: PROTOCOL_VERSION });
  assertEqual(g.width, 512);
  assertEqual(g.height, 256);
  assertEqual(g.channels, 3, "默认通道数");
  assertEqual(g.numFrames, 254);
  assert(Array.isArray(g.background) && g.background.length === 3);
});

test("config 拒绝非法宽高", () => {
  assertThrows(() => geometryFromConfig({ width: 0, height: 10 }), /宽高非法/);
  assertThrows(() => geometryFromConfig({ width: 10, height: -1 }), /宽高非法/);
  assertThrows(() => geometryFromConfig({ width: 1.5, height: 10 }), /宽高非法/);
});

test("config 拒绝协议版本不匹配", () => {
  assertThrows(
    () => geometryFromConfig({ width: 8, height: 8, version: PROTOCOL_VERSION + 1 }),
    /协议版本不匹配/);
});

test("帧解析：seq 小端 + 像素偏移正确", () => {
  const w = 4, h = 2, geom = { width: w, height: h, channels: 3 };
  const buf = new ArrayBuffer(HEADER_BYTES + w * h * 3);
  const dv = new DataView(buf);
  dv.setUint8(0, 1);                       // 类型 = 渲染帧
  dv.setUint32(1, 0x01020304, true);       // seq
  dv.setFloat32(5, 21.5, true);            // psnr
  const px = new Uint8ClampedArray(buf, HEADER_BYTES);
  for (let i = 0; i < px.length; i++) px[i] = i % 256;

  const frame = parseFrame(buf, geom);
  assertEqual(frame.kind, 1, "类型字节");
  assertEqual(frame.seq, 0x01020304, "seq 应按小端读出");
  assertEqual(frame.psnr, 21.5, "psnr 应能读出");
  assertEqual(frame.pixels.length, w * h * 3, "像素长度");
  assertEqual(frame.pixels[0], 0);
  assertEqual(frame.pixels[5], 5);
});

test("帧解析拒绝长度不符（宁可报错也不画错位图）", () => {
  const geom = { width: 4, height: 2, channels: 3 };
  assertThrows(() => parseFrame(new ArrayBuffer(10), geom), /帧长度不符/);
  assertThrows(() => parseFrame(new ArrayBuffer(HEADER_BYTES), geom), /帧长度不符/);
});

test("帧解析在未收到 config 时明确报错", () => {
  assertThrows(() => parseFrame(new ArrayBuffer(8), null), /尚未收到 config/);
});

test("消息解析：非法 JSON 返回 null 而不抛", () => {
  assertEqual(parseMessage("这不是 JSON"), null);
  assertEqual(parseMessage("{}"), null, "缺 type");
  const ok = parseMessage('{"type":"status","fps":60}');
  assert(ok && ok.type === "status");
});

test("未知 type 不崩（便于后端加消息）", () => {
  const m = parseMessage('{"type":"brand_new","x":1}');
  assert(m && m.type === "brand_new");
});

// ------------------------------------------------------------- fps.js --

console.log("\nfps.js（前端自算画面帧率）");

test("空窗口与单帧都返回 0（而不是保留旧值或 inf）", () => {
  const m = new FpsMeter(1000);
  assertEqual(m.value, 0, "空窗口");
  assertEqual(m.display, "0");
  m.tick(0);
  assertEqual(m.value, 0, "只有 1 帧无法算速率");
});

test("稳定 60 FPS 时应测出 60", () => {
  const m = new FpsMeter(1000);
  for (let i = 0; i <= 60; i++) m.tick(i * (1000 / 60));
  const v = m.value;
  assert(Math.abs(v - 60) < 1.5, `应约 60，实际 ${v.toFixed(2)}`);
  assertEqual(m.display, "60");
});

test("窗口滚动：帧率变化后 1–2 个窗口内收敛", () => {
  const m = new FpsMeter(1000);
  // 先在 [0,1)s 内 100 FPS
  for (let i = 0; i <= 100; i++) m.tick(i * 10);
  assert(Math.abs(m.value - 100) < 5, `第一段应约 100，实际 ${m.value.toFixed(1)}`);
  // 再降为 10 FPS，跑满一个窗口
  let t = 1010;
  for (let i = 0; i < 12; i++) { m.tick(t); t += 100; }
  assert(Math.abs(m.value - 10) < 3, `降速后应约 10，实际 ${m.value.toFixed(1)}`);
});

test("长时间无帧 → 窗口清空，显示 0", () => {
  const m = new FpsMeter(500);
  for (let i = 0; i < 30; i++) m.tick(i * 16);
  assert(m.value > 0, "先有帧率");
  m.tick(100000);                       // 很久之后来一帧，旧的全过期
  assertEqual(m.value, 0, "只剩 1 帧 → 0");
  assertEqual(m.display, "0");
});

test("非法窗口长度应报错", () => {
  assertThrows(() => new FpsMeter(0), /windowMs/);
  assertThrows(() => new FpsMeter(-1), /windowMs/);
});

test("reset 清空", () => {
  const m = new FpsMeter(1000);
  for (let i = 0; i < 10; i++) m.tick(i * 16);
  m.reset();
  assertEqual(m.value, 0);
});

// ---------------------------------------------------------------- mock --

console.log("\nmock.js（复刻真实协议）");

/** 等待某个事件，带超时 —— 避免测试挂死。 */
function waitFor(src, type, timeoutMs = 2000) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error(`等待 ${type} 事件超时（${timeoutMs}ms）`)), timeoutMs);
    src.addEventListener(type, (ev) => {
      clearTimeout(timer);
      resolve(ev.detail);
    }, { once: true });
  });
}

const tests = [];

tests.push((async () => {
  const src = new MockSource({ width: 64, height: 32, numFrames: 10 });
  const cfgPromise = waitFor(src, "config");
  const framePromise = waitFor(src, "frame");
  src.start();
  const cfg = await cfgPromise;
  const raw = await framePromise;
  src.stop();

  test("mock config 含几何与版本", () => {
    assertEqual(cfg.width, 64);
    assertEqual(cfg.height, 32);
    assertEqual(cfg.version, PROTOCOL_VERSION);
    assertEqual(cfg.numFrames, 10);
  });

  test("mock 帧能被同一个 parseFrame 解析（协议一致）", () => {
    // mock 直接给 {seq, pixels}；这里按真实二进制再走一遍解析路径
    const bytes = HEADER_BYTES + 64 * 32 * 3;
    const buf = new ArrayBuffer(bytes);
    const dv = new DataView(buf);
    dv.setUint8(0, 1);
    dv.setUint32(1, raw.seq, true);
    dv.setFloat32(5, Number.NaN, true);
    new Uint8ClampedArray(buf, HEADER_BYTES).set(raw.pixels);
    const parsed = parseFrame(buf, { width: 64, height: 32, channels: 3 });
    assertEqual(parsed.seq, raw.seq);
    assertEqual(parsed.pixels.length, 64 * 32 * 3);
  });

  test("mock 的 seq 递增（丢帧统计依赖它）", async () => {
    // 由上面的 raw 推导：起始 seq 应为 1
    assertEqual(raw.seq, 1, "第一帧 seq 应为 1");
  });
})());

tests.push((async () => {
  const src = new MockSource({ width: 16, height: 16 });
  src.start();
  await waitFor(src, "config");

  test("mock 接受 params 并改变背景色", async () => {
    src.send({ type: "params", render: { background: [1, 1, 1] },
               drive: { mode: "dataset", frame: 3, amplitude: 0.2 } });
    const f = await waitFor(src, "frame");
    // 左上角不在椭圆内，应为背景色
    assertEqual(f.pixels[0], 255, "背景应为白");
    assertEqual(f.pixels[1], 255);
    assertEqual(f.pixels[2], 255);
  });

  test("mock 暂停后不再产帧", async () => {
    src.send({ type: "stream", paused: true });
    let got = false;
    const onFrame = () => { got = true; };
    src.addEventListener("frame", onFrame);
    await new Promise((r) => setTimeout(r, 120));
    src.removeEventListener("frame", onFrame);
    src.stop();
    assert(!got, "暂停后仍在产帧");
  });
})());

tests.push((async () => {
  const src = new MockSource({ width: 32, height: 32 });
  src.start();
  await waitFor(src, "config");

  test("mock status 字段齐全（前端 updateStats 依赖）", async () => {
    const st = await waitFor(src, "status", 1500);
    assertEqual(st.type, "status");
    assert(typeof st.fps === "number", "fps 应为数字");
    assert(st.ms && typeof st.ms === "object", "ms 应为对象");
    assert(typeof st.dropped === "number", "dropped 应为数字");
    assert(typeof st.paused === "boolean", "paused 应为布尔");
    src.stop();
  });
})());


// ------------------------------------------------------------- client.js --

console.log("\nclient.js（错误事件形状）");

/**
 * 直接调 `_onMessage` 喂消息。
 *
 * 为什么要测这个：`error` 事件有**两个来源** ——
 * 传输层（`_emitError`，`detail.error` 是 Error）与后端业务错误
 * （`{type:"error", message}`，直接转发时 `detail.error` **不存在**）。
 * 监听方统一按 `detail.error.message` 取值，于是后者会让**错误处理自己抛 TypeError**，
 * 把真正的错误信息盖掉 —— 这是最糟糕的一类 bug：报错时看不到真正的错。
 */
// `RenderClient` 构造时按 `location` 推导默认 ws 地址；Node 里没有它，给个桩。
globalThis.location = { protocol: "http:", host: "127.0.0.1:8000" };

function fakeClient() {
  const c = new RenderClient();
  c.geometry = { width: 8, height: 8, channels: 3, totalFrames: 10 };
  return c;
}

test("后端 error 消息 → detail.error.message 可读", () => {
  const c = fakeClient();
  let received = null;
  c.addEventListener("error", (ev) => { received = ev.detail; });
  c._onMessage({ data: JSON.stringify({ type: "error", message: "模型尚未加载" }) });
  assert(received !== null, "应派发 error 事件");
  assert(received.error instanceof Error, "detail.error 应是 Error");
  assertEqual(received.error.message, "模型尚未加载", "错误信息应保留原文");
  assert(received.error.fromServer === true, "应标记来源为后端");
});

test("后端 error 缺 message 字段时给出兜底文案（不抛错）", () => {
  const c = fakeClient();
  let received = null;
  c.addEventListener("error", (ev) => { received = ev.detail; });
  c._onMessage({ data: JSON.stringify({ type: "error" }) });
  assert(received !== null);
  assert(/未知错误/.test(received.error.message), "应有兜底文案");
});

test("传输层错误与后端错误形状一致（同一种读取方式）", () => {
  const c = fakeClient();
  const seen = [];
  c.addEventListener("error", (ev) => seen.push(ev.detail.error.message));
  c._onMessage({ data: JSON.stringify({ type: "error", message: "后端说" }) });
  c._emitError(new Error("传输层说"));
  assertEqual(seen.length, 2);
  assertEqual(seen[0], "后端说");
  assertEqual(seen[1], "传输层说");
});

test("非 error 类型的消息仍原样转发", () => {
  const c = fakeClient();
  let got = null;
  c.addEventListener("status", (ev) => { got = ev.detail; });
  c._onMessage({ data: JSON.stringify({ type: "status", frame: 3, playing: false }) });
  assert(got !== null);
  assertEqual(got.type, "status");
  assertEqual(got.frame, 3);
});

await Promise.all(tests);

console.log(`\n${"─".repeat(52)}`);
if (failures.length === 0) {
  console.log(`\x1b[32m通过 ${passed}，失败 0\x1b[0m`);
  process.exit(0);
} else {
  console.log(`通过 ${passed}，\x1b[31m失败 ${failures.length}\x1b[0m`);
  for (const f of failures) console.log(`\n--- ${f.name} ---\n${f.err.stack}`);
  process.exit(1);
}
