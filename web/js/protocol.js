/**
 * 帧协议解析（见 docs/GUI_PROTOCOL.md §3.1）。
 *
 * 二进制帧布局：
 *   偏移 0   4 字节   seq，uint32 小端
 *   偏移 4   W*H*3    RGB8，行优先
 *
 * 设计要点：几何（宽高）不在帧里重复，由先到的 JSON `config` 提供一次。
 */

/** 帧头长度（字节）。 */
export const HEADER_BYTES = 4;

/** 协议版本；与后端不一致时应显式报错，而不是画出错位的图。 */
export const PROTOCOL_VERSION = 1;

/**
 * 前端构建标记。
 *
 * 用途只有一个：**一眼判断浏览器加载的是不是最新代码**。
 * 改了前端却看不到变化时，先看这里 —— 如果标记没变，就是缓存问题
 * （后端的静态服务已设 `Cache-Control: no-store`，仍可能是浏览器旧标签页）。
 *
 * 改前端时请顺手更新此值。
 */
export const BUILD_TAG = "2026-02-gui5";

/**
 * 解析一条二进制帧。
 *
 * @param {ArrayBuffer} buffer 原始消息
 * @param {{width:number, height:number, channels:number}} geom 由 `config` 得到
 * @returns {{seq:number, pixels:Uint8ClampedArray}}
 * @throws {Error} 几何未配置或长度不符时抛出（宁可报错也不画出错位的图）
 */
export function parseFrame(buffer, geom) {
  if (!geom || !geom.width || !geom.height) {
    throw new Error("尚未收到 config，无法解析帧");
  }
  const channels = geom.channels ?? 3;
  const expected = HEADER_BYTES + geom.width * geom.height * channels;
  if (buffer.byteLength !== expected) {
    throw new Error(
      `帧长度不符：期望 ${expected} 字节（${geom.width}×${geom.height}×${channels} + ${HEADER_BYTES}），` +
      `实际 ${buffer.byteLength}`
    );
  }

  const view = new DataView(buffer);
  const seq = view.getUint32(0, /* littleEndian = */ true);
  const pixels = new Uint8ClampedArray(buffer, HEADER_BYTES);
  return { seq, pixels };
}

/**
 * 解析服务端 JSON 文本消息。
 *
 * 未知 `type` 返回 `null`（不抛错）—— 便于后端加新消息时前端不崩。
 *
 * @param {string} text
 * @returns {object|null}
 */
export function parseMessage(text) {
  let msg;
  try {
    msg = JSON.parse(text);
  } catch (err) {
    console.warn("[protocol] 收到无法解析的 JSON：", text.slice(0, 120));
    return null;
  }
  if (!msg || typeof msg.type !== "string") {
    console.warn("[protocol] 消息缺少 type 字段：", msg);
    return null;
  }
  return msg;
}

/**
 * 从 `config` 消息取出几何信息，并做基本校验。
 *
 * @param {object} msg
 * @returns {{width:number, height:number, channels:number, background:number[], numFrames:number|null}}
 */
export function geometryFromConfig(msg) {
  const width = Number(msg.width);
  const height = Number(msg.height);
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new Error(`config 里的宽高非法：width=${msg.width} height=${msg.height}`);
  }
  if (msg.version !== undefined && msg.version !== PROTOCOL_VERSION) {
    throw new Error(
      `协议版本不匹配：后端 ${msg.version}，前端 ${PROTOCOL_VERSION}。请同步前后端。`
    );
  }
  const bg = Array.isArray(msg.background) ? msg.background.map(Number) : [0, 0, 0];
  return {
    width,
    height,
    channels: msg.channels ?? 3,
    background: bg,
    numFrames: Number.isInteger(msg.num_frames) ? msg.num_frames : null,
  };
}
