/**
 * 前端画面帧率（滑动时间窗）。
 *
 * **为什么由前端算**：它反映"对方实际看到的画面到达率"——包括网络与浏览器的
 * 影响，与后端的渲染吞吐不是一回事。后端不提供 `fps` 字段，只把统计写到
 * stdout 日志（见 docs/APP.md）。
 *
 * **为什么用滑动窗口而不是累计平均**：改帧数上限后累计平均要很久才收敛；
 * 时间窗在 1–2 个窗口内就能反映新状态。
 */

export class FpsMeter {
  /** @param {number} windowMs 时间窗长度；默认 1 秒 */
  constructor(windowMs = 1000) {
    if (!(windowMs > 0)) throw new Error(`windowMs 必须 > 0，实际 ${windowMs}`);
    this.windowMs = windowMs;
    this.stamps = [];
  }

  /** 登记一帧（应在**真正画到画布之后**调用）。 */
  tick(now = performance.now()) {
    this.stamps.push(now);
    this._trim(now);
  }

  /**
   * 当前窗口内的帧率。
   *
   * 窗口内不足 2 帧时返回 **0** 而不是保留旧值 —— 保留旧值会让人以为
   * 画面还在更新。
   */
  get value() {
    const n = this.stamps.length;
    if (n < 2) return 0;
    const span = (this.stamps[n - 1] - this.stamps[0]) / 1000;
    return span > 0 ? (n - 1) / span : 0;
  }

  /** 显示用：整数，或 0。 */
  get display() {
    const v = this.value;
    return v > 0 ? v.toFixed(0) : "0";
  }

  reset() { this.stamps.length = 0; }

  _trim(now) {
    const cutoff = now - this.windowMs;
    let drop = 0;
    for (const s of this.stamps) {
      if (s < cutoff) drop += 1; else break;
    }
    if (drop) this.stamps.splice(0, drop);
  }
}
