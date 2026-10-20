"""追踪服务 ↔ 本项目的二进制协议（**纯函数，不碰 socket**）。

## 设计：自描述的请求/响应，没有握手

原来的设计是"桥接进程推送、本端被动接收"，为此需要一次握手告知顶点数/表情维数。
改成**通用追踪服务**（画面进、结果出）后就不需要了：

- 谁都能连：数据集回放、摄像头、单张图片，只要按格式发一帧即可
- **回复自带计数**（`n_verts`、`n_exp`），客户端不必预先知道 RDF 的配置
- 无状态：没有"必须先握手否则解析错位"的隐式约束

## 布局

请求（客户端 → 服务）::

    magic     4s   b"RDFQ"
    seq       I    帧序号（服务原样回填）
    width     H    画面宽
    height    H    画面高
    rgb       w*h*3 uint8   RGB888，行优先

回复（服务 → 客户端）::

    magic     4s   b"RDFF"
    seq       I    与请求相同
    timestamp d    服务侧时间戳（秒，perf_counter 域，仅供诊断）
    n_verts   H    顶点数（RDF 原生 FLAME 为 5023）
    n_exp     H    表情维数（100）
    x         n_exp+18 f    RDF 参数
    verts     n_verts*3 f   RDF 用**它自己的** FLAME2023 算出的顶点

顶点由服务侧算好再发 —— 本端因此不需要 FLAME2023，
也就不存在"用 2020 的基算 2023 的参数"这种错误。

`timestamp` 用服务侧的 `perf_counter`：两端在不同进程/可能不同机器，
墙钟不可比，判断新鲜度请用**本端到达时间**。
"""

from __future__ import annotations

import struct

import numpy as np

#: 回复魔数。保留它是为了让读端在错位时能重新同步。
MAGIC_FRAME = b"RDFF"
#: 请求（客户端 → 追踪服务）：一帧原始画面。
MAGIC_REQUEST = b"RDFQ"

#: 协议版本；两端不一致时应显式报错，而不是画出错位的图。
VERSION = 1

#: 姿态段维数（5 关节 × 3）。
POSE_DIM = 15
#: 平移段维数。
TRANS_DIM = 3
#: 除表情外的固定部分。
EXTRA_DIM = POSE_DIM + TRANS_DIM

_REQUEST_HEAD = struct.Struct("<4sIHH")
_FRAME_HEAD = struct.Struct("<4sIdHH")


def x_dim(num_expressions: int) -> int:
    """参数向量维数：表情 + 18。"""
    return int(num_expressions) + EXTRA_DIM


# --------------------------------------------------------------- 请求 --


def request_bytes(seq: int, rgb: np.ndarray) -> bytes:
    """打包一帧请求。`rgb` 为 `[H,W,3]` uint8。"""
    arr = np.ascontiguousarray(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"rgb 形状应为 [H,W,3]，实际 {tuple(arr.shape)}")
    h, w = int(arr.shape[0]), int(arr.shape[1])
    if not (0 < w < 65536 and 0 < h < 65536):
        raise ValueError(f"尺寸超出协议范围：{w}x{h}")
    head = _REQUEST_HEAD.pack(MAGIC_REQUEST, int(seq), w, h)
    return head + arr.tobytes()


def request_size(width: int, height: int) -> int:
    return _REQUEST_HEAD.size + int(width) * int(height) * 3


def request_head_size() -> int:
    return _REQUEST_HEAD.size


def unpack_request_head(buf: bytes) -> tuple[int, int, int]:
    """解析请求头，返回 `(seq, width, height)`。"""
    if len(buf) != _REQUEST_HEAD.size:
        raise ValueError(f"请求头长度应为 {_REQUEST_HEAD.size}，实际 {len(buf)}")
    magic, seq, w, h = _REQUEST_HEAD.unpack(buf)
    if magic != MAGIC_REQUEST:
        raise ValueError(f"请求魔数不对：{magic!r}（期望 {MAGIC_REQUEST!r}）")
    if w == 0 or h == 0:
        raise ValueError(f"请求尺寸非法：{w}x{h}")
    return int(seq), int(w), int(h)


def unpack_request(body: bytes, *, width: int, height: int) -> dict:
    """解析请求体（**不含**头），返回 `{"rgb": [H,W,3] uint8}`。

    **零拷贝**：返回的数组直接视图到 `body`，调用方需保证 `body` 在这期间存活。
    """
    expect = int(width) * int(height) * 3
    if len(body) != expect:
        raise ValueError(f"请求体长度应为 {expect}，实际 {len(body)}")
    rgb = np.frombuffer(body, dtype=np.uint8).reshape(int(height), int(width), 3)
    return {"rgb": rgb}


# --------------------------------------------------------------- 回复 --


def frame_bytes(seq: int, timestamp: float, x: np.ndarray,
                verts: np.ndarray) -> bytes:
    """打包一帧回复。计数从 `x` / `verts` 的形状推出来，调用方不必传。"""
    x = np.asarray(x, np.float32).reshape(-1)
    verts = np.asarray(verts, np.float32)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"verts 形状应为 [V,3]，实际 {tuple(verts.shape)}")
    n_verts = int(verts.shape[0])
    n_exp = int(x.size - EXTRA_DIM)
    if n_exp < 0:
        raise ValueError(f"x 维数 {x.size} 小于固定部分 {EXTRA_DIM}")

    head = _FRAME_HEAD.pack(MAGIC_FRAME, int(seq), float(timestamp),
                            n_verts, n_exp)
    return head + x.tobytes() + verts.astype(np.float32, copy=False).tobytes()


def frame_size(num_vertices: int, num_expressions: int) -> int:
    return (_FRAME_HEAD.size + x_dim(num_expressions) * 4
            + int(num_vertices) * 3 * 4)


def frame_head_size() -> int:
    """回复头长度（读端先读它，再按头里的计数读剩余部分）。"""
    return _FRAME_HEAD.size


def unpack_frame_head(buf: bytes) -> tuple[int, float, int, int]:
    """解析回复头，返回 `(seq, timestamp, n_verts, n_exp)`。"""
    if len(buf) != _FRAME_HEAD.size:
        raise ValueError(f"回复头长度应为 {_FRAME_HEAD.size}，实际 {len(buf)}")
    magic, seq, ts, n_verts, n_exp = _FRAME_HEAD.unpack(buf)
    if magic != MAGIC_FRAME:
        raise ValueError(f"回复魔数不对：{magic!r}（数据流可能已错位）")
    return int(seq), float(ts), int(n_verts), int(n_exp)


def unpack_frame(body: bytes, *, num_vertices: int, num_expressions: int) -> dict:
    """解析回复体（**不含**头），返回 `{"x": [D], "verts": [V,3]}`。"""
    dim_x = x_dim(num_expressions)
    expect = dim_x * 4 + int(num_vertices) * 3 * 4
    if len(body) != expect:
        raise ValueError(f"回复体长度应为 {expect}，实际 {len(body)}")

    x = np.frombuffer(body, dtype=np.float32, count=dim_x).copy()
    verts = np.frombuffer(body, dtype=np.float32,
                          count=int(num_vertices) * 3,
                          offset=dim_x * 4).reshape(int(num_vertices), 3).copy()
    return {"x": x, "verts": verts}


def body_size(num_vertices: int, num_expressions: int) -> int:
    """回复体长度（头之后的部分）。"""
    return frame_size(num_vertices, num_expressions) - frame_head_size()
