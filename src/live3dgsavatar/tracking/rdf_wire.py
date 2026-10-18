"""桥接进程 ↔ 本项目的二进制帧格式（**纯函数，不碰 socket**）。

## 为什么自己定格式而不上 protobuf/msgpack

每帧 61 KB、30 fps，回环上瓶颈在光栅化而不在序列化。自定义定长二进制
零依赖、可 `struct` 直接解析，也便于用 `nc` 抓包调试。

## 布局

握手（连接后一次）::

    magic     4s   b"RDFH"
    version   H    协议版本
    n_verts   I    顶点数（RDF 原生 FLAME 为 5023）
    n_exp     I    表情维数（100）
    width     H    输入图像宽
    height    H    输入图像高
    identity  300f 该序列的 identity 参数（本端不用于计算，仅记录/诊断）

每帧::

    magic     4s   b"RDFF"
    seq       I    帧序号（回绕可忽略）
    timestamp d    桥接侧的时间戳（秒，time.perf_counter 域）
    x         n_exp+18 f    RDF 的 118 维参数
    verts     n_verts*3 f   RDF 用**它自己的** FLAME2023 算出的顶点

顶点由桥接侧算好再发 —— 本端因此不需要 FLAME2023，
也就不存在"用 2020 的基算 2023 的参数"这种错误。

`timestamp` 用桥接侧的 `perf_counter`：两端在不同进程/可能不同机器，
墙钟不可比，但"两帧之间的间隔"仍可用（用到达时间即可判断新鲜度）。
"""

from __future__ import annotations

import struct

import numpy as np

#: 握手魔数。
MAGIC_HELLO = b"RDFH"
#: 帧魔数。保留它是为了让读端在错位时能重新同步。
MAGIC_FRAME = b"RDFF"

#: 协议版本；两端不一致时应显式报错，而不是画出错位的图。
VERSION = 1

#: RDF 的姿态段维数（5 关节 × 3）。
POSE_DIM = 15
#: RDF 的平移段维数。
TRANS_DIM = 3
#: 除表情外的固定部分。
EXTRA_DIM = POSE_DIM + TRANS_DIM
#: identity 维数。
DIM_IDENTITY = 300

_HELLO_HEAD = struct.Struct("<4sHIIHH")
_FRAME_HEAD = struct.Struct("<4sId")


def hello_bytes(
    *,
    num_vertices: int,
    num_expressions: int,
    width: int,
    height: int,
    identity: np.ndarray | None = None,
) -> bytes:
    """打包握手包。"""
    ident = (np.zeros(DIM_IDENTITY, np.float32) if identity is None
             else np.asarray(identity, np.float32).reshape(-1))
    if ident.size != DIM_IDENTITY:
        raise ValueError(f"identity 应为 {DIM_IDENTITY} 维，实际 {ident.size}")
    head = _HELLO_HEAD.pack(MAGIC_HELLO, VERSION, int(num_vertices),
                            int(num_expressions), int(width), int(height))
    return head + ident.tobytes()


def hello_size() -> int:
    return _HELLO_HEAD.size + DIM_IDENTITY * 4


def unpack_hello(buf: bytes) -> dict:
    """解析握手包（长度必须正好）。"""
    if len(buf) != hello_size():
        raise ValueError(f"握手掌长度应为 {hello_size()}，实际 {len(buf)}")
    magic, version, n_verts, n_exp, width, height = _HELLO_HEAD.unpack_from(buf, 0)
    if magic != MAGIC_HELLO:
        raise ValueError(f"握手掌魔数不对：{magic!r}")
    if version != VERSION:
        raise ValueError(
            f"协议版本不匹配：桥接 {version}，本端 {VERSION}。请同步两端代码。")
    ident = np.frombuffer(buf, dtype=np.float32, count=DIM_IDENTITY,
                          offset=_HELLO_HEAD.size).copy()
    return {
        "version": version,
        "num_vertices": int(n_verts),
        "num_expressions": int(n_exp),
        "width": int(width),
        "height": int(height),
        "identity": ident,
    }


def x_dim(num_expressions: int) -> int:
    return int(num_expressions) + EXTRA_DIM


def frame_bytes(seq: int, timestamp: float, x: np.ndarray, verts: np.ndarray) -> bytes:
    """打包一帧。"""
    x = np.asarray(x, np.float32).reshape(-1)
    verts = np.asarray(verts, np.float32)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"verts 形状应为 [V,3]，实际 {tuple(verts.shape)}")
    head = _FRAME_HEAD.pack(MAGIC_FRAME, int(seq), float(timestamp))
    return head + x.tobytes() + verts.astype(np.float32, copy=False).tobytes()


def frame_size(num_vertices: int, num_expressions: int) -> int:
    return _FRAME_HEAD.size + x_dim(num_expressions) * 4 + int(num_vertices) * 3 * 4


def frame_head_size() -> int:
    """帧头长度（读端先读它，再按已知尺寸读剩余部分）。"""
    return _FRAME_HEAD.size


def unpack_frame_head(buf: bytes) -> tuple[int, float]:
    """"解析帧头，返回 `(seq, timestamp)`。"""
    if len(buf) != _FRAME_HEAD.size:
        raise ValueError(f"帧头长度应为 {_FRAME_HEAD.size}，实际 {len(buf)}")
    magic, seq, ts = _FRAME_HEAD.unpack(buf)
    if magic != MAGIC_FRAME:
        raise ValueError(f"帧魔数不对：{magic!r}（数据流可能已错位）")
    return int(seq), float(ts)


def unpack_frame(body: bytes, *, num_vertices: int, num_expressions: int) -> dict:
    """解析帧体（**不含**帧头），返回 `{"x": [D], "verts": [V,3]}`。"""
    dim_x = x_dim(num_expressions)
    expect = dim_x * 4 + int(num_vertices) * 3 * 4
    if len(body) != expect:
        raise ValueError(f"帧体长度应为 {expect}，实际 {len(body)}")

    x = np.frombuffer(body, dtype=np.float32, count=dim_x).copy()
    verts = np.frombuffer(body, dtype=np.float32,
                          count=int(num_vertices) * 3,
                          offset=dim_x * 4).reshape(int(num_vertices), 3).copy()
    return {"x": x, "verts": verts}


# --------------------------------------------------------------- 便捷 --


class FrameCodec:
    """按握手信息把"帧头 + 帧体"合成一个对象，读端用起来更省事。"""

    def __init__(self, hello: dict) -> None:
        self.num_vertices = int(hello["num_vertices"])
        self.num_expressions = int(hello["num_expressions"])
        self.width = int(hello["width"])
        self.height = int(hello["height"])
        self.identity = hello["identity"]

    @property
    def body_size(self) -> int:
        return frame_size(self.num_vertices, self.num_expressions) - frame_head_size()

    def decode(self, head: bytes, body: bytes) -> dict:
        seq, ts = unpack_frame_head(head)
        payload = unpack_frame(body, num_vertices=self.num_vertices,
                               num_expressions=self.num_expressions)
        return {"seq": seq, "timestamp": ts, **payload}
