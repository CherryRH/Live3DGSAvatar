"""推流编码（`streaming/`）：JPEG 编码与协议打包。

带宽是远端部署的头号瓶颈（raw 一帧 768 KiB，30 fps 就要 190 Mbps）。
这里守三件事：

1. **形状/dtype 校验**：编码器只接受 `[H,W,3] uint8`，错了要立刻报
2. **压缩率与成本**：在真实渲染图上必须显著小于 raw，且单帧耗时不失控
3. **协议**：编码帧与原始帧共用帧头，前端按类型字节区分
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import live3dgsavatar  # noqa: E402, F401

from live3dgsavatar.app import protocol as P  # noqa: E402
from live3dgsavatar.streaming import codec  # noqa: E402

_IMAGE = Path.home() / "Datasets" / "INSTA" / "crh_2" / "images" / "00000.png"


def _photo() -> np.ndarray:
    """真实渲染图（黑底人像）；没有数据集时退化为平滑渐变。"""
    if _IMAGE.is_file():
        from PIL import Image

        return np.asarray(Image.open(_IMAGE).convert("RGB"))
    # 渐变比噪声更接近渲染结果，压缩率也更真实
    y, x = np.mgrid[0:512, 0:512]
    rgb = np.stack([x * 255 // 511, y * 255 // 511,
                    ((x + y) * 255 // 1022)], axis=-1)
    return rgb.astype(np.uint8)


# ------------------------------------------------------------- 编码 --


def test_encode_jpeg_rejects_bad_shape_and_dtype() -> None:
    with pytest.raises(ValueError, match="H,W,3"):
        codec.encode_jpeg(np.zeros((8, 8), np.uint8))
    with pytest.raises(ValueError, match="uint8"):
        codec.encode_jpeg(np.zeros((8, 8, 3), np.float32))
    with pytest.raises(ValueError, match="quality"):
        codec.encode_jpeg(np.zeros((8, 8, 3), np.uint8), 0)


def test_encode_jpeg_returns_decodable_bytes() -> None:
    """编出来必须能被解回来，且尺寸一致。"""
    from io import BytesIO

    from PIL import Image

    rgb = _photo()
    data = codec.encode_jpeg(rgb, 85)
    assert data[:2] == b"\xff\xd8", "应是 JPEG（SOI 标记）"

    back = np.asarray(Image.open(BytesIO(data)).convert("RGB"))
    assert back.shape == rgb.shape


def test_encode_jpeg_is_much_smaller_than_raw() -> None:
    """压缩比是这次改动的**目的**：至少 10×，否则不如发 raw。"""
    rgb = _photo()
    raw_bytes = rgb.size
    jpeg = codec.encode_jpeg(rgb, 85)
    ratio = raw_bytes / len(jpeg)
    assert ratio > 10, f"压缩只有 {ratio:.1f}×（{len(jpeg)} 字节），不值得"


def test_encode_cost_is_bounded() -> None:
    """单帧编码要远小于一帧的渲染时间预算（否则变成新瓶颈）。"""
    import time

    rgb = _photo()
    codec.encode_jpeg(rgb, 85)                    # 预热
    t0 = time.perf_counter()
    for _ in range(10):
        codec.encode_jpeg(rgb, 85)
    per_frame_ms = (time.perf_counter() - t0) / 10 * 1000
    assert per_frame_ms < 20.0, f"编码 {per_frame_ms:.1f} ms，太慢了"


def test_quality_controls_size() -> None:
    rgb = _photo()
    low = codec.encode_jpeg(rgb, 60)
    high = codec.encode_jpeg(rgb, 95)
    assert len(high) > len(low), "高质量应该更大"


# ------------------------------------------------------------- 协议 --


def test_pack_blob_roundtrip() -> None:
    blob = b"\xff\xd8\xff\xe0fakejpeg"
    buf = P.pack_blob(42, blob, kind=P.MSG_RENDER_JPEG, psnr=22.5)
    kind, seq, psnr, body = P.unpack_frame(buf)

    assert kind == P.MSG_RENDER_JPEG
    assert seq == 42
    assert abs(psnr - 22.5) < 1e-6
    assert body == blob
    assert len(buf) == P.FRAME_HEADER_BYTES + len(blob)


def test_pack_blob_rejects_empty_and_wrong_type() -> None:
    with pytest.raises(ValueError, match="为空"):
        P.pack_blob(1, b"", kind=P.MSG_RENDER_JPEG)
    with pytest.raises(TypeError, match="bytes"):
        P.pack_blob(1, "not-bytes", kind=P.MSG_RENDER_JPEG)  # type: ignore[arg-type]


def test_encoded_kinds_are_distinct_from_raw() -> None:
    """两个编码类型都必须不是 raw（否则前端会按定长去校验而报错）。"""
    assert P.MSG_RENDER_JPEG not in (P.MSG_RENDER, P.MSG_TARGET)
    assert P.MSG_TARGET_JPEG not in (P.MSG_RENDER, P.MSG_TARGET)
    assert len({P.MSG_RENDER, P.MSG_TARGET,
                P.MSG_RENDER_JPEG, P.MSG_TARGET_JPEG}) == 4
