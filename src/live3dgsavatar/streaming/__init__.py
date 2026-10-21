"""传输层：把渲染帧压成适合跨网络发送的字节流。

与 `core/render` 的分工：那边负责"算出图"，这边只负责"把图变小"。
两者都不该知道对方的存在 —— 编码器只接受 `[H,W,3] uint8`。
"""

from .codec import encode_jpeg, jpeg_available

__all__ = ["encode_jpeg", "jpeg_available"]
