"""给追踪服务提供画面的**取帧器**。

追踪服务是"画面进、参数出"的通用服务，所以"画面从哪来"完全不归它管：
数据集回放、摄像头、单张图片都可以。这里先实现数据集（`input/` 原始画面），
摄像头将来只需再加一个同类实现。

## 为什么不复用 `data/scene.py`

`scene` 读的是**已经追踪好的参数与网格**（`render_inputs.npz` / `checkpoint`）。
走追踪服务时我们要的恰恰是**未追踪的画面**，两者来源不同、用途相反，
混在一起会让"到底走没走 RDF"变得不清楚。

## 用哪个目录

| 目录 | 内容 | 优先级 |
|---|---|---|
| `input/` | 视频裁出的**原始画面**（自然背景）| 首选（与真摄像头同一分布，已实测可追踪）|
| `images/` | **已抠图**的人像（黑底 + alpha）| 回退（有些数据集只有这一份）|

默认 `auto`：有 `input/` 就用它，否则退到 `images/`。这样**每个数据集都能
喂追踪服务**，同时优先走验证过的那条路。可用 `tracking.frame_subdir` 强制其一。

已知差异：`images/` 是黑底、边缘带 alpha 过渡，与追踪器训练时的自然图像分布不同；
人脸检测仍能工作，但属于**未实测**的输入，所以默认不优先选它。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

#: 支持的图片后缀
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


class DatasetFrames:
    """按顺序（可循环）读取数据集的 `input/` 原始画面。

    Args:
        data_dir: 数据集目录（需含 `input/`）
        fps: 取帧节奏上限；``0`` 表示不限（以追踪速度为准）
        loop: 读完后是否回到第一帧
        start: 起始帧

    Raises:
        FileNotFoundError: 没有 `input/` 目录，或里面没有图片
    """

    kind = "dataset"

    def __init__(self, data_dir: Path, *, fps: float = 30.0, loop: bool = True,
                 start: int = 0, subdir: str = "auto") -> None:
        data_dir = Path(data_dir)
        self.dir = self._pick_dir(data_dir, subdir)
        self.subdir = self.dir.name
        self.files = sorted(p for p in self.dir.iterdir()
                            if p.suffix.lower() in IMAGE_SUFFIXES)
        if not self.files:
            raise FileNotFoundError(f"{self.dir.name}/ 里没有图片：{self.dir}")
        self.fps = float(fps)
        self.loop = bool(loop)
        self._index = max(0, int(start)) % len(self.files)
        self._last = 0.0

    @staticmethod
    def _pick_dir(data_dir: Path, subdir: str) -> Path:
        """按 `auto` / `input` / `images` 选目录，缺失时报可执行的错。"""
        choice = str(subdir or "auto").strip().lower()
        if choice not in ("auto", "input", "images"):
            raise ValueError(
                f"tracking.frame_subdir 只能是 auto | input | images，实际 {subdir!r}")

        candidates = (["input", "images"] if choice == "auto" else [choice])
        for name in candidates:
            d = data_dir / name
            if d.is_dir() and any(p.suffix.lower() in IMAGE_SUFFIXES
                                  for p in d.iterdir()):
                return d

        tried = " / ".join(f"{n}/" for n in candidates)
        raise FileNotFoundError(
            f"数据集里没有可用画面目录（找过 {tried}）：{data_dir}\n"
            "实时驱动需要逐帧画面；`input/` 是原始画面，`images/` 是抠图人像。")

    @property
    def num_frames(self) -> int:
        return len(self.files)

    @property
    def index(self) -> int:
        return self._index

    def read_at(self, index: int) -> np.ndarray | None:
        """读**指定帧号**的图（不推进自身索引、不节流）。

        播放条要能跳帧与暂停，所以取帧必须能"按号取"。
        节流由调用方控制：实时驱动是靠追踪本身的耗时（约 60–70 ms）限速的，
        不需要再叠一层。
        """
        i = int(index) % len(self.files) if self.loop else max(0, int(index))
        if i >= len(self.files):
            return None
        return _read_rgb(self.files[i])

    def read(self) -> np.ndarray | None:
        """返回 `[H,W,3]` uint8；不循环且读完时返回 ``None``。

        内部按 `fps` 节流：追踪服务比取帧快时不该空转把画面一次性灌进去。
        """
        if self._index >= len(self.files):
            if not self.loop:
                return None
            self._index = 0

        if self.fps > 0:
            interval = 1.0 / self.fps
            rest = interval - (time.perf_counter() - self._last)
            if rest > 0:
                time.sleep(rest)
        self._last = time.perf_counter()

        path = self.files[self._index]
        self._index += 1
        return _read_rgb(path)

    def close(self) -> None:
        """无资源可释放（对称接口，方便以后换成摄像头）。"""


def _read_rgb(path: Path) -> np.ndarray | None:
    """读成 `[H,W,3]` uint8 RGB。失败返回 ``None``（由调用方跳过）。"""
    from PIL import Image

    try:
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError) as e:
        log.warning("读图失败 %s：%s", path, e)
        return None
