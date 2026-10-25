"""训练数据源：把"数据集 / RDF 导出"统一成可以按 batch 取的样本流。

## 为什么单独一层

"数据长什么样"（网格 / 驱动参数 / 图像 / 掩码）与"训练怎么取"
（切分、shuffle、batch、验证遍历）是两件事。分开之后：


- `ManifestSource`：读 `render_inputs.npz` + `images/`，**末尾 N 帧作验证集**
- 图像按需读取（小 LRU），不必把几千帧全塞进内存

样本契约：

```
mesh          [V,3] float32   世界空间顶点（5083，带牙齿）
blend_weight  [D]   float32   驱动参数（129）
image         [3,H,W] float32 前景已叠到背景上的 RGB，值域 [0,1]
mask          [1,H,W] bool    前景
```

背景合成也在这里做：读取时就把掩码外填成背景色，
`Objective` 只处理像素，不必关心"这张图有没有透明通道"。
"""

from __future__ import annotations

import queue
import random
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

#: 默认验证集帧数：末尾 N 帧（`images[:-350]` 训练 / `images[-350:]` 验证）
DEFAULT_TEST_FRAMES = 350


@dataclass
class Sample:
    """一个训练样本。"""

    mesh: torch.Tensor
    blend_weight: torch.Tensor
    image: torch.Tensor
    mask: torch.Tensor
    name: str = ""


@dataclass
class Batch:
    """一批样本（已堆叠）。"""

    mesh: torch.Tensor
    blend_weight: torch.Tensor
    image: torch.Tensor
    mask: torch.Tensor

    def __len__(self) -> int:
        return self.image.shape[0]

    @property
    def rgb(self) -> torch.Tensor:
        """前景已合成到背景上的 RGB（数据源已做，这里只是语义别名）。"""
        return self.image


class ManifestSource:
    """读 `<data_dir>/render_inputs.npz` 与 `<data_dir>/images/`。

    Args:
        data_dir: 数据集目录
        split: ``"train"`` / ``"test"`` / ``"all"``
        test_frames: 末尾多少帧归测试集
        device: 样本放到哪个设备
        image_cache: 图像缓存条数（0 = 不缓存）
        background: 背景色 `[3]`；掩码外填它
        seed: shuffle 的随机种子
    """

    #: 图像缓存的默认文件名（`scripts/cache_images.py` 生成）
    CACHE_NAME = "images.npy"

    def __init__(self, data_dir: Path, *, split: str = "train",
                 test_frames: int = DEFAULT_TEST_FRAMES,
                 device: str | torch.device = "cpu",
                 image_cache: int = 64,
                 background=(0.0, 0.0, 0.0),
                 seed: int = 0,
                 use_cache: bool = True,
                 preload: bool = True) -> None:
        if split not in ("train", "test", "all"):
            raise ValueError(f"split 只能是 train/test/all，实际 {split!r}")
        self.root = Path(data_dir)
        self.split = split
        self.device = device
        self.background = torch.tensor(background, dtype=torch.float32)
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._cache_limit = int(image_cache)
        self._memmap = None
        self._preload = bool(preload)
        self._use_cache = bool(use_cache)
        self._rng = random.Random(seed)

        manifest = self.root / "render_inputs.npz"
        if not manifest.is_file():
            raise FileNotFoundError(f"找不到 {manifest}")
        with np.load(manifest, allow_pickle=True) as d:
            self._mesh = np.asarray(d["mesh"], dtype=np.float32)
            self._weight = np.asarray(d["blend_weight"], dtype=np.float32)
            self._names = [str(n) for n in d["names"]] if "names" in d.files else [
                f"{i:05d}" for i in range(self._mesh.shape[0])]
        self._n_all = self._mesh.shape[0]
        if self._use_cache:
            cache = self.root / self.CACHE_NAME
            if cache.is_file():
                self._memmap = np.load(cache, mmap_mode="r")
                if self._memmap.shape[0] < self._n_all:
                    raise ValueError(
                        f"{cache} 只有 {self._memmap.shape[0]} 帧，"
                        f"少于数据集的 {self._n_all} 帧")
        # 网格/驱动参数常驻设备：每帧都要用，反复 H2D 是浪费
        self._preloaded = None
        if self._preload:
            self._preloaded = (
                torch.from_numpy(self._mesh).to(self.device),
                torch.from_numpy(self._weight).to(self.device),
            )
        self._index = self._select_split(split, test_frames)
        if not self._index:
            raise ValueError(f"{self.root} 的 {split} 切分为空")

    # ------------------------------------------------------------ 基本信息 --

    def _select_split(self, split: str, test_frames: int) -> list[int]:
        n = self._n_all
        k = min(int(test_frames), n - 1) if n > 1 else 0
        if split == "all":
            return list(range(n))
        if split == "train":
            return list(range(0, n - k)) if k > 0 else list(range(n))
        return list(range(n - k, n))

    def __len__(self) -> int:
        return len(self._index)

    @property
    def names(self) -> list[str]:
        return [self._names[i] for i in self._index]

    # ------------------------------------------------------------ 取样本 --

    @property
    def has_image_cache(self) -> bool:
        """是否存在 `images.npy` 缓存（有则跳过 PNG 解码）。"""
        return self._memmap is not None

    def _read_array(self, i: int) -> np.ndarray:
        """取第 i 帧的原始像素（HWC, uint8）。有缓存走 memmap，否则解 PNG。"""
        if self._memmap is not None:
            return np.asarray(self._memmap[i])
        from PIL import Image

        path = self.root / "images" / f"{self._names[i]}.png"
        if not path.is_file():
            path = self.root / "images" / self._names[i]
        return np.array(Image.open(path))

    def _load_image(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """读第 i 帧并合成背景。返回 `(rgb [3,H,W], mask [1,H,W])`。"""
        cached = self._cache.get(i)
        if cached is not None:
            return cached

        arr = self._read_array(i)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        # HWC → CHW，与渲染输出 [3,H,W] 一致
        rgb = torch.from_numpy(arr[..., :3].copy()).float().permute(2, 0, 1) / 255.0
        if arr.shape[2] == 4:
            mask = torch.from_numpy((arr[..., 3] > 127).copy())[None].bool()
        else:
            mask = torch.ones(1, *rgb.shape[-2:], dtype=torch.bool)
        bg = self.background.reshape(3, 1, 1)
        rgb = torch.where(mask, rgb, bg.expand_as(rgb))
        item = (rgb, mask)
        if self._cache_limit > 0:
            if len(self._cache) >= self._cache_limit:
                self._cache.pop(next(iter(self._cache)))
            self._cache[i] = item
        return item

    def sample(self, i: int) -> Sample:
        rgb, mask = self._load_image(i)
        if self._preloaded is not None:
            mesh_all, weight_all = self._preloaded
            mesh, weight = mesh_all[i], weight_all[i]
        else:
            mesh = torch.from_numpy(self._mesh[i].copy()).to(self.device)
            weight = torch.from_numpy(self._weight[i].copy()).to(self.device)
        return Sample(
            mesh=mesh,
            blend_weight=weight,
            image=rgb.to(self.device),
            mask=mask.to(self.device),
            name=self._names[i],
        )

    def batch(self, indices: list[int]) -> Batch:
        items = [self.sample(i) for i in indices]
        return Batch(
            mesh=torch.stack([s.mesh for s in items]),
            blend_weight=torch.stack([s.blend_weight for s in items]),
            image=torch.stack([s.image for s in items]),
            mask=torch.stack([s.mask for s in items]),
        )

    # ------------------------------------------------------------ 迭代 --

    def random_indices(self, n: int) -> list[int]:
        """随机取 n 个帧号（有放回，训练用）。"""
        return [self._rng.choice(self._index) for _ in range(int(n))]

    def random_batch(self, batch_size: int) -> Batch:
        return self.batch(self.random_indices(batch_size))

    def iter_batches(self, batch_size: int, *, shuffle: bool = False):
        """顺序/打乱遍历整份切分（验证用）。"""
        idx = list(self._index)
        if shuffle:
            self._rng.shuffle(idx)
        for start in range(0, len(idx), batch_size):
            yield self.batch(idx[start:start + batch_size])


class PrefetchSource:
    """在后台线程里预取 batch，让取数（PNG 解码 / H2D 拷贝）与 GPU 计算重叠。

    训练时 GPU 利用率长期上不去，绝大多数情况是**取数在主线程串行做**：
    解一帧 512² PNG 要几毫秒，一个 batch 就是几十毫秒，而一步计算只有十几
    毫秒 —— GPU 大部分时间在等数据。

    用法：

        src = PrefetchSource(ManifestSource(...), batch_size=10, workers=2, depth=4)
        ... Trainer(source=src, ...)      # 训练结束会自动 close()

    `random_batch(n)` 的 `n` 会被忽略（必须与 `batch_size` 一致），
    因为预取队列按固定批大小生产。
    """

    def __init__(self, inner, batch_size: int, *, workers: int = 2,
                 depth: int = 4) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size 必须为正，实际 {batch_size}")
        self.inner = inner
        self.batch_size = int(batch_size)
        self.workers = max(1, int(workers))
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(depth)))
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._error: BaseException | None = None
        self._started = False

    # ------------------------------------------------------------ 生命周期 --

    def start(self) -> "PrefetchSource":
        if self._started:
            return self
        self._started = True
        for k in range(self.workers):
            th = threading.Thread(target=self._work, name=f"prefetch-{k}",
                                  daemon=True)
            th.start()
            self._threads.append(th)
        return self

    def close(self) -> None:
        self._stop.set()
        # 唤醒可能阻塞在 put 上的线程
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        for th in self._threads:
            th.join(timeout=1.0)
        self._threads.clear()

    def __enter__(self) -> "PrefetchSource":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------ 取数 --

    def _work(self) -> None:
        while not self._stop.is_set():
            try:
                batch = self.inner.random_batch(self.batch_size)
            except BaseException as e:                        # noqa: BLE001
                self._error = e
                return
            while not self._stop.is_set():
                try:
                    self._queue.put(batch, timeout=0.2)
                    break
                except queue.Full:
                    continue

    def random_batch(self, batch_size: int):
        if not self._started:
            self.start()
        if int(batch_size) != self.batch_size:
            raise ValueError(
                f"PrefetchSource 固定批大小 {self.batch_size}，"
                f"调用方给了 {batch_size}")
        while True:
            if self._error is not None:
                raise self._error
            try:
                return self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

    def __len__(self) -> int:
        return len(self.inner)

    def iter_batches(self, batch_size: int, shuffle: bool = False):
        """验证遍历直接透传（预取对顺序遍历没意义，反而占内存）。"""
        return self.inner.iter_batches(batch_size, shuffle=shuffle)
