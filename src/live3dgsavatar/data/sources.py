"""训练数据源：把"数据集 / RDF 导出"统一成可以按 batch 取的样本流。

## 为什么单独一层

RGBAvatar 把取数逻辑写在 `dataset/flame_dataset.py` + 自制的
`DataLoader.get_batch()` 里：没有 epoch、没有 shuffle、没有验证集，
训练与评测各写一套。这里把"训练/验证怎么取"和"数据长什么样"分开：

- `ManifestSource`：读我们的 `render_inputs.npz` + `images/`（沿用 RGBAvatar 的切分：**最后 N 帧为测试**）
- 图像按需读取（小 LRU），避免把 3000 帧全塞进内存

样本契约：

```
mesh          [V,3] float32   世界空间顶点（5083，带牙齿）
blend_weight  [D]   float32   驱动参数（129）
image         [3,H,W] float32 前景已叠到背景上的 RGB，值域 [0,1]
mask          [1,H,W] bool    前景
```

背景合成在这里做：RGBAvatar 在 `step()` 里用 `gt_rgb * mask + bg * (1-mask)`，
我们提前合成，`Objective` 就只关心像素。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

#: 默认测试集帧数（RGBAvatar 的切分：`images[:-350]` 训练 / `[-350:]` 测试）
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

    def __init__(self, data_dir: Path, *, split: str = "train",
                 test_frames: int = DEFAULT_TEST_FRAMES,
                 device: str | torch.device = "cpu",
                 image_cache: int = 64,
                 background=(0.0, 0.0, 0.0),
                 seed: int = 0) -> None:
        if split not in ("train", "test", "all"):
            raise ValueError(f"split 只能是 train/test/all，实际 {split!r}")
        self.root = Path(data_dir)
        self.split = split
        self.device = device
        self.background = torch.tensor(background, dtype=torch.float32)
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._cache_limit = int(image_cache)
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

    def _load_image(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """读第 i 帧并合成背景。返回 `(rgb [3,H,W], mask [1,H,W])`。"""
        cached = self._cache.get(i)
        if cached is not None:
            return cached
        from PIL import Image

        path = self.root / "images" / f"{self._names[i]}.png"
        if not path.is_file():
            # 名字可能带了扩展名
            path = self.root / "images" / self._names[i]
        arr = np.array(Image.open(path))
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
        return Sample(
            mesh=torch.from_numpy(self._mesh[i].copy()).to(self.device),
            blend_weight=torch.from_numpy(self._weight[i].copy()).to(self.device),
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
