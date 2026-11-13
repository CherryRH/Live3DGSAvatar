"""数据集读取、场景、样本源、FLAME 与姿态。

显式包标记（`__init__.py`）是**打包**需要的：`setuptools.find_packages`
默认只收集含 `__init__.py` 的目录。缺了它，可编辑安装仍能靠命名空间包
import，但 wheel/非 editable 安装会静默漏掉整个 `data/`。
"""
