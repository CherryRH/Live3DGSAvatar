"""`scripts/render_test.py` 里可以离线验证的部分（不需要 GPU）。

模型目录与 work-name 是**一一对应**的（前端也按这个约定找模型），
所以渲染测试只认一个 `model.ply`；`--ply` 只是显式覆盖。

这里覆盖三件事：

1. 帧偏移时 GT 必须跟着走（曾经只偏移渲染、不偏移 GT，PSNR 全是错的）
2. 每个实验的产物落在自己的目录里（不同 work-name 互不覆盖）
3. 对比图是"原图 | 渲染"两格横排，格子上带 PSNR 标签
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from script_loader import load_script

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "render_test.py"


def _load():
    return load_script(SCRIPT.name)


# ------------------------------------------------------------------ 拼图 --


def test_comparison_has_two_panels_side_by_side(tmp_path) -> None:
    m = _load()
    from PIL import Image

    panel = np.zeros((8, 8, 3), np.uint8)
    out = tmp_path / "diff" / "00000.png"
    m.write_comparison(out, [("target 00000", panel),
                             ("render  27.18 dB", panel)])

    img = Image.open(out)
    # 两格横排 + 4 px 间隔；高度 = 图高 + 标签条
    assert img.size == (8 * 2 + 4, 8 + 20)


def test_comparison_ignores_empty_panels(tmp_path) -> None:
    m = _load()
    out = tmp_path / "c.png"
    m.write_comparison(out, [])
    assert not out.exists()


# -------------------------------------------------------------- 产物目录 --


def test_output_dir_is_per_experiment() -> None:
    """产物默认按 `<对象>/<版本>` 分目录，避免不同实验互相覆盖。"""
    src = SCRIPT.read_text(encoding="utf-8")
    assert '"render_test"' in src
    assert '/ str(cfg.get("subject")) / str(cfg.get("work_name") or "default")' in src


def test_single_model_only() -> None:
    """模型目录与 work-name 一一对应：不再扫描多个 `.ply`。"""
    src = SCRIPT.read_text(encoding="utf-8")
    assert "resolve_models" not in src
    assert "model_aug" not in src


# ------------------------------------------------------------------ GT 偏移 --


def test_load_gt_follows_frame_offset(tmp_path) -> None:
    """`--frame-offset N` 时 GT 必须取第 N 帧起，否则 PSNR 无意义。"""
    from PIL import Image

    m = _load()
    gt_dir = tmp_path / "images"
    gt_dir.mkdir()
    for i in range(4):
        arr = np.full((4, 4, 4), i, np.uint8)
        Image.fromarray(arr).save(gt_dir / f"{i:05d}.png")

    class Args:
        pass

    args = Args()
    args.gt_dir = gt_dir
    got = m.load_gt(args, [0, 1], offset=2)
    assert sorted(got) == [0, 1]
    # 第 0 格应是数据集第 2 帧（像素值 2）
    assert int(got[0][0][0, 0, 0]) == 2
    assert int(got[1][0][0, 0, 0]) == 3
