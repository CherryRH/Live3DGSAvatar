"""统一指标（`core/metrics.py`）。

这些函数被三处共用：渲染测试、在线更新对比、GUI 实时对比。口径必须唯一，
否则"同一个模型在两个界面显示不同 PSNR"这种问题极难排查。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import live3dgsavatar  # noqa: E402, F401

from live3dgsavatar.core.metrics import (PSNR_CAP, Accum,  # noqa: E402
                                         max_abs_diff, psnr, psnr_masked,
                                         psnr_pair)


def test_psnr_matches_definition() -> None:
    a = np.zeros((4, 4, 3), np.uint8)
    b = np.full((4, 4, 3), 128, np.uint8)
    # MSE = 128² → PSNR = 10·log10(255²/128²) ≈ 5.99 dB
    assert psnr(a, b) == pytest.approx(10 * np.log10(255 ** 2 / 128 ** 2),
                                       rel=1e-5)


def test_identical_images_use_cap_not_inf() -> None:
    """完全相同记为 99 dB：inf 无法聚合，也不能写进 JSON。"""
    a = np.zeros((2, 2, 3), np.uint8)
    assert psnr(a, a) == PSNR_CAP
    assert psnr_pair(a, a, np.ones((2, 2), bool)) == (PSNR_CAP, PSNR_CAP)


def test_psnr_masked_ignores_outside_mask() -> None:
    target = np.zeros((8, 8, 3), np.uint8)
    pred = target.copy()
    pred[:4] = 255                      # 掩码外
    mask = np.zeros((8, 8), bool)
    mask[4:] = True                     # 只看下半
    assert psnr_masked(pred, target, mask) == PSNR_CAP


def test_psnr_masked_without_mask_returns_none() -> None:
    a = np.zeros((4, 4, 3), np.uint8)
    assert psnr_masked(a, a, None) is None
    assert psnr_masked(a, a, np.zeros((4, 4), bool)) is None


def test_psnr_pair_reports_shape_mismatch() -> None:
    a = np.zeros((4, 4, 3), np.uint8)
    b = np.zeros((8, 8, 3), np.uint8)
    assert psnr_pair(a, b) == (None, None)


def test_max_abs_diff_is_in_255_scale() -> None:
    a = np.zeros((4, 4, 3), np.uint8)
    b = np.full((4, 4, 3), 200, np.uint8)
    assert max_abs_diff(a, b) == 200.0


def test_accum_ignores_none_and_nan() -> None:
    acc = Accum()
    for v in (1.0, None, float("nan"), 3.0):
        acc.add(v)
    s = acc.summary()
    assert s is not None and s["n"] == 2 and s["median"] == 2.0
    assert len(acc) == 2


def test_accum_empty_summary_is_none() -> None:
    assert Accum().summary() is None
