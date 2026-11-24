"""修复图参考帧选择的纯 CPU 测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def test_pick_ref_frames_uses_training_range_and_splits() -> None:
    """左右取训练段极值；ref=4 再加区间中点，ref=5 加两个三等分点。"""
    from make_repairs import _pick_ref_frames

    yaw = np.array([-10.0, 0.0, 10.0, -30.0, 40.0, 5.0, 100.0])
    # 训练段只到前 6 帧 → 区间 [-30, 40]，最后那个 100° 的留出帧不能选
    picked2 = _pick_ref_frames(yaw, 6, base_idx=1, count=2)
    assert [round(float(yaw[i])) for i in picked2] == [-30, 40]
    picked3 = _pick_ref_frames(yaw, 6, base_idx=1, count=3)
    assert sorted(round(float(yaw[i])) for i in picked3) == [-30, 5, 40]
    picked4 = _pick_ref_frames(yaw, 6, base_idx=1, count=4)
    assert 100.0 not in [float(yaw[i]) for i in picked4]
    assert len(picked4) == 4


def test_pick_ref_frames_skips_base_and_duplicates() -> None:
    from make_repairs import _pick_ref_frames

    yaw = np.array([0.0, 20.0, -20.0])
    picked = _pick_ref_frames(yaw, len(yaw), base_idx=0, count=2)
    assert sorted(picked) == [1, 2]
