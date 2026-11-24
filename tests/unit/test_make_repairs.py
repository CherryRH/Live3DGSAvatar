"""修复图参考帧选择的纯 CPU 测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def test_pick_ref_frames_uses_signed_yaw() -> None:
    """±20 必须分别落在左右两侧，不能都挑到正脸附近。"""
    from make_repairs import _pick_ref_frames

    # 前 10 帧：正面附近；后面是 ±20/±35
    yaw = np.array([0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 5.0,
                    20.0, -20.0, 35.0, -35.0])
    picked = _pick_ref_frames(yaw, len(yaw), base_idx=0,
                              targets=[20.0, -20.0, 35.0, -35.0])
    assert len(picked) == 4
    assert yaw[picked[0]] == 20.0
    assert yaw[picked[1]] == -20.0
    assert yaw[picked[2]] == 35.0
    assert yaw[picked[3]] == -35.0


def test_pick_ref_frames_skips_base_and_duplicates() -> None:
    from make_repairs import _pick_ref_frames

    yaw = np.array([0.0, 20.0, -20.0])
    picked = _pick_ref_frames(yaw, len(yaw), base_idx=0, targets=[20.0, -20.0, 20.0])
    assert picked == [1, 2]
