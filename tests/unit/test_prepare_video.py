"""视频预处理里可离线验证的裁剪统计。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from script_loader import load_script


def _load():
    return load_script("prepare_video.py")


def test_person_stats_finds_head_band() -> None:
    m = _load()
    alpha = np.zeros((100, 80), np.float32)
    alpha[20:80, 20:60] = 1.0          # 人像
    alpha[20:35, 25:55] = 1.0          # 头部带
    st = m.person_stats(alpha)
    assert st["top"] == 20
    assert 20 <= st["head_width"] <= 48
    assert st["center"] > 25


def test_smooth_stats_has_one_value_per_frame() -> None:
    m = _load()
    stats = [
        {"top": 10.0, "center": 30.0, "head_width": 20.0},
        {"top": 12.0, "center": 32.0, "head_width": 22.0},
        {"top": 14.0, "center": 34.0, "head_width": 24.0},
    ]
    top, center, width = m.smooth_stats(stats, 5, 12)
    assert top.shape == center.shape == width.shape == (12,)
    assert top[0] == 10.0 and width[-1] >= width[0]


def test_resolve_video_accepts_bare_stem(tmp_path: Path) -> None:
    m = _load()
    video = tmp_path / "crh_1.mp4"
    video.write_bytes(b"x")
    assert m.resolve_video(tmp_path / "crh_1") == video
