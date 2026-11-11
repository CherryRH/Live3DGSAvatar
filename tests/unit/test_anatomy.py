"""高斯解剖的纯统计工具（不需要 GPU）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from live3dgsavatar.evaluation.anatomy import (  # noqa: E402
    DEAD_OPACITY,
    allocate_targets,
    dead_mask,
    faces_without_gaussians,
    histogram_counts,
    rgb_from_sh_dc,
    sh_dc_from_rgb,
    summarize,
)


def test_summarize_handles_empty_and_values() -> None:
    assert summarize([]) == {"n": 0}
    s = summarize([1.0, 2.0, 3.0, 4.0], quantiles=(50,))
    assert s["n"] == 4 and s["min"] == 1.0 and s["max"] == 4.0
    assert s["p50"] == pytest.approx(2.5)
    assert "p90" not in s


def test_histogram_counts_matches_numpy_and_rejects_bad_edges() -> None:
    v = np.array([0.0, 0.5, 1.0, 2.5, 10.0])
    h = histogram_counts(v, [0.0, 1.0, 2.0, 11.0])
    assert h["counts"] == [2, 1, 2]
    with pytest.raises(ValueError, match="递增"):
        histogram_counts(v, [0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="两个"):
        histogram_counts(v, [0.0])


def test_dead_mask_uses_physical_opacity() -> None:
    op = np.array([0.0, DEAD_OPACITY * 0.5, DEAD_OPACITY, 0.5])
    assert dead_mask(op).tolist() == [True, True, False, False]


def test_faces_without_gaussians() -> None:
    face_id = np.array([0, 0, 2, 4])          # 面 1、3 没有高斯
    mask = faces_without_gaussians(face_id, 5)
    assert mask.tolist() == [False, True, False, True, False]
    # 越界/负值不算"使用过"
    assert faces_without_gaussians([-1, 99], 3).tolist() == [True, True, True]


def test_sh_dc_roundtrip() -> None:
    rgb = np.array([[0.0, 0.5, 1.0], [0.2, 0.4, 0.6]])
    back = rgb_from_sh_dc(sh_dc_from_rgb(rgb))
    assert np.allclose(back, rgb)


def test_allocate_targets_truncates_and_handles_empty() -> None:
    assert allocate_targets([3, 1, 2], 0).tolist() == []
    assert allocate_targets([], 5).tolist() == []
    assert allocate_targets([3, 1, 2], 5).tolist() == [3, 1, 2]
    assert allocate_targets([3, 1, 2], 2).tolist() == [3, 1]
