"""PoseDriver：头转必须与烘焙数据闭环，且只转头、不动胸。

评审 A2 的黄金测试：`turn(idx, 0)` 必须逐位等于数据集里的 mesh；大角度时
根权重高的颈根几乎不动，头骨顶点整体转动。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from live3dgsavatar.tracking.pose_drive import PoseDriver

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = REPO_ROOT / "datasets" / "bala"
FLAME = REPO_ROOT / "data" / "FLAME2020"


def _driver() -> PoseDriver:
    if not (DATASET / "render_inputs.npz").is_file():
        pytest.skip("缺少 datasets/bala（仅本地可用）")
    if not (FLAME / "generic_model.pkl").is_file():
        pytest.skip("缺少 FLAME 资产")
    return PoseDriver(DATASET, flame_dir=FLAME)


def test_turn_zero_is_identity() -> None:
    """dyaw=dpitch=0 时必须与烘焙 mesh 逐位一致（评审 A2 的闭环）。"""
    driver = _driver()
    with np.load(DATASET / "render_inputs.npz", allow_pickle=True) as d:
        mesh = np.asarray(d["mesh"], dtype=np.float32)
        weight = np.asarray(d["blend_weight"], dtype=np.float32)
    for idx in (0, min(500, driver.frames - 1)):
        turned = driver.turn(idx, 0.0, 0.0)
        err = float((turned.mesh - mesh[idx]).abs().max())
        assert err == 0.0, f"frame {idx}: turn(0) 与烘焙 mesh 差 {err:.3e} m"
        assert np.array_equal(turned.weight.numpy(), weight[idx]), \
            f"frame {idx}: turn(0) 不应改驱动参数"


def test_turn_rotates_head_not_neck_base() -> None:
    """大角度：头骨顶点明显移动，根权重高的颈根几乎不动。"""
    driver = _driver()
    with np.load(DATASET / "render_inputs.npz", allow_pickle=True) as d:
        mesh = np.asarray(d["mesh"], dtype=np.float32)
    idx = driver.base_index()
    turned = driver.turn(idx, 60.0, 0.0)
    moved = (turned.mesh - mesh[idx]).norm(dim=-1)
    w = driver.head_weight
    head = w > 0.99
    root = w < 0.2
    assert bool(head.any()) and bool(root.any())
    assert float(moved[head].mean()) > 0.03      # 头骨至少 3 cm
    assert float(moved[root].mean()) < 0.02      # 颈根最多 2 cm


def test_turn_is_cached() -> None:
    """同样参数第二次调用应命中缓存（同一对象）。"""
    driver = _driver()
    a = driver.turn(0, 30.0, 0.0)
    b = driver.turn(0, 30.0, 0.0)
    assert a is b


def test_head_weight_covers_teeth() -> None:
    """补牙后的 60 个牙齿顶点也必须跟随头转（权重来自 FLAME 的 LBS）。"""
    driver = _driver()
    assert driver.head_weight.shape[0] == 5083
    teeth = driver.head_weight[-60:]
    assert float(teeth.min()) > 0.5
