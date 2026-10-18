"""测试的共用小工具。

**不再依赖任何外部仓库**：本项目的测试只验证本项目自己的代码。
"""

from __future__ import annotations

from pathlib import Path

# 项目根 = tests/support.py 的上一级
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"


def compare(a, b, atol: float, what: str = "") -> tuple[bool, str]:
    """比较两个张量是否在容差内接近。

    **自洽工具**：只用于"同一量用两种写法算，结果应一致"这类检查，
    不涉及任何外部参照实现。
    """
    import torch

    if a.shape != b.shape:
        return False, f"{what}: 形状不一致 {tuple(a.shape)} vs {tuple(b.shape)}"
    if not torch.allclose(a, b, atol=atol, rtol=0.0):
        d = float((a - b).abs().max())
        return False, f"{what}: max|Δ|={d:.3e} 超过容差 {atol:.0e}"
    return True, ""
