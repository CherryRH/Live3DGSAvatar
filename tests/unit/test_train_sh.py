"""训练入口：`train_base.sh` 只做基础训练，不掺后训练/加密流程。"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SH = REPO_ROOT / "train_base.sh"


def _run(script: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(script), *args], cwd=str(REPO_ROOT),
                          capture_output=True, text=True, check=False)


def test_train_sh_is_base_only() -> None:
    p = _run(TRAIN_SH, "--help")
    assert p.returncode == 0
    assert "--post-train" not in p.stdout
    assert "post_train" not in p.stdout
    # base 是"各向同性 + 高 lr + spill/alpha"的 RGBAvatar 风格配方，
    # 不参与加密/修复图；加密只在 train_avatar.sh 里
    body = TRAIN_SH.read_text()
    assert "--densify" not in body
    assert "model_rgb.yaml" in body
    assert "repair" not in body
