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
    # 加密已收敛到 train_avatar.sh，train_base.sh 不再提它
    assert "post_train" not in p.stdout
    assert "train_avatar.sh" in p.stdout
