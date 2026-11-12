"""训练/后训练入口：train.sh 只训练，post_train.sh 独立跑后训练。"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SH = REPO_ROOT / "train.sh"
POST_SH = REPO_ROOT / "post_train.sh"


def _run(script: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(script), *args], cwd=str(REPO_ROOT),
                          capture_output=True, text=True, check=False)


def _plan(*extra: str) -> dict[str, str]:
    p = _run(POST_SH, "--subject", "s", "--work-name", "w", "--plan", *extra)
    assert p.returncode == 0, p.stderr
    out = {}
    for line in p.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    return out


def _post_cfg() -> dict | None:
    try:
        import yaml
    except ImportError:
        return None
    with (REPO_ROOT / "configs" / "train.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)["train"]["pipeline"]["post_train"]


def test_train_sh_is_base_only() -> None:
    p = _run(TRAIN_SH, "--help")
    assert p.returncode == 0
    assert "--post-train" not in p.stdout
    assert "post_train.sh" in p.stdout


def test_post_plan_defaults() -> None:
    plan = _plan()
    assert plan["post_repair_mode"] == "mask_diffusion"
    assert plan["post_rings"] == "auto"
    assert plan["post_steps"] == "100"
    assert plan["post_syncfix_ready"].lower() == "false"
    assert plan["post_prefix"] == "w_post"


def test_post_in_place_and_joint_mode() -> None:
    plan = _plan("--in-place")
    assert plan["post_prefix"] == "w"
    plan = _plan("--repair-mode", "joint_syncfix")
    assert plan["post_repair_mode"] == "joint_syncfix"


def test_post_config_matches_plan() -> None:
    cfg = _post_cfg()
    if cfg is None:
        return
    plan = _plan()
    assert plan["post_repair_mode"] == str(cfg["repair_mode"])
    assert plan["post_rings"] == str(cfg["rings"])
    assert int(plan["post_steps"]) == int(cfg["steps"])
    assert float(plan["post_quality_tol"]) == float(cfg["quality_tol"])
    assert plan["post_base_model"] == str(cfg["base_model"])
