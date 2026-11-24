"""训练入口不得再硬编码"共用训练参数"（评审：改了 train.yaml 不生效）。"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ("train_base.sh", "train_rgb.sh", "train_avatar.sh")

# 这些必须来自 configs/train.yaml（或命令行 -- 透传），不能写在脚本里
FORBIDDEN = (
    "--iterations 10000",
    "--batch-size 8",
    "--patience 4",
    "--min-delta 0.05",
    "--eval-every 500",
    "--blend-start 0",
)


def test_train_scripts_do_not_hardcode_common_params() -> None:
    for name in SCRIPTS:
        text = (REPO_ROOT / name).read_text(encoding="utf-8")
        for token in FORBIDDEN:
            assert token not in text, f"{name} 仍硬编码 {token!r}"


def test_train_yaml_holds_common_params() -> None:
    import yaml

    cfg = yaml.safe_load((REPO_ROOT / "configs/train.yaml").read_text())
    pipe = cfg["train"]["pipeline"]
    assert pipe["iterations"] > 0
    # 配方文件里不应再重复步数/早停这类共用项
    for recipe in ("rgb", "base", "avatar"):
        data = yaml.safe_load(
            (REPO_ROOT / "configs" / f"train_{recipe}.yaml").read_text())
        for key in ("iterations", "batch_size", "patience", "eval_every",
                    "min_delta", "blend_start"):
            assert key not in data, f"train_{recipe}.yaml 不应包含共用项 {key}"
