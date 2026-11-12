"""`train.sh` 的后训练稳定路线：入口能解析、和 `configs/train.yaml` 一致。

这些参数曾经在三处各写一份（train.sh / train.py 默认 / yaml），结果
mask 外扩在 12/16/24 之间漂移，文档和实际跑的对不上。这里把 train.sh 的
`--plan` 输出当作唯一事实来源，和 yaml 逐项比对，防止再漂。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SH = REPO_ROOT / "train.sh"

# train.sh --plan 的键 → configs/train.yaml 的 post_train 键
PLAN_TO_CONFIG = {
    "post_base_model": "base_model",
    "post_ip_adapter": "ip_adapter",
    "post_top_n": "top_n",
    "post_rings": "rings",
    "post_steps": "steps",
    "post_real_per_batch": "real_per_batch",
    "post_refined_per_batch": "refined_per_batch",
    "post_lr_xyz": "lr_xyz",
    "post_mask_dilate": "mask_dilate",
    "post_core_dilate": "core_dilate",
    "post_strength": "strength",
    "post_min_fill": "min_fill",
    "post_trusted_tol": "trusted_tol",
    "post_sharpen": "sharpen",
    "post_min_sharp": "min_sharp",
    "post_min_mask_sharp": "min_mask_sharp",
    "post_quality_tol": "quality_tol",
    "post_psnr_weight": "psnr_weight",
    "post_feather": "feather",
    "post_color_ring": "color_ring",
    "post_head_cut": "head_cut",
    "post_head_cut_margin": "head_cut_margin",
}


def _plan(*extra: str) -> dict[str, str]:
    p = subprocess.run(
        ["bash", str(TRAIN_SH), "--subject", "s", "--work-name", "w",
         "--plan", *extra],
        cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
    assert p.returncode == 0, p.stderr
    out = {}
    for line in p.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key] = value
    return out


def _post_train_config() -> dict | None:
    try:
        import yaml
    except ImportError:  # 零依赖测试环境：跳过一致性检查
        return None
    path = REPO_ROOT / "configs" / "train.yaml"
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)["train"]["pipeline"]["post_train"]


def _same(plan_value: str, cfg_value: object) -> bool:
    if isinstance(cfg_value, bool):
        return plan_value.lower() == str(cfg_value).lower()
    try:
        return float(plan_value) == float(cfg_value)
    except (TypeError, ValueError):
        return plan_value == str(cfg_value)


def test_train_sh_has_valid_shell_syntax() -> None:
    p = subprocess.run(["bash", "-n", str(TRAIN_SH)],
                       capture_output=True, text=True, check=False)
    assert p.returncode == 0, p.stderr


def test_plan_is_the_stable_epic_route() -> None:
    """路线不变量：epiC(SD1.5) + normal ControlNet，无身份/颜色对齐。

    具体数值由 configs/train.yaml 决定（`test_config_matches_train_sh_route`
    逐项比对），这里只钉住不应该随调参变化的部分。
    """
    plan = _plan()
    assert plan["post_base_model"] == "emilianJR/epiCRealism"
    assert plan["post_ip_adapter"] == "none"
    assert plan["post_color_match"].lower() == "false"
    assert plan["post_controlnet"] == "lllyasviel/control_v11p_sd15_normalbae"
    rings = plan["post_rings"]
    assert rings == "auto" or rings.count(";") == 3, rings


def test_plan_defaults_to_base_only() -> None:
    plan = _plan()
    assert plan["post_enabled"] == "0"
    assert plan["post_only"] == "0"
    assert plan["post_in_place"] == "0"


def test_post_flags_toggle_the_second_stage() -> None:
    plan = _plan("--post-train")
    assert plan["post_enabled"] == "1" and plan["post_only"] == "0"
    plan = _plan("--post-only")
    assert plan["post_enabled"] == "1" and plan["post_only"] == "1"
    in_place = _plan("--post-train", "--in-place")
    assert in_place["post_in_place"] == "1"
    # 原地更新时 prefix 就是 work 名（报告叫 best_report.json，不是 best_post_*）
    assert in_place["post_prefix"] == "w"
    assert in_place["post_dir"] == "output/post/s/w"
    # 默认 prefix 跟随 work 名；显式 --prefix 覆盖
    assert plan["post_prefix"] == "w_post"
    assert plan["post_dir"] == "output/post/s/w_post"
    plan = _plan("--post-train", "--prefix", "mix")
    assert plan["post_prefix"] == "mix"


def test_config_matches_train_sh_route() -> None:
    """两处默认值必须一致：train.sh 是入口，yaml 是 train.py 直调的后备。"""
    cfg = _post_train_config()
    if cfg is None:
        return
    plan = _plan()
    for plan_key, cfg_key in PLAN_TO_CONFIG.items():
        assert cfg_key in cfg, f"configs/train.yaml 缺少 post_train.{cfg_key}"
        assert _same(plan[plan_key], cfg[cfg_key]), (
            f"{cfg_key}: train.sh={plan[plan_key]!r} yaml={cfg[cfg_key]!r}")
    assert cfg["enabled"] is False, "train.sh 自己编排后训练，yaml 默认应关闭"
