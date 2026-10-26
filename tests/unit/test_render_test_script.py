"""`scripts/render_test.py` 里可以离线验证的部分（不需要 GPU）。

模型目录与 work-name 是**一一对应**的（前端也按这个约定找模型），
所以渲染测试只认一个 `model.ply`；`--ply` 只是显式覆盖。

这里覆盖三件事：

1. 帧偏移时 GT 必须跟着走（曾经只偏移渲染、不偏移 GT，PSNR 全是错的）
2. 每个实验的产物落在自己的目录里（不同 work-name 互不覆盖）
3. 对比图是"原图 | 渲染"两格横排，格子上带 PSNR 标签
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from script_loader import load_script

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "render_test.py"


def _load():
    return load_script(SCRIPT.name)


# ------------------------------------------------------------------ 拼图 --


def test_comparison_has_two_panels_side_by_side(tmp_path) -> None:
    m = _load()
    from PIL import Image

    panel = np.zeros((8, 8, 3), np.uint8)
    out = tmp_path / "diff" / "00000.png"
    m.write_comparison(out, [("target 00000", panel),
                             ("render  27.18 dB", panel)])

    img = Image.open(out)
    # 两格横排 + 4 px 间隔；高度 = 图高 + 标签条
    assert img.size == (8 * 2 + 4, 8 + 20)


def test_comparison_ignores_empty_panels(tmp_path) -> None:
    m = _load()
    out = tmp_path / "c.png"
    m.write_comparison(out, [])
    assert not out.exists()


# -------------------------------------------------------------- 产物目录 --


def test_output_dir_is_per_experiment() -> None:
    """产物按 `<对象>/<版本>/<数据集>` 分层，不同实验互不覆盖。"""
    src = SCRIPT.read_text(encoding="utf-8")
    assert '"render_test"' in src
    assert '/ args.subject / args.work_name / args.dataset' in src


def test_single_model_only() -> None:
    """模型目录与 work-name 一一对应：不再扫描多个 `.ply`。"""
    src = SCRIPT.read_text(encoding="utf-8")
    assert "resolve_models" not in src
    assert "model_aug" not in src


# ------------------------------------------------------------------ GT 偏移 --


def test_load_gt_follows_frame_offset(tmp_path) -> None:
    """`--frame-offset N` 时 GT 必须取第 N 帧起，否则 PSNR 无意义。"""
    from PIL import Image

    m = _load()
    gt_dir = tmp_path / "images"
    gt_dir.mkdir()
    for i in range(4):
        arr = np.full((4, 4, 4), i, np.uint8)
        Image.fromarray(arr).save(gt_dir / f"{i:05d}.png")

    class Args:
        pass

    args = Args()
    args.gt_dir = gt_dir
    got = m.load_gt(args, [0, 1], offset=2)
    assert sorted(got) == [0, 1]
    # 第 0 格应是数据集第 2 帧（像素值 2）
    assert int(got[0][0][0, 0, 0]) == 2
    assert int(got[1][0][0, 0, 0]) == 3


# ------------------------------------------------------ 参数与产物组织 --


def test_cli_takes_model_and_dataset(tmp_path) -> None:
    """两个参数：模型（subject/work-name）与数据集（datasets/<名字>）。"""
    m = _load()
    args = m.parse_args(["--subject", "crh_1", "--work-name", "run",
                         "--dataset", "crh_2", "--frames", "20"])
    assert args.subject == "crh_1"
    assert args.work_name == "run"
    assert args.dataset == "crh_2"
    assert args.frames == 20


def test_no_arbitrary_directory_option() -> None:
    """不再接受任意目录：数据集必须先由 build_dataset.sh 建好。

    这条约束保证"渲染的输入都是标准数据集"，避免两条并行的数据通道。
    """
    src = SCRIPT.read_text(encoding="utf-8")
    assert "--frames-dir" not in src
    assert "--cap4d" not in src
    assert 'add_argument("--data"' not in src


def test_output_dir_has_three_levels() -> None:
    """产物目录 = <对象>/<版本>/<数据集>，内部只有 report/images/diff。"""
    src = SCRIPT.read_text(encoding="utf-8")
    assert '/ args.subject / args.work_name / args.dataset' in src
    assert 'out_root / "images"' in src
    assert 'out_root / "diff"' in src
    assert 'out_root / "report.json"' in src


def test_scene_is_loaded_from_selected_dataset() -> None:
    """`--dataset` 选的数据集与模型 subject 可以不同名，场景必须按它加载。

    真实故障：日志显示 `data = datasets/crh_1_cap4d`（212 帧），
    却渲染了 2863 帧 —— `load_scene(cfg)` 只按 cfg 的 subject 找数据集，
    于是读成了 crh_1。
    """
    src = SCRIPT.read_text(encoding="utf-8")
    assert "data_dir=args.data" in src, "load_scene 必须显式传 --dataset 的目录"


def test_two_datasets_load_independently() -> None:
    """同一份 cfg，换 `data_dir` 就换帧数（功能层面锁住上面的契约）。"""
    from live3dgsavatar.config import load_config
    from live3dgsavatar.data.scene import load_scene

    root = REPO_ROOT / "datasets"
    a, b = root / "crh_1", root / "crh_2"
    if not (a.is_dir() and b.is_dir()):
        pytest.skip("缺少仓库内数据集")

    cfg = load_config()
    cfg.set("subject", "crh_1")          # 故意固定 subject
    cfg.set("runtime.device", "cpu")
    scene_a = load_scene(cfg, frames=4, data_dir=a)
    scene_b = load_scene(cfg, frames=4, data_dir=b)
    assert scene_a.num_frames == scene_b.num_frames == 4
    assert scene_a.total_frames != scene_b.total_frames, "两个数据集帧数应当不同"


def test_clean_previous_outputs_removes_stale_images(tmp_path) -> None:
    """换帧数/换数据集重跑时，旧的渲染图与对比图必须清掉。

    实测踩过：先跑 2863 帧中断，再跑 212 帧，images/ 里留了 1025 张混杂图。
    """
    m = _load()
    out = tmp_path / "exp"
    images = out / "images"
    diff = out / "diff"
    for d, n in ((images, 3), (diff, 2)):
        d.mkdir(parents=True)
        for i in range(n):
            (d / f"{i:05d}.png").write_bytes(b"x")
    (out / "report.json").write_text("{}", encoding="utf-8")

    removed = m.clean_previous_outputs(out, images)

    assert removed == 5
    assert list(images.glob("*.png")) == []
    assert list(diff.glob("*.png")) == []
    assert (out / "report.json").is_file(), "指标文件不该被清理"
