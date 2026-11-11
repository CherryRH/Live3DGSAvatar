"""`scripts/train.py` 里可离线验证的部分：结构配置的解析与回退。

真实训练要 GPU（CUDA 扩展 + nvdiffrast），这里只测"配置从哪来"。
这条回退链是有来由的：训练**新模型**时 `<models_dir>/<subject>/<work>/`
还不存在，若直接走 `resolve_model_config` 会报"还没有选择模型" ——
那句错误会让人以为数据或模型选错了，其实是"还没训过"。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from script_loader import load_script

from live3dgsavatar.data.models import promote_model

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "train.py"


def _load():
    return load_script(SCRIPT.name)


def _write(path: Path, text: str = "model:\n  tex_size: 128\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ------------------------------------------------------------------ 回退链 --


def test_explicit_config_wins(tmp_path) -> None:
    m = _load()
    explicit = _write(tmp_path / "custom.yaml")
    models = tmp_path / "models"
    _write(models / "s" / "w" / "config.yaml")
    assert m.resolve_structure_config(models, "s", "w", explicit=explicit) == explicit


def test_uses_matching_work_config_when_present(tmp_path) -> None:
    m = _load()
    models = tmp_path / "models"
    direct = _write(models / "s" / "w" / "config.yaml")
    _write(models / "s" / "other" / "config.yaml")
    assert m.resolve_structure_config(models, "s", "w") == direct


def test_falls_back_to_sibling_work(tmp_path) -> None:
    """训练新版本时没有 `<work>/config.yaml`，用同对象下任一版本的结构。"""
    m = _load()
    models = tmp_path / "models"
    sibling = _write(models / "s" / "test" / "config.yaml")
    assert m.resolve_structure_config(models, "s", "smoke") == sibling


def test_falls_back_to_repo_default(tmp_path) -> None:
    m = _load()
    models = tmp_path / "models"
    models.mkdir()
    default = _write(tmp_path / "configs" / "model.yaml")
    assert m.resolve_structure_config(models, "s", "w",
                                      repo_default=default) == default


def test_reports_clearly_when_nothing_found(tmp_path) -> None:
    """全都没有时要说明"训练新模型需要结构配置"，而不是"没选模型"。"""
    m = _load()
    models = tmp_path / "models"
    models.mkdir()
    with pytest.raises(FileNotFoundError, match="model-config"):
        m.resolve_structure_config(models, "s", "w")


def test_explicit_missing_file_is_reported(tmp_path) -> None:
    m = _load()
    with pytest.raises(FileNotFoundError, match="不存在"):
        m.resolve_structure_config(tmp_path, "s", "w",
                                   explicit=tmp_path / "nope.yaml")


# --------------------------------------------------------------- 仓库模板 --


def test_repo_template_is_valid() -> None:
    """`configs/model.yaml` 必须能被 `AvatarConfig` 读出来。"""
    from live3dgsavatar.core.avatar import AvatarConfig

    path = REPO_ROOT / "configs" / "model.yaml"
    if not path.is_file():
        pytest.skip("没有仓库模板")
    cfg = AvatarConfig.from_yaml(path)
    assert cfg.tex_size > 0 and cfg.num_basis_in > 0 and cfg.num_basis_blend > 0


# --------------------------------------------------------------- 模型落地 --


def test_promote_copies_model_structure_and_report(tmp_path) -> None:
    """把训练产物装进 `models/<对象>/<版本>/`，渲染侧才认得出来。"""
    m = _load()
    src = tmp_path / "out"
    src.mkdir()
    (src / "model.ply").write_bytes(b"ply-data")
    (src / "config.yaml").write_text("model:\n  tex_size: 256\n", encoding="utf-8")
    (src / "report.json").write_text("{\"best_psnr\": 29.0}", encoding="utf-8")

    target = tmp_path / "models" / "s" / "w"
    out = promote_model(source_ply=src / "model.ply",
                          structure_config=src / "config.yaml",
                          report=src / "report.json", target=target)

    assert out == target
    assert (target / "model.ply").read_bytes() == b"ply-data"
    assert (target / "config.yaml").is_file()
    assert (target / "report.json").is_file()


def test_promote_refuses_to_overwrite_without_force(tmp_path) -> None:
    """覆盖已有模型是破坏性操作，必须显式 --force。"""
    m = _load()
    src = tmp_path / "model.ply"
    src.write_bytes(b"x")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model: {}\n", encoding="utf-8")
    target = tmp_path / "models" / "s" / "w"
    promote_model(source_ply=src, structure_config=cfg, report=None, target=target)

    with pytest.raises(FileExistsError, match="force"):
        promote_model(source_ply=src, structure_config=cfg, report=None,
                        target=target)
    # 加了 force 就可以
    promote_model(source_ply=src, structure_config=cfg, report=None,
                    target=target, force=True)


def test_promote_reports_missing_source(tmp_path) -> None:
    m = _load()
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model: {}\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="找不到"):
        promote_model(source_ply=tmp_path / "nope.ply",
                        structure_config=cfg, report=None,
                        target=tmp_path / "models" / "s" / "w")


def test_promote_without_report_is_allowed(tmp_path) -> None:
    m = _load()
    src = tmp_path / "model.ply"
    src.write_bytes(b"x")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model: {}\n", encoding="utf-8")
    target = tmp_path / "models" / "s" / "w"
    promote_model(source_ply=src, structure_config=cfg, report=None, target=target)
    assert not (target / "report.json").exists()


# ------------------------------------------------------ 数据集构建脚本 --


def test_build_dataset_new_dataset_from_baked_source(tmp_path) -> None:
    """已烘焙的来源 → 数据集（images/ + render_inputs.npz）。

    这是 `build_dataset.sh` 的后半段：前端只负责拿到跟踪结果，整理交给这里。
    """
    import numpy as np

    m = load_script("build_dataset.py")
    src = tmp_path / "src"
    (src / "images").mkdir(parents=True)
    n, v = 3, 6
    np.savez(src / "render_inputs.npz",
             mesh=np.zeros((n, v, 3), np.float32),
             blend_weight=np.zeros((n, 129), np.float32),
             K=np.eye(3), w2v=np.eye(4),
             width=np.int64(8), height=np.int64(8),
             names=np.array([f"{i:05d}" for i in range(n)]))
    from PIL import Image

    for i in range(n):
        Image.fromarray(np.zeros((8, 8, 4), np.uint8)).save(
            src / "images" / f"{i:05d}.png")

    out = tmp_path / "datasets" / "ds1"
    m.build(source=src, out_dir=out, flame_dir=tmp_path,
            base_dataset=None, limit=0, force=False)

    assert (out / "images").is_dir()
    with np.load(out / "render_inputs.npz", allow_pickle=True) as d:
        for key in ("mesh", "blend_weight", "K", "w2v", "width", "height", "names"):
            assert key in d.files, key
        assert int(d["mesh"].shape[0]) == n


def test_build_dataset_refuses_to_overwrite(tmp_path) -> None:
    m = load_script("build_dataset.py")
    out = tmp_path / "ds"
    out.mkdir()
    (out / "render_inputs.npz").write_bytes(b"x")
    with pytest.raises(FileExistsError, match="force"):
        m.build(source=tmp_path, out_dir=out, flame_dir=tmp_path,
                base_dataset=None, force=False)


def test_build_dataset_requires_images_for_every_frame(tmp_path) -> None:
    """图像与跟踪结果必须一一对应：缺一张就报错，不能悄悄跳过。"""
    import numpy as np

    m = load_script("build_dataset.py")
    src = tmp_path / "src"
    (src / "images").mkdir(parents=True)
    np.savez(src / "render_inputs.npz",
             mesh=np.zeros((2, 6, 3), np.float32),
             blend_weight=np.zeros((2, 129), np.float32),
             K=np.eye(3), w2v=np.eye(4),
             width=np.int64(8), height=np.int64(8),
             names=np.array(["0", "1"]))
    with pytest.raises(FileNotFoundError, match="缺少第 0 帧"):
        m.build(source=src, out_dir=tmp_path / "out", flame_dir=tmp_path,
                base_dataset=None, force=False)


def test_build_dataset_renames_images_to_five_digits(tmp_path) -> None:
    """构建出的数据集图像一律 `00000.png …`，与 npz 行序一致。"""
    import numpy as np

    m = load_script("build_dataset.py")
    src = tmp_path / "src"
    (src / "images").mkdir(parents=True)
    n = 2
    np.savez(src / "render_inputs.npz",
             mesh=np.zeros((n, 6, 3), np.float32),
             blend_weight=np.zeros((n, 129), np.float32),
             K=np.eye(3), w2v=np.eye(4),
             width=np.int64(8), height=np.int64(8),
             names=np.array(["frame_1", "frame_2"]))
    from PIL import Image

    for name in ("frame_1", "frame_2"):
        Image.fromarray(np.zeros((8, 8, 4), np.uint8)).save(
            src / "images" / f"{name}.png")

    out = tmp_path / "ds"
    m.build(source=src, out_dir=out, flame_dir=tmp_path, base_dataset=None)
    assert sorted(p.name for p in (out / "images").iterdir()) == [
        "00000.png", "00001.png"]
    with np.load(out / "render_inputs.npz", allow_pickle=True) as d:
        assert list(d["names"]) == ["00000", "00001"]


# ------------------------------------------------------- 缺陷定位脚本 --


def test_find_defects_dry_run_plans_without_gpu() -> None:
    """`--dry-run` 必须在没有 GPU 的机器上把计划算出来（候选数、产物路径）。"""
    import io
    import contextlib

    root = REPO_ROOT / "datasets" / "crh_1"
    if not (root / "render_inputs.npz").is_file():
        pytest.skip("缺少仓库内数据集")

    m = load_script("find_defects.py")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = m.main(["--subject", "crh_1", "--work-name", "run",
                       "--dataset", "crh_1", "--dry-run"])

    out = buf.getvalue()
    assert code == 0
    assert "候选" in out and "55 视角" in out
    assert "output/defects" in out


def test_find_defects_requires_model_and_dataset() -> None:
    """模型或数据集不存在时给出单行错误，而不是堆栈。"""
    import io
    import contextlib

    m = load_script("find_defects.py")
    with pytest.raises(SystemExit) as err:
        with contextlib.redirect_stdout(io.StringIO()):
            m.main(["--subject", "crh_1", "--work-name", "nope",
                    "--dataset", "crh_1", "--dry-run"])
    assert "[error]" in str(err.value.code)


# ------------------------------------------------ 精修图微调（脚本工具） --


def test_foreground_iou_and_color_shift() -> None:
    import numpy as np

    m = load_script("refine_from_images.py")
    a = np.zeros((4, 4), bool)
    a[:2] = True
    b = np.zeros((4, 4), bool)
    b[1:3] = True
    assert m.foreground_iou(a, a) == pytest.approx(1.0)
    assert m.foreground_iou(a, b) == pytest.approx(1 / 3)
    assert m.foreground_iou(np.zeros((2, 2), bool), np.zeros((2, 2), bool)) == 1.0

    render = np.zeros((4, 4, 3), np.uint8)
    refined = np.full((4, 4, 3), 10, np.uint8)
    mask = np.ones((4, 4), bool)
    assert m.color_shift(render, refined, mask) == pytest.approx(10.0)


def test_composite_score_penalises_sharpness_loss() -> None:
    m = load_script("refine_from_images.py")
    assert m.composite_score(28.0, 0.90, guard=0.85, penalty=20) == 28.0
    # 锐度比 0.80，比 guard 低 0.05 → 扣 1 dB
    assert m.composite_score(28.0, 0.80, guard=0.85,
                             penalty=20) == pytest.approx(27.0)
    assert m.composite_score(28.0, 0.60, guard=0.85,
                             penalty=20) == pytest.approx(23.0)


def test_sharpness_ratio_of_same_image_is_one() -> None:
    import numpy as np

    m = load_script("refine_from_images.py")
    img = np.random.default_rng(0).integers(0, 255, (16, 16, 3), dtype=np.uint8)
    mask = np.ones((16, 16), bool)
    assert m.sharpness_ratio(img, img, mask) == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------------ 增强流程（顶层脚本） --


def test_syncfix_command_places_paths_and_extra_args() -> None:
    m = load_script("enhance_model.py")
    args = m.parse_args(["--subject", "a", "--work-name", "b",
                         "--syncfix-dir", "/tmp/SyncFix",
                         "--syncfix-model", "/tmp/w.pt",
                         "--syncfix-arg=--foo=1"])
    cmd = m.syncfix_command(args, views=7)
    assert cmd[0] == "python" and cmd[1].endswith("SyncFix/inference.py")
    # 默认 --syncfix-batch 4：批量不能超过它
    assert cmd[cmd.index("--n_per_pass") + 1] == "4"
    full = m.syncfix_command(m.parse_args(["--subject", "a", "--work-name", "b",
                                           "--syncfix-dir", "/tmp/S",
                                           "--syncfix-batch", "0"]), views=7)
    assert full[full.index("--n_per_pass") + 1] == "7", "0 表示不限"
    assert cmd[cmd.index("--model_path") + 1] == "/tmp/w.pt"
    assert "--foo=1" in cmd
    assert "{views}" in cmd and "{refs}" in cmd and "{raw}" in cmd


def test_syncfix_command_requires_dir() -> None:
    m = load_script("enhance_model.py")
    args = m.parse_args(["--subject", "a", "--work-name", "b"])
    with pytest.raises(RuntimeError, match="syncfix-dir"):
        m.syncfix_command(args, views=1)


def test_collect_refined_matches_by_stem(tmp_path) -> None:
    from PIL import Image

    m = load_script("enhance_model.py")
    raw = tmp_path / "raw"
    (raw / "sub").mkdir(parents=True)
    labels = ["v1", "v2"]
    for lab in labels:
        Image.new("RGB", (4, 4), (10, 20, 30)).save(raw / "sub" / f"{lab}.png")
    out = tmp_path / "refined"
    got = m.collect_refined(raw, out, labels)
    assert got == labels
    assert (out / "v1.png").is_file() and (out / "v2.png").is_file()


def test_collect_refined_falls_back_to_order(tmp_path) -> None:
    from PIL import Image

    m = load_script("enhance_model.py")
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in range(2):
        Image.new("RGB", (4, 4)).save(raw / f"result_{i}.png")
    out = tmp_path / "refined"
    got = m.collect_refined(raw, out, ["a", "b"])
    assert got == ["a", "b"]
    assert (out / "a.png").is_file() and (out / "b.png").is_file()


def test_collect_refined_rejects_incomplete_output(tmp_path) -> None:
    from PIL import Image

    m = load_script("enhance_model.py")
    raw = tmp_path / "raw"
    raw.mkdir()
    Image.new("RGB", (4, 4)).save(raw / "x.png")
    with pytest.raises(RuntimeError, match="少于"):
        m.collect_refined(raw, tmp_path / "refined", ["a", "b"])


def test_render_and_defect_metrics_extraction() -> None:
    m = load_script("enhance_model.py")
    report = {"quality": {"psnr_masked": {"median": 27.1, "min": 25.0,
                                          "max": 30.0, "n": 286},
                          "sharpness": {"ratio": 0.86}},
              "summary": {"hole_base": {"mean": 0.17}, "hole_new": {"mean": 0.05},
                          "hole_rel": -0.7,
                          "outside_base": {"mean": 0.28},
                          "outside_new": {"mean": 0.29},
                          "sharp_rel": -0.02}}
    rm = m.render_metrics(report)
    assert rm["psnr"] == 27.1 and rm["sharp_ratio"] == 0.86
    dm = m.defect_metrics(report)
    assert dm["hole_before"] == 0.17 and dm["hole_after"] == 0.05
    assert dm["outside_after"] == 0.29
    assert m.render_metrics(None) == {}
    assert m.defect_metrics(None) == {}


def test_syncfix_prefix_selection() -> None:
    """解释器优先级：显式路径 > 指定环境 > 自动检测 > 当前 python。"""
    m = load_script("enhance_model.py")

    base = ["--subject", "a", "--work-name", "b", "--syncfix-dir", "/tmp/S"]
    assert m.syncfix_prefix(m.parse_args(base)) == ["python"]

    env = m.parse_args(base + ["--syncfix-env", "syncfix"])
    assert m.syncfix_prefix(env) == ["conda", "run", "--no-capture-output",
                                     "-n", "syncfix", "python"]

    both = m.parse_args(base + ["--syncfix-env", "syncfix",
                                "--syncfix-python", "/x/python"])
    assert m.syncfix_prefix(both) == ["/x/python"], "显式解释器优先级最高"


def test_clear_images_removes_pngs_keeps_others(tmp_path) -> None:
    """清理旧产物：只删图片，保留 json 等。"""
    from PIL import Image

    m = load_script("enhance_model.py")
    d = tmp_path / "refined"
    d.mkdir()
    Image.new("RGB", (2, 2)).save(d / "a.png")
    Image.new("RGB", (2, 2)).save(d / "b.png")
    (d / "keep.json").write_text("{}", encoding="utf-8")

    assert m.clear_images(d) == 2
    assert not list(d.glob("*.png"))
    assert (d / "keep.json").is_file()
    assert m.clear_images(tmp_path / "nope") == 0


def test_syncfix_pass_size_limits_batch() -> None:
    """一次前向不能塞全部视角：84 张会 OOM，必须按批切。"""
    m = load_script("enhance_model.py")
    assert m._pass_size(84, 8) == 8
    assert m._pass_size(4, 8) == 4
    assert m._pass_size(84, 0) == 84, "0 表示不限（保持旧行为）"

    args = m.parse_args(["--subject", "a", "--work-name", "b",
                         "--syncfix-dir", "/tmp/S", "--syncfix-batch", "8"])
    cmd = m.syncfix_command(args, views=84)
    assert cmd[cmd.index("--n_per_pass") + 1] == "8"
    # 显式覆盖（重试时用）
    cmd2 = m.syncfix_command(args, views=84, batch=2)
    assert cmd2[cmd2.index("--n_per_pass") + 1] == "2"


# ------------------------------------------------ 伪标签脚本的小工具 --


def test_pseudo_label_mask_helpers() -> None:
    import numpy as np

    m = load_script("pseudo_label.py")
    hole = np.zeros((9, 9), bool)
    hole[4, 4] = True
    grown = m.dilate(hole, 2)
    assert grown.sum() > 1 and grown[2, 4] and grown[6, 4]

    a = np.zeros((4, 4), bool); a[:2] = True
    b = np.zeros((4, 4), bool); b[1:3] = True
    assert m.foreground_iou(a, b) == pytest.approx(1 / 3)
    assert m.masked_ssim(np.zeros((8, 8, 3), np.uint8),
                         np.zeros((8, 8, 3), np.uint8),
                         np.ones((8, 8), bool)) == pytest.approx(1.0)
