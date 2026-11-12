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


def test_snapshot_structure_config_survives_source_removal(tmp_path) -> None:
    """训练期间源模型目录可能被清理；开训前必须留一份结构配置快照。"""
    m = _load()
    src = _write(tmp_path / "models" / "s" / "w" / "config.yaml",
                 "model:\n  tex_size: 128\n")
    snap = m.snapshot_structure_config(src, tmp_path / "out")
    src.unlink()
    assert snap.is_file()
    assert "tex_size: 128" in snap.read_text(encoding="utf-8")


def test_snapshot_structure_config_reports_missing_source(tmp_path) -> None:
    m = _load()
    with pytest.raises(FileNotFoundError):
        m.snapshot_structure_config(tmp_path / "nope.yaml", tmp_path / "out")


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


def test_refine_plan_and_objective_follow_flags() -> None:
    """默认只动外观；只有显式给 --lr-xyz / --ssim-weight / --lpips-weight 才变。"""
    m = load_script("refine_from_images.py")
    base = ["--subject", "a", "--work-name", "b"]
    a = m.parse_args(base)
    p = m.build_plan(a)
    assert "xyz" not in p.groups, "默认不该动位置"
    assert p.groups["f_dc"].lr > 0 and p.groups["opacity"].lr > 0
    o = m.build_objective(a)
    assert (o.l1, o.ssim, o.lpips) == (1.0, 0.0, 0.0)

    a2 = m.parse_args(base + ["--lr-xyz", "2e-5", "--ssim-weight", "0.2",
                              "--lpips-weight", "0.1"])
    p2 = m.build_plan(a2)
    assert "xyz" in p2.groups
    assert p2.groups["xyz"].lr == pytest.approx(2e-5)
    o2 = m.build_objective(a2)
    assert o2.ssim == pytest.approx(0.2)
    assert o2.lpips == pytest.approx(0.1)


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


def test_keep_interior_holes_drops_border_and_large() -> None:
    """只保留内部小洞：碰边界的、面积过大的都丢掉。"""
    import numpy as np

    m = load_script("pseudo_label.py")
    hole = np.zeros((40, 40), bool)
    hole[18:22, 18:22] = True          # 内部小洞：保留
    hole[0:4, 10:14] = True            # 碰上边界：丢
    hole[25:39, 25:39] = True          # 大片：丢（>15%）

    kept, dropped = m.keep_interior_holes(hole, max_area=0.15, border=2)
    assert dropped == 2
    assert kept[19, 19] and kept.sum() == 16
    assert not kept[1, 11] and not kept[30, 30]


def test_pseudo_label_prompt_suppresses_clothes_and_makeup() -> None:
    m = load_script("pseudo_label.py")
    args = m.parse_args(["--subject", "a", "--work-name", "b"])
    assert "bare face" in args.prompt
    for word in ("clothing", "jewelry", "earrings", "makeup", "lipstick"):
        assert word in args.negative, word


def test_pseudo_label_defaults_cover_ghosting() -> None:
    """默认应修"破碎/虚影"而不只是洞：mask-mode=full、strength 足够动结构。"""
    m = load_script("pseudo_label.py")
    args = m.parse_args(["--subject", "a", "--work-name", "b"])
    assert args.mask_mode == "full"
    assert args.strength >= 0.5, "0.4 只改纹理，动不了虚影"
    assert args.spike_threshold > 0


def test_build_defect_mask_covers_ghosting() -> None:
    """full 模式比 holes 多覆盖"模型自评的高缺陷面"。"""
    import numpy as np

    m = load_script("pseudo_label.py")
    a = np.full((60, 60), 1.0, np.float32)
    a[25:30, 25:30] = 0.0                 # 内部洞
    surf = np.zeros((60, 60), bool)
    surf[10:50, 10:50] = True
    ids = np.full((60, 60), -1, np.int32)
    ids[10:30, 10:50] = 3                 # 高缺陷面
    ids[30:50, 10:50] = 4                 # 正常面
    face_mass = np.array([0.0, 0.0, 0.0, 0.9, 0.1])

    holes, _, _, _ = m.build_defect_mask(a, surf, ids, None, mode="holes")
    full, _, _, st = m.build_defect_mask(
        a, surf, ids, face_mass, mode="full", defect_quantile=0.5)
    assert full[15, 15] == 1, "高缺陷面应被覆盖"
    assert holes[15, 15] == 0, "holes 模式不该因为 face mass 扩进来"
    assert full[27, 27] and holes[27, 27], "两者都应包含内部洞"
    assert st["mask_ratio"] > float(holes.mean())


def test_build_defect_mask_respects_observed_faces() -> None:
    """未观测表面（脖子/肩部）不补：即使 alpha 有洞也不进掩码。"""
    import numpy as np

    m = load_script("pseudo_label.py")
    a = np.ones((60, 60), np.float32)
    a[20:25, 20:25] = 0.0                 # 观测过的洞：要补
    a[40:45, 40:45] = 0.0                 # 未观测的洞：不补
    surf = np.zeros((60, 60), bool)
    surf[10:50, 10:50] = True
    ids = np.full((60, 60), -1, np.int32)
    ids[10:50, 10:50] = 1
    observed = np.zeros((60, 60), bool)
    observed[10:50, 10:50] = True
    observed[40:45, 40:45] = False

    mask, _, _, st = m.build_defect_mask(
        a, surf, ids, None, observed=observed, mode="holes", dilate_px=0)
    assert mask[22, 22], "观测面的洞应保留"
    assert not mask[42, 42], "未观测面（缺失解剖结构）不该补"
    assert st["observed_px"] == int(observed.sum())


def test_build_defect_mask_rejects_bool_ids() -> None:
    """面 id 被误写成布尔掩码时必须报错，而不是用布尔数组索引 face_mass。"""
    import numpy as np

    m = load_script("pseudo_label.py")
    with pytest.raises(ValueError, match="面 id"):
        m.build_defect_mask(np.ones((8, 8), np.float32),
                            np.ones((8, 8), bool),
                            np.ones((8, 8), bool),
                            np.zeros(4, np.float32))


def test_observed_pixels_maps_face_ids() -> None:
    import numpy as np

    m = load_script("pseudo_label.py")
    ids = np.array([[-1, 0], [1, 2]])
    out = m.observed_pixels(ids, np.array([True, False, True]))
    assert out.tolist() == [[False, True], [False, True]]
    assert m.observed_pixels(ids).tolist() == [[False, True], [True, True]]


def test_composite_labels_keeps_render_outside_mask() -> None:
    """掩码外必须逐像素等于渲染图，否则 SD 的整图色偏会被学进去。"""
    import numpy as np

    m = load_script("pseudo_label.py")
    render = np.zeros((4, 4, 3), np.uint8)
    gen = np.full((4, 4, 3), 200, np.uint8)
    mask = np.zeros((4, 4), bool)
    mask[1:3, 1:3] = True
    out = m.composite_labels(render, gen, mask)
    assert out[0, 0].tolist() == [0, 0, 0]
    assert out[2, 2].tolist() == [200, 200, 200]


def test_unsharp_restores_gradient_energy() -> None:
    import numpy as np

    m = load_script("pseudo_label.py")
    img = np.full((32, 32, 3), 60.0, np.float32)
    img[:, 16:] = 200.0                     # 一个边缘
    full = np.ones((32, 32), bool)
    before = m.gradient_energy(img, full)
    after = m.gradient_energy(m.unsharp(img, 0.5), full)
    assert after > before * 1.1, "unsharp 应把梯度能量提上来"
    assert m.unsharp(img, 0.0).shape == img.shape
