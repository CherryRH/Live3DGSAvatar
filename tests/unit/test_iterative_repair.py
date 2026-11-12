"""`iterative_repair.py` 的 in-place 落盘：覆盖模型但保留 config。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from script_loader import load_script


def test_install_in_place_overwrites_model_keeps_config(tmp_path) -> None:
    m = load_script("iterative_repair.py")
    models = tmp_path / "models"
    target = models / "biden" / "best"
    target.mkdir(parents=True)
    (target / "config.yaml").write_text("model:\n  tex_size: 256\n",
                                        encoding="utf-8")
    (target / "model.ply").write_bytes(b"old")

    src = tmp_path / "finetune" / "model.ply"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"new")
    rep = src.parent / "report.json"
    rep.write_text("{\"best_seconds\": 12.3}", encoding="utf-8")

    out = m.install_in_place(models, "biden", "best", src, rep)

    assert out == target
    assert (target / "model.ply").read_bytes() == b"new"
    assert (target / "report.json").is_file()
    # config 是模型结构，不能被 fine-tune 的 report 覆盖
    assert "tex_size: 256" in (target / "config.yaml").read_text(encoding="utf-8")


def test_crop_below_face_keeps_only_above_observed() -> None:
    """未观测的脖子/胸口不修：裁到最低观测像素附近。"""
    m = load_script("pseudo_label.py")
    mask = np.ones((10, 6), bool)
    core = np.ones((10, 6), bool)
    observed = np.zeros((10, 6), bool)
    observed[4, 2] = True
    m2, c2 = m.crop_below_face(mask, core, observed, margin=0)
    assert m2[:5].all() and not m2[5:].any()
    assert c2[:5].all() and not c2[5:].any()
    # margin 向下多保留 2 行
    m3, _ = m.crop_below_face(mask, core, observed, margin=2)
    assert m3[:7].all() and not m3[7:].any()
    # 没有观测信息时原样返回
    same, _ = m.crop_below_face(mask, core, None)
    assert np.array_equal(same, mask)


def test_merge_selected_reads_pseudo_accepted(tmp_path) -> None:
    """新布局不再写 defects/refined，直接从 rN/pseudo/accepted 合并到 cum。"""
    m = load_script("iterative_repair.py")
    run = tmp_path / "r1"
    src = run / "defects"
    (src / "views").mkdir(parents=True)
    (src / "refs").mkdir()
    (src / "selected.json").write_text(
        json.dumps({"views": {"chosen": ["a", "b"]}}), encoding="utf-8")
    accepted = run / "pseudo" / "accepted"
    accepted.mkdir(parents=True)
    (accepted / "a.png").write_bytes(b"a")
    (accepted / "b.png").write_bytes(b"b")

    dst = tmp_path / "cum"
    m._merge_selected(dst, src, keep={"a"}, refined_src=accepted)
    assert (dst / "refined" / "a.png").is_file()
    assert not (dst / "refined" / "b.png").exists()
    chosen = json.loads((dst / "selected.json").read_text(encoding="utf-8"))
    assert chosen["views"]["chosen"] == ["a"]

    # 不显式给 refined_src 时，也能从兄弟目录 pseudo/accepted 找到
    dst2 = tmp_path / "cum2"
    m._merge_selected(dst2, src, keep={"b"})
    assert (dst2 / "refined" / "b.png").is_file()
