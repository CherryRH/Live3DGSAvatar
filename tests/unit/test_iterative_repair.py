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


def test_merge_selected_reads_syncfix_labels(tmp_path) -> None:
    """SyncFix 标签直接合并到 cum/refined。"""
    m = load_script("iterative_repair.py")
    run = tmp_path / "r1"
    src = run / "defects"
    (src / "views").mkdir(parents=True)
    (src / "refs").mkdir()
    (src / "selected.json").write_text(
        json.dumps({"views": {"chosen": ["a", "b"]}}), encoding="utf-8")
    accepted = run / "syncfix"
    accepted.mkdir(parents=True)
    (accepted / "a.png").write_bytes(b"a")
    (accepted / "b.png").write_bytes(b"b")

    dst = tmp_path / "cum"
    m._merge_selected(dst, src, keep={"a"}, refined_src=accepted)
    assert (dst / "refined" / "a.png").is_file()
    assert not (dst / "refined" / "b.png").exists()
    chosen = json.loads((dst / "selected.json").read_text(encoding="utf-8"))
    assert chosen["views"]["chosen"] == ["a"]

    # 不显式给 refined_src 时，也能从 src/refined 找到（兼容旧布局）
    legacy = src / "refined"
    legacy.mkdir()
    (legacy / "b.png").write_bytes(b"b")
    dst2 = tmp_path / "cum2"
    m._merge_selected(dst2, src, keep={"b"})
    assert (dst2 / "refined" / "b.png").is_file()
