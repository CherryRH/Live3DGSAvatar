"""`iterative_repair.py` 的 in-place 落盘：覆盖模型但保留 config。"""

from __future__ import annotations

from pathlib import Path

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
