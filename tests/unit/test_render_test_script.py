"""`scripts/render_test.py` 的多模型流程（不需要 GPU 的部分）。

一次渲染模型目录下的**所有** PLY（`model.ply` 与 `model_aug.ply` 等），
每个模型自己的 `images/` 与 `report.json` 分开，最后再出一份统一的
横向对比图（原图 | 模型1 | 模型2 …，每格带 PSNR 标签）。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "render_test.py"


def _load():
    if "_render_test_script" in sys.modules:
        return sys.modules["_render_test_script"]
    spec = importlib.util.spec_from_file_location("_render_test_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_render_test_script"] = mod
    spec.loader.exec_module(mod)
    return mod


class _Cfg:
    def __init__(self, **kw):
        self._v = kw

    def get(self, key, default=None):
        return self._v.get(key, default)


class _Args:
    def __init__(self, ply=None):
        self.ply = ply


def _make_models(tmp_path: Path, names: list[str]) -> Path:
    d = tmp_path / "models" / "crh_1" / "run"
    d.mkdir(parents=True)
    for name in names:
        (d / name).write_bytes(b"ply")
    (d / "notes.txt").write_text("ignore me", encoding="utf-8")
    return d


def _cfg(tmp_path: Path) -> _Cfg:
    return _Cfg(**{"paths.models_dir": str(tmp_path / "models"),
                   "subject": "crh_1", "work_name": "run"})


# --------------------------------------------------------------- 模型发现 --


def test_discovers_all_ply_and_puts_model_first(tmp_path) -> None:
    m = _load()
    _make_models(tmp_path, ["model_aug.ply", "model.ply", "model_old.ply"])
    found = m.resolve_models(_Args(), _cfg(tmp_path))

    assert [n for n, _ in found] == ["model", "model_aug", "model_old"]
    assert all(p.suffix == ".ply" for _, p in found)


def test_explicit_ply_short_circuits_discovery(tmp_path) -> None:
    m = _load()
    d = _make_models(tmp_path, ["model.ply", "model_aug.ply"])
    found = m.resolve_models(_Args(d / "model_aug.ply"), _cfg(tmp_path))
    assert found == [("model_aug", d / "model_aug.ply")]


def test_missing_models_reports_how_to_fix(tmp_path) -> None:
    m = _load()
    (tmp_path / "models" / "crh_1" / "run").mkdir(parents=True)
    with pytest.raises(SystemExit, match="train.py --promote"):
        m.resolve_models(_Args(), _cfg(tmp_path))


# ------------------------------------------------------------------ 拼图 --


def test_comparison_is_one_row_with_labels(tmp_path) -> None:
    m = _load()
    from PIL import Image

    panel = np.zeros((8, 8, 3), np.uint8)
    out = tmp_path / "compare" / "00000.png"
    m.write_comparison(out, [("target 00000", panel),
                             ("model 30.95 dB", panel),
                             ("model_aug 31.02 dB", panel)])

    img = Image.open(out)
    # 三格横排 + 两处 4 px 间隔；高度 = 图高 + 标签条
    assert img.size == (8 * 3 + 4 * 2, 8 + 20)


def test_comparison_ignores_empty_panels(tmp_path) -> None:
    m = _load()
    out = tmp_path / "c.png"
    m.write_comparison(out, [])
    assert not out.exists()


# ------------------------------------------------------------ 多模型循环 --


def test_loop_uses_per_model_output_dirs(tmp_path) -> None:
    """每个模型的 images/ 与 report.json 必须落在自己的子目录里。"""
    src = SCRIPT.read_text(encoding="utf-8")
    assert 'model_out = out_root / name' in src
    assert 'report_path = model_out / "report.json"' in src
    assert 'share_dir = None if args.no_save else model_out / "images"' in src
    # 统一对比图与汇总表
    assert 'compare_dir = out_root / "compare"' in src
    assert 'write_comparison(compare_dir' in src
