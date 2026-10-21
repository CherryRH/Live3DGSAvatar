"""综合性能脚本（`scripts/bench.py`）的可测部分 —— **不需要 GPU / RDF**。

能在这里测的是"数字怎么算"和"参数怎么校验"，真正跑 GPU 的部分只能在
目标机器上跑（本仓库的 CI 环境没有 GPU）。

守三件事：

1. **统计口径**：用中位数而不是均值（单帧抖动会把均值拉高），帧率由中位数折算
2. **参数校验**：未知的段、非法帧数要立刻报错，而不是跑到一半才发现
3. **路径解析**：`--dry-run` 不依赖 GPU，且数据集缺失时**说清楚原因**
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "bench.py"


def _load():
    """当模块加载脚本（注册到 `sys.modules`：dataclass/相对导入都需要它）。"""
    if "_bench_under_test" in sys.modules:
        return sys.modules["_bench_under_test"]
    spec = importlib.util.spec_from_file_location("_bench_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_bench_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- 统计 --


def test_summarize_uses_median_not_mean() -> None:
    """一个离群值（卡顿帧）不该明显拉低帧率。"""
    bench = _load()
    samples = [10.0] * 99 + [200.0]              # 99 帧 10ms + 1 次卡顿
    r = bench.summarize(samples)

    assert r["n"] == 100
    assert r["median_ms"] == 10.0
    assert r["mean_ms"] > 11.0, "均值会被卡顿拉高"
    assert abs(r["fps"] - 100.0) < 1e-6, "帧率应按中位数折算"
    assert r["max_ms"] == 200.0
    assert r["p95_ms"] <= 200.0


def test_summarize_p95_is_within_range() -> None:
    bench = _load()
    r = bench.summarize(list(range(1, 101)))
    assert r["min_ms"] == 1.0
    assert 94.0 <= r["p95_ms"] <= 100.0


def test_summarize_empty_is_empty() -> None:
    bench = _load()
    assert bench.summarize([]) == {}


# ------------------------------------------------------------ 帧号选取 --


def test_pick_indices_wraps_when_more_frames_than_dataset() -> None:
    """要测的帧数超过数据集长度时循环覆盖，而不是只测一遍就停。"""
    bench = _load()
    assert bench._pick_indices(3, 7) == [0, 1, 2, 0, 1, 2, 0]
    assert bench._pick_indices(0, 5) == []


# ------------------------------------------------------------ 参数校验 --


def test_parse_args_rejects_unknown_part() -> None:
    bench = _load()
    with pytest.raises(SystemExit):
        bench.parse_args(["--parts", "render,cpu"])


def test_parse_args_rejects_zero_frames() -> None:
    bench = _load()
    with pytest.raises(SystemExit):
        bench.parse_args(["--frames", "0"])


def test_parse_args_accepts_subset_of_parts() -> None:
    bench = _load()
    args = bench.parse_args(["--parts", "render"])
    assert args.parts == ["render"]
    assert args.frames == 50 and args.warmup == 5


# ------------------------------------------------------- 环境探测 --


def test_default_rdf_python_prefers_env_var(tmp_path) -> None:
    """显式环境变量优先，且只在文件存在时采纳。"""
    bench = _load()
    fake = tmp_path / "bin" / "python"
    fake.parent.mkdir(parents=True)
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)

    old = bench.os.environ.get("LIVE3DGS_RDF_PYTHON")
    bench.os.environ["LIVE3DGS_RDF_PYTHON"] = str(fake)
    try:
        assert bench.default_rdf_python() == fake
        bench.os.environ["LIVE3DGS_RDF_PYTHON"] = str(tmp_path / "missing")
        # 指向不存在的文件时不采纳（回退到自动探测，可能为 None）
        assert bench.default_rdf_python() is None
    finally:
        if old is None:
            bench.os.environ.pop("LIVE3DGS_RDF_PYTHON", None)
        else:
            bench.os.environ["LIVE3DGS_RDF_PYTHON"] = old


# ---------------------------------------------------------- dry run --


def test_dry_run_needs_no_gpu() -> None:
    """`--dry-run` 只解析路径，不应触碰 CUDA。"""
    import contextlib
    import io

    bench = _load()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert bench.main(["--parts", "render", "--dry-run",
                           "--frames", "3"]) == 0
    out = buf.getvalue()
    assert "dry-run" in out, out
    assert "未运行任何测试" in out, out


def test_dry_run_reports_dataset_error() -> None:
    """数据集不存在时要**说明原因**（而不是空白或堆栈）。"""
    from live3dgsavatar.config import load_config

    bench = _load()
    cfg = load_config(files=("system.yaml", "render.yaml", "train.yaml"),
                      env=False)
    info = bench.plan(
        bench.parse_args(["--data-subject", "no-such-dataset", "--dry-run"]),
        cfg)
    assert info.get("dataset") is None
    assert info.get("dataset_error"), "应记录数据集为何不可用"


# ------------------------------------------------- 追踪服务预检 --


def test_check_tracker_ready_reports_missing_assets(tmp_path) -> None:
    """缺权重时要在**启动之前**报清楚，而不是让子进程崩在 torch.load。

    真实事故：服务器上只部署了 RealDenseFace 的代码、没拷 weights/，
    结果 bench 只看到"子进程退出码 1"，真正原因埋在第三层调用栈里。
    """
    bench = _load()
    python = bench.default_rdf_python()
    if python is None:
        pytest.skip("没有 realdenseface 环境")

    fake = tmp_path / "rdf"
    (fake / "tracker").mkdir(parents=True)      # 只有代码目录，没有任何资产

    try:
        bench.check_tracker_ready(python, fake)
    except SystemExit as e:
        msg = str(e)
        assert "资产" in msg, msg
        assert "vits.pth" in msg, msg
    else:
        raise AssertionError("缺资产时应当报错退出")
