"""在线更新脚本的对比图与跨对象设置（**不需要 GPU**）。

脚本本身不是包，测试通过 `importlib` 加载（必须注册到 `sys.modules`，
否则 `@dataclass` 解析类型注解时会 `AttributeError` —— 这是 importlib 的已知行为）。

这里守三件事：

1. **模型侧与数据侧分开解析**：`--data-subject crh_2` 不能把模型路径也带到 crh_2
2. **掩码 PSNR**：验收集只看掩码内（背景不一致时全图 PSNR 没有意义）
3. **对比图布局**：一排三格 + 标签条；尺寸算错会静默截断画面
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "online_update.py"
_DATA_ROOT = Path.home() / "Datasets" / "INSTA"


def _load_script():
    """把脚本当模块加载（注册进 sys.modules 以便 dataclass 正常工作）。"""
    if "_ou_under_test" in sys.modules:
        return sys.modules["_ou_under_test"]
    spec = importlib.util.spec_from_file_location("_ou_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_ou_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------ 掩码内 PSNR --


def test_masked_psnr_ignores_outside_mask() -> None:
    """掩码外改成天差地别，掩码内 PSNR 不应变化 —— 否则背景会污染验收集。"""
    m = _load_script()
    h, w = 32, 32
    target = np.zeros((h, w, 3), np.uint8)
    pred = target.copy()
    pred[0:8, :] = 255                       # 掩码外（上方）

    mask = np.zeros((h, w), bool)
    mask[16:, :] = True                      # 只看下半部分

    assert m._masked_psnr(pred, target, mask) == float("inf")

    pred[mask] = 128                         # 掩码内改动：差值 128/255
    val = m._masked_psnr(pred, target, mask)
    # MSE = (128/255)^2 ≈ 0.252 → PSNR = −10·log10(0.252) ≈ 5.99 dB
    assert val is not None and 5.5 < val < 6.5, f"实际 {val}"


def test_masked_psnr_returns_none_for_empty_mask() -> None:
    m = _load_script()
    z = np.zeros((8, 8, 3), np.uint8)
    assert m._masked_psnr(z, z, np.zeros((8, 8), bool)) is None
    assert m._masked_psnr(z, z, None) is None


# --------------------------------------------------------- 类型转换 --


def test_to_uint8_hwc_shape_and_range() -> None:
    import torch

    m = _load_script()
    t = torch.zeros(1, 3, 5, 7)
    t[:, 0] = 1.5                            # 超上界应被 clamp
    out = m._to_uint8_hwc(t)
    assert out.shape == (5, 7, 3)
    assert out.dtype == np.uint8
    assert out.max() == 255


# ----------------------------------------------------------- 对比图 --


def test_comparison_layout_is_three_panels() -> None:
    """一排三格 + 顶部标签条。尺寸算错会静默截断画面。"""
    import tempfile

    from PIL import Image

    m = _load_script()
    h, w = 40, 60
    img = np.zeros((h, w, 3), np.uint8)

    out = Path(tempfile.mkdtemp()) / "c.png"
    m._write_comparison(out, target_rgb=img, before_rgb=img, after_rgb=img,
                        before_psnr=10.0, after_psnr=20.0, frame_name="00001")
    assert out.exists()

    with Image.open(out) as im:
        assert im.size == (w * 3 + 4 * 2, h + 20), f"实际 {im.size}"


def test_comparison_accepts_missing_psnr() -> None:
    """没有掩码时 PSNR 为 None —— 标签里不该出现 'None dB'。"""
    from PIL import Image

    m = _load_script()
    img = np.zeros((10, 10, 3), np.uint8)
    import tempfile

    out = Path(tempfile.mkdtemp()) / "c.png"
    m._write_comparison(out, target_rgb=img, before_rgb=img, after_rgb=img,
                        before_psnr=None, after_psnr=None, frame_name="0")
    with Image.open(out):
        pass


def test_setup_separates_model_and_data_subject() -> None:
    """`--data-subject` 只改数据侧，模型路径必须仍属于 `--subject`。"""
    model_dir = REPO_ROOT / "models" / "crh_1"
    data_dir = _DATA_ROOT / "crh_2"
    if not model_dir.is_dir() or not data_dir.is_dir():
        pytest.skip("缺少 crh_1 模型或 crh_2 数据集")

    m = _load_script()
    args = m.parse_args(["--subject", "crh_1", "--data-subject", "crh_2",
                         "--work-name", "test"])
    setup = m._setup(args)

    assert setup.model_subject == "crh_1"
    assert setup.data_subject == "crh_2"
    assert "crh_1" in str(setup.ply), f"模型路径串了：{setup.ply}"
    assert str(setup.data_dir).endswith("crh_2"), f"数据路径不对：{setup.data_dir}"
    # 切换 subject 之后，cfg 里的 subject 是数据侧 —— 模型侧信息已在 Setup 里固化
    assert str(setup.cfg.subject) == "crh_2"
    assert setup.model_config is not None and "crh_1" in str(setup.model_config)


def test_setup_defaults_data_subject_to_model_subject() -> None:
    if not (REPO_ROOT / "models" / "crh_1").is_dir():
        pytest.skip("缺少 crh_1 模型")
    m = _load_script()
    setup = m._setup(m.parse_args(["--subject", "crh_1", "--work-name", "test"]))
    assert setup.data_subject == setup.model_subject == "crh_1"


# --------------------------------------------------- 合成（叠背景）--


def test_delta_report_aligns_per_gaussian() -> None:
    """`opacity` 是 `[N,1]`、dop 也是 `[N,1]`：必须逐高斯对齐。

    真实事故：只对一边 `squeeze` 会广播成 `[N,N]` —— **不报错**，
    但统计全错（数量级和分布都不对）。
    """
    import contextlib
    import io

    import torch

    m = _load_script()

    class _Av:
        opacity = torch.tensor([[-5.0], [0.0], [2.0], [-9999.0]])

    class _Cfg:
        update_opacity = True

    class _Up:
        avatar = _Av()
        config = _Cfg()

    class _St:
        color_delta = torch.zeros(4, 1, 3)
        opacity_delta = torch.tensor([[0.0], [0.5], [-1.0], [3.0]])

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m._delta_report(_Up(), _St())
    text = buf.getvalue()

    # 手算：sigmoid([-5,0,2]) → sigmoid([-5,0.5,1])
    #   |Δ| = [0, 0.1225, 0.1497] → max 0.1497
    assert "有效高斯 3 个" in text, text
    assert "最大=0.150" in text or "最大=0.149" in text, text


def test_write_comparisons_end_to_end(tmp_path) -> None:
    """跑通 `_write_comparisons`：抓函数内 `import numpy` 的位置错误。

    真实事故：函数中段写 `import numpy as np` 会让 `np` 在整个函数里成为
    **局部变量**，于是中段之前的 `np.concatenate(...)` 抛
    `UnboundLocalError` —— 只有在真正渲染对比图时才会踩到，
    `--dry-run` 和单元测试都发现不了。
    """
    from types import SimpleNamespace

    import torch

    m = _load_script()

    class _Avatar:
        color = torch.zeros(4, 1, 3)
        num_gaussians = 4

    class _Updater:
        avatar = _Avatar()
        config = SimpleNamespace(update_opacity=False)

        def render_observation(self, obs, bg, state):   # noqa: ANN001
            # 直接回显目标：PSNR 应为 inf，且不依赖 GPU
            return SimpleNamespace(color=obs.target,
                                   alpha=torch.ones(1, 1, *obs.target.shape[-2:]))

    # 现在对比走"逐帧读图"：给一个假 scene + 假图列表，替换 `_make_observation`
    class _Scene:
        num_frames = 3

    def fake_obs(scene, files, i, device):   # noqa: ANN001
        return SimpleNamespace(name=f"{i:05d}",
                               target=torch.rand(1, 3, 8, 8),
                               mask=torch.ones(1, 1, 8, 8))

    original = m._make_observation
    m._make_observation = fake_obs
    try:
        args = SimpleNamespace(compare_dir=tmp_path, compare_frames=2)
        setup = SimpleNamespace(cfg={"paths.output_dir": str(tmp_path)})
        n = m._write_comparisons(args, setup, _Updater(), _Scene(),
                                 SimpleNamespace(width=8, height=8),
                                 [], torch.zeros(3), None, 2, 3, tmp_path)
    finally:
        m._make_observation = original

    assert n == 2, f"应写 2 张，实际 {n}"
    assert (tmp_path / "00000.png").exists()
    assert (tmp_path / "00001.png").exists()
    assert not (tmp_path / "00002.png").exists(), "只应输出前 2 帧"


def test_experiment_name_encodes_key_args() -> None:
    """实验名要自解释：模型、数据、帧数、门控、损失、对比范围。

    同参数重跑应落到**同一目录**（便于比较）；参数一变目录就变
    （不会把不同实验混在一起）。
    """
    from types import SimpleNamespace

    m = _load_script()
    base = dict(model_subject="crh_1", data_subject="crh_2", n_obs=300,
                update_opacity=False, use_l1=False, n_cmp=-1)

    name = m.experiment_name(SimpleNamespace(exp_name=None, sample="uniform"), **base)
    assert name == "crh_1__from-crh_2__f300-uniform__noop__L2__all", name

    # 同对象时不出现 from-
    name = m.experiment_name(SimpleNamespace(exp_name=None, sample="uniform"),
                             **{**base, "data_subject": "crh_1"})
    assert "from-" not in name, name

    # 开不透明度 / L1 / 有限对比都会改变名字
    for over, token in ((dict(update_opacity=True), "op"),
                        (dict(use_l1=True), "L1"),
                        (dict(n_cmp=8), "cmp8")):
        got = m.experiment_name(SimpleNamespace(exp_name=None, sample="uniform"),
                                **{**base, **over})
        assert f"__{token}" in got or got.endswith(token), (over, got)

    # 显式 --exp-name 优先
    got = m.experiment_name(SimpleNamespace(exp_name="my-run", sample="uniform"), **base)
    assert got == "my-run"


def test_image_files_returns_sorted_paths() -> None:
    m = _load_script()
    files = m._image_files(_DATA_ROOT / "crh_2")
    assert files, "数据集里应有图"
    assert [f.name for f in files] == sorted(f.name for f in files)


# ------------------------------------------------------- 观测帧采样 --


def test_sampling_uniform_covers_whole_range() -> None:
    """均匀采样要覆盖整段视频（含首尾），而不是扎堆在某一段。

    这是"泛化"的前提：连续几秒的片段姿态/光照几乎不变，
    只在那上面拟合出的颜色偏移换个时段就失效。
    """
    m = _load_script()
    idx = m.sample_indices(3524, 60, "uniform")
    assert len(idx) == 60
    assert idx[0] == 0
    assert idx[-1] == 3523, f"最后一帧应在末段，实际 {idx[-1]}"
    assert idx == sorted(idx)
    assert len(set(idx)) == len(idx), "不应重复"

    # 相邻间隔大致均匀（允许取整误差）
    gaps = [b - a for a, b in zip(idx, idx[1:])]
    assert max(gaps) - min(gaps) <= 2, f"间隔差异过大：{gaps}"


def test_sampling_head_is_contiguous() -> None:
    m = _load_script()
    assert m.sample_indices(3524, 5, "head") == [0, 1, 2, 3, 4]


def test_sampling_random_is_reproducible() -> None:
    m = _load_script()
    a = m.sample_indices(3524, 20, "random", seed=7)
    b = m.sample_indices(3524, 20, "random", seed=7)
    c = m.sample_indices(3524, 20, "random", seed=8)
    assert a == b, "同种子必须可复现"
    assert a != c, "不同种子应不同"
    assert len(set(a)) == 20


def test_sampling_edge_cases() -> None:
    m = _load_script()
    assert m.sample_indices(10, 10, "uniform") == list(range(10))
    assert m.sample_indices(10, 99, "uniform") == list(range(10))
    assert m.sample_indices(10, 0, "uniform") == []
    assert m.sample_indices(0, 5, "uniform") == []
    assert m.sample_indices(10, 1, "uniform") == [0]


# --------------------------------------------------------- 涂色实验 --


def test_paint_mask_shapes_and_coverage() -> None:
    m = _load_script()
    h, w = 32, 32

    left = m.paint_mask(h, w, "left", 8)
    assert left.shape == (h, w) and left.dtype == bool
    assert left[:, : w // 2].all() and not left[:, w // 2:].any()

    right = m.paint_mask(h, w, "right", 8)
    assert (right == ~left).all(), "left 与 right 应互补"

    stripes = m.paint_mask(h, w, "stripes", 4)
    assert abs(stripes.mean() - 0.5) < 0.02, "竖条覆盖率应约 50%"
    # 同一列整列同色（竖条，不是棋盘）
    assert all(stripes[:, 0].tolist()) and not any(stripes[:, 4].tolist())

    checker = m.paint_mask(h, w, "checker", 4)
    assert abs(checker.mean() - 0.5) < 0.02
    assert checker[0, 0] != checker[0, 4], "棋盘相邻格应相反"

    assert not m.paint_mask(h, w, "none").any()


def test_paint_apply_only_touches_region() -> None:
    import numpy as np

    m = _load_script()
    img = np.zeros((8, 8, 3), np.float32)
    out = m.paint_apply(img.copy(), "left", (0.0, 1.0, 0.0))
    assert out[0, 0].tolist() == [0.0, 1.0, 0.0]
    assert out[0, 7].tolist() == [0.0, 0.0, 0.0]


def test_parse_color_maps_255_to_1() -> None:
    m = _load_script()
    assert m.parse_color("0,255,0") == (0.0, 1.0, 0.0)
    assert m.parse_color("255,255,255") == (1.0, 1.0, 1.0)
    for bad in ("1,2", "300,0,0", "a,b,c"):
        try:
            m.parse_color(bad)
        except ValueError:
            continue
        raise AssertionError(f"应拒绝 {bad!r}")


def test_paint_leaves_alpha_mask_untouched(tmp_path) -> None:
    """涂色必须**只改 RGB**，不动 alpha。

    若把涂色区域也塞进 mask，背景区域（模型永远渲染不到）会进入损失，
    产生一个无法降低的残差 —— 实验结论会被误导。
    """
    import numpy as np
    import torch

    m = _load_script()

    from PIL import Image

    from live3dgsavatar.core.types import Mesh

    src = _DATA_ROOT / "crh_2" / "images"
    if not src.is_dir():
        pytest.skip("缺少 crh_2 数据集")
    files = sorted(src.iterdir())
    orig = np.array(Image.open(files[0]).convert("RGBA"))

    class _Scene:
        faces = torch.zeros(1, 3, dtype=torch.int32)
        uvs = torch.zeros(1, 2)
        uv_faces = torch.zeros(1, 3, dtype=torch.int32)

        def frames_tensor(self, idx):
            return torch.zeros(len(idx), 4, 3)

        def weights_tensor(self, idx):
            return torch.zeros(len(idx), 129)

    obs = m._make_observation(_Scene(), files, 0, "cpu", paint="left",
                              paint_color=(0.0, 1.0, 0.0), paint_param=16)
    # 生产代码是在 float(0–1) 上判 `> 0.5`，等价于 uint8 上 `> 127`。
    # （写成 uint8 `> 0.5` 几乎全真 —— 阈值单位搞错了。）
    expected = torch.from_numpy(orig[..., 3] > 127)
    assert torch.equal(obs.mask[0, 0], expected), "alpha 掩码被涂色影响了"

    # 涂色区域应变成纯色
    painted = obs.target[0].permute(1, 2, 0).numpy()
    mk = m.paint_mask(painted.shape[0], painted.shape[1], "left", 16)
    assert np.allclose(painted[mk].mean(0), [0.0, 1.0, 0.0], atol=1e-5)
