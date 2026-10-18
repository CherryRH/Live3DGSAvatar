"""数据集格式的加载契约。

支持两种格式，且**互不干扰**：

1. **manifest**（自包含）：`render_inputs.npz` 里已算好逐帧 `mesh` / `blend_weight`。
   不需要 3DMM 参数、不需要 FLAME 前向、不需要 chumpy、**不需要 GPU**。
2. **checkpoint**（参照格式）：`{images,checkpoint}/`，需经 FLAME 前向算网格。

这些测试只覆盖**格式识别与校验逻辑**，不依赖 GPU：
真正的帧数据用合成的小文件。真实数据集（`crh_1`/`crh_2`）的加载另在
`tests/unit/test_scene_manifest_real.py` 里做**存在性**检查（缺数据则跳过）。
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import torch

from live3dgsavatar.config import load_config
from live3dgsavatar.data import flame_topology as FT
from live3dgsavatar.data.scene import MANIFEST_NAME, _flame_topology, load_scene

# 本项目自带的 FLAME 资产（不再依赖任何外部仓库）
_HERE = Path(__file__).resolve().parents[2]
_FLAME_DIR = _HERE / "data" / "FLAME2020"
_CACHE_DIR = _HERE / "output" / "cache"


def _raises(exc_type: type[BaseException]):
    """零依赖的 `pytest.raises` 替代（运行器的 pytest 桩不支持 raises）。"""
    holder = {"error": None}

    @contextlib.contextmanager
    def _cm():
        try:
            yield holder
        except exc_type as e:          # noqa: BLE001 - 只捕获期望的类型
            holder["error"] = e
            return
        raise AssertionError(f"期望 {exc_type.__name__}，但没有抛出")

    return _cm()


def _make_manifest(tmp_path, *, n=3, v=8, d=129, bad_d=False):
    """造一个最小 manifest；顶点数故意很小，只测校验逻辑。"""
    rng = np.random.default_rng(0)
    np.savez(
        tmp_path / MANIFEST_NAME,
        mesh=rng.normal(size=(n, v, 3)).astype(np.float32),
        blend_weight=rng.normal(size=(n, d if not bad_d else d + 5)).astype(np.float32),
        K=np.eye(3, dtype=np.float64),
        w2v=np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64),
        width=np.int64(512),
        height=np.int64(512),
        names=np.array([f"{i:05d}" for i in range(n)]),
    )


def _cfg_with(tmp_path):
    cfg = load_config()
    cfg.set("subject", "synthetic")
    cfg.set("work_name", "test")
    cfg.set("paths.data_dir", tmp_path)
    return cfg


def test_manifest_is_detected_by_filename(tmp_path) -> None:
    """格式靠 `render_inputs.npz` 是否存在来判断，不需要额外配置项。"""
    _make_manifest(tmp_path)
    assert (tmp_path / MANIFEST_NAME).exists()


def test_manifest_rejects_blend_weight_dim_mismatch(tmp_path) -> None:
    """`blend_weight` 维度必须等于配置的 `num_basis_in`。

    错维会在 MLP 的 `Linear` 里以形状错误爆掉，但那时已经过了数据层，
    报错信息与根因相距很远 —— 所以在加载时显式拦截。
    """
    _make_manifest(tmp_path, bad_d=True)
    cfg = _cfg_with(tmp_path)
    with _raises(ValueError) as box:
        load_scene(cfg, frames=1, device="cpu")
    assert "num_basis_in" in str(box["error"])


def test_manifest_requires_all_fields(tmp_path) -> None:
    np.savez(tmp_path / MANIFEST_NAME, mesh=np.zeros((2, 8, 3), np.float32))
    cfg = _cfg_with(tmp_path)
    with _raises(ValueError) as box:
        load_scene(cfg, frames=1, device="cpu")
    assert "缺少字段" in str(box["error"])


def test_manifest_missing_dataset_dir_is_reported(tmp_path) -> None:
    cfg = load_config()
    cfg.set("paths.data_dir", tmp_path / "nope")
    with _raises(FileNotFoundError) as box:
        load_scene(cfg, frames=1, device="cpu")
    assert "数据集目录不存在" in str(box["error"])


def test_data_dir_overrides_subject_path(tmp_path) -> None:
    """`paths.data_dir` 优先于 `<data_root>/<subject>`。

    数据集目录名常与 `subject` 不一致（例如 `Datasets/INSTA/crh_1` 而
    subject 也是 `crh_1`，但 `data_root` 下并没有同名父目录），
    因此必须允许显式覆盖。
    """
    cfg = load_config()
    cfg.set("paths.data_root", tmp_path / "root")
    cfg.set("paths.data_dir", tmp_path / "explicit")
    from live3dgsavatar.config import resolve_data_dir

    assert resolve_data_dir(cfg) == (tmp_path / "explicit").resolve()

    cfg.set("paths.data_dir", None)
    assert resolve_data_dir(cfg) == (tmp_path / "root" / str(cfg.subject)).resolve()


def test_manifest_rejects_wrong_vertex_count(tmp_path) -> None:
    """网格顶点数不足时应报错，而不是让 `face_id` 越界。

    `face_id` 是**面索引**，指向 FLAME 的面表；顶点数少于面表引用范围时，
    绑定会插值到不存在的顶点上 —— 画面上表现为模型错乱，极难定位。
    """
    # 故意只给 4 个顶点，远少于 FLAME 面表的引用范围
    _make_manifest(tmp_path, v=4)
    cfg = _cfg_with(tmp_path)
    with _raises(ValueError) as box:
        load_scene(cfg, frames=1, device="cpu")
    assert "网格顶点数" in str(box["error"])


def test_extract_texture_requires_texel_mask() -> None:
    """从 `.ply` 加载时无法重建 UV 纹理图，必须显式报错。

    `.ply` 只保存有效 texel（N 个高斯），原始 texel 位置没有存盘，
    因此 `valid_binding_mask` 退化为全 True。若静默继续，
    `flat[mask] = color` 会因维度不匹配抛 `IndexError`，或者更糟——
    形状恰好凑巧时画出一张错位的纹理图。
    """
    import torch

    from live3dgsavatar.core.avatar import AvatarConfig, GaussianAvatar
    from live3dgsavatar.core.deform.bind import Binding

    n = 6
    binding = Binding(
        face_id=torch.zeros(n, dtype=torch.long),
        face_bary=torch.tensor([[1.0, 0.0, 0.0]] * n),
        valid_mask=torch.ones(n, dtype=torch.bool),      # [N]，不是 [H*W]
    )
    av = GaussianAvatar(AvatarConfig(tex_size=4), binding)
    with _raises(RuntimeError) as box:
        av.extract_texture()
    assert "valid_binding_mask" in str(box["error"])
    assert "tex_size²" in str(box["error"])


def test_scene_tensors_share_one_device(tmp_path) -> None:
    """**Scene 的所有张量必须在同一设备上。**

    历史 bug：`load_scene(device="cuda")` 把逐帧 `mesh` 移到了 GPU，
    却没移 `faces`/`uvs`/`uv_faces`，于是 `mesh.verts[:, mesh.faces[used]]` 抛
    `RuntimeError: indices should be either on cpu or on the same device`。
    这里用 CPU 作为"目标设备"验证契约，不需要 GPU。
    """
    cfg = _cfg_with(tmp_path)
    topo = _flame_topology(cfg, "cpu")
    v = int(topo["faces"].max()) + 1
    _make_manifest(tmp_path, n=2, v=v)

    scene = load_scene(cfg, frames=2, device="cpu")
    devices = {
        "frames.mesh": scene.frames[0]["mesh"].device,
        "frames.blend_weight": scene.frames[0]["blend_weight"].device,
        "faces": scene.faces.device,
        "uvs": scene.uvs.device,
        "uv_faces": scene.uv_faces.device,
        "K": scene.K.device,
        "R": scene.R.device,
        "T": scene.T.device,
    }
    distinct = {str(d) for d in devices.values()}
    assert len(distinct) == 1, f"张量分散在多个设备上：{ {k: str(v) for k, v in devices.items()} }"
    assert distinct == {"cpu"}


def test_frames_tensor_follows_scene_device(tmp_path) -> None:
    """`frames_tensor()` 不传 device 时应**跟随 Scene**，不再默认 cuda。"""
    cfg = _cfg_with(tmp_path)
    v = int(_flame_topology(cfg, "cpu")["faces"].max()) + 1
    _make_manifest(tmp_path, n=2, v=v)
    scene = load_scene(cfg, frames=2, device="cpu")
    assert scene.frames_tensor([0, 1]).device.type == "cpu"
    assert scene.weights_tensor([0, 1]).device.type == "cpu"


def test_flame_topology_is_device_independent() -> None:
    """拓扑缓存以 CPU 为准，返回时再搬设备。

    缓存若持有 GPU 张量，会让"先 cuda 后 cpu"的调用拿到错误设备。
    """
    FT.reset_memory_cache()
    for _ in range(2):
        topo = FT.load(_FLAME_DIR, _CACHE_DIR, "cpu")
        for name in ("faces", "uvs", "uv_faces"):
            assert topo[name].device.type == "cpu"


def test_flame_topology_roundtrip_and_fingerprint(tmp_path) -> None:
    """拓扑应能落盘复用；FLAME 资产变了则缓存失效。

    **为什么要缓存**：拓扑只取决于 FLAME 资产、与数据无关，每次启动都读
    50 MB 的 pickle 是纯浪费。**为什么必须校验**：资产从 2020 换成 2023
    而缓存未失效时，`face_id` 会指向错误的面 —— 表现为"模型错乱"，极难定位。
    """
    cache = tmp_path / "cache"
    FT.reset_memory_cache()
    FT.clear_cache(cache)

    # 1) 首次：构建并落盘
    t1 = FT.load(_FLAME_DIR, cache, "cpu")
    assert FT.cache_path(cache).exists(), "首次调用应写缓存"
    # 面表 10032 行（含 56 个牙齿面）—— `model.ply` 里 face_id 上界 10031 的来源
    assert tuple(t1["faces"].shape) == (10032, 3)
    assert tuple(t1["uv_faces"].shape) == (10032, 3)
    assert int(t1["faces"].max()) < 5083

    # 2) 二次：命中缓存，内容一致
    FT.reset_memory_cache()
    t2 = FT.load(_FLAME_DIR, cache, "cpu")
    assert torch.equal(t1["faces"], t2["faces"])
    assert torch.equal(t1["uvs"], t2["uvs"])

    # 3) 指纹一致 → 有效
    assert FT.read_cache(cache, _FLAME_DIR) is not None

    # 4) 指纹变化 → 失效
    import os
    import time as _time

    uv = _FLAME_DIR / "flame_uv.npz"
    stat = uv.stat()
    os.utime(uv, (_time.time() + 3600, _time.time() + 3600))
    try:
        assert FT.read_cache(cache, _FLAME_DIR) is None, (
            "FLAME 资产变化后缓存必须失效，否则面表可能与模型不匹配")
    finally:
        os.utime(uv, (stat.st_atime, stat.st_mtime))

    FT.reset_memory_cache()
    FT.clear_cache(cache)


# ------------------------------------------------------- checkpoint 格式 --


def test_flame_model_matches_known_invariants() -> None:
    """FLAME 前向的**自洽**不变量（不需要外部参照）。

    渲染"参照格式"数据集时必须把 3DMM 参数算成网格顶点，因此这条路径必须正确。
    这里用数学上必然成立的性质验证，而不是对照任何外部实现：

    1. 全零参数 + 单位旋转 → 回到 `v_template`
    2. 只改 shape → 关节位置随之改变（shape 确实作用到网格）
    3. 只改 jaw → 只有嘴部区域明显位移（不会整体平移）
    """
    from live3dgsavatar.data.flame_model import FlameModel

    fm = FlameModel(_FLAME_DIR, dtype=torch.float64)
    b = 1
    z300 = torch.zeros(b, 300, dtype=torch.float64)
    z100 = torch.zeros(b, 100, dtype=torch.float64)
    I3 = torch.eye(3, dtype=torch.float64)

    v = fm.forward(z300, z100, I3[None], I3[None], I3[None].expand(b, 2, 3, 3))
    assert torch.allclose(v[0], fm.v_template, atol=1e-10), (
        "全零参数应回到 template")

    # shape 影响关节位置
    j0 = fm.joints(z300, z100)
    j1 = fm.joints(torch.ones(b, 300, dtype=torch.float64) * 0.5, z100)
    assert not torch.allclose(j0, j1), "shape 应改变关节位置"

    # 牙齿顶点数正确
    assert fm.v_template.shape[0] == 5023 + 60


def test_checkpoint_dataset_requires_checkpoint_dir(tmp_path) -> None:
    from live3dgsavatar.data.checkpoint_loader import CheckpointDataset

    with _raises(FileNotFoundError) as box:
        CheckpointDataset(tmp_path)
    assert "checkpoint" in str(box["error"])


def test_load_scene_reports_unknown_format(tmp_path) -> None:
    """既没有 npz 也没有 checkpoint 时，报错要指明"格式无法识别"。"""
    cfg = _cfg_with(tmp_path)
    with _raises(FileNotFoundError) as box:
        load_scene(cfg, frames=1, device="cpu")
    assert "无法识别" in str(box["error"]) or "缺少" in str(box["error"])
