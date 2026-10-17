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

import numpy as np

from live3dgsavatar.config import load_config
from live3dgsavatar.data.scene import MANIFEST_NAME, load_scene


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
    却没移 `faces`/`uvs`/`uv_faces`（它们来自 FLAME 拓扑）。
    于是 `mesh.verts[:, mesh.faces[used]]` 抛

        RuntimeError: indices should be either on cpu or on the same device

    —— 报错发生在 `core/` 深处，离根因（`data/` 少搬了一个张量）很远。
    这里用 CPU 作为"目标设备"来验证契约，不需要 GPU。
    """
    import torch

    # 用真实的 FLAME 拓扑（与真实数据集同源），只造帧数据
    cfg = _cfg_with(tmp_path)
    ref_root = cfg.get("paths.reference_root")
    if ref_root is None:
        from pytest import skip  # noqa: PLC0415
        skip("未配置 paths.reference_root，无法取 FLAME 拓扑")

    from live3dgsavatar.data.scene import _flame_topology

    topo = _flame_topology(ref_root, "cpu")
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
    """`frames_tensor()` 不传 device 时应**跟随 Scene**，不再默认 cuda。

    旧实现默认 `device="cuda"`，导致"只是想取几帧"在没有 GPU 的环境里也炸，
    且与 Scene 已就位的设备可能不一致。
    """
    cfg = _cfg_with(tmp_path)
    ref_root = cfg.get("paths.reference_root")
    if ref_root is None:
        from pytest import skip  # noqa: PLC0415
        skip("未配置 paths.reference_root")

    from live3dgsavatar.data.scene import _flame_topology

    v = int(_flame_topology(ref_root, "cpu")["faces"].max()) + 1
    _make_manifest(tmp_path, n=2, v=v)
    scene = load_scene(cfg, frames=2, device="cpu")

    assert scene.frames_tensor([0, 1]).device.type == "cpu"
    assert scene.weights_tensor([0, 1]).device.type == "cpu"


def test_flame_topology_is_device_independent() -> None:
    """**拓扑缓存必须以 CPU 为准，返回时再搬设备。**

    历史 bug：`manifest` 路径把逐帧 `mesh` 放到了 GPU，却直接返回
    缓存的拓扑（CPU），于是 `mesh.verts[:, mesh.faces[used]]` 抛

        RuntimeError: indices should be either on cpu or on the same device

    缓存若意外持有 GPU 张量，还会让"先 cuda 后 cpu"的调用拿到错误设备。
    这里验证：请求哪个设备，返回就在哪个设备；且反复请求结果稳定。
    """
    from live3dgsavatar.config import load_config
    from live3dgsavatar.data.scene import _flame_topology

    cfg = load_config()
    ref = cfg.get("paths.reference_root")
    if ref is None:
        from pytest import skip  # noqa: PLC0415
        skip("未配置 paths.reference_root")

    for dev in ("cpu", "cpu"):          # 重复请求：缓存不能"绑架"设备
        topo = _flame_topology(ref, dev)
        for name in ("faces", "uvs", "uv_faces"):
            assert topo[name].device.type == dev, (
                f"{name} 应在 {dev}，实际 {topo[name].device}")
