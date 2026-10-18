"""在线更新（热更新）的离线可测部分。

**不依赖 GPU**：闭式颜色校正、增量容器、报告逻辑都是纯张量运算。
真正的梯度微调需要光栅化器（CUDA 扩展），其端到端验证在
`scripts/render_test.py` / GUI 侧完成。
"""

from __future__ import annotations

import torch

from live3dgsavatar.training.color_fit import ColorCorrection, fit_color_correction
from live3dgsavatar.training.losses import masked_mse, masked_psnr
from live3dgsavatar.training.observations import BatchObservation, FrameObservation
from live3dgsavatar.training.updater import UpdateConfig, UpdateReport, UpdateState
from live3dgsavatar.core.types import Camera, Mesh


def _img(b=1, h=8, w=8, value=0.5):
    return torch.full((b, 3, h, w), float(value))


def test_masked_mse_ignores_outside_mask() -> None:
    """掩码外像素不应影响损失 —— 背景预测通常不准。"""
    pred = _img(); tgt = _img()
    m = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    m[..., :4, :] = True
    tgt[..., :4, :] = 1.0          # 掩码内差 0.5
    tgt[..., 4:, :] = 0.0          # 掩码外差 0.5，但应被忽略
    loss = float(masked_mse(pred, tgt, m))
    assert abs(loss - 0.25) < 1e-6, f"损失应只算掩码内（0.25），实际 {loss}"


def test_masked_psnr_matches_definition() -> None:
    pred = _img(value=0.5); tgt = _img(value=0.5)
    assert masked_psnr(pred, tgt) > 90, "完全相同时 PSNR 应极高"
    tgt2 = _img(value=0.6)
    psnr = masked_psnr(pred, tgt2)
    # MSE = 0.01 → PSNR = 20 dB
    assert abs(psnr - 20.0) < 0.1, f"应约 20 dB，实际 {psnr:.2f}"


def test_color_correction_identity_is_noop() -> None:
    c = ColorCorrection.identity()
    x = torch.rand(1, 3, 4, 4)
    assert torch.allclose(c(x), x, atol=1e-6)


def test_fit_color_correction_recovers_known_gain() -> None:
    """给定一个已知的通道增益，拟合应把它恢复出来，PSNR 显著提升。"""
    torch.manual_seed(0)
    base = torch.rand(1, 3, 16, 16) * 0.8 + 0.1
    gain = torch.tensor([1.2, 0.8, 1.0]).view(1, 3, 1, 1)
    target = (base * gain).clamp(0, 1)

    corr, m = fit_color_correction(base, target, with_bias=False)
    assert m["psnr_delta"] > 3.0, f"应明显提升，实际 {m['psnr_delta']:+.2f} dB"
    # 恢复出的对角项应接近真实增益
    diag = corr.matrix.diag()
    for got, want in zip(diag.tolist(), [1.2, 0.8, 1.0]):
        assert abs(got - want) < 0.05, f"增益恢复不准：{got:.3f} vs {want:.3f}"


def test_fit_color_correction_with_mask() -> None:
    """只在掩码内拟合：掩码外的脏数据不应污染结果。"""
    torch.manual_seed(1)
    base = torch.rand(1, 3, 12, 12) * 0.6 + 0.2
    target = (base * 1.15).clamp(0, 1)
    target[..., :6, :] = 0.0                     # 一半是垃圾
    m = torch.zeros(1, 1, 12, 12, dtype=torch.bool)
    m[..., 6:, :] = True

    corr, metrics = fit_color_correction(base, target, m, with_bias=False)
    # 掩码覆盖下半 6 行 × 12 列 = 72 个像素（不是 ×3 通道 —— 掩码是逐像素的）
    assert metrics["n_pixels"] == 6 * 12, f"实际 {metrics['n_pixels']}"
    assert metrics["n_params"] == 9, "with_bias=False 时只有 3×3"
    assert metrics["psnr_delta"] > 1.0
    # 掩码外的垃圾不应污染拟合：恢复出的增益应接近 1.15
    for d in corr.matrix.diag().tolist():
        assert abs(d - 1.15) < 0.05, f"增益被污染：{d:.3f}"


def test_fit_color_correction_rejects_empty_mask() -> None:
    base = _img(); target = _img()
    m = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    try:
        fit_color_correction(base, target, m)
    except ValueError as e:
        assert "有效像素太少" in str(e)
    else:
        raise AssertionError("空掩码应报错")


def test_update_config_validates() -> None:
    for bad in ({"steps": 0}, {"time_budget_s": 0}, {"frame_chunk": 0}):
        try:
            UpdateConfig(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} 应报错")


def test_update_state_roundtrip() -> None:
    """增量应能序列化/反序列化（流式场景要跨轮传递）。"""
    st = UpdateState(color_delta=torch.rand(5, 1, 3, requires_grad=True),
                     opacity_delta=torch.rand(5, 1, requires_grad=True), step=7)
    d = st.to_dict()
    st2 = UpdateState.from_dict(d)
    assert torch.allclose(st2.color_delta, st.color_delta.detach())
    assert torch.allclose(st2.opacity_delta, st.opacity_delta.detach())
    assert st2.step == 7


def test_update_report_summary_mentions_delta() -> None:
    r = UpdateReport(before={"psnr": 30.0, "loss": 1.0},
                     after={"psnr": 32.5, "loss": 0.5}, steps=100, elapsed_s=9.5)
    assert abs(r.psnr_delta - 2.5) < 1e-9
    s = r.summary()
    assert "30.00" in s and "32.50" in s and "+2.50" in s


def test_observation_validates_shapes() -> None:
    mesh = Mesh(verts=torch.zeros(1, 4, 3), faces=torch.zeros(1, 3, dtype=torch.int32),
                uvs=torch.zeros(4, 2), uv_faces=torch.zeros(1, 3, dtype=torch.int32))
    good = FrameObservation(mesh=mesh, blend_weight=torch.zeros(1, 129),
                            target=torch.zeros(1, 3, 8, 8))
    assert good.batch_size == 1 and good.height == 8
    # 批大小与 target 不一致：必须报错（否则"这帧参数驱动了那帧画面"）
    try:
        FrameObservation(mesh=mesh, blend_weight=torch.zeros(2, 129),
                         target=torch.zeros(1, 3, 8, 8))
    except ValueError as e:
        assert "批大小" in str(e)
    else:
        raise AssertionError("批大小不一致应报错")

    # 一维 blend_weight 应报错
    try:
        FrameObservation(mesh=mesh, blend_weight=torch.zeros(129),
                         target=torch.zeros(1, 3, 8, 8))
    except ValueError as e:
        assert "[B, D]" in str(e)
    else:
        raise AssertionError("一维 blend_weight 应报错")


def test_batch_observation_rejects_mixed_resolution() -> None:
    mesh = Mesh(verts=torch.zeros(1, 4, 3), faces=torch.zeros(1, 3, dtype=torch.int32),
                uvs=torch.zeros(4, 2), uv_faces=torch.zeros(1, 3, dtype=torch.int32))
    f1 = FrameObservation(mesh=mesh, blend_weight=torch.zeros(1, 129),
                          target=torch.zeros(1, 3, 8, 8))
    f2 = FrameObservation(mesh=mesh, blend_weight=torch.zeros(1, 129),
                          target=torch.zeros(1, 3, 16, 16))
    cam = Camera.from_intrinsics_extrinsics(
        K=torch.eye(3), R=torch.eye(3), T=torch.zeros(3), width=8, height=8)
    try:
        BatchObservation(frames=[f1, f2], camera=cam)
    except ValueError as e:
        assert "分辨率" in str(e)
    else:
        raise AssertionError("混合分辨率应报错")


# --------------------------------------------------------- 更新循环（假渲染器）--

class FakeRasterizer:
    """最小可微"渲染器"：每帧输出一块常数颜色。

    用它验证 `OnlineUpdater` 的**优化逻辑**（梯度、增量注入、早停、报告），
    而这些与真实光栅化无关。真实渲染的端到端验证放在 GPU 侧脚本里。
    """

    def __init__(self, image):
        self.image = image

    def render(self, gaussians, camera, background, target_image=None):
        # 用高斯颜色的均值填充图像 —— 对颜色有梯度，形状与真实输出一致
        b = gaussians.batch_size
        h, w = self.image.shape[2], self.image.shape[3]
        # color: [B, N, 1, 3] → 对 N 求均值，保留通道 → [B, 1, 3] → [B,3,1,1]
        c = gaussians.color.mean(dim=1).reshape(b, 3, 1, 1)
        color = c.expand(b, 3, h, w)
        alpha = torch.ones(b, 1, h, w, dtype=color.dtype, device=color.device)
        from live3dgsavatar.core.types import RenderOutput
        return RenderOutput(color=color, alpha=alpha)


def _tiny_avatar(n=16):
    from live3dgsavatar.core.avatar import AvatarConfig, GaussianAvatar
    from live3dgsavatar.core.deform.bind import Binding

    binding = Binding(face_id=torch.zeros(n, dtype=torch.long),
                      face_bary=torch.tensor([[1.0, 0.0, 0.0]] * n),
                      valid_mask=torch.ones(n, dtype=torch.bool))
    return GaussianAvatar(AvatarConfig(tex_size=4, num_basis_in=8, num_basis_blend=2),
                          binding)


def _tiny_batch(target_value=0.7, n=16, bs=2):
    from live3dgsavatar.core.avatar import AvatarConfig  # noqa: F401
    from live3dgsavatar.core.types import Camera, Mesh

    mesh = Mesh(verts=torch.zeros(bs, 4, 3),
                faces=torch.zeros(1, 3, dtype=torch.int32),
                uvs=torch.zeros(4, 2),
                uv_faces=torch.zeros(1, 3, dtype=torch.int32))
    cam = Camera.from_intrinsics_extrinsics(
        K=torch.eye(3), R=torch.eye(3), T=torch.zeros(3), width=8, height=8)
    frames = [
        FrameObservation(mesh=mesh, blend_weight=torch.zeros(bs, 8),
                         target=torch.full((bs, 3, 8, 8), target_value),
                         name=f"f{i}")
        for i in range(2)
    ]
    return BatchObservation(frames=frames, camera=cam)


def test_updater_reduces_loss_and_raises_psnr() -> None:
    """核心断言：更新后 PSNR 必须上升、loss 必须下降。

    假渲染器把"高斯颜色均值"铺满图像，而目标是一块常数 —— 所以
    最优解是**把颜色调到目标值**。这恰好是热更新要做的事的微缩版。
    """
    from live3dgsavatar.training.updater import OnlineUpdater

    avatar = _tiny_avatar()
    batch = _tiny_batch(target_value=0.7)
    updater = OnlineUpdater(
        avatar, FakeRasterizer(batch.frames[0].target),
        config=UpdateConfig(steps=200, time_budget_s=30, lr_color=5e-2,
                            update_opacity=False, log_every=0))
    report, state = updater.fit(batch, background=torch.zeros(3))

    assert report.steps > 0, "应当真的迭代了"
    assert report.after["loss"] < report.before["loss"], "loss 应下降"
    assert report.psnr_delta > 10.0, (
        f"应显著提升 PSNR，实际 {report.psnr_delta:+.2f} dB")
    # 增量应把颜色推向目标值（0.7 对应的 SH DC ≈ (0.7-0.5)/SH_C0）
    assert float(state.color_delta.abs().max()) > 0.1, "增量应明显非零"


def test_updater_respects_time_budget() -> None:
    """时间预算必须生效 —— 热更新不能把通话卡住。"""
    import time as _time
    from live3dgsavatar.training.updater import OnlineUpdater

    avatar = _tiny_avatar()
    batch = _tiny_batch()
    updater = OnlineUpdater(
        avatar, FakeRasterizer(batch.frames[0].target),
        config=UpdateConfig(steps=10_000, time_budget_s=0.2,
                            lr_color=1e-3, update_opacity=False, log_every=0))
    t0 = _time.perf_counter()
    report, _ = updater.fit(batch, background=torch.zeros(3))
    elapsed = _time.perf_counter() - t0
    assert elapsed < 5.0, f"预算 0.2s，实际耗时 {elapsed:.2f}s"
    assert report.steps < 10_000, "应在预算内提前停止"


def test_updater_can_continue_from_state() -> None:
    """流式场景：从已有增量继续，而不是每次从零开始。"""
    from live3dgsavatar.training.updater import OnlineUpdater

    avatar = _tiny_avatar()
    batch = _tiny_batch(target_value=0.7)
    updater = OnlineUpdater(
        avatar, FakeRasterizer(batch.frames[0].target),
        config=UpdateConfig(steps=50, time_budget_s=30, lr_color=5e-2,
                            update_opacity=False, log_every=0))
    r1, st1 = updater.fit(batch, background=torch.zeros(3))
    r2, st2 = updater.fit(batch, background=torch.zeros(3), state=st1)
    assert r2.before["psnr"] > r1.before["psnr"], "第二轮应从更好的起点开始"
    # `state.step` 是**累计**步数；`report.steps` 只是本轮步数
    assert st2.step > st1.step, f"累计步数应增长：{st1.step} → {st2.step}"
    assert st2.step == st1.step + r2.steps
