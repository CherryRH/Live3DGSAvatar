"""评估协议：把"渲染结果 vs 真值"整理成统一的报告。

两种模式：

- **有真值**（`summarize_quality`）：PSNR（全图 / 掩码内）、max|Δ|。
  用于留出集、真实姿态、以及任何有对应原图的场合。
- **无真值**（`summarize_no_reference`）：只统计图像自身的可信度代理——
  前景占比、锐度（平均梯度）、饱和比例、色彩偏离。用于新视角：
  那里没有 GT，但这些量能抓住"糊 / 空 / 过曝"这类明显退化。

两者都只接收 **CPU numpy 图像**，因此在没有 GPU 的测试里可以直接验证。
渲染与落盘留给调用方（脚本 / GUI / 训练钩子），这样这一层不依赖 CUDA 扩展。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.metrics import Accum, max_abs_diff, psnr, psnr_masked, ssim

__all__ = ["EvaluationReport", "summarize_quality", "summarize_no_reference",
           "frame_metrics", "lpips_score"]


def summarize_quality(images: dict[int, np.ndarray],
                      gt: dict[int, tuple[np.ndarray, np.ndarray]]) -> dict:
    """逐帧与真值比对，返回 `{psnr_full, psnr_masked, max_abs_diff}`。

    Args:
        images: `{帧号: HxWx3 uint8 渲染图}`
        gt: `{帧号: (HxWx3 uint8 原图, HxW bool 掩码)}`；缺帧会被跳过

    Returns:
        每项都是 `Accum.summary()` 的结果（`median/min/max/mean/n`）；
        没有可比对的帧时返回空字典。
    """
    full, masked, mad = Accum(), Accum(), Accum()
    struct = Accum()
    sharp_r, sharp_g = Accum(), Accum()
    for index, image in images.items():
        item = gt.get(index)
        if item is None:
            continue
        ref_rgb, ref_mask = item
        if image.shape != ref_rgb.shape:
            continue
        full.add(psnr(image, ref_rgb))
        masked.add(psnr_masked(image, ref_rgb, ref_mask))
        mad.add(max_abs_diff(image, ref_rgb))
        struct.add(ssim(image, ref_rgb, ref_mask))
        # 锐度：只在掩码内统计平均梯度。PSNR 会奖励"更平滑"，必须同时看它，
        # 否则"PSNR 涨了但细节糊了"这种退化会被漏掉（踩过）。
        if ref_mask is not None and bool(ref_mask.any()):
            sharp_r.add(_mean_gradient(image, ref_mask))
            sharp_g.add(_mean_gradient(ref_rgb, ref_mask))

    out: dict = {"psnr_full": full.summary(),
                 "psnr_masked": masked.summary(),
                 "max_abs_diff": mad.summary(),
                 "ssim": struct.summary()}
    sr, sg = sharp_r.summary(), sharp_g.summary()
    if sr and sg and sg["median"] > 0:
        out["sharpness"] = {"render": sr, "gt": sg,
                            "ratio": round(sr["median"] / sg["median"], 3)}
    return {k: v for k, v in out.items() if v is not None}


def frame_metrics(image: np.ndarray, ref_rgb: np.ndarray,
                  ref_mask: np.ndarray | None) -> dict:
    """单帧的 PSNR / SSIM / LPIPS；算不出的项为 `None`。

    视频叠加、报告与逐帧对比都走这一个函数，避免各处口径漂移。
    """
    if image.shape != ref_rgb.shape:
        return {"psnr": None, "ssim": None, "lpips": None}
    value = psnr_masked(image, ref_rgb, ref_mask)
    if value is None:
        value = psnr(image, ref_rgb)
    return {
        "psnr": round(float(value), 2),
        "ssim": (lambda v: round(float(v), 4) if v is not None else None)(
            ssim(image, ref_rgb, ref_mask)),
        "lpips": (lambda v: round(float(v), 4) if v is not None else None)(
            lpips_score(image, ref_rgb)),
    }


_LPIPS_MODEL = None
_LPIPS_FAILED = False


def lpips_score(image: np.ndarray, ref_rgb: np.ndarray) -> float | None:
    """LPIPS（AlexNet backbone），越低越像；不可用时返回 ``None``。

    依赖 `lpips` 包与首次运行时的权重下载。失败只降级，不影响其他指标。
    输入按 0–255 的 NHWC uint8 处理（与其他指标同口径）。
    """
    global _LPIPS_MODEL, _LPIPS_FAILED
    if _LPIPS_FAILED:
        return None
    try:
        import torch
        import lpips as lpips_lib
    except Exception:                                  # pragma: no cover
        _LPIPS_FAILED = True
        return None
    try:
        if _LPIPS_MODEL is None:
            _LPIPS_MODEL = lpips_lib.LPIPS(net="alex", verbose=False)
            _LPIPS_MODEL.eval()
            if torch.cuda.is_available():
                _LPIPS_MODEL = _LPIPS_MODEL.cuda()
        a = torch.from_numpy(np.asarray(image, np.float32) / 127.5 - 1.0)
        b = torch.from_numpy(np.asarray(ref_rgb, np.float32) / 127.5 - 1.0)
        a = a.permute(2, 0, 1).unsqueeze(0)
        b = b.permute(2, 0, 1).unsqueeze(0)
        device = next(_LPIPS_MODEL.parameters()).device
        with torch.no_grad():
            value = _LPIPS_MODEL(a.to(device), b.to(device))
        return float(value.item())
    except Exception:                                  # pragma: no cover
        _LPIPS_FAILED = True
        return None


def _mean_gradient(image: np.ndarray, mask: np.ndarray) -> float:
    """掩码内的平均梯度（0–255 尺度）；越小越糊。"""
    grey = np.asarray(image, dtype=np.float32).mean(axis=-1)
    gy, gx = np.gradient(grey)
    mag = np.hypot(gx, gy)
    return float(mag[np.asarray(mask, bool)].mean())


def summarize_no_reference(images: dict[int, np.ndarray],
                           masks: dict[int, np.ndarray] | None = None) -> dict:
    """无真值时的图像代理量（全部是"越大越正常"，除了 `clip_ratio`）。

    Args:
        images: `{帧号: HxWx3 uint8}`
        masks: 可选的 `{帧号: HxW bool}` 前景；缺省时用"非纯白/非纯黑"近似

    Returns:
        `{foreground, sharpness, saturation, clip_ratio}`，每项为
        `Accum.summary()`（没有可用帧时为 `None`）
    """
    fg, sharp, sat, clip = Accum(), Accum(), Accum(), Accum()
    for index, image in images.items():
        rgb = np.asarray(image, dtype=np.float32)
        mask = None if masks is None else masks.get(index)
        if mask is None:
            mask = rgb.max(axis=-1) > 8          # 近似前景：非全黑
        if not bool(mask.any()):
            continue

        grey = rgb.mean(axis=-1)
        gy, gx = np.gradient(grey)
        sharp.add(float(np.hypot(gx, gy)[mask].mean()) * 255.0)
        fg.add(float(mask.mean()))

        px = rgb[mask]
        sat.add(float((px.max(axis=1) - px.min(axis=1)).mean()))
        clip.add(float(((px <= 1).all(axis=1) | (px >= 254).all(axis=1)).mean()))

    out = {"foreground": fg.summary(), "sharpness": sharp.summary(),
           "saturation": sat.summary(), "clip_ratio": clip.summary()}
    return {k: v for k, v in out.items() if v is not None}


@dataclass
class EvaluationReport:
    """一次评估的完整结果（可直接序列化成 `report.json`）。

    `quality` 为空表示这次没有真值，只做了无真值统计。
    """

    subject: str
    work_name: str
    dataset: str
    num_frames: int
    dataset_frames: int
    width: int
    height: int
    quality: dict = field(default_factory=dict)
    no_reference: dict = field(default_factory=dict)
    perf: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    @property
    def has_reference(self) -> bool:
        return bool(self.quality)

    def to_dict(self) -> dict:
        out: dict = {
            "subject": self.subject,
            "work_name": self.work_name,
            "dataset": self.dataset,
            "num_frames": int(self.num_frames),
            "dataset_frames": int(self.dataset_frames),
            "width": int(self.width),
            "height": int(self.height),
        }
        if self.quality:
            out["quality"] = self.quality
        if self.no_reference:
            out["no_reference"] = self.no_reference
        if self.perf:
            out["perf"] = self.perf
        if self.extra:
            out["extra"] = self.extra
        return out
