#!/usr/bin/env python
"""几何条件扩散生成伪标签：在"有表面、模型没铺满"的区域补出可信纹理。

这是 GAF（CVPR 2025）思路的**轻量版**：不训练任何网络，用冻结的
SD + Normal ControlNet 在低强度下重绘缺陷视角，再用几何与一致性检查过滤，
通过的图作为微调目标（落到 `<缺陷目录>/refined/`，交给 `refine_from_images.py`）。
视角标签里的 `yaw/pitch` 是绝对头部角度；渲染时会减掉基帧头姿。

## 与 GAF 的差别（以及为什么这样取舍）

| | GAF | 这里 |
|---|---|---|
| 扩散先验 | 自训练的多视角头部扩散（跨视角注意力） | 公开 SD1.5 + ControlNet-normal（单视角） |
| 条件 | FLAME 法线 + 输入图 VAE latent | FLAME 法线 + 低强度 img2img（渲染图做初始化） |
| 用法 | 迭代去噪图当伪 GT | 只采信**筛选通过**的图 |

单视角没有跨视角耦合，所以一致性必须靠**筛选**补回来：
轮廓 IoU、可信区域不变、洞内结构非平坦、两次采样一致性。

## 用法

```bash
# 只生成伪标签（需要 diffusers；首次会下载 SD1.5 与 ControlNet 权重）
python scripts/pseudo_label.py --subject <对象> --work-name <版本> \
    --defects output/defects/<对象>/<版本>/<数据集> \
    --install-refined

# 只渲染法线与掩码（不需要 diffusers，可先检查几何是否对齐）
python scripts/pseudo_label.py ... --dry-run
```

产物：`<缺陷目录>/pseudo/{accepted/,debug/,report.json}`。
`--install-refined` 会把通过的图复制到 `<缺陷目录>/refined/`，
之后交给 `refine_from_images.py` / `iterative_repair.py` 微调。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import live3dgsavatar  # noqa: F401


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="几何条件扩散生成伪标签")
    p.add_argument("--subject", default=None)
    p.add_argument("--work-name", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--defects", type=Path, default=None,
                   help="缺陷目录（含 selected.json）")
    p.add_argument("--out", type=Path, default=None,
                   help="输出目录（默认 <缺陷目录>/pseudo）")
    p.add_argument("--install-refined", action="store_true",
                   help="把通过筛选的图复制到 <缺陷目录>/refined/")
    p.add_argument("--limit", type=int, default=0, help="只处理前 N 个视角（调试）")
    p.add_argument("--seed", type=int, default=0)

    # ---- 扩散 ----
    p.add_argument("--base-model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--ip-adapter", choices=("none", "faceid", "faceid_plus"),
                   default="none",
                   help="身份条件：faceid=简单 FaceID；faceid_plus=Plus v2（CLIP+FaceID）")
    p.add_argument("--ip-adapter-path", default="h94/IP-Adapter-FaceID")
    p.add_argument("--ip-adapter-weight", default="ip-adapter-faceid_sd15.bin")
    p.add_argument("--ip-adapter-image-encoder",
                   default="h94/IP-Adapter/sdxl_models/image_encoder",
                   help="faceid_plus 的 CLIP ViT-H image encoder（本地或 HF 目录）")
    p.add_argument("--ip-adapter-scale", type=float, default=0.7,
                   help="身份适配器强度（默认 0.7；plus 可试 0.6）")
    p.add_argument("--controlnet", default="lllyasviel/control_v11p_sd15_normalbae")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--strength", type=float, default=0.85,
                   help="重绘强度；SD1.5 人脸加强版默认 0.85")
    p.add_argument("--guidance", type=float, default=5.5)
    p.add_argument("--prompt",
                   default="close-up of a bare face and neck, natural bare "
                           "skin, visible pores, eyes looking straight ahead, "
                           "soft light, plain background, sharp focus")
    p.add_argument("--negative",
                   default="looking at camera, looking at viewer, eye contact, "
                           "side glance, cross-eyed, clothing, shirt, collar, "
                           "jacket, scarf, necklace, earrings, jewelry, hat, "
                           "glasses, makeup, lipstick, eyeshadow, blush, "
                           "painted face, beauty filter, "
                           "blurry, smooth plastic, deformed, extra eyes, low "
                           "resolution, watermark")

    # ---- 掩码与筛选 ----
    p.add_argument("--normal-convention", choices=("opengl", "camera"),
                   default="opengl",
                   help="法线编码约定（ControlNet 用 opengl；camera 仅用于对比调试）")
    p.add_argument("--raster-flip", choices=("auto", "true", "false"),
                   default="auto",
                   help="光栅化是否翻转 y。auto = 用 alpha 轮廓自动判定（推荐）")
    p.add_argument("--mask-mode",
                   choices=("holes", "full", "unsupported", "structure"),
                   default="full",
                   help="holes=只修 alpha 不足的洞；full=洞 + 破碎/虚影（默认）")
    p.add_argument("--defect-quantile", type=float, default=0.7,
                   help="full 模式下，缺陷质量高于该分位的区域也算进掩码")
    p.add_argument("--spike-threshold", type=float, default=0.15,
                   help="alpha 梯度超过该值视为破碎/虚影")
    p.add_argument("--mask-dilate", type=int, default=12,
                   help="洞掩码向外扩多少像素（默认 16）")
    p.add_argument("--core-dilate", type=int, default=12,
                   help="真正替换回渲染图的 core 外扩（默认 16 = 整掩码替换）")
    p.add_argument("--color-match", dest="color_match", action="store_true",
                   default=False,
                   help="把生成区域颜色分布对齐到周围渲染图（默认开）")
    p.add_argument("--no-color-match", dest="color_match",
                   action="store_false", help="关闭颜色对齐")
    p.add_argument("--color-ring", type=int, default=8,
                   help="取渲染图周围多少像素的分布做对齐（默认 8）")
    p.add_argument("--color-mean-only", action="store_true",
                   help="只对齐颜色均值，保留生成图对比度（推荐先试）")
    p.add_argument("--feather", type=int, default=4,
                   help="core 边界羽化半径（默认 4；0=硬替换）")
    p.add_argument("--identity-gate", dest="identity_gate", action="store_true",
                   default=True,
                   help="用 ArcFace 检查生成图是否保持参考身份（默认开）")
    p.add_argument("--no-identity-gate", dest="identity_gate",
                   action="store_false", help="关闭身份门")
    p.add_argument("--identity-threshold", type=float, default=0.30,
                   help="ArcFace 余弦相似度下限（默认 0.30）")
    p.add_argument("--coverage-min", type=int, default=1,
                   help="面被真实数据观测的最少次数；低于此视为缺失解剖结构，不补。"
                        "0 或 --no-observed 关闭该限制（消融用）")
    p.add_argument("--photometric-frames", type=int, default=60,
                   help="用多少帧真实图统计'面被前景支持过'（alpha>127）。"
                        "0 = 退回 find_defects 的几何可见性")
    p.add_argument("--photometric-min-px", type=int, default=16,
                   help="一个面累计至少多少前景像素才算被数据支持")
    p.add_argument("--no-observed", action="store_true",
                   help="关闭观测面限制（默认只补被数据看过的表面）")
    p.add_argument("--sharpen", type=float, default=0.35,
                   help="unsharp 增益，补偿 SD 往返的梯度损失。实测 SD 往返把标签"
                        "梯度压到渲染的 0.935，0.30 可回到约 0.99；0 关闭")
    p.add_argument("--trusted-tol", type=float, default=3.0,
                   help="可信区域允许的平均改动（0-255）")
    p.add_argument("--min-iou", type=float, default=0.90,
                   help="整幅轮廓 IoU 的参考下限（只记录，不拒绝：补洞会改轮廓）")
    p.add_argument("--min-outside-iou", type=float, default=0.97,
                   help="**掩码外**轮廓召回率下限（渲染的前景有多少还在）；"
                        "掩码内允许改轮廓。用召回而不是 IoU：SD 会把背景抬亮，"
                        "IoU 会把背景噪声算进去")
    p.add_argument("--max-outside-extra", type=float, default=0.10,
                   help="掩码外新增前景（诊断项；合成后掩码外会被换回渲染图，"
                        "不再作为拒绝理由）")
    p.add_argument("--min-fill", type=float, default=0.45,
                   help="掩码内生成前景（>8）占比下限；低于此说明扩散没补上，"
                        "这样的标签喂给微调只会教出噪声")
    p.add_argument("--min-sharp", type=float, default=0.95,
                   help="伪标签/渲染的**整幅**前景梯度能量比下限")
    p.add_argument("--min-mask-sharp", type=float, default=0.90,
                   help="伪标签/渲染在**掩码内**的梯度能量比下限。整幅比会被"
                        "掩码外的渲染图主导，这条才反映补出来的区域够不够锐")
    p.add_argument("--max-hole-area", type=float, default=0.15,
                   help="单个洞连通域的像素占比上限；更大的块是缺失解剖结构，不修")
    p.add_argument("--border-margin", type=int, default=2,
                   help="碰到图像边缘的洞不修（通常是画面外的脖子/肩膀）")
    p.add_argument("--head-cut", action="store_true",
                   help="unsupported 模式：只修到最低观测像素附近，"
                        "避免扩散补出脖子/胸口/衣服/首饰")
    p.add_argument("--head-cut-margin", type=int, default=0,
                   help="head-cut 时向下保留的像素（0 = 正好裁在最低观测处）")
    p.add_argument("--min-std", type=float, default=3.0,
                   help="洞内像素标准差下限（太平坦说明糊了）")
    p.add_argument("--consistency", type=int, default=2,
                   help="每个视角采样几次做一致性检查（1 = 关闭）")
    p.add_argument("--min-consistency", type=float, default=0.60,
                   help="多次采样在洞内的 SSIM 下限")
    p.add_argument("--dry-run", action="store_true",
                   help="只渲染 RGB/法线/掩码，不跑扩散（无需 diffusers）")
    p.add_argument("--force", action="store_true", help="覆盖已有产物")
    return p.parse_args(argv)


# ------------------------------------------------------------------ 工具 --


def dilate(mask, radius: int):
    """二值膨胀（纯 numpy，圆形结构元近似为方形，够用）。"""
    import numpy as np

    m = np.asarray(mask, bool)
    if radius <= 0 or not m.any():
        return m
    out = m.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx > radius * radius:
                continue
            out |= np.roll(np.roll(m, dy, axis=0), dx, axis=1)
    return out


def _erode(mask, radius: int):
    """二值腐蚀（与 dilate 对称的方形近似）。"""
    import numpy as np

    m = np.asarray(mask, bool)
    if radius <= 0 or not m.any():
        return m
    out = m.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx > radius * radius:
                continue
            out &= np.roll(np.roll(m, dy, axis=0), dx, axis=1)
    return out


def crop_below_face(mask, core, observed, margin: int = 0):
    """把掩码裁到「最低观测像素 + margin」以上（y 越大越靠下）。

    未观测的脖子/胸口是扩散最容易补出衣服、项链、胸口的地方；宁可在那里
    留空，也不要生成假内容。返回裁完的 `(mask, core)`。
    """
    import numpy as np

    m = np.asarray(mask, bool)
    c = np.asarray(core, bool)
    obs = None if observed is None else np.asarray(observed, bool)
    if obs is None or not obs.any():
        return m, c
    rows = np.arange(m.shape[0])[:, None]
    y_cut = int(np.where(obs)[0].max()) + int(margin)
    y_cut = min(max(y_cut, 0), m.shape[0] - 1)
    keep = rows <= y_cut
    return m & keep, c & keep


def observed_pixels(ids_np, observed_face=None):
    """面 id 图 → 像素级"该表面被真实数据看过"掩码。

    `observed_face` 是逐面 bool；`None` 表示不限制。**缺失解剖结构**
    （脖子/肩部：FLAME 有面、数据从未拍到）在这里被排除，扩散不去补。
    """
    import numpy as np

    ids = np.asarray(ids_np)
    if ids.dtype == bool:
        raise ValueError("observed_pixels 需要面 id（int），不是布尔掩码")
    out = np.zeros(ids.shape, bool)
    if observed_face is None:
        out[...] = ids >= 0
        return out
    valid = ids >= 0
    if valid.any():
        out[valid] = np.asarray(observed_face, bool)[ids[valid].astype(np.int64)]
    return out


def gradient_energy(img, mask) -> float:
    """前景内平均梯度幅值（锐度的代理）。"""
    import numpy as np

    m = np.asarray(mask, bool)
    if not m.any():
        return 0.0
    grey = np.asarray(img, np.float32).mean(axis=-1)
    gy, gx = np.gradient(grey)
    return float(np.hypot(gx, gy)[m].mean())


def unsharp(img, gain: float, sigma: float = 1.0):
    """固定增益 unsharp；只补救 SD 往返造成的低通（标定见模块注释）。"""
    import numpy as np
    from scipy.ndimage import gaussian_filter

    g = np.asarray(img, np.float32)
    if gain <= 0:
        return g.copy()
    blur = gaussian_filter(g, sigma=(float(sigma), float(sigma), 0.0))
    return np.clip(g + float(gain) * (g - blur), 0.0, 255.0)


def composite_labels(render, gen, mask):
    """掩码内用生成图、掩码外保留渲染图。

    SD 的整图输出在可信区也有 ~1.3/255 的色偏；直接当标签会让微调把色偏
    也学进去（实测只调颜色时 PSNR −0.63 dB）。合成后可信区严格不变。
    """
    import numpy as np

    out = np.array(render, copy=True)
    m = np.asarray(mask, bool)
    if m.any():
        out[m] = np.asarray(gen)[m]
    return out


def match_color(src, tgt, src_mask, tgt_mask, *, space: str = "lab",
                mean_only: bool = False):
    """把 src 在 src_mask 内的颜色分布对齐到 tgt 在 tgt_mask 内的分布。

    Reinhard 风格：逐通道 (x - μ_s) / σ_s * σ_t + μ_t。
    只改颜色统计，不改结构；默认在 Lab 空间做，避免 RGB 三通道耦合。
    """
    import cv2
    import numpy as np

    a = np.asarray(src, np.float32)
    b = np.asarray(tgt, np.float32)
    if space == "lab":
        a = cv2.cvtColor(a, cv2.COLOR_RGB2LAB)
        b = cv2.cvtColor(b, cv2.COLOR_RGB2LAB)
    sp = a[np.asarray(src_mask, bool)]
    tp = b[np.asarray(tgt_mask, bool)]
    if sp.size == 0 or tp.size == 0:
        return np.asarray(src, np.float32)
    sm, ss = sp.mean(0), sp.std(0) + 1e-6
    tm, ts = tp.mean(0), tp.std(0) + 1e-6
    if mean_only:
        # 只对齐均值：保留生成图自身对比度，避免被周围平缓区域压平
        out = (a - sm) + tm
    else:
        # std 比值限幅：允许轻微统一色调，但不摧毁结构/纹理对比
        ratio = np.clip(ts / ss, 0.8, 1.25)
        out = (a - sm) * ratio + tm
    if space == "lab":
        out = cv2.cvtColor(out.astype(np.float32), cv2.COLOR_LAB2RGB)
    return np.clip(out, 0.0, 255.0)


def photometric_coverage(scene, data_dir, faces, glctx, *,
                         n_frames: int = 60,
                         device: str = "cuda") -> tuple:
    """逐面统计"落在真实图像前景（alpha>127）里的次数"。

    与 `find_defects` 的 `coverage` 不同：后者是**几何可见性**——网格被投影到
    就算看见。脖子/胸口在正面视图里几何可见、但图像是黑底，会被误当成"看过"。
    这里要求像素真的落在前景里，才算"数据支持过这块表面"。
    """
    import numpy as np
    import torch
    from PIL import Image

    from live3dgsavatar.core.render.faceid import detect_flip, face_id_map
    from live3dgsavatar.core.types import Camera

    total = int(scene.num_frames)
    n = max(1, min(int(n_frames), total))
    idx = np.linspace(0, total - 1, n).astype(int).tolist()
    n_faces = int(scene.faces.shape[0])
    counts = np.zeros(n_faces, np.int64)
    w2c = torch.eye(4, dtype=torch.float32)
    w2c[:3, :3] = scene.R.detach().cpu()
    w2c[:3, 3] = scene.T.detach().cpu()
    flip = None
    for i in idx:
        frame = scene.frames[i]
        verts = frame["mesh"].unsqueeze(0).to(device)
        cam = Camera(K=scene.K, w2c=w2c.unsqueeze(0).to(device),
                     width=scene.width, height=scene.height).to(device)
        rgba = np.asarray(Image.open(data_dir / "images" / f"{i:05d}.png")
                          .convert("RGBA"))
        fg = rgba[..., 3] > 127
        if flip is None:
            flip, _ = detect_flip(verts, faces, cam, glctx, fg)
        ids = face_id_map(verts, faces, cam, glctx, flip_y=flip)
        ids_np = ids.detach().cpu().numpy()
        sel = fg & (ids_np >= 0)
        if sel.any():
            counts += np.bincount(ids_np[sel].reshape(-1), minlength=n_faces)
    return counts, bool(flip)


def build_defect_mask(alpha, surf, ids_np, face_mass=None, *,
                      observed=None, mode: str = "full", spike: float = 0.15,
                      defect_quantile: float = 0.7, max_hole_area: float = 0.15,
                      border: int = 2, dilate_px: int = 6,
                      core_dilate: int = 4):
    """构造"要重绘"的像素掩码，返回 `(mask, core, dropped, stats)`。

    - `mask`：扩散上下文（目标外扩 `dilate_px`，ControlNet/重绘用）；
    - `core`：真正替换回渲染图的区域（目标外扩 `core_dilate`，默认 4px）；
      眼睛/嘴巴/脸等已观测结构留在 core 外，由渲染图保留，避免换脸。
    - `holes`：只修 alpha 不足的内部小洞；
    - `full`（默认）：再并入 alpha 梯度破碎区与模型自评的高缺陷面；
    - `observed`：像素级"该表面被数据看过"掩码。给了就只在这些像素里找缺陷，
      于是 `alpha≈0` 的缺失解剖结构（脖子/肩膀）不会被重绘。
    """
    import numpy as np

    ids = np.asarray(ids_np)
    if ids.dtype == bool:
        raise ValueError(
            "build_defect_mask 需要面 id（int），不是布尔掩码；"
            "面 id 才能索引 face_mass")
    a = np.asarray(alpha, np.float32)
    s = np.asarray(surf, bool)
    obs = None if observed is None else np.asarray(observed, bool)
    gy, gx = np.gradient(a)
    broken = s & (np.hypot(gx, gy) > spike)
    if mode == "structure":
        # 语义区域：只修耳朵/头发边缘/脖子下端，五官（眼/嘴/鼻）排除。
        # 用 2D 头部 bbox 做近似，不依赖 FLAME landmark。
        from scipy.ndimage import binary_erosion
        ys, xs = np.where(s)
        if ys.size == 0:
            z = np.zeros_like(s, bool)
            return z, z.copy(), 0, {"hole_px": 0,
                                    "observed_px": (int(obs.sum())
                                                    if obs is not None
                                                    else int(s.sum())),
                                    "mask_ratio": 0.0, "core_ratio": 0.0}
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        h = max(1, y1 - y0)
        w = max(1, x1 - x0)
        yy, xx = np.mgrid[0:s.shape[0], 0:s.shape[1]]
        # 五官：脸中央（相对头部 bbox）
        feat = (s & (xx >= x0 + 0.32 * w) & (xx <= x0 + 0.68 * w)
                & (yy >= y0 + 0.22 * h) & (yy <= y0 + 0.70 * h))
        # 耳朵：左右侧带
        ears = (s & ((xx <= x0 + 0.18 * w) | (xx >= x0 + 0.82 * w))
                & (yy >= y0 + 0.15 * h) & (yy <= y0 + 0.70 * h))
        # 脖子下端：下部
        neck = s & (yy >= y0 + 0.78 * h)
        # 头发边缘：外轮廓带（1px，避免带进额头）
        edge = s & ~binary_erosion(s, iterations=1)
        # 五官保护带 8px：必须大于等于 core 的膨胀，否则 mask 会扩回脸上
        feat_safe = dilate(feat, 8)
        target = (ears | neck | edge) & ~feat_safe
        mask = dilate(target, dilate_px) & ~feat_safe
        core = dilate(target, core_dilate) & ~feat_safe
        stats = {"hole_px": int((s & (a < 0.5)).sum()),
                 "observed_px": (int(obs.sum()) if obs is not None
                                 else int(s.sum())),
                 "mask_ratio": round(float(mask.mean()), 4),
                 "core_ratio": round(float(core.mean()), 4)}
        return mask, core, 0, stats
    if mode == "unsupported":
        if obs is None:
            raise ValueError("unsupported 模式需要 observed（逐面数据支持）")
        # 生成目标：FLAME 有面、但真实数据从未支持过的区域
        hole = s & ~obs
        # 脖子/胸口通常连到图像底边：去掉碰底边的连通域，只留头部/后脑
        from scipy.ndimage import label as _label
        lab, nlab = _label(hole)
        kept = np.zeros_like(hole)
        for k in range(1, nlab + 1):
            comp = lab == k
            if comp[-1].any():
                continue
            kept |= comp
        hole = kept
        limit, use_border = 1.0, 0
    else:
        hole = s & (a < 0.5)
        if obs is not None:
            hole &= obs
            broken &= obs
        if mode == "full":
            hole = hole | dilate(broken, 3)
            if face_mass is not None:
                fm = np.asarray(face_mass, np.float64)
                valid = ids >= 0
                if valid.any():
                    vis = fm[ids[valid]]
                    cut = float(np.quantile(vis, defect_quantile))
                    pm = np.zeros_like(a)
                    pm[valid] = fm[ids[valid]]
                    extra = s & (pm > cut)
                    if obs is not None:
                        extra &= obs
                    hole = hole | extra
        limit = max_hole_area if mode == "holes" else 0.5
        use_border = border
    if mode == "unsupported":
        # 生成目标本来就是"大块未观测表面"（含轮廓/脖子），不做内部小洞过滤
        keepped, dropped = hole, 1 if hole.any() else 0
    else:
        keepped, dropped = keep_interior_holes(hole, max_area=limit,
                                               border=use_border)
    mask = dilate(keepped, dilate_px)
    core = dilate(keepped, core_dilate)
    if obs is not None and mode != "unsupported":
        # 膨胀不能越界到未观测表面（否则又会把脖子带进来）
        mask &= dilate(obs, dilate_px)
        core &= dilate(obs, core_dilate)
    stats = {"hole_px": int((s & (a < 0.5)).sum()),
             "observed_px": int(obs.sum()) if obs is not None else int(s.sum()),
             "mask_ratio": round(float(mask.mean()), 4),
             "core_ratio": round(float(core.mean()), 4)}
    return mask, core, dropped, stats


def keep_interior_holes(mask, *, max_area: float = 0.15,
                        border: int = 2):
    """只保留"内部的小洞"，返回 `(mask, 丢弃的连通域数)`。

    大块空洞（脖子、锁骨、画面外）不是"漏画"，而是**模型本来就没有的解剖结构**；
    让扩散去补它，等于给伪标签注入想象，微调学到的也是想象。这里按两条规则筛：
    碰图像边缘的丢掉、面积超过 `max_area` 的丢掉。
    """
    import numpy as np
    from scipy.ndimage import label

    m = np.asarray(mask, bool)
    if not m.any():
        return m, 0
    lab, n = label(m)
    keep = np.zeros_like(m)
    dropped = 0
    for k in range(1, n + 1):
        comp = lab == k
        if comp.sum() > max_area * m.size:
            dropped += 1
            continue
        b = max(1, int(border))
        if (comp[:b].any() or comp[-b:].any()
                or comp[:, :b].any() or comp[:, -b:].any()):
            dropped += 1
            continue
        keep |= comp
    return keep, dropped


def foreground_iou(a, b) -> float:
    import numpy as np

    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = int((a | b).sum())
    return float((a & b).sum()) / union if union else 1.0


def masked_ssim(a, b, mask) -> float:
    import numpy as np

    from live3dgsavatar.core.metrics import ssim

    m = np.asarray(mask, bool)
    if not m.any():
        return 1.0
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    # 只比较洞内：把外面填成同一张图，避免边界影响
    aa = a.copy()
    bb = b.copy()
    aa[~m] = a[~m]
    bb[~m] = a[~m]
    return float(ssim(aa.astype(np.uint8), bb.astype(np.uint8)))


def build_pipeline(args):
    """加载 SD + ControlNet inpainting 管线（fp16、冻结）。"""
    try:
        import torch
        from diffusers import (ControlNetModel,
                               StableDiffusionControlNetInpaintPipeline)
    except ImportError as e:  # pragma: no cover - 依赖缺失
        raise SystemExit(
            "[error] 需要 diffusers / transformers / accelerate。\n"
            "        先跑 bash setup_env.sh（requirements.txt 里已列出）。\n"
            f"        原始错误：{e}") from None

    dtype = torch.float16
    controlnet = ControlNetModel.from_pretrained(args.controlnet,
                                                 torch_dtype=dtype)
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        args.base_model, controlnet=controlnet, torch_dtype=dtype,
        safety_checker=None, requires_safety_checker=False)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    face_app = None
    if args.ip_adapter in ("faceid", "faceid_plus"):
        try:
            from insightface.app import FaceAnalysis
        except ImportError as e:  # pragma: no cover
            raise SystemExit(
                "[error] IP-Adapter FaceID 需要 insightface/onnxruntime：\n"
                "        pip install insightface onnxruntime opencv-python-headless\n"
                f"        原始错误：{e}") from None
        # faceid 只用 ArcFace embedding；faceid_plus 额外用 CLIP image encoder
        image_encoder_folder = None
        if args.ip_adapter == "faceid_plus":
            from transformers import CLIPVisionModelWithProjection
            enc = Path(args.ip_adapter_image_encoder)
            if not enc.is_dir():
                from huggingface_hub import snapshot_download
                snap = snapshot_download(
                    "h94/IP-Adapter",
                    allow_patterns=["sdxl_models/image_encoder/*"])
                enc = Path(snap) / "sdxl_models" / "image_encoder"
            # diffusers 的 image_encoder_folder 是 ip_adapter_path 的子目录；
            # 本地 CLIP 直接注册到 pipeline，load_ip_adapter 就会跳过加载。
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                str(enc)).to(pipe.device, dtype=pipe.dtype)
            pipe.register_modules(image_encoder=image_encoder)
        pipe.load_ip_adapter(args.ip_adapter_path, subfolder=None,
                             weight_name=args.ip_adapter_weight,
                             image_encoder_folder=None)
        if args.ip_adapter == "faceid_plus":
            # Plus 必须同时给 CLIP 图 + ArcFace embedding；该互斥校验只适用于
            # 普通 IP-Adapter，这里跳过（输入由调用处保证）。
            pipe.check_inputs = lambda *a, **k: None
        pipe.set_ip_adapter_scale(float(args.ip_adapter_scale))
        face_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        face_app.prepare(ctx_id=0, det_size=(640, 640))
    return pipe, face_app


# ------------------------------------------------------------------ 主流程 --


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import numpy as np
    import torch
    from PIL import Image

    from live3dgsavatar.config import load_config, resolve_model_ply
    from live3dgsavatar.core.camera import (orbit_w2c_matrix, roll_w2c,
                                           rotation_x, rotation_y)
    from live3dgsavatar.core.io import load_ply
    from live3dgsavatar.core.io.ply import read_embedded_config
    from live3dgsavatar.core.render import SimpleRasterizer
    from live3dgsavatar.core.render.faceid import face_id_map
    from live3dgsavatar.core.render.normals import normal_map
    from live3dgsavatar.core.types import Camera, Mesh
    from live3dgsavatar.data.datasets import resolve_dataset_dir
    from live3dgsavatar.data.pose import rot6d_to_matrix
    from live3dgsavatar.data.scene import load_scene

    cfg = load_config()
    subject = args.subject or str(cfg.get("subject") or "")
    work = args.work_name or str(cfg.get("work_name") or "")
    if not (subject and work):
        raise SystemExit("[error] 需要 --subject 与 --work-name")
    cfg.set("subject", subject)
    cfg.set("work_name", work)
    dataset = args.dataset or subject
    data_dir = resolve_dataset_dir(cfg, dataset)
    ply = resolve_model_ply(cfg)
    defects_dir = Path(args.defects or (
        Path(str(cfg.get("paths.output_dir", "output"))) / "defects"
        / subject / work / dataset))
    selected = defects_dir / "selected.json"
    if not selected.is_file():
        raise SystemExit(f"[error] 找不到 {selected}（先跑 find_defects.py）")
    out_dir = Path(args.out or (defects_dir / "pseudo"))
    acc_dir = out_dir / "accepted"
    dbg_dir = out_dir / "debug"
    for d in (acc_dir, dbg_dir):
        d.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        # 上一次跑通过、这一次被拒的视角不能留在 accepted/ 里（会混进 refined/）
        stale = len(list(acc_dir.glob("*.png")))
        for p in acc_dir.glob("*.png"):
            p.unlink()
        if stale:
            print(f"  清理旧 accepted {stale} 张", flush=True)
    payload = json.loads(selected.read_text(encoding="utf-8"))
    # 光栅 y 翻转是**整段场景固定**的约定；find_defects 已在覆盖统计里
    # 用正面帧判定过一次。逐视角按 alpha IoU 重判在大角度/近似对称轮廓上
    # 会翻错（crh_1 frame01509: 0.768 vs 0.765，normal 图上下颠倒）。
    forced_flip = payload.get("raster_flip")
    forced_flip = None if forced_flip is None else bool(forced_flip)
    labels = list(payload["views"]["chosen"])
    if args.limit:
        labels = labels[: args.limit]
    print(f"模型  : {ply}")
    print(f"数据  : {data_dir}")
    print(f"缺陷  : {defects_dir}（{len(labels)} 个视角）")
    print(f"产物  : {out_dir}")

    device = "cuda"
    scene = load_scene(cfg, frames=-1, data_dir=data_dir)
    avatar = load_ply(ply, config=read_embedded_config(ply), device=device)
    rasterizer = SimpleRasterizer()
    from live3dgsavatar.evaluation.views import parse_view_label

    w2c_data = torch.eye(4, dtype=torch.float32)
    w2c_data[:3, :3] = scene.R.detach().cpu()
    w2c_data[:3, 3] = scene.T.detach().cpu()
    bg = torch.tensor([float(x) for x in cfg.get("render.background", [0, 0, 0])],
                      dtype=torch.float32, device=device)
    faces = scene.faces.to(torch.int32)
    import nvdiffrast.torch as dr
    glctx = dr.RasterizeCudaContext()

    def pick_flip(alpha_np, verts, cam):
        """用 alpha 轮廓判定光栅化方向：两种 y 翻转里，轮廓重合更好的那个。"""
        a = alpha_np > 0.5
        masks = {}
        for flip in (False, True):
            _, m = normal_map(verts, faces, cam, glctx, flip_y=flip,
                              convention=args.normal_convention)
            masks[flip] = m.detach().cpu().numpy()
        ious = {k: foreground_iou(v, a) for k, v in masks.items()}
        if args.raster_flip == "true":
            best = True
        elif args.raster_flip == "false":
            best = False
        else:
            best = max(ious, key=ious.get)
        return best, ious

    # 逐面缺陷质量与观测次数：mass.npz 是"模型自己认为哪里差"与
    # "哪些表面真的被数据看过"的直接信号。观测次数用来排除缺失解剖结构。
    face_mass = None
    observed_face = None
    mass_path = defects_dir / "mass.npz"
    if mass_path.is_file():
        with np.load(mass_path, allow_pickle=True) as d:
            gmass = np.asarray(d["mass"], np.float32)
            gcov = (np.asarray(d["coverage"], np.float32)
                    if "coverage" in d.files else None)
        fmap = avatar.binding_face_id.detach().cpu().numpy()
        n_faces = int(scene.faces.shape[0])
        acc = np.zeros(n_faces, np.float64)
        cnt = np.zeros(n_faces, np.float64)
        np.add.at(acc, fmap, gmass)
        np.add.at(cnt, fmap, 1.0)
        nz = cnt > 0
        face_mass = np.zeros_like(acc)
        face_mass[nz] = acc[nz] / cnt[nz]
        print(f"  缺陷质量：{mass_path.name}（逐面中位 {np.median(face_mass):.3f}）")
        if (gcov is not None and args.photometric_frames <= 0
                and not args.no_observed and args.coverage_min > 0):
            fcov = np.zeros(n_faces, np.float64)
            np.maximum.at(fcov, fmap, gcov)
            observed_face = fcov >= int(args.coverage_min)
            print(f"  观测面：{observed_face.mean():.1%} 的面被数据看过"
                  f"（coverage ≥ {args.coverage_min}；其余不补）")
    if args.photometric_frames > 0 and not args.no_observed:
        print(f"  前景覆盖统计（{args.photometric_frames} 帧）…", flush=True)
        fg_counts, flip_cov = photometric_coverage(
            scene, data_dir, faces, glctx, n_frames=args.photometric_frames)
        observed_face = fg_counts >= int(args.photometric_min_px)
        print(f"  前景支持的面：{observed_face.mean():.1%}"
              f"（≥{args.photometric_min_px} px；脖子/胸口不补，"
              f"光栅 y 翻转={flip_cov}）")
    if observed_face is None and not args.no_observed and args.coverage_min > 0:
        print("  [warn] 无法建立观测面限制——会补缺失解剖结构", flush=True)

    pipe, face_app = ((None, None) if args.dry_run
                      else build_pipeline(args))
    records = []
    for label in labels:
        info = parse_view_label(label)
        idx = int(info["expr"][5:]) if info["expr"].startswith("frame") else 0
        idx = min(max(idx, 0), scene.num_frames - 1)
        frame = scene.frames[idx]
        center = frame["mesh"].mean(dim=0).detach().cpu()
        bw = frame["blend_weight"].detach().cpu().numpy()[None]
        head_R = torch.from_numpy(rot6d_to_matrix(bw[:, 100:106])[0]).float()
        R_target = (rotation_x(info["pitch"])
                    @ rotation_y(info["yaw"]))[:3, :3]
        view = orbit_w2c_matrix(w2c_data, center, R_target @ head_R.T)
        view = roll_w2c(view, info["roll"])
        cam = Camera(K=scene.K, w2c=view.unsqueeze(0).to(device),
                     width=scene.width, height=scene.height).to(device)
        mesh = Mesh(verts=frame["mesh"].unsqueeze(0), faces=scene.faces,
                    uvs=scene.uvs, uv_faces=scene.uv_faces)
        weight = frame["blend_weight"].unsqueeze(0).to(device)
        with torch.no_grad():
            gs = avatar.deform(mesh, weight)
            out = rasterizer.render(gs, cam, bg)
            alpha_np = out.alpha[0, 0].detach().cpu().numpy()
            if forced_flip is None:
                flip, ious = pick_flip(alpha_np, mesh.verts, cam)
                flip_note = (f"IoU: 翻 {ious[True]:.3f} / "
                             f"不翻 {ious[False]:.3f}")
            else:
                flip = forced_flip
                flip_note = "来自 find_defects"
            ids = face_id_map(mesh.verts, faces, cam, glctx, flip_y=flip)
            ids_int = ids.detach().cpu().numpy()
            ids_np = ids_int >= 0
            normal, surf = normal_map(mesh.verts, faces, cam, glctx,
                                      flip_y=flip,
                                      convention=args.normal_convention)
        img = (out.color[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
               * 255.0).astype(np.uint8)
        surf_np = surf.detach().cpu().numpy()
        obs_px = (observed_pixels(ids_int, observed_face)
                  if observed_face is not None else None)
        print(f"  {label:40s} 光栅 y 翻转={flip}（{flip_note}）", flush=True)
        mask, core, dropped, mstats = build_defect_mask(
            alpha_np, surf_np, ids_int, face_mass, observed=obs_px,
            mode=args.mask_mode,
            spike=args.spike_threshold, defect_quantile=args.defect_quantile,
            max_hole_area=args.max_hole_area, border=args.border_margin,
            dilate_px=args.mask_dilate, core_dilate=args.core_dilate)
        if (args.mask_mode == "unsupported" and args.head_cut
                and obs_px is not None and obs_px.any()):
            mask, core = crop_below_face(
                mask, core, obs_px, margin=args.head_cut_margin)
        if dropped:
            print(f"  {label:40s} 丢弃 {dropped} 块大洞（缺失解剖结构/画面外）",
                  flush=True)
        if mask.any():
            print(f"  {label:40s} 掩码 {mstats['mask_ratio']:.1%}"
                  f"（alpha<0.5 的面 {mstats['hole_px']} px）", flush=True)
        normal_u8 = (normal.detach().cpu().numpy() * 255).astype(np.uint8)

        Image.fromarray(img).save(dbg_dir / f"{label}.render.png")
        Image.fromarray(normal_u8).save(dbg_dir / f"{label}.normal.png")
        Image.fromarray((mask * 255).astype(np.uint8)).save(
            dbg_dir / f"{label}.mask.png")
        if obs_px is not None:
            Image.fromarray((obs_px * 255).astype(np.uint8)).save(
                dbg_dir / f"{label}.observed.png")
        # 叠加图：在渲染图上画出掩码边界，便于**直接检查是否上下颠倒**
        overlay = img.copy()
        edge = mask ^ _erode(mask, 2)
        overlay[edge] = np.array([255, 0, 0], np.uint8)
        Image.fromarray(overlay).save(dbg_dir / f"{label}.overlay.png")
        # 自检：法线掩码与面 id 掩码必须几乎一致（同一套光栅化）
        agree = float((surf_np & ids_np).sum()) / max(1, int((surf_np | ids_np).sum()))
        if agree < 0.98:
            print(f"  [warn] {label}: 法线掩码与面 id 掩码一致性 {agree:.3f}"
                  "（若明显偏低说明光栅化方向不一致）", flush=True)
        if args.dry_run:
            records.append({"label": label, "status": "dry-run",
                            "mask_px": int(mask.sum()),
                            "mask_ratio": mstats["mask_ratio"],
                            "observed_px": mstats["observed_px"]})
            continue
        if not mask.any():
            # 没有可修区域：不跑扩散，也不产出标签（否则等于拿渲染图监督自己）
            records.append({"label": label, "accepted": False,
                            "status": "no-defect", "reasons": [], "mask_px": 0,
                            "observed_px": (int(obs_px.sum())
                                            if obs_px is not None else None)})
            print(f"  {label:40s} - 无缺陷（掩码为空，不跑扩散）", flush=True)
            continue

        face_emb = None
        ref_pil = None
        if face_app is not None:
            ref_path = defects_dir / "refs" / f"{idx:05d}.png"
            if ref_path.is_file():
                ref_rgb = np.asarray(Image.open(ref_path).convert("RGB"))
                ref_pil = Image.fromarray(ref_rgb)
                faces = face_app.get(ref_rgb[:, :, ::-1])  # BGR
                if faces:
                    face = max(faces, key=lambda f: float(
                        (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))
                    face_emb = torch.from_numpy(face.normed_embedding)
                    face_emb = (face_emb.unsqueeze(0).unsqueeze(0)
                                .to(device=device, dtype=pipe.unet.dtype))
            if face_emb is None:
                print(f"  {label:40s} [warn] 找不到参考脸，退回无身份条件",
                      flush=True)

        if args.ip_adapter == "faceid_plus" and ref_pil is not None:
            with torch.no_grad():
                clip_embeds, _ = pipe.encode_image(
                    ref_pil, device=pipe.device, num_images_per_prompt=1)
            proj = pipe.unet.encoder_hid_proj.image_projection_layers[0]
            proj.clip_embeds = (clip_embeds.unsqueeze(1).unsqueeze(1)
                                .repeat(2, 1, 1, 1)
                                .to(device=pipe.device, dtype=pipe.dtype))

        pseudo_list = []
        for k in range(max(1, args.consistency)):
            g = torch.Generator(device="cuda").manual_seed(args.seed + k)
            kwargs = dict(
                prompt=args.prompt, negative_prompt=args.negative,
                image=Image.fromarray(img),
                mask_image=Image.fromarray((mask * 255).astype(np.uint8)),
                control_image=Image.fromarray(normal_u8),
                strength=float(args.strength),
                num_inference_steps=int(args.steps),
                guidance_scale=float(args.guidance),
                generator=g)
            if face_emb is not None:
                # CFG：负分支用零 embedding，正分支用真实身份；相同会互相抵消
                kwargs["ip_adapter_image_embeds"] = [
                    torch.cat([torch.zeros_like(face_emb), face_emb], dim=0)]
                if args.ip_adapter == "faceid_plus" and ref_pil is not None:
                    kwargs["ip_adapter_image"] = ref_pil
            res = pipe(**kwargs).images[0]
            pseudo_list.append(np.asarray(res.convert("RGB"), np.float32))
        raw = pseudo_list[0]
        gen = unsharp(raw, args.sharpen)
        if args.color_match:
            from scipy.ndimage import binary_dilation
            ring = (binary_dilation(core, iterations=max(1, args.color_ring))
                    & ~core)
            tgt = ring & (alpha_np > 0.5)   # 只取渲染前景，避开黑背景
            if tgt.sum() < 64:
                tgt = ring
            gen = match_color(gen, img, core, tgt,
                              mean_only=args.color_mean_only)
        if args.feather > 0:
            from scipy.ndimage import gaussian_filter
            w = gaussian_filter(core.astype(np.float32), float(args.feather))
            w = np.clip(w, 0.0, 1.0)[..., None]
            pseudo = img.astype(np.float32) * (1.0 - w) + gen * w
            pseudo = np.where(mask[..., None], pseudo, img.astype(np.float32))
        else:
            pseudo = composite_labels(img, gen, core)
        Image.fromarray(raw.astype(np.uint8)).save(dbg_dir / f"{label}.gen.png")
        Image.fromarray(pseudo.astype(np.uint8)).save(
            dbg_dir / f"{label}.pseudo.png")

        # ---- 筛选 ----
        reasons = []
        # ArcFace 身份门：以**最终标签**为准。core 外的脸来自渲染图，
        # 生成图的脸只作诊断（保守合成下它本来就不会进入标签）。
        id_sim = None
        gen_sim = None
        if args.identity_gate and face_app is not None and face_emb is not None:
            re = face_emb.detach().float().cpu().numpy().reshape(-1)

            def _face_sim(im):
                faces = face_app.get(im.astype(np.uint8)[:, :, ::-1])
                if not faces:
                    return None
                f = max(faces, key=lambda x: float(
                    (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])))
                ge = np.asarray(f.normed_embedding, np.float32)
                denom = float(np.linalg.norm(re) * np.linalg.norm(ge)) + 1e-8
                return float(np.dot(re, ge) / denom)

            gen_sim = _face_sim(raw)
            id_sim = _face_sim(pseudo)
            if id_sim is None:
                reasons.append("身份门：标签里找不到脸")
            elif id_sim < args.identity_threshold:
                reasons.append(f"身份相似度 {id_sim:.3f} < "
                               f"{args.identity_threshold}")
        cons = 1.0
        trusted = ~mask
        if trusted.any():
            diff = np.abs(raw - img.astype(np.float32)).mean(axis=-1)
            tdiff = float(diff[trusted].mean())
            if tdiff > args.trusted_tol:
                reasons.append(f"可信区域改动 {tdiff:.2f} > {args.trusted_tol}")
        gen_fg = raw.max(axis=-1) > 8
        render_fg = alpha_np > 0.5
        iou = foreground_iou(gen_fg, render_fg)
        iou_flame = foreground_iou(gen_fg, surf_np)
        # 掩码外只要求"渲染的轮廓还在"，不要求背景逐像素一致：
        # SD 的 img2img 会把整幅背景抬亮 1–2/255，用"前景并集/整幅图"算 IoU
        # 会把背景噪声算进去（实测大角度因此被误杀到 0.83）。
        outside_zone = ~dilate(mask, 2)
        rf_zone = render_fg & outside_zone
        gf_zone = gen_fg & outside_zone
        outside_iou = foreground_iou(gf_zone, rf_zone)
        outside_recall = (float((rf_zone & gf_zone).sum()) / rf_zone.sum()
                          if rf_zone.any() else 1.0)
        outside_extra = (float((gf_zone & ~render_fg).sum()) / rf_zone.sum()
                         if rf_zone.any() else 0.0)
        fill = float(gen_fg[mask].mean())
        inside = float(raw[mask].std())
        e_ref = gradient_energy(img, render_fg)
        sharp_ratio = (gradient_energy(pseudo, render_fg) / e_ref
                       if e_ref > 0 else 1.0)
        e_mask = gradient_energy(img, mask)
        mask_sharp = (gradient_energy(pseudo, mask) / e_mask
                      if e_mask > 0 else 1.0)
        if outside_recall < args.min_outside_iou:
            reasons.append(f"掩码外轮廓召回 {outside_recall:.3f} < "
                           f"{args.min_outside_iou}")
        if fill < args.min_fill:
            reasons.append(f"掩码内填充率 {fill:.2f} < {args.min_fill}")
        if mask_sharp < args.min_mask_sharp:
            reasons.append(f"掩码内锐度比 {mask_sharp:.3f} < "
                           f"{args.min_mask_sharp}")
        if sharp_ratio < args.min_sharp:
            reasons.append(f"锐度比 {sharp_ratio:.3f} < {args.min_sharp}")
        if inside < args.min_std:
            reasons.append(f"洞内过平 std {inside:.2f} < {args.min_std}")
        if len(pseudo_list) > 1:
            cons = masked_ssim(pseudo_list[0], pseudo_list[1], mask)
            if cons < args.min_consistency:
                reasons.append(
                    f"多次采样不一致 SSIM {cons:.3f} < {args.min_consistency}")
        status = "accepted" if not reasons else "rejected"
        if status == "accepted":
            Image.fromarray(pseudo.astype(np.uint8)).save(
                acc_dir / f"{label}.png")
        records.append({"label": label, "status": status,
                        "accepted": status == "accepted",
                        "reasons": reasons,
                        "mask_px": int(mask.sum()),
                        "observed_px": (int(obs_px.sum())
                                        if obs_px is not None else None),
                        "silhouette_iou": round(float(iou), 4),
                        "silhouette_iou_outside": round(float(outside_iou), 4),
                        "outside_recall": round(float(outside_recall), 4),
                        "outside_extra": round(float(outside_extra), 4),
                        "silhouette_iou_flame": round(float(iou_flame), 4),
                        "fill_ratio": round(float(fill), 4),
                        "sharp_ratio": round(float(sharp_ratio), 4),
                        "mask_sharp_ratio": round(float(mask_sharp), 4),
                        "trusted_diff": round(float(tdiff), 3),
                        "hole_std": round(float(inside), 2),
                        "consistency": round(float(cons), 4),
                        "identity_sim": (None if id_sim is None
                                         else round(float(id_sim), 4)),
                        "identity_sim_gen": (None if gen_sim is None
                                             else round(float(gen_sim), 4))})
        print(f"  {label:40s} {'✓' if status == 'accepted' else '✗ ' + '; '.join(reasons)}",
              flush=True)

    report = {"subject": subject, "work_name": work, "dataset": dataset,
              "defects": str(defects_dir), "dry_run": bool(args.dry_run),
              "gates": {"trusted_tol": args.trusted_tol,
                        "min_iou": args.min_iou,
                        "min_outside_recall": args.min_outside_iou,
                        "min_fill": args.min_fill,
                        "min_sharp": args.min_sharp,
                        "min_mask_sharp": args.min_mask_sharp,
                        "min_std": args.min_std,
                        "min_consistency": args.min_consistency,
                        "identity_threshold": args.identity_threshold,
                        "core_dilate": args.core_dilate},
              "observed": {"enabled": observed_face is not None,
                           "coverage_min": int(args.coverage_min)},
              "sharpen": float(args.sharpen),
              "views": records}
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    ok = sum(1 for r in records if r.get("status") == "accepted")
    n_defect = sum(1 for r in records if r.get("status") == "no-defect")
    n_reject = sum(1 for r in records if r.get("status") == "rejected")
    print(f"\n完成：{len(records)} 个视角，通过 {ok}，无缺陷 {n_defect}，"
          f"拒绝 {n_reject}")
    print(f"产物：{out_dir}")

    if args.install_refined and ok:
        import shutil

        refined = defects_dir / "refined"
        refined.mkdir(parents=True, exist_ok=True)
        # 与 accepted/ 同一理由：本次没通过的视角不能沿用上一轮的文件
        labels_set = set(labels)
        for p in refined.glob("*.png"):
            if p.stem in labels_set:
                p.unlink()
        n = 0
        for r in records:
            if r.get("status") != "accepted":
                continue
            shutil.copyfile(acc_dir / f"{r['label']}.png",
                            refined / f"{r['label']}.png")
            n += 1
        print(f"已装入 refined/：{n} 张（可交给 refine_from_images.py / iterative_repair.py）")
    return 0 if (args.dry_run or ok or n_defect) else 1


if __name__ == "__main__":
    raise SystemExit(main())
