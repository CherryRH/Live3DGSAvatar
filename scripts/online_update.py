#!/usr/bin/env python
"""在线更新（热更新）的实际运行入口。

用数据集里的一小段帧模拟"通话开始时的观测"，在时间预算内微调模型外观，
并报告 PSNR 的前后变化 —— 这就是热更新的验收判据。

两种模式：

1. ``--dry-run``：**不需要 GPU**。只测闭式颜色校正（纯张量运算），
   用数据集原图与模型颜色做点对点比较，验证拟合流程与指标。
2. 默认：完整流程（闭式颜色校正 + 梯度微调）。需要 GPU。

用法::

    python scripts/online_update.py --dry-run
    python scripts/online_update.py --frames 10 --time-budget 10
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# ⚠️ 必须在导入参照实现之前施加 numpy 兼容补丁（chumpy）。
import live3dgsavatar  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="在线更新（热更新）")
    p.add_argument("--subject", default=None,
                   help="**模型**的对象名。默认取配置")
    p.add_argument("--work-name", default=None)
    p.add_argument("--data-subject", default=None,
                   help="**观测/驱动数据**的数据集名。默认与 --subject 相同；"
                        "两者不同时即「用别的一段视频更新这个模型」")
    p.add_argument("--compare-frames", type=int, default=0,
                   help="对比多少帧：从**第 0 帧开始**连续取 N 帧；"
                        "-1 = 全部数据集帧（可看泛化，未参与拟合的帧也在内）；"
                        "0 = 不输出。与 --frames 独立")
    p.add_argument("--compare-dir", type=Path, default=None,
                   help="对比图目录；默认 <output_dir>/online_update/<实验名>/")
    p.add_argument("--exp-name", default=None,
                   help="实验名（输出子目录）。默认由关键参数自动生成，"
                        "例如 `<对象>__from-<数据集>__f300__noop__L2__all`")
    # 默认值取自 configs/system.yaml 的 online_update 节（None 表示用配置）
    p.add_argument("--frames", type=int, default=None,
                   help="用于更新的帧数（现实中几十帧通常就够）")
    p.add_argument("--sample", choices=("uniform", "random", "head"),
                   default="uniform",
                   help="观测帧怎么选：uniform=在整个数据集上等距（默认）；"
                        "random=随机；head=从头连续取")
    p.add_argument("--seed", type=int, default=0,
                   help="--sample random 的随机种子")
    p.add_argument("--paint", choices=PAINT_KINDS, default="none",
                   help="**能力测试**：把目标画面的指定区域涂成纯色，"
                        "看更新能否把渲染也刷成该色（默认 none）")
    p.add_argument("--paint-color", default="0,255,0",
                   help="涂色颜色，`R,G,B`（0–255），默认绿")
    p.add_argument("--paint-param", type=int, default=16,
                   help="条纹/棋盘格的格子边长（像素）")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--time-budget", type=float, default=None, help="墙钟预算（秒）")
    p.add_argument("--lr-color", type=float, default=None)
    p.add_argument("--lr-opacity", type=float, default=None)
    p.add_argument("--no-opacity", action="store_true", help="不更新不透明度")
    p.add_argument("--mask-weight", type=float, default=None,
                   help="掩码外像素权重（0 = 只用前景）")
    p.add_argument("--dry-run", action="store_true",
                   help="只测闭式颜色校正（不需要 GPU）")
    p.add_argument("--force", action="store_true",
                   help="即使 train.online_update.enabled=false 也运行")
    p.add_argument("--l2", action="store_true",
                   help="用 L2 而非配置里的 L1（覆盖 train.online_update.color_l1）")
    p.add_argument("--no-color", action="store_true",
                   help="不更新颜色（覆盖 train.online_update.update_color）")
    p.add_argument("--frames-chunk", type=int, default=None,
                   help="每步用几帧（覆盖 train.online_update.frame_chunk）")
    p.add_argument("--dump", type=Path, default=None,
                   help="把更新增量存成 npz")
    return p.parse_args(argv)


def _cfg_get(args, cfg, name, key, default):
    """命令行优先，其次配置（`train.online_update.*`），最后兜底默认值。"""
    v = getattr(args, name.replace("-", "_"), None)
    return v if v is not None else cfg.get(f"train.online_update.{key}", default)


@dataclass
class Setup:
    """模型侧与数据侧**分开**解析后的路径集合。

    为什么要分开：这是"用另一个数据集的画面更新本模型"。
    `cfg.subject` 同时决定模型目录和数据集目录，一旦切换就会互相干扰，
    所以先把模型侧全部解析完，再把 `subject` 指向数据侧。
    """

    cfg: object
    model_subject: str
    data_subject: str
    ply: Path
    model_config: Path | None
    data_dir: Path


def _setup(args) -> Setup:
    from live3dgsavatar.config import (load_config, resolve_data_dir,
                                       resolve_model_config, resolve_model_ply)

    cfg = load_config()
    if args.work_name:
        cfg.set("work_name", args.work_name)
    if args.subject:
        cfg.set("subject", args.subject)
    if not (str(cfg.get("subject") or "").strip()
            and str(cfg.get("work_name") or "").strip()):
        raise SystemExit(
            "[error] 需要指定模型：--subject 与 --work-name\n"
            "        默认配置不再写死对象名，所以必须显式给出（或写进 configs/local.yaml）。")
    model_subject = str(cfg.get("subject"))

    # ---- 模型侧（趁 subject 还指向模型）----
    ply = resolve_model_ply(cfg)
    model_config = resolve_model_config(cfg)

    # ---- 数据侧 ----
    data_subject = args.data_subject or model_subject
    cfg.set("subject", data_subject)
    data_dir = resolve_data_dir(cfg)
    return Setup(cfg=cfg, model_subject=model_subject,
                 data_subject=data_subject, ply=ply,
                 model_config=model_config, data_dir=data_dir)


def _load_model_config(setup: Setup) -> dict:
    import yaml

    if setup.model_config is None:
        raise SystemExit(
            f"[error] 找不到模型随附的 config.yaml"
            f"（models/{setup.model_subject}/{setup.cfg.work_name}/config.yaml）")
    return yaml.safe_load(open(setup.model_config))


def experiment_name(args, *, model_subject: str, data_subject: str,
                    n_obs: int, update_opacity: bool, use_l1: bool,
                    n_cmp: int) -> str:
    """由**关键参数**生成实验名，让输出目录自解释。

    为什么不用时间戳：同参数重跑应当覆盖同一目录（便于比较、不会堆一堆结果）；
    参数一变目录就变，不会把不同实验混在一起。

    形如 `<对象>__from-<数据集>__f300__noop__L2__all`。
    """
    if args.exp_name:
        return str(args.exp_name)
    parts = [model_subject]
    if data_subject != model_subject:
        parts.append(f"from-{data_subject}")
    parts.append(f"f{n_obs}-{getattr(args, 'sample', 'uniform')}")
    parts.append("op" if update_opacity else "noop")
    parts.append("L1" if use_l1 else "L2")
    paint = getattr(args, "paint", "none")
    if paint != "none":
        parts.append(f"paint-{paint}")
    parts.append("all" if n_cmp < 0 else f"cmp{n_cmp}")
    return "__".join(parts)


#: 涂色实验的图案。用于**能力测试**：更新机制若正常，应当能把渲染刷成该颜色。
PAINT_KINDS = ("none", "left", "right", "stripes", "checker")


def paint_mask(height: int, width: int, kind: str, param: int = 16):
    """生成涂色区域掩码 `[H,W]` bool（纯函数）。

    ⚠️ 与 `images/` 的 alpha 掩码是**两回事**：alpha 决定"哪些像素参与损失"，
    这里决定"哪些像素被涂色"。涂色只改目标 RGB，**不改 alpha** ——
    否则背景区域（模型永远渲染不到）会进入损失，产生无法降低的残差。
    """
    import numpy as np

    if kind == "none":
        return np.zeros((height, width), bool)
    xs = np.arange(width)[None, :].repeat(height, axis=0)
    if kind == "left":
        return xs < width // 2
    if kind == "right":
        return xs >= width // 2
    if kind == "stripes":
        step = max(1, int(param))
        return (xs // step) % 2 == 0
    if kind == "checker":
        step = max(1, int(param))
        ys = np.arange(height)[:, None].repeat(width, axis=1)
        return ((xs // step) + (ys // step)) % 2 == 0
    raise ValueError(f"未知涂色图案：{kind}")


def paint_apply(rgb, kind: str, color, param: int = 16):
    """把 `rgb [H,W,3]`（float 0–1）的指定区域涂成 `color [3]`（0–1）。原地修改并返回。"""
    import numpy as np

    if kind == "none":
        return rgb
    h, w = rgb.shape[:2]
    m = paint_mask(h, w, kind, param)
    rgb[m] = np.asarray(color, np.float32)
    return rgb


def parse_color(text: str) -> tuple[float, float, float]:
    """`255,0,0` → `(1.0, 0.0, 0.0)`（输入按 0–255）。"""
    parts = [float(x) for x in str(text).split(",")]
    if len(parts) != 3:
        raise ValueError(f"颜色需要 3 个分量（0–255），实际：{text!r}")
    if any(not (0 <= v <= 255) for v in parts):
        raise ValueError(f"颜色分量应在 0–255，实际：{parts}")
    return tuple(v / 255.0 for v in parts)


def sample_indices(total: int, n: int, strategy: str = "uniform",
                   seed: int = 0) -> list[int]:
    """从 `total` 帧里选 `n` 帧用于更新。

    | 策略 | 行为 | 用途 |
    |---|---|---|
    | `uniform` | 在整个序列上**等距**取（含首尾） | 默认：覆盖各种姿态/光照，泛化更好 |
    | `random` | 随机不重复（`seed` 可复现） | 想看抽样方差时 |
    | `head`   | 从头连续取 `n` 帧 | 复现旧行为 / 只关心某一片段 |

    为什么默认不是 `head`：连续几秒的片段姿态与光照几乎不变，
    在上面拟合出的颜色偏移等于只拟合那一小段，换个时段就失效
    （实测：前 300 帧 +0.8 dB，未参与的帧只有 +0.05 dB）。
    均匀采样把有限预算摊到整段视频上，同样的帧数更有代表性。
    """
    if total <= 0 or n <= 0:
        return []
    if n >= total:
        return list(range(total))

    if strategy == "head":
        return list(range(n))

    if strategy == "random":
        import random

        return sorted(random.Random(seed).sample(range(total), n))

    # uniform：n 个等距点，含首尾；用 round 后去重（保持升序）
    if n == 1:
        return [0]
    return sorted({int(round(i * (total - 1) / (n - 1))) for i in range(n)})


def _image_files(data_dir: Path) -> list[Path]:
    """数据集里抠图后的原图列表（`images/`，按文件名排序）。"""
    from live3dgsavatar.config import load_config

    cfg = load_config()
    sub = str(cfg.get("paths.image_subdir", "images"))
    return sorted((data_dir / sub).iterdir())


def _make_observation(scene, files: list[Path], i: int, device: str,
                      paint: str = "none", paint_color=(0.0, 1.0, 0.0),
                      paint_param: int = 16):
    """构造**单帧**观测（图从磁盘读，用完即可释放）。

    ## 为什么要单帧构造

    每帧的目标图是 `3×512×512 float32` = **3 MB**。3524 帧就是 **10.3 GB** ——
    直接把全部帧都塞进 batch 会 OOM（实测："11.24 GiB allocated by PyTorch"）。
    而 mesh 只有 61 KB/帧（3524 帧共 0.2 GB），可以整体留在显存里。

    所以内存策略是：**mesh 全留、图按需读**。
    """
    import numpy as np
    import torch
    from PIL import Image

    from live3dgsavatar.core.types import Mesh
    from live3dgsavatar.training.observations import FrameObservation

    img = np.array(Image.open(files[i]).convert("RGBA")).astype("float32") / 255.0
    target = img[..., :3]
    if paint != "none":
        # 只改 RGB，**不动 alpha** —— alpha 是"哪些像素参与损失"，
        # 把背景也纳入损失会产生模型永远达不到的残差。
        target = paint_apply(target.copy(), paint, paint_color, paint_param)
    rgb = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0).to(device)
    mask = torch.from_numpy(img[..., 3] > 0.5).unsqueeze(0).unsqueeze(0).to(device)
    return FrameObservation(
        mesh=Mesh(verts=scene.frames_tensor([i]),
                  faces=scene.faces, uvs=scene.uvs, uv_faces=scene.uv_faces),
        blend_weight=scene.weights_tensor([i]),
        target=rgb, mask=mask, name=f"{i:05d}")


def _scene_and_camera(cfg, data_dir: Path, n_frames: int, device: str):
    """载入场景（mesh 全留）与相机。`n_frames=-1` 表示全部。"""
    from live3dgsavatar.core.types import Camera
    from live3dgsavatar.data.scene import load_scene

    scene = load_scene(cfg, frames=n_frames, device=device)
    # ⚠️ 必须显式 `.to(device)`：`Camera` 的 `w2c` 若留在 CPU 而 `K` 在 GPU，
    #    光栅化内核会把它当作 CUDA 指针 → `illegal memory access`。
    cam = Camera.from_intrinsics_extrinsics(
        K=scene.K, R=scene.R, T=scene.T,
        width=scene.width, height=scene.height).to(device)
    return scene, cam


def _load_observations(cfg, data_dir: Path, n: int, device: str):
    """构造前 `n` 帧的观测（拟合用）。`n=-1` 表示全部。

    ⚠️ 目标图会**全部驻留显存**（每帧 3 MB）。`n` 很大时会 OOM ——
    对比图刻意走"逐帧读"的另一条路径，见 `_write_comparisons`。
    """
    from live3dgsavatar.training.observations import BatchObservation

    scene, cam = _scene_and_camera(cfg, data_dir, n, device)
    files = _image_files(data_dir)
    count = scene.num_frames if n < 0 else min(n, scene.num_frames)
    frames = [_make_observation(scene, files, i, device) for i in range(count)]
    return BatchObservation(frames=frames, camera=cam)


def _to_uint8_hwc(tensor) -> "object":
    """`[1,3,H,W]` 浮点（0–1）→ `[H,W,3]` uint8。"""
    import numpy as np

    img = tensor[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (img * 255.0 + 0.5).astype(np.uint8)


def _masked_psnr(pred, target, mask) -> float | None:
    """掩码内 PSNR（口径见 `core.metrics`）。"""
    from live3dgsavatar.core.metrics import psnr_masked

    return psnr_masked(pred, target, mask)


def _label_strip(width: int, text: str, height: int = 20):
    """在面板上方画一条深色标签条（ASCII，避免字体依赖）。"""
    from PIL import Image, ImageDraw

    strip = Image.new("RGB", (width, height), (24, 26, 32))
    ImageDraw.Draw(strip).text((6, 4), text, fill=(230, 232, 238))
    return strip


def _write_comparison(path: Path, *, target_rgb, before_rgb, after_rgb,
                      before_psnr: float | None, after_psnr: float | None,
                      frame_name: str) -> None:
    """一排三图：`target | 更新前 | 更新后`，每格上方带标签与 PSNR。

    `target` 是**抠图后的目标**（`images/`，黑底）—— 与两次渲染同背景，
    这样"更新前后"的差异不会被背景差异干扰。
    """
    from PIL import Image

    panels = [
        (target_rgb, f"target {frame_name}"),
        (before_rgb, "before" + ("" if before_psnr is None
                                 else f"  {before_psnr:.2f} dB")),
        (after_rgb, "after" + ("" if after_psnr is None
                                else f"  {after_psnr:.2f} dB")),
    ]
    tiles = [Image.fromarray(img) for img, _ in panels]
    w, h = tiles[0].size
    gap = 4

    total_w = w * len(tiles) + gap * (len(tiles) - 1)
    strip_h = 20
    canvas = Image.new("RGB", (total_w, h + strip_h), (10, 11, 14))
    x = 0
    for tile, (_, label) in zip(tiles, panels):
        canvas.paste(_label_strip(w, label, strip_h), (x, 0))
        canvas.paste(tile, (x, strip_h))
        x += w + gap

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _delta_report(updater, state) -> None:
    """打印增量本身有多大 —— 用来判断"PSNR 涨了但看着差"是哪种改动造成的。"""
    import torch

    from live3dgsavatar.core.avatar import sh0_to_rgb

    av = updater.avatar
    dcol = state.color_delta.detach()
    # SH DC → RGB 的增益（`sh0_to_rgb` 里是 0.2821）——换算成 0-255 更直观
    rgb_gain = 0.2821
    dcol_rgb = dcol * rgb_gain * 255.0
    print(f"\n[增量规模] 颜色 |Δ| 平均={float(dcol_rgb.abs().mean()):.2f} "
          f"最大={float(dcol_rgb.abs().max()):.2f} (0-255)")

    if state.opacity_delta is None:
        print("  不透明度：未更新")
        return
    dop = state.opacity_delta.detach()
    op0 = av.opacity.squeeze(1)          # [N]
    dop = dop.squeeze(1)                 # [N] —— 与 op0 同形状！
    # ⚠️ `op0` 是 `[N,1]` 而 `dop.squeeze(1)` 是 `[N]`，直接相加会广播成
    #    `[N,N]`（不报错，但统计全错）。两边都 squeeze 才是逐高斯对齐。
    live = op0 > -10                     # 只统计有效高斯（其余本来就不可见）
    before = torch.sigmoid(op0[live])
    after = torch.sigmoid((op0 + dop)[live])
    change = (after - before).abs()
    print(f"  不透明度 Δ（有效高斯 {int(live.sum())} 个）："
          f"平均={float(change.mean()):.4f}  p95={float(torch.quantile(change, 0.95)):.4f} "
          f"最大={float(change.max()):.4f}")
    print(f"  不透明度 均值 {float(before.mean()):.3f} → {float(after.mean()):.3f}")
    big = int((change > 0.2).sum())
    if big:
        print(f"  ⚠ {big} 个高斯的不透明度变化超过 0.2 —— "
              "透明度大幅变动容易让画面发虚/发暗，但可能降低 MSE。\n"
              "    可试 `--no-opacity` 对比：只改颜色通常更稳。")


def _write_comparisons(args, setup, updater, scene, camera, files, bg,
                       state_after, n_cmp: int, n_obs: int,
                       out_dir: Path, paint_kw: dict | None = None) -> int:
    """渲染并写出 `target | 更新前 | 更新后` 对比图。返回写出的张数。

    对比帧**从第 0 帧开始连续取 N 帧**（`N = -1` 表示全部）。
    若 N 超过观测帧数，多出来的帧**没有参与拟合** —— 那正是看泛化能力的部分。

    三格全部用**抠图后的目标**（`images/`，黑底）：这样三格同背景、同口径，
    PSNR 也正好是在这同一批像素上算的。原始 `input/` 背景是白墙，
    混进来只会让观感对比失真（背景差异被误读成模型变差）。

    ⚠️ `numpy` 必须在函数**开头**导入：Python 会把函数内任何 `import numpy as np`
    当作局部变量声明，若写在函数中段，前面所有对 `np` 的使用都会
    `UnboundLocalError`（曾因此崩过）。
    """
    import numpy as np
    import torch

    from live3dgsavatar.training.updater import UpdateState

    paint_kw = paint_kw or {}

    total = scene.num_frames
    n = total if n_cmp < 0 else max(0, min(int(n_cmp), total))
    if n == 0:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)

    state_before = UpdateState.zeros(
        updater.avatar, update_opacity=updater.config.update_opacity)

    print(f"\n[对比图] {n} 帧 → {out_dir}")
    rows = []
    paint_stats: dict[str, list] = {}
    for i in range(n):
        # **逐帧读图、用完即弃**：全部驻留会是 10 GB+（见 `_make_observation`）
        obs = _make_observation(scene, files, i, bg.device, **paint_kw)
        before = updater.render_observation(obs, bg, state_before)
        after = updater.render_observation(obs, bg, state_after)

        # 三格同口径：目标与两次渲染都是黑底、掩码内像素
        target = _to_uint8_hwc(obs.target)
        b_rgb = _to_uint8_hwc(before.color)
        a_rgb = _to_uint8_hwc(after.color)

        mask_np = None if obs.mask is None else obs.mask[0, 0].cpu().numpy().astype(bool)
        psnr_b = _masked_psnr(b_rgb, target, mask_np)
        psnr_a = _masked_psnr(a_rgb, target, mask_np)

        # ---- 涂色实验的三组指标（各回答不同问题，别混着看）----
        #
        #   ① 被涂区域 vs 涂色值  → "学会了没有"（目标函数优化得如何）
        #   ② 未涂区域 vs 原图    → "附带损伤"（没让改的地方被改坏多少）
        #   ③ 整体 vs 原图        → 参考值（含被涂区域，必然很低）
        #
        # 只报 ③ 会得到 8–9 dB 这种吓人的数字，但它衡量的是"渲染得多像一张
        # 半绿半真的图"，本来就不可能高 —— 不是缺陷，是问题问错了。
        if paint_kw and paint_kw.get("paint", "none") != "none":
            from PIL import Image

            hh, ww = target.shape[:2]
            pm = paint_mask(hh, ww, paint_kw["paint"], paint_kw.get("paint_param", 16))
            fg = mask_np if mask_np is not None else np.ones((hh, ww), bool)
            painted_region = pm & fg
            clean_region = (~pm) & fg

            glyph = np.asarray(paint_kw["paint_color"], np.float32) * 255.0
            if painted_region.any():
                for tag, img in (("before", b_rgb), ("after", a_rgb)):
                    err = np.abs(img[painted_region].astype(np.float32) - glyph).mean()
                    near = (np.abs(img[painted_region].astype(np.float32) - glyph)
                            .max(axis=-1) < 40).mean()
                    paint_stats.setdefault(tag, []).append((err, near))

            # 未涂区域的"原图"从磁盘重读（涂色只改 RGB，不影响 alpha）
            raw = np.array(Image.open(files[i]).convert("RGBA")).astype(np.float32) / 255.0
            orig_rgb = (raw[..., :3] * 255.0 + 0.5).astype(np.uint8)
            for tag, img in (("before", b_rgb), ("after", a_rgb)):
                val = _masked_psnr(img, orig_rgb, clean_region)
                if val is not None:
                    paint_stats.setdefault(f"clean_{tag}", []).append((val, 0.0))

        _write_comparison(out_dir / f"{obs.name}.png",
                          target_rgb=target, before_rgb=b_rgb, after_rgb=a_rgb,
                          before_psnr=psnr_b, after_psnr=psnr_a,
                          frame_name=obs.name)
        rows.append((i, psnr_b, psnr_a))
        del obs, before, after, b_rgb, a_rgb, target

    # ---- 汇总（帧多时不逐行刷屏）----
    valid = [(i, b, a) for i, b, a in rows if b is not None and a is not None]
    if valid:
        deltas = np.array([a - b for _, b, a in valid])
        if paint_kw and paint_kw.get("paint", "none") != "none":
            print("  （下表 PSNR 是对**涂色后**的目标算的 —— 它是优化目标，"
                  "数值低是正常的；\n"
                  "     '有没有学会'看下面的①，'有没有改坏别处'看②）")
        print(f"  {'帧':>6}  {'更新前':>8}  {'更新后':>8}  {'Δ':>7}")
        show = valid if len(valid) <= 24 else (valid[:8] + valid[-8:])
        for i, b, a in show:
            tag = "  (未参与拟合)" if i >= n_obs else ""
            print(f"  {i:>6}  {b:8.2f}  {a:8.2f}  {a - b:+7.2f}{tag}")
        if len(valid) > len(show):
            print(f"  …（共 {len(valid)} 帧，省略 {len(valid) - len(show)} 行）")
        print(f"  平均 {np.mean([b for _, b, _ in valid]):.2f} → "
              f"{np.mean([a for _, _, a in valid]):.2f} dB  "
              f"(Δ 均值 {deltas.mean():+.2f}, 中位 {np.median(deltas):+.2f}, "
              f"最差 {deltas.min():+.2f})")
        worse = int((deltas < -0.1).sum())
        if worse:
            print(f"  ⚠ {worse}/{len(valid)} 帧变差 —— PSNR 是平均值，"
                  "个别帧下降是正常的；若大面积下降说明更新方向不对")

    if paint_stats:
        b = paint_stats.get("before")
        a = paint_stats.get("after")
        if b and a:
            e0 = float(np.mean([v[0] for v in b]))
            e1 = float(np.mean([v[0] for v in a]))
            n0 = float(np.mean([v[1] for v in b])) * 100.0
            n1 = float(np.mean([v[1] for v in a])) * 100.0
            print("\n[涂色实验] 三组指标（各回答不同问题，不要混着看）")
            print(f"  ① 被涂区域 vs 涂色值：色差 {e0:.1f} → {e1:.1f} /255  "
                  f"（{e1 - e0:+.1f}）")
            print(f"     接近目标色(±40)的像素 {n0:.1f}% → {n1:.1f}%"
                  "   ← 这一项才说明'学会了没有'")
            if n1 < 20.0 and e1 > 40.0:
                print("     ⚠ 基本没学会涂色 —— 问题在更新机制或学习率，"
                      "而不是泛化")

        cb = paint_stats.get("clean_before")
        ca = paint_stats.get("clean_after")
        if cb and ca:
            c0 = float(np.mean([v[0] for v in cb]))
            c1 = float(np.mean([v[0] for v in ca]))
            print(f"  ② 未涂区域 vs **原图**：PSNR {c0:.2f} → {c1:.2f} dB "
                  f"（{c1 - c0:+.2f}）   ← 附带损伤")
            print("     这一项掉得少，说明更新只改了该改的地方；掉得多说明"
                  "颜色自由度在互相牵连")

        print("  ③ 整体 vs 涂色目标（上面的 PSNR 表）只有 8–9 dB 是**正常的**：")
        print("     它衡量'渲染得多像一张半绿半真的图'，本来就不可能高。")

    print(f"  已写出 {n} 张（每张：target | 更新前 | 更新后）")
    return n


def _dry_run(args) -> int:
    """不需要 GPU：用模型颜色与数据集原图做点对点比较，验证闭式拟合。"""
    import torch

    from live3dgsavatar.core.avatar import AvatarConfig, sh0_to_rgb
    from live3dgsavatar.core.io import load_ply
    from live3dgsavatar.core.render.camera_utils import project_points
    from live3dgsavatar.training.color_fit import fit_color_correction
    import yaml

    setup = _setup(args)
    cfg, data_dir, ply = setup.cfg, setup.data_dir, setup.ply
    print(f"[dry-run] model={ply.relative_to(REPO_ROOT)}  (subject={setup.model_subject})")
    print(f"[dry-run] data ={data_dir}  (subject={setup.data_subject})")

    n_frames = int(_cfg_get(args, cfg, "frames", "frames", 8))
    batch = _load_observations(cfg, data_dir, min(n_frames, 3), "cpu")
    mcfg = _load_model_config(setup)
    avatar = load_ply(ply, config=AvatarConfig(
        tex_size=mcfg["model"]["tex_size"],
        num_basis_in=mcfg["model"]["num_basis_in"],
        num_basis_blend=mcfg["model"]["num_basis_blend"]), device="cpu")

    from live3dgsavatar.core.types import Mesh
    obs = batch.frames[0]
    with torch.no_grad():
        gs = avatar.deform(obs.mesh, obs.blend_weight)
    rgb = sh0_to_rgb(gs.color[0].squeeze(1)).clamp(0, 1)
    ndc, _ = project_points(gs.xyz, batch.camera.full_proj, batch.camera.width, batch.camera.height)
    px = ((ndc[0, :, 0] * 0.5 + 0.5) * batch.camera.width).round().long().clamp(0, batch.camera.width - 1)
    py = ((ndc[0, :, 1] * 0.5 + 0.5) * batch.camera.height).round().long().clamp(0, batch.camera.height - 1)
    target = obs.target[0].permute(1, 2, 0)[py, px]              # [N,3]

    corr, metrics = fit_color_correction(rgb, target)
    print("\n[闭式颜色校正]（逐高斯点对点，非真实渲染）")
    print(f"  像素数={metrics['n_pixels']}  参数={metrics['n_params']}")
    print(f"  PSNR {metrics['psnr_before']:.2f} → {metrics['psnr_after']:.2f} dB "
          f"({metrics['psnr_delta']:+.2f})")
    print(f"  矩阵对角={[round(float(x), 3) for x in corr.matrix.diag()]}")
    print("\n  注意：绝对值无意义（不是真实渲染），只看相对提升与矩阵形状。")
    return 0


def main() -> int:
    args = parse_args()
    if args.dry_run:
        return _dry_run(args)

    # ---- 完整流程（需要 GPU）----
    import torch
    from live3dgsavatar.core.avatar import AvatarConfig
    from live3dgsavatar.core.io import load_ply
    from live3dgsavatar.core.render import SimpleRasterizer
    from live3dgsavatar.training import (OnlineUpdater, UpdateConfig, fit_color_correction)
    from live3dgsavatar.training.observations import BatchObservation
    import yaml

    if not torch.cuda.is_available():
        print("[error] 完整热更新需要 GPU（光栅化无法降级到 CPU）。"
              "可先用 --dry-run 验证闭式颜色校正。", file=sys.stderr)
        return 1
    # 崩溃排查辅助：CUDA 内核错误是异步上报的，建议带 CUDA_LAUNCH_BLOCKING=1 运行
    if not os.environ.get("CUDA_LAUNCH_BLOCKING"):
        print("[提示] 若出现 CUDA 非法访问，请用 "
              "`CUDA_LAUNCH_BLOCKING=1 python scripts/online_update.py ...` "
              "获取准确的出错位置")

    setup = _setup(args)
    cfg, data_dir, ply = setup.cfg, setup.data_dir, setup.ply

    # ---- 门控：总开关 ----
    enabled = bool(cfg.get("train.online_update.enabled", True))
    if not enabled and not args.force:
        print("[skip] 在线更新已关闭（train.online_update.enabled = false）。\n"
              "       命令行加 --force 可强制运行一次。")
        return 0

    update_color = bool(cfg.get("train.online_update.update_color", True)) \
        and not args.no_color
    update_opacity = (not args.no_opacity) and bool(
        cfg.get("train.online_update.update_opacity", True))
    if not update_color and not update_opacity:
        print("[skip] 颜色与不透明度都被关闭 —— 在线更新是空操作。\n"
              "       检查 train.online_update.update_color / update_opacity。")
        return 0

    print(f"model = {ply.relative_to(REPO_ROOT)}  (subject={setup.model_subject})")
    print(f"data  = {data_dir}  (subject={setup.data_subject})")
    if setup.data_subject != setup.model_subject:
        print(f"  注意：用 **{setup.data_subject}** 的画面驱动 **{setup.model_subject}** "
              "的模型（跨身份）。\n"
              "        在线更新只改颜色与不透明度，改不了身份/形状 —— "
              "PSNR 上限因此受限，看**前后差值**而不是绝对值。")

    device = str(cfg.get("runtime.device", "cuda"))
    n_obs = int(_cfg_get(args, cfg, "frames", "frames", 8))
    n_cmp = int(args.compare_frames)

    # 场景只载 **mesh**（61 KB/帧）；目标图按需读，见 `_make_observation`。
    # 对比帧可能超出观测帧（那正是"泛化"要看的部分），所以按需多载几帧 mesh。
    if n_cmp < 0:
        load_n = -1
    elif n_cmp > n_obs:
        load_n = n_cmp
    else:
        load_n = n_obs

    scene, camera = _scene_and_camera(cfg, data_dir, load_n, device)
    files = _image_files(data_dir)
    print(f"场景: {scene.num_frames} 帧 mesh 已就绪  "
          f"{camera.width}x{camera.height}")

    n_obs = min(n_obs, scene.num_frames) if n_obs > 0 else scene.num_frames
    obs_idx = sample_indices(scene.num_frames, n_obs, args.sample, args.seed)
    n_obs = len(obs_idx)
    img_gb = n_obs * 3 * camera.height * camera.width * 4 / 1024 ** 3
    if img_gb > 1.5:
        print(f"  ⚠ 拟合集目标图约占 {img_gb:.1f} GB 显存；"
              "若 OOM，减小 --frames 或分批")
    paint_color = parse_color(args.paint_color)
    paint_kw = dict(paint=args.paint, paint_color=paint_color,
                    paint_param=args.paint_param)
    if args.paint != "none":
        print(f"涂色: {args.paint}  颜色={args.paint_color}  "
              f"格子={args.paint_param}px  ← **能力测试**，不是泛化实验")
    fit_frames = [_make_observation(scene, files, i, device, **paint_kw)
                  for i in obs_idx]
    fit_batch = BatchObservation(frames=fit_frames, camera=camera)
    span = f"{obs_idx[0]}–{obs_idx[-1]}" if obs_idx else "-"
    print(f"拟合: {len(fit_frames)} 帧（采样={args.sample}，覆盖 {span}"
          f"，共 {scene.num_frames} 帧）")
    use_l1 = (False if args.l2
              else bool(cfg.get("train.online_update.color_l1", False)))
    print(f"门控  : 颜色={'on' if update_color else 'off'}  "
          f"不透明度={'on' if update_opacity else 'off'}  "
          f"损失={'L1' if use_l1 else 'L2'}")

    # ---- 实验目录：输出按"实验名"分层，而不是全堆在 online_update/ 下 ----
    exp = experiment_name(args, model_subject=setup.model_subject,
                          data_subject=setup.data_subject, n_obs=n_obs,
                          update_opacity=update_opacity, use_l1=use_l1,
                          n_cmp=n_cmp)
    out_dir = (Path(args.compare_dir) if args.compare_dir
               else Path(str(cfg.get("paths.output_dir"))) / "online_update" / exp)
    print(f"实验: {exp}")
    print(f"输出: {out_dir}")

    mcfg = _load_model_config(setup)
    avatar = load_ply(ply, config=AvatarConfig(
        tex_size=mcfg["model"]["tex_size"],
        num_basis_in=mcfg["model"]["num_basis_in"],
        num_basis_blend=mcfg["model"]["num_basis_blend"]), device=device)
    rasterizer = SimpleRasterizer(sh_degree=int(cfg.get("render.sh_degree", 0)))
    bg = torch.tensor([float(x) for x in cfg.get("render.background", [0, 0, 0])],
                      device=device)

    updater = OnlineUpdater(avatar, rasterizer, config=UpdateConfig(
        steps=int(_cfg_get(args, cfg, "steps", "steps", 300)),
        time_budget_s=float(_cfg_get(args, cfg, "time-budget", "time_budget_s", 10.0)),
        lr_color=float(_cfg_get(args, cfg, "lr-color", "lr_color", 2e-3)),
        lr_opacity=float(_cfg_get(args, cfg, "lr-opacity", "lr_opacity", 2e-2)),
        update_opacity=update_opacity,
        update_color=update_color,
        mask_weight=float(_cfg_get(args, cfg, "mask-weight", "mask_weight", 0.0)),
        color_l1=use_l1,
        frame_chunk=int(_cfg_get(args, cfg, "frames-chunk", "frame_chunk", 8))))

    report, state = updater.fit(fit_batch, background=bg)
    print(f"\n{report.summary()}")
    if report.history:
        print("  过程:", ", ".join(f"@{h['step']}:{h['psnr']:.2f}" for h in report.history))

    _delta_report(updater, state)

    if n_cmp != 0:
        _write_comparisons(args, setup, updater, scene, camera, files, bg,
                           state, n_cmp, len(fit_frames), out_dir, paint_kw)

    # 增量只有 ~1 MB（颜色 0.69 + 不透明 0.23），存下来便于复现与复用
    import json

    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.dump) if args.dump else (out_dir / "state.npz")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    # `to_dict()` 里既有 tensor（增量）也有 int（`step`）——不能一律 `.numpy()`
    arrays = {}
    for key, value in state.to_dict().items():
        arrays[key] = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
    np.savez(state_path, **arrays)
    print(f"  增量已存: {state_path}")

    metrics = {
        "experiment": exp,
        "model_subject": setup.model_subject,
        "data_subject": setup.data_subject,
        "frames_obs": len(fit_frames),
        "compare_frames": n_cmp,
        "update_color": update_color,
        "update_opacity": update_opacity,
        "loss": "L1" if use_l1 else "L2",
        "steps": report.steps,
        "elapsed_s": round(report.elapsed_s, 3),
        "psnr_before": round(report.before["psnr"], 4),
        "psnr_after": round(report.after["psnr"], 4),
        "psnr_delta": round(report.after["psnr"] - report.before["psnr"], 4),
        "history": report.history,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  指标已存: {out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
