#!/usr/bin/env python
"""一次性生成修复图：按固定 yaw 渲染当前模型 -> SyncFix -> unsharp -> 输出目录。

按固定间隔（如 5°）逐个视角渲染、修复，供加密管线做**初始化**监督。
视角是"头转"（`tracking.pose_drive`），不是绕头心转相机。

产物（syncfix_label 需要的 defects 目录结构）::

    <out>/views/yaw+45_pitch+0_roll+0_frame02104.png   # 待修复的渲染图
    <out>/refs/*.png                                    # 身份参考图（真实帧）
    <out>/selected.json                                 # 本次视角清单
    <out>/syncfix/*.png                                 # SyncFix 输出
    --out-dir/*.png                                     # unsharp 后的最终标签
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import live3dgsavatar  # noqa: F401,E402


def parse_args(argv=None):
    """命令行只留"跑哪个对象/哪一版/输出到哪"；调参项全部走 config。"""
    p = argparse.ArgumentParser(description="按固定头姿渲染 + SyncFix 生成修复图")
    p.add_argument("--subject", default=None)
    p.add_argument("--work-name", default=None,
                   help="要渲染的模型版本名（models/<subject>/<work-name>）")
    p.add_argument("--model", type=Path, default=None,
                   help="直接给 model.ply；给定时忽略 --work-name")
    p.add_argument("--dataset", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="只打印角度与标签，不渲染/不修复")
    p.add_argument("--auto", action="store_true",
                   help="按训练段分位数自动排角度（推荐）；给了就忽略 --yaws/--pitches")
    p.add_argument("--yaws", default="",
                   help="逗号分隔的相对 yaw（度），如 45,-45；只做 pitch 时给 0")
    p.add_argument("--pitches", default="0",
                   help="逗号分隔的相对 pitch（度），如 -30,...,30；与 yaws 做叉积")
    p.add_argument("--base-frame", default="auto",
                   help="基帧：auto=训练段里 |yaw| 最小的帧；或给帧号")
    p.add_argument("--refs", type=Path, default=None,
                   help="参考图目录（真实帧）；缺省用 datasets/<obj>/repair/refs")
    p.add_argument("--out", type=Path, required=True,
                   help="中间目录（views/ refs/ selected.json syncfix/）")
    p.add_argument("--repair-root", type=Path, default=None,
                   help="同时恢复数据集的 repair/ 结构：views/ refs/ masks/ repairs/ + manifest")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="最终修复图目录（unsharp 后）；缺省 <out>/final")
    args = p.parse_args(argv)

    from live3dgsavatar.config import load_config

    from live3dgsavatar.config import CONFIG_DIR

    cfg = load_config()
    # 修复图参数只在 avatar 配方里（train_avatar.yaml 的 repair 块）
    recipe_path = CONFIG_DIR / "train_avatar.yaml"
    rcfg: dict = {}
    if recipe_path.is_file():
        import yaml
        recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8")) or {}
        rcfg = recipe.get("repair", {}) or {}
    if not rcfg:
        rcfg = cfg.get("train.pipeline.repair", {}) or {}
    sxf = cfg.get("syncfix", {}) or {}
    # 调参项：config 为唯一来源（脚本不再挂这些命令）
    args.auto_quantile = float(rcfg.get("auto_quantile", 90.0))
    args.auto_extrapolate = float(rcfg.get("auto_extrapolate", 60.0))
    args.auto_yaw_only = bool(rcfg.get("yaw_only", True))
    args.angle_step = float(rcfg.get("step", 5.0))
    args.view_bg = float(rcfg.get("view_bg", 0.8))
    args.refs_auto = int(rcfg.get("refs_auto", 2))
    args.max_hole_px = int(rcfg.get("max_hole_px", 3000))
    args.max_invented_px = int(rcfg.get("max_invented_px", 1000))
    args.invented_dilate = int(rcfg.get("invented_dilate", 8))
    args.max_head_black = float(rcfg.get("max_head_black", 0.0))  # 旧比例判据，默认关
    args.min_coverage = float(rcfg.get("min_coverage", 0.65))
    args.mask_mode = str(rcfg.get("mask", "key"))
    args.key_threshold = int(rcfg.get("key_threshold", 18))
    args.edge_shrink = int(rcfg.get("edge_shrink", 1))
    args.no_inpaint = not bool(rcfg.get("inpaint", True))
    args.syncfix_batch = int(rcfg.get("syncfix_batch", 4))
    args.ref_size = max(int(rcfg.get("ref_size", 3)),
                        max(3, int(rcfg.get("refs_auto", 3))))
    args.sharpen = float(rcfg.get("sharpen", 0.35))
    args.syncfix_repo = Path(str(sxf.get("repo", "/home/test/Projects/SyncFix")))
    args.syncfix_python = str(sxf.get("python", "syncfix"))
    args.syncfix_model = Path(str(sxf.get("model",
                                          "/home/test/Projects/SyncFix/models/paper")))
    return args


def _auto_angles(data_dir, train_end: int, step: float = 5.0,
                 max_yaw: float = 90.0, max_pitch: float = 45.0,
                 quantile: float = 90.0, extrapolate: float = 45.0):
    """按训练段四个方向的分位数（默认 p90）往外排修复角度。

    修复图最初是补**数据里没有**的角度，所以默认从 p90 往外走。
    但实际观察发现：模型在 p50 附近就开始出瑕疵，修复图在 6:2 里起的是
    "锚定+修轮廓"的作用——想让修复图覆盖到 p50，就把 quantile 调低。
    """
    import numpy as _np

    from live3dgsavatar.data.pose import head_pitch, head_yaw

    with np.load(Path(data_dir) / "render_inputs.npz", allow_pickle=True) as d:
        w = np.asarray(d["blend_weight"][:max(1, int(train_end))], dtype=np.float32)
    yaw = head_yaw(w)
    pitch = head_pitch(w)

    def _side(values, positive: bool, top: float) -> list[float]:
        v = values if positive else -values
        v = v[v > 0]
        pct = float(_np.percentile(v, quantile)) if v.size else 0.0
        start = float(_np.ceil(pct / step) * step)
        # 只外推到 p90 + extrapolate：再往外 base 纯属瞎猜，修复图只会教坏模型
        stop = min(float(top), pct + float(extrapolate))
        return [float(s) for s in _np.arange(start, stop + 1e-6, step)]

    yaw_pos = _side(yaw, True, max_yaw)
    yaw_neg = [-v for v in _side(yaw, False, max_yaw)]
    pitch_pos = _side(pitch, True, max_pitch)
    pitch_neg = [-v for v in _side(pitch, False, max_pitch)]
    return yaw_pos + yaw_neg, pitch_pos + pitch_neg


def _head_hole_px(out: Path, label: str, matte_path: Path,
                  y_ratio: float = 0.6) -> tuple[int, float]:
    """FLAME 可见头部区里"最大的一块没被 mask 覆盖"的像素数 + 占比。"""
    from PIL import Image
    from scipy import ndimage

    surf_path = out / "flame_surface" / f"{label}.png"
    if not surf_path.is_file():
        return 0, 0.0
    surf = np.asarray(Image.open(surf_path)) > 127
    head = surf.copy()
    head[int(y_ratio * head.shape[0]):, :] = False
    if not bool(head.any()):
        return 0, 0.0
    im = np.asarray(Image.open(matte_path).convert("RGBA"))
    miss = head & ~(im[..., 3] > 127)
    if not bool(miss.any()):
        return 0, 0.0
    lab, n = ndimage.label(miss)
    biggest = int(ndimage.sum(miss, lab, range(1, n + 1)).max()) if n else 0
    return biggest, float(miss.sum()) / float(head.sum())


def _key_rgba(src_dir: Path, dst_dir: Path, *, threshold: int = 18,
              shrink: int = 1) -> int:
    """按背景色键控出 alpha，不跑 BiRefNet。

    背景是我们自己渲染的灰底，SyncFix 基本会保留它，所以用"与背景色的
    最大通道差"就能把人物抠出来；极端角度下比人像 matting 稳得多。
    """
    from PIL import Image
    from scipy import ndimage

    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src_dir.glob("*.png")):
        rgb = np.asarray(Image.open(p).convert("RGB"))
        # 背景色取四条边的中位数，抵消 SyncFix 的整体色调偏移
        border = np.concatenate([
            rgb[0, :, :], rgb[-1, :, :], rgb[:, 0, :], rgb[:, -1, :]])
        bg = np.median(border, axis=0)
        fg = np.abs(rgb.astype(np.int16) - bg).max(axis=-1) > int(threshold)
        fg = ndimage.binary_fill_holes(fg)
        lab, k = ndimage.label(fg)
        if k > 1:
            sizes = ndimage.sum(fg, lab, range(1, k + 1))
            fg = lab == (int(np.argmax(sizes)) + 1)
        fg = ndimage.binary_closing(fg, iterations=1)
        fg = ndimage.binary_fill_holes(fg)
        # 去掉边界附近"接近背景色"的像素（SyncFix 把头发和灰底混出来的浅色描边）
        band = fg & ~ndimage.binary_erosion(fg, iterations=10)
        near_bg = np.abs(rgb.astype(np.int16) - bg).max(axis=-1) < 30
        fg = fg & ~(band & near_bg)
        fg = ndimage.binary_fill_holes(fg)
        if shrink > 0:
            fg = ndimage.binary_erosion(fg, iterations=int(shrink))
            fg = ndimage.binary_fill_holes(fg)
        # 剩下的边界像素用最近的内部颜色替换，去掉残留的背景染色
        rgb_out = rgb.copy()
        if int(shrink) >= 0:
            interior = ndimage.binary_erosion(fg, iterations=2)
            if bool(interior.any()):
                idx = ndimage.distance_transform_edt(
                    ~interior, return_distances=False, return_indices=True)
                rim = fg & ~interior
                rgb_out[rim] = rgb[idx[0], idx[1]][rim]
        alpha = (fg.astype(np.uint8) * 255)
        out = np.dstack([rgb_out, alpha])
        out[alpha == 0, :3] = 0
        Image.fromarray(out).save(dst_dir / p.name)
        n += 1
    return n


def _invented_px(out: Path, label: str, matte_path: Path,
                 dilate: int = 8) -> int:
    """修复图在"渲染轮廓之外"凭空造出的像素数（SyncFix 幻觉检测）。

    极端角度下 SyncFix 会补出不存在的结构（biden +70 起的"第二只耳朵"）。
    渲染里根本没有的东西，修复图却实心画了出来——这就是幻觉。
    """
    from PIL import Image
    from scipy import ndimage

    ap = out / "render_alpha" / f"{label}.png"
    if not ap.is_file():
        return 0
    ref = np.asarray(Image.open(ap)) > 127
    ref = ndimage.binary_dilation(ref, iterations=int(dilate))
    m = np.asarray(Image.open(matte_path).convert("RGBA"))[..., 3] > 127
    return int((m & ~ref).sum())


def _head_black_fraction(out: Path, label: str, matte_path: Path,
                         y_ratio: float = 0.6) -> float:
    """该视角在 FLAME 可见头部区里"有轮廓但没内容"的比例。

    `matte` 是 BiRefNet 在修复图上的结果；纯黑像素会被它判成背景，所以
    修复图没补出内容的地方就表现为"FLAME 该有、mask 没有、RGB 全黑"。
    """
    from PIL import Image

    surf_path = out / "flame_surface" / f"{label}.png"
    if not surf_path.is_file():
        return 0.0
    surf = np.asarray(Image.open(surf_path)) > 127
    head = surf.copy()
    head[int(y_ratio * head.shape[0]):, :] = False
    if not bool(head.any()):
        return 0.0
    im = np.asarray(Image.open(matte_path).convert("RGBA"))
    black = (im[..., :3].max(axis=-1) <= 12) & ~(im[..., 3] > 127)
    return float((black & head).sum()) / float(head.sum())


def _fill_dark_holes(rgb: np.ndarray, alpha: np.ndarray,
                     thresh: int = 12, radius: int = 4) -> np.ndarray:
    """把 mask 内的纯黑块用周围颜色补上。

    SyncFix 在极端视角会把某些区域留成纯黑（大角度后脑/发顶常见），BiRefNet
    又把它判成前景，最终表现为"修复图上有个黑洞"。直接抠掉会在头上打洞，
    所以按周围颜色修补更稳。
    """
    import cv2

    m = np.asarray(alpha) > 127
    dark = m & (np.asarray(rgb).max(axis=-1) <= int(thresh))
    if not bool(dark.any()):
        return np.asarray(rgb, np.uint8).copy()
    return cv2.inpaint(np.ascontiguousarray(np.asarray(rgb, np.uint8)),
                       dark.astype(np.uint8), int(radius), cv2.INPAINT_TELEA)


def _clean_alpha(alpha: np.ndarray) -> np.ndarray:
    """清理抠图：只保留最大连通域、填内洞、闭运算补 1–2 px 缺口。

    BiRefNet 在头转视角上偶尔会漏判一小块（脖子侧、发梢），直接用会污染
    修复图的 mask，进而污染训练。
    """
    from scipy import ndimage

    if alpha.ndim != 2:
        return alpha
    m = alpha > 127
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == (int(np.argmax(sizes)) + 1)
    m = ndimage.binary_fill_holes(m)
    m = ndimage.binary_closing(m, iterations=1)
    m = ndimage.binary_fill_holes(m)
    out = np.zeros_like(alpha)
    out[m] = 255
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    import torch
    from PIL import Image

    from live3dgsavatar.config import load_config, resolve_model_ply
    from live3dgsavatar.core.avatar import AvatarConfig
    from live3dgsavatar.core.camera import orbit_w2c_matrix, rotation_y
    from live3dgsavatar.core.io import load_ply
    from live3dgsavatar.core.render import SimpleRasterizer, render_frame
    from live3dgsavatar.core.types import Camera, Mesh
    from live3dgsavatar.data.datasets import resolve_dataset_dir
    from live3dgsavatar.data.pose import head_yaw, rot6d_to_matrix
    from live3dgsavatar.data.scene import load_scene

    cfg = load_config()
    subject = args.subject or str(cfg.get("subject") or "")
    if not subject:
        raise SystemExit("[error] 需要 --subject 或配置里的 subject")
    cfg.set("subject", subject)
    work = args.work_name or str(cfg.get("work_name") or "")
    if work:
        cfg.set("work_name", work)
    data_dir = resolve_dataset_dir(cfg, args.dataset or subject)
    ply = args.model.expanduser() if args.model is not None else resolve_model_ply(cfg)
    if not Path(ply).is_file():
        raise SystemExit(f"[error] 找不到模型：{ply}")

    # 角度在读到数据后再确定（--auto 需要训练段的 p90）

    with np.load(data_dir / "render_inputs.npz", allow_pickle=True) as d:
        meshes = torch.from_numpy(d["mesh"].copy()).float()
        weights = torch.from_numpy(d["blend_weight"].copy()).float()
    total = int(meshes.shape[0])

    yaw_all = np.abs(head_yaw(weights.numpy()))
    held = max(1, int(round(total * float(cfg.get("train.pipeline.test_ratio", 0.3)))))
    train_end = total - held
    if str(args.base_frame).lower() == "auto":
        base_idx = int(np.argmin(yaw_all[:train_end]))
    else:
        base_idx = int(args.base_frame)
    print(f"  数据集 : {data_dir}（{total} 帧，训练段 {train_end}）")
    print(f"  模型   : {ply}")
    print(f"  基帧   : {base_idx:05d}（|yaw| {yaw_all[base_idx]:.1f}°）")
    if args.auto:
        yaws, pitches = _auto_angles(
            data_dir, train_end, step=float(args.angle_step),
            quantile=float(args.auto_quantile),
            extrapolate=float(args.auto_extrapolate))
        if args.auto_yaw_only:
            pitches = []
        print(f"  角度   : auto  yaw {yaws}  pitch {pitches}")
    else:
        yaws = [float(v) for v in str(args.yaws).split(",") if str(v).strip()]
        pitches = [float(v) for v in str(args.pitches).split(",")
                   if str(v).strip()]
    if not yaws:
        raise SystemExit("[error] --yaws 为空")


    device = "cuda"
    if not torch.cuda.is_available():
        raise SystemExit("[error] 需要 GPU（nvdiffrast）")
    scene = load_scene(cfg, frames=1, data_dir=data_dir)
    avatar = load_ply(Path(ply),
                      config=AvatarConfig.from_yaml(Path(ply).parent / "config.yaml"),
                      device=device)
    rasterizer = SimpleRasterizer()

    # 头转（pose 驱动）：语义与 GT 一致；没有 RDF 跟踪结果时退回相机绕转
    cam_fixed = Camera.from_intrinsics_extrinsics(
        K=scene.K, R=scene.R, T=scene.T,
        width=scene.width, height=scene.height).to(device)
    try:
        from live3dgsavatar.tracking.pose_drive import PoseDriver
        driver = PoseDriver(data_dir, device=device)
        print("  视角   : FLAME 头姿驱动（头转、胸不动）")
    except Exception as exc:
        driver = None
        print(f"  视角   : 相机绕转（pose 驱动不可用：{exc}）")

    w2c = torch.eye(4, dtype=torch.float32)
    w2c[:3, :3] = torch.as_tensor(scene.R).detach().cpu()
    w2c[:3, 3] = torch.as_tensor(scene.T).detach().cpu()
    base_verts = meshes[base_idx:base_idx + 1].to(device)
    weight0 = weights[base_idx:base_idx + 1].to(device)
    center = base_verts[0].mean(dim=0).detach().cpu()
    head_R = torch.from_numpy(
        rot6d_to_matrix(weights[base_idx:base_idx + 1, 100:106].numpy())[0]).float()

    out = args.out
    views_dir = out / "views"
    alpha_dir = out / "render_alpha"
    refs_dir = out / "refs"
    if not args.dry_run:
        views_dir.mkdir(parents=True, exist_ok=True)
        refs_dir.mkdir(parents=True, exist_ok=True)
        for stale in list(views_dir.glob("*.png")) + list(alpha_dir.glob("*.png")):
            stale.unlink()
        # 中间产物也必须清空：否则上一轮的角度会作为"孤児标签"混进最终目录
        for sub in ("syncfix", "matte", "final", "flame_surface"):
            d = out / sub
            if d.is_dir():
                for stale in d.glob("*.png"):
                    stale.unlink()

    bg = torch.full((1, 3, 1, 1), float(args.view_bg), device=device)
    import nvdiffrast.torch as _dr
    from live3dgsavatar.core.render.faceid import face_id_map
    glctx = _dr.RasterizeCudaContext()
    labels: list[str] = []
    # 只做"十字"：yaw 扇 + pitch 扇，而不是 yaw×pitch 全网格
    # （全网格会让 22×12 张里大部分是重复的斜向角，SyncFix 成本翻倍）。
    views = [(y, 0.0) for y in yaws]
    views += [(0.0, p) for p in pitches if abs(p) > 1e-9]
    for yaw, pitch in views:
        label = f"yaw{yaw:+.0f}_pitch{pitch:+.0f}_roll+0_frame{base_idx:05d}"
        labels.append(label)
        if args.dry_run:
            print(f"  [dry-run] {label}")
            continue
        if driver is not None:
            turned = driver.turn(base_idx, yaw, pitch)
            cam = cam_fixed
            verts = turned.mesh.unsqueeze(0)
            weight = turned.weight.unsqueeze(0)
        else:
            R_target = rotation_y(yaw)[:3, :3]
            view = orbit_w2c_matrix(w2c, center, R_target @ head_R.T)
            cam = Camera(K=scene.K, w2c=view.unsqueeze(0).to(device),
                         width=scene.width, height=scene.height).to(device)
            verts, weight = base_verts, weight0
        mesh = Mesh(verts=verts, faces=scene.faces.to(device),
                    uvs=scene.uvs.to(device), uv_faces=scene.uv_faces.to(device))
        with torch.no_grad():
            res = render_frame(avatar, rasterizer, mesh, weight, cam, bg)
        img = (res.color[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0
               ).astype(np.uint8)
        # base 在该角度有没有画出东西：覆盖率太低时修复图只会留洞，直接跳过
        with torch.no_grad():
            ids = face_id_map(verts, scene.faces.to(device), cam, glctx,
                              flip_y=False)
        ids = ids[0] if ids.ndim == 3 else ids
        surf = (ids >= 0).cpu().numpy()
        cov = float((res.alpha[0, 0].cpu().numpy()[surf] > 0.5).mean()) \
            if surf.any() else 0.0
        if cov < float(args.min_coverage):
            print(f"  {label}  跳过（覆盖率 {cov:.2f} < {args.min_coverage}）")
            labels.pop()
            continue
        (out / "flame_surface").mkdir(parents=True, exist_ok=True)
        Image.fromarray((surf.astype(np.uint8) * 255)).save(
            out / "flame_surface" / f"{label}.png")
        Image.fromarray(img).save(views_dir / f"{label}.png")
        # 渲染 alpha 作为抠图先验：修复图只能保留"渲染里真的有的东西"，
        # 避免 BiRefNet 把背景/漂浮高斯误判成前景（B 站里那块黑斑就是这么来的）。
        amask = (res.alpha[0, 0].clamp(0, 1).cpu().numpy() * 255.0
                 ).astype(np.uint8)
        alpha_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(amask).save(alpha_dir / f"{label}.png")
        print(f"  {label}  (yaw {yaw:+.1f}°, pitch {pitch:+.1f}°)")

    # 参考图：至少 3 张（中/左/右），refs_auto 更大时继续加
    if not args.dry_run:
        from live3dgsavatar.data.pose import head_yaw

        refs_src = args.refs or (data_dir / "repair" / "refs")
        copied = 0
        if Path(refs_src).is_dir():
            for p in sorted(Path(refs_src).glob("*.png")):
                shutil.copyfile(p, refs_dir / p.name)
                copied += 1
        target_refs = max(3, int(args.refs_auto))
        test_ratio = float(cfg.get("train.pipeline.test_ratio", 0.3))

        def _save_ref(frame_idx: int, name: str) -> None:
            arr = np.asarray(Image.open(data_dir / "images" / f"{frame_idx:05d}.png")
                             .convert("RGBA"))
            rgb = arr[..., :3].astype(np.float32)
            rgb[arr[..., 3] <= 127] = float(args.view_bg) * 255.0
            Image.fromarray(rgb.astype(np.uint8)).save(refs_dir / name)

        # 中：基帧（始终作为身份参考）
        base_ref = refs_dir / f"base_{base_idx:05d}.png"
        if not base_ref.is_file() and copied < target_refs:
            _save_ref(base_idx, base_ref.name)
            copied += 1

        # 左/右（及更多）：按目标 |yaw| 从真实帧里挑，左右交替
        picked: list[int] = []
        if copied < target_refs:
            yaw_all = np.abs(head_yaw(np.asarray(
                np.load(data_dir / "render_inputs.npz",
                        allow_pickle=True)["blend_weight"], dtype=np.float32)))
            train_end = int(len(yaw_all) * (1.0 - test_ratio))
            targets = [20.0, -20.0, 35.0, -35.0, 50.0, -50.0,
                       0.0, 65.0, -65.0, 80.0, -80.0]
            for target in targets:
                if copied >= target_refs:
                    break
                cand = np.abs(yaw_all[:train_end] - target)
                for idx in np.argsort(cand)[:8]:
                    idx = int(idx)
                    if idx == base_idx or idx in picked:
                        continue
                    picked.append(idx)
                    _save_ref(idx, f"auto_{idx:05d}.png")
                    copied += 1
                    break
            if picked:
                print(f"  参考图 : 追加 {len(picked)} 张真实帧 "
                      f"{['%05d' % i for i in picked]}")
        print(f"  参考图 : {copied} 张（目标 ≥{target_refs}）-> {refs_dir}")
        (out / "selected.json").write_text(json.dumps({
            "subject": subject, "model": str(ply), "base_frame": base_idx,
            "yaws": yaws, "views": labels, "refs": copied,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  输出   : {views_dir}")

        # ---- SyncFix 修复：按**角度从里往外**分批；某一侧一旦出现破洞就整侧停 ----
        import subprocess
        from live3dgsavatar.evaluation.views import parse_view_label

        group = max(1, int(args.syncfix_batch))
        rgb_dir = out / "syncfix"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        matte_dir = out / "matte"
        matte_dir.mkdir(parents=True, exist_ok=True)
        labels = sorted(p.stem for p in views_dir.glob("*.png"))
        yaw_of = {lab: float(parse_view_label(lab)["yaw"]) for lab in labels}
        sides = {
            "pos": sorted([v for v in labels if yaw_of[v] > 0],
                          key=lambda v: abs(yaw_of[v])),
            "neg": sorted([v for v in labels if yaw_of[v] < 0],
                          key=lambda v: abs(yaw_of[v])),
            "zero": [v for v in labels if abs(yaw_of[v]) < 1e-9],
        }
        kept: list[str] = []
        for side, labs in sides.items():
            if not labs:
                continue
            stopped = False
            for gi in range(0, len(labs), group):
                grp = labs[gi:gi + group]
                gdir = out / f"_g_{side}_{gi // group}"
                (gdir / "views").mkdir(parents=True, exist_ok=True)
                (gdir / "refs").mkdir(parents=True, exist_ok=True)
                for lab in grp:
                    shutil.copyfile(views_dir / f"{lab}.png",
                                    gdir / "views" / f"{lab}.png")
                for ref in refs_dir.glob("*.png"):
                    shutil.copyfile(ref, gdir / "refs" / ref.name)
                subprocess.run([
                    sys.executable, str(REPO_ROOT / "scripts" / "syncfix_label.py"),
                    "--defects", str(gdir), "--out", str(gdir / "fix"),
                    "--repo", str(args.syncfix_repo),
                    "--python", str(args.syncfix_python),
                    "--model", str(args.syncfix_model),
                    "--ref-size", str(args.ref_size),
                    "--n-per-pass", str(min(group, 4)),
                ], check=True)
                for f in sorted((gdir / "fix").glob("*.png")):
                    shutil.copyfile(f, rgb_dir / f.name)
                if args.mask_mode == "key":
                    _key_rgba(gdir / "fix", gdir / "matte",
                              threshold=args.key_threshold,
                              shrink=args.edge_shrink)
                else:
                    subprocess.run([
                        sys.executable, str(REPO_ROOT / "scripts" / "matte_birefnet.py"),
                        "--input", str(gdir / "fix"), "--out", str(gdir / "matte"),
                        "--force",
                    ], check=True)
                    if args.mask_mode == "auto":
                        key_dir = gdir / "matte_key"
                        _key_rgba(gdir / "fix", key_dir,
                                  threshold=args.key_threshold,
                                  shrink=args.edge_shrink)
                        for kp in key_dir.glob("*.png"):
                            bp = gdir / "matte" / kp.name
                            if not bp.is_file():
                                shutil.copyfile(kp, bp)
                                continue
                            b = np.asarray(Image.open(bp).convert("RGBA"))
                            k2 = np.asarray(Image.open(kp).convert("RGBA"))
                            b[..., 3] = np.maximum(b[..., 3], k2[..., 3])
                            Image.fromarray(b).save(bp)
                for lab in grp:
                    src = gdir / "matte" / f"{lab}.png"
                    if not src.is_file():
                        continue
                    biggest, frac = _head_hole_px(out, lab, src)
                    invented = _invented_px(out, lab, src,
                                            dilate=args.invented_dilate)
                    too_big = (args.max_hole_px > 0
                               and biggest > int(args.max_hole_px))
                    if too_big:
                        print(f"  {lab}  破洞 {biggest}px（占头部 {frac:.1%}）> "
                              f"{args.max_hole_px}px → {side} 侧从该角度起停止")
                    elif (args.max_invented_px > 0
                          and invented > int(args.max_invented_px)):
                        print(f"  {lab}  幻觉 {invented}px > "
                              f"{args.max_invented_px}px（渲染外凭空造内容）"
                              f" → {side} 侧从该角度起停止")
                        too_big = True
                    if too_big:
                        stopped = True
                        break
                    shutil.copyfile(src, matte_dir / f"{lab}.png")
                    kept.append(lab)
                if stopped:
                    break
        if not kept:
            raise SystemExit("[error] 所有角度都被判为破洞")
        final_src = matte_dir

        # ---- unsharp + 输出到 --out-dir（有 alpha 就保留）----
        final = args.out_dir or (out / "final")
        final.mkdir(parents=True, exist_ok=True)
        from PIL import ImageFilter
        n = 0
        for p in sorted(final_src.glob("*.png")):
            rgb = Image.open(p).convert("RGB")
            if args.sharpen > 0:
                arr = np.asarray(rgb, np.float32)
                blur = np.asarray(rgb.filter(ImageFilter.GaussianBlur(radius=1.0)),
                                  np.float32)
                rgb = Image.fromarray(np.clip(arr + args.sharpen * (arr - blur),
                                              0, 255).astype(np.uint8))
            surf_path = out / "flame_surface" / f"{p.stem}.png"
            if surf_path.is_file() and args.max_head_black > 0:
                surf_px = np.asarray(Image.open(surf_path)) > 127
                head = surf_px.copy()
                head[int(0.6 * head.shape[0]):, :] = False
                if head.any():
                    rgb_now = np.asarray(rgb, np.uint8)
                    black = (rgb_now.max(-1) <= 12) & head
                    frac = float(black.sum()) / float(head.sum())
                    if frac > float(args.max_head_black):
                        print(f"  {p.stem}  跳过（头部黑洞 {frac:.1%} "
                              f"> {args.max_head_black:.0%}）")
                        continue
            alpha_np = _clean_alpha(
                np.asarray(Image.open(p).getchannel("A"), np.uint8))
            if not args.no_inpaint:
                rgb = Image.fromarray(_fill_dark_holes(
                    np.asarray(rgb, np.uint8), alpha_np))
            alpha = Image.fromarray(alpha_np)
            out_im = rgb.convert("RGBA")
            out_im.putalpha(alpha)
            # alpha 外的 RGB 置 0，与数据集约定一致
            a = np.array(out_im, dtype=np.uint8, copy=True)
            a[a[..., 3] <= 127, :3] = 0
            out_im = Image.fromarray(a)
            out_im.save(final / p.name)
            n += 1
        print(f"  修复图 : {n} 张（mask={args.mask_mode}）-> {final}")

        if args.repair_root is not None:
            root = Path(args.repair_root)
            for sub in ("views", "refs", "masks", "repairs"):
                (root / sub).mkdir(parents=True, exist_ok=True)
            # 先清掉上一轮的标签，避免角度改了之后留下孤儿文件。
            # ⚠️ `repairs/` 可能就是 --out-dir，不能整个删掉；只删不在本次集合里的。
            # 只保留"渲染过且通过破洞判定"的标签；被破洞拦下的也要从数据集里清掉
            cur = {q.stem for q in views_dir.glob("*.png")} & set(kept)
            for sub in ("views", "masks", "repairs"):
                for stale in (root / sub).glob("*.png"):
                    if stale.stem not in cur:
                        stale.unlink()
            for p in views_dir.glob("*.png"):
                shutil.copyfile(p, root / "views" / p.name)
            for p in refs_dir.glob("*.png"):
                shutil.copyfile(p, root / "refs" / p.name)
            # masks/ 直接取最终 RGBA 的 alpha（已做渲染先验 + 去暗块），
            # 保证 masks/ 与 repairs/ 完全一致
            for p in final.glob("*.png"):
                if p.stem not in cur:
                    continue
                im = Image.open(p)
                if im.mode == "RGBA":
                    im.getchannel("A").save(root / "masks" / p.name)
            for p in final.glob("*.png"):
                if p.stem not in cur:
                    continue
                dst = root / "repairs" / p.name
                if p.resolve() != dst.resolve():   # --out-dir 可能就是 root/repairs
                    shutil.copyfile(p, dst)
            (root / "manifest.json").write_text(json.dumps({
                "subject": subject, "base_frame": base_idx, "yaws": yaws,
                "pitches": pitches,
                "source": "SyncFix (batched, refs per batch) + BiRefNet alpha",
                "files": {"views": "views/", "repairs": "repairs/",
                          "masks": "masks/", "refs": "refs/"},
            }, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  repair/ 结构：{root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
