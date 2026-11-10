#!/usr/bin/env python
"""把一个"RDF 跟踪结果目录"整理成标准数据集。

## 统一的数据集结构

```
datasets/<对象>/
├── images/             图像
└── render_inputs.npz   mesh(N,5083,3) / blend_weight(N,129) / K / w2v / size / names
```



## 输入

`--from` 给一个目录，含 `images/` 与 `render_inputs.npz`。该 npz 可以是两种 schema：

- `rdf`：`x`(N,118) + `verts`(N,5023,3) —— `rdf_dump.py` 的产物（需要转换）
- `baked`：`mesh` + `blend_weight` —— 已经是训练格式（只做拷贝与校验）

## 用法

```bash
# 新数据集：datasets/<name>/
python scripts/build_dataset.py --from <目录> --name <name>

# 就地烘焙（跟踪结果已经落在 datasets/<对象>/）
python scripts/build_dataset.py --from datasets/<对象> --name <对象>
```

通常不直接调用本脚本，而是用根目录的 `build_dataset.sh`：它会先把 RDF 跑好。

**来源就是目标目录时**（跟踪结果直接落在 `datasets/<对象>/`）：npz 已是训练格式就
直接跳过；还是 RDF 原始格式就**原地烘焙**，原文件备份为 `render_inputs.rdf.npz`。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import live3dgsavatar  # noqa: F401

#: 标准数据集必须有的字段
MANIFEST_KEYS = ("mesh", "blend_weight", "K", "w2v", "width", "height", "names")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="把 RDF 跟踪结果整理成标准数据集")
    p.add_argument("--from", dest="source", type=Path, required=True,
                   help="含 images/ 与 render_inputs.npz 的目录")
    p.add_argument("--name", default=None,
                   help="新数据集名（写到 datasets/<name>/）")
    p.add_argument("--out", type=Path, default=None,
                   help="直接指定输出目录（优先于 --name）")
    p.add_argument("--images", type=Path, default=None,
                   help="图像目录（默认 <来源>/images；可以直接给图像目录本身）")
    p.add_argument("--keep-images", action="store_true",
                   help="图像用复制而不是移动（默认移动，源目录里的图像会被搬走）")
    p.add_argument("--limit", type=int, default=0, help="最多用多少帧；0 全部")
    p.add_argument("--force", action="store_true", help="目标已存在时覆盖")
    return p.parse_args(argv)


def camera_from_dump(dump_path: Path) -> tuple:
    """没有基础数据集时，从 RDF dump 里的虚拟相机推 K / w2v。

    RDF 的相机是 fov_y、位于 `(0,0,d)`、`rot=diag(1,-1,-1)` —— 与数据集的
    虚拟相机等价，所以新数据集可以直接由它定出内参与视图变换。
    """
    import numpy as np

    with np.load(dump_path, allow_pickle=True) as d:
        width = int(np.asarray(d["image_width"]).reshape(-1)[0])
        height = int(np.asarray(d["image_height"]).reshape(-1)[0])
        fov_y = float(np.asarray(d["camera_fov_y"]).reshape(-1)[0])
        rot = np.asarray(d["camera_rot"], dtype=np.float32).reshape(3, 3)
        pos = np.asarray(d["camera_pos"], dtype=np.float32).reshape(3)
    f = 0.5 * height / np.tan(np.radians(fov_y) * 0.5)
    K = np.array([[f, 0.0, (width - 1) / 2.0],
                  [0.0, f, (height - 1) / 2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    # w2v：把世界坐标转到相机坐标（RDF 用 rot 与相机位置）
    w2v = np.eye(4, dtype=np.float32)
    w2v[:3, :3] = rot
    w2v[:3, 3] = -rot @ pos
    return K, w2v, width, height


def place_image(src: Path, dst: Path, *, keep_images: bool = False) -> str:
    """把一张图放到数据集里，返回动作：`skip` / `copy` / `move`。

    - 源与目标是同一个文件 → `skip`（重复运行不会报错）；
    - `keep_images` → 复制；否则**移动**（源目录里的图像被搬走，省一份空间）。
    """
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve():
        return "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if keep_images:
        shutil.copyfile(src, dst)
        return "copy"
    try:
        shutil.move(str(src), str(dst))
    except OSError:            # 跨设备等：退回复制 + 删除
        shutil.copyfile(src, dst)
        src.unlink()
    return "move"


def check_destination(out_dir: Path, source: Path, *,
                      force: bool = False) -> str:
    """检查目标目录，返回动作：`proceed` / `bake_in_place` / `already_ready`。

    - 目标与来源是同一目录、且 npz 已是 baked：`already_ready`（不必重做）；
    - 目标与来源是同一目录、npz 是 rdf：`bake_in_place`（允许原地烘焙）；
    - 目标已有 npz 但来源是别处：抛 `FileExistsError`（覆盖需 `--force`）。

    Raises:
        FileExistsError: 会覆盖别的数据集
    """
    from live3dgsavatar.data.tracked import frames_schema

    out_dir, source = Path(out_dir), Path(source)
    npz = out_dir / "render_inputs.npz"
    if force or not npz.is_file():
        return "proceed"
    if out_dir.resolve() == source.resolve():
        return ("already_ready" if frames_schema(npz) == "baked"
                else "bake_in_place")
    raise FileExistsError(f"{out_dir} 已有数据集；确认覆盖请加 --force")


def build(*, source: Path, out_dir: Path, flame_dir: Path,
          base_dataset: Path | None = None, limit: int = 0,
          force: bool = False, images_dir: Path | None = None,
          keep_images: bool = False) -> Path:
    """把 `source` 整理进 `out_dir`（images/ + render_inputs.npz）。"""
    import numpy as np
    from PIL import Image

    from live3dgsavatar.data.tracked import (find_frames_file, frames_schema,
                                            load_frames_dir)

    source = Path(source)
    action = check_destination(out_dir, source, force=force)
    if action == "already_ready":
        print(f"{out_dir} 已是标准数据集，跳过")
        return out_dir
    if action == "bake_in_place":
        # 原地烘焙：先把原始 RDF 结果备份一份，再覆盖成训练格式
        backup = out_dir / "render_inputs.rdf.npz"
        if not backup.is_file():
            shutil.copyfile(out_dir / "render_inputs.npz", backup)
        print(f"原地烘焙：原始结果已备份为 {backup.name}")

    dump_path = find_frames_file(source)
    schema = frames_schema(dump_path)
    print(f"来源：{source}（schema={schema}）")

    frames = load_frames_dir(source, flame_dir, device="cpu")
    if limit > 0:
        frames = frames[:limit]
    if not frames:
        raise ValueError("没有可用的帧")
    num_verts = int(frames[0]["mesh"].numel() // 3)

    # 相机与尺寸：优先沿用基础数据集（保证与已有模型同一套相机）
    if base_dataset is not None:
        with np.load(base_dataset / "render_inputs.npz", allow_pickle=True) as d:
            K = np.asarray(d["K"], np.float32)
            w2v = np.asarray(d["w2v"], np.float32)
            width, height = int(d["width"]), int(d["height"])
        print(f"相机：沿用数据集 {base_dataset.name}")
    elif schema == "baked":
        # 来源已经是标准格式：相机直接沿用，不做推断
        with np.load(dump_path, allow_pickle=True) as d:
            K = np.asarray(d["K"], np.float32)
            w2v = np.asarray(d["w2v"], np.float32)
            width, height = int(d["width"]), int(d["height"])
        print(f"相机：沿用来源（{width}x{height}）")
    else:
        K, w2v, width, height = camera_from_dump(dump_path)
        print(f"相机：由 RDF 虚拟相机推出（{width}x{height}）")

    # 图像一律重命名为 `00000.png …`：帧序与 npz 的行序严格一致，
    # 不依赖来源目录的命名（可能是 00000.png，也可能是 1.png）。
    images_out = out_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    meshes, weights = [], []
    src_images = Path(images_dir) if images_dir is not None else (source / "images")
    for i, frame in enumerate(frames):
        name = f"{i:05d}"
        src = src_images / f"{frame.get('name', name)}.png"
        if not src.is_file():
            src = src_images / f"{name}.png"
        if not src.is_file():
            # 图像与跟踪结果必须一一对应，缺一张就停
            raise FileNotFoundError(
                f"缺少第 {i} 帧的图像：{src}（source/images 与帧数据必须对齐）")
        dst = images_out / f"{name}.png"
        if not dst.exists() or force:
            place_image(src, dst, keep_images=keep_images)
        names.append(name)
        meshes.append(frame["mesh"].numpy())
        weights.append(frame["blend_weight"].numpy())

    np.savez_compressed(
        out_dir / "render_inputs.npz",
        mesh=np.stack(meshes).astype(np.float32),
        blend_weight=np.stack(weights).astype(np.float32),
        K=np.asarray(K, np.float32), w2v=np.asarray(w2v, np.float32),
        width=np.int64(width), height=np.int64(height),
        names=np.array(names),
    )

    # 命名自检：必须是 `00000.png …` 且与帧数一致，否则下游会按字典序错位
    expected = {f"{i:05d}.png" for i in range(len(names))}
    actual = {p.name for p in images_out.glob("*.png")}
    if actual != expected:
        extra = sorted(actual - expected)[:3]
        missing = sorted(expected - actual)[:3]
        raise ValueError(
            f"{images_out} 里的图像命名不符合 00000.png 规范或数量不符"
            f"（多出 {extra}，缺少 {missing}）")

    # 自检：形状、图像尺寸、字段齐全
    with np.load(out_dir / "render_inputs.npz", allow_pickle=True) as d:
        missing = [k for k in MANIFEST_KEYS if k not in d.files]
        if missing:
            raise ValueError(f"写出的 npz 缺少字段：{missing}（内部错误）")
        n = int(d["mesh"].shape[0])
        if int(d["mesh"].shape[1]) != num_verts:
            raise ValueError("补牙后的顶点数与第一帧不一致")
    for name in names[:3]:
        arr = np.asarray(Image.open(images_out / f"{name}.png"))
        if (arr.shape[1], arr.shape[0]) != (width, height):
            raise ValueError(
                f"{name}.png 尺寸 {arr.shape[1]}x{arr.shape[0]} 与声明的 "
                f"{width}x{height} 不一致")
    print(f"写出：{out_dir}（{n} 帧，{width}x{height}）")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from live3dgsavatar.config import load_config, repo_root

    cfg = load_config()
    flame_dir = Path(str(cfg.get("paths.flame_dir")))
    datasets_root = repo_root() / str(cfg.get("paths.data_root", "datasets"))

    base_dataset = None
    if args.out is not None:
        out_dir = args.out.expanduser()
    elif args.name is not None:
        out_dir = datasets_root / args.name
    else:
        print("[error] 需要 --name（新数据集）或 --out（指定目录）",
              file=sys.stderr)
        return 1

    try:
        build(source=args.source.expanduser().resolve(), out_dir=out_dir,
              flame_dir=flame_dir, base_dataset=base_dataset, limit=args.limit,
              force=args.force,
              images_dir=(args.images.expanduser().resolve()
                          if args.images is not None else None),
              keep_images=bool(args.keep_images))
    except (FileExistsError, FileNotFoundError, ValueError) as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1

    print(f"提示：bash train.sh --subject {out_dir.name} --work-name <版本>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
