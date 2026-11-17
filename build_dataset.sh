#!/usr/bin/env bash
# 构建数据集：一个入口把"图像 → RDF 跟踪 → 标准数据集"整条流程跑完。
#
# 标准结构：
#   datasets/<对象>/images/ + render_inputs.npz
#
# 用法：
#   ./build_dataset.sh <images 目录> --name <对象>              # 新建数据集
#   ./build_dataset.sh <images 目录> --name <对象> --limit 300 # 只取前 300 帧
#   ./build_dataset.sh --video videos/biden.mp4 --name biden    # 从视频直接构建
#   ./build_dataset.sh --video <视频> --name <对象> --fps 30 --limit 300
#       [--head-factor 2.8] [--top-margin 0.12]   # 调节头/肩构图
#   ... --keep-stage 保留 output/build/<name> 供检查
#
# 已经跟踪过的目录（含 images/ 与 render_inputs.npz）会跳过 RDF，直接整理：
#   ./build_dataset.sh <已跟踪的目录> --name <数据集名>
#
# 图像默认**移动**进数据集（省空间）；想保留原图加 --keep-images：
#   ./build_dataset.sh <图像目录> --name biden --keep-images
#
# 环境变量：
#   RDF_ENV     跑 RDF 的 conda 环境（默认 realdenseface）
#   MAIN_ENV    跑本项目脚本的 conda 环境（默认 live3dgs）
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

RDF_ENV="${RDF_ENV:-realdenseface}"
MAIN_ENV="${MAIN_ENV:-live3dgs}"

die() { echo "[error] $*" >&2; exit 1; }
info() { echo "[build_dataset] $*"; }

usage() {
  sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
}

SOURCE=""
VIDEO=""
NAME=""
LIMIT=0
STRIDE=1
FPS=30
FORCE=""
KEEP_IMAGES=""
KEEP_STAGE=0
HEAD_FACTOR=""
TOP_MARGIN=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video)   VIDEO="${2:-}"; shift 2 ;;
    --name)    NAME="${2:-}"; shift 2 ;;
    --limit)   LIMIT="${2:-0}"; shift 2 ;;
    --stride)  STRIDE="${2:-1}"; shift 2 ;;
    --fps)     FPS="${2:-30}"; shift 2 ;;
    --head-factor) HEAD_FACTOR="${2:-}"; shift 2 ;;
    --top-margin)  TOP_MARGIN="${2:-}"; shift 2 ;;
    --keep-stage) KEEP_STAGE=1; shift ;;
    --keep-images) KEEP_IMAGES="--keep-images"; shift ;;
    --force)   FORCE="--force"; shift ;;
    -h|--help) usage ;;
    -*)        die "未知参数：$1（--help 查看用法）" ;;
    *)         SOURCE="$1"; shift ;;
  esac
done

# ---------------------------------------------------------------- 视频入口 --
if [[ -n "$VIDEO" ]]; then
  [[ -n "$NAME" ]] || die "视频入口需要 --name（数据集名）"
  STAGE="$REPO_ROOT/output/build/$NAME"
  info "视频预处理：$VIDEO → $STAGE"
  PRE_ARGS=(--video "$VIDEO" --out "$STAGE" --fps "$FPS"
            --limit "$LIMIT" --stride "$STRIDE")
  [[ -n "$HEAD_FACTOR" ]] && PRE_ARGS+=(--head-factor "$HEAD_FACTOR")
  [[ -n "$TOP_MARGIN" ]] && PRE_ARGS+=(--top-margin "$TOP_MARGIN")
  [[ -n "$FORCE" ]] && PRE_ARGS+=(--force)
  conda run -n "$MAIN_ENV" --no-capture-output \
    python scripts/prepare_video.py "${PRE_ARGS[@]}"

  info "MODNet 抠图（人像发丝/首饰边缘）"
  conda run -n "$MAIN_ENV" --no-capture-output \
    python scripts/matte_modnet.py --input "$STAGE/input" \
      --out "$STAGE/images_modnet" --model "$REPO_ROOT/models/matting/modnet.onnx" \
      --force
  rm -rf "$STAGE/images"
  mv "$STAGE/images_modnet" "$STAGE/images"

  info "RDF 跟踪裁好的 RGB 帧（环境：$RDF_ENV）"
  conda run -n "$RDF_ENV" --no-capture-output \
    python scripts/rdf_dump.py --images "$STAGE/input" --out "$STAGE"

  info "整理为标准数据集（环境：$MAIN_ENV）"
  BUILD_ARGS=(--from "$STAGE" --images "$STAGE/images" --name "$NAME")
  [[ -n "$KEEP_IMAGES" ]] && BUILD_ARGS+=("$KEEP_IMAGES")
  [[ -n "$FORCE" ]] && BUILD_ARGS+=(--force)
  conda run -n "$MAIN_ENV" --no-capture-output \
    python scripts/build_dataset.py "${BUILD_ARGS[@]}"

  # 保留原始裁图，方便与 images/ 对照；不留在 datasets 里避免混淆训练输入
  mkdir -p "$REPO_ROOT/datasets/$NAME/input"
  cp -a "$STAGE/input/." "$REPO_ROOT/datasets/$NAME/input/"
  cp -a "$STAGE/crop_stats.json" "$REPO_ROOT/datasets/$NAME/crop_stats.json"
  if (( KEEP_STAGE == 0 )); then rm -rf "$STAGE"; fi
  info "完成：datasets/$NAME（images/ + input/ + render_inputs.npz）"
  exit 0
fi

# ------------------------------------------------------------- RDF 跟踪 --
if [[ -f "$SOURCE/render_inputs.npz" || -f "$SOURCE/rdf.npz" ]]; then
  info "已有跟踪结果，跳过 RDF"
else
  info "运行 RDF 跟踪（环境：$RDF_ENV）"
  command -v conda >/dev/null || die "找不到 conda"
  # 跟踪结果写进源目录：<源>/render_inputs.npz
  conda run -n "$RDF_ENV" --no-capture-output \
    python scripts/rdf_dump.py \
      --images "$IMAGES_DIR" \
      --out "$SOURCE" \
      --limit "$LIMIT" --stride "$STRIDE"
fi

# --------------------------------------------------------- 整理进 datasets --
info "整理为标准数据集（环境：$MAIN_ENV）"
ARGS=(--from "$SOURCE" --limit "$LIMIT" --images "$IMAGES_DIR")
if [[ -n "$KEEP_IMAGES" ]]; then ARGS+=("$KEEP_IMAGES"); fi
ARGS+=(--name "$NAME")
[[ -n "$FORCE" ]] && ARGS+=("$FORCE")

conda run -n "$MAIN_ENV" --no-capture-output \
  python scripts/build_dataset.py "${ARGS[@]}"
