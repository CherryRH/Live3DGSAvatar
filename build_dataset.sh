#!/usr/bin/env bash
# 构建数据集：一个入口把"图像 → RDF 跟踪 → 标准数据集"整条流程跑完。
#
# 标准结构（见 docs/CONVENTIONS.md §6）：
#   datasets/<对象>/images/ + render_inputs.npz
# 增强集同结构，放在 datasets/<对象>/augmented/。
#
# 用法：
#   ./build_dataset.sh <images 目录> --name crh_3              # 新建数据集
#   ./build_dataset.sh <images 目录> --into crh_1             # 作为 crh_1 的增强集
#   ./build_dataset.sh <images 目录> --name crh_3 --limit 300 # 只取前 300 帧
#   ./build_dataset.sh --video <视频文件> --name crh_4        # 预留（暂未实现）
#
# 已经跟踪过的目录（含 images/ 与 render_inputs.npz）会跳过 RDF，直接整理：
#   ./build_dataset.sh ~/Projects/cap4d/output/crh_1_200/generated_images --name crh_1_cap4d
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
INTO=""
LIMIT=0
STRIDE=1
FORCE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video)   VIDEO="${2:-}"; shift 2 ;;
    --name)    NAME="${2:-}"; shift 2 ;;
    --into)    INTO="${2:-}"; shift 2 ;;
    --limit)   LIMIT="${2:-0}"; shift 2 ;;
    --stride)  STRIDE="${2:-1}"; shift 2 ;;
    --force)   FORCE="--force"; shift ;;
    -h|--help) usage ;;
    -*)        die "未知参数：$1（--help 查看用法）" ;;
    *)         SOURCE="$1"; shift ;;
  esac
done

# ---------------------------------------------------------------- 视频入口 --
if [[ -n "$VIDEO" ]]; then
  cat >&2 <<'EOF'
[error] 视频入口尚未实现。

规划：先用 ffmpeg 抽帧到临时目录，再走与本脚本相同的图像流程。
      目前请先自行抽帧，然后把输出目录传进来：
        ffmpeg -i <视频> -vf fps=30 <目录>/%05d.png
EOF
  exit 2
fi

[[ -n "$SOURCE" ]] || { usage; }
[[ -d "$SOURCE" ]] || die "目录不存在：$SOURCE"
[[ -n "$NAME" || -n "$INTO" ]] || die "需要 --name（新数据集）或 --into（增强集）"
[[ -z "$NAME" || -z "$INTO" ]] || die "--name 与 --into 只能给一个"

SOURCE="$(cd "$SOURCE" && pwd)"
IMAGES_DIR=""
if [[ -d "$SOURCE/images" ]]; then
  IMAGES_DIR="$SOURCE/images"
else
  IMAGES_DIR="$SOURCE"      # 直接给了图像目录
fi

# ------------------------------------------------------------- RDF 跟踪 --
if [[ -f "$SOURCE/render_inputs.npz" || -f "$SOURCE/rdf.npz" || -f "$SOURCE/cap4d_rdf.npz" ]]; then
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
ARGS=(--from "$SOURCE" --limit "$LIMIT")
if [[ -n "$INTO" ]]; then
  ARGS+=(--into "$INTO")
else
  ARGS+=(--name "$NAME")
fi
[[ -n "$FORCE" ]] && ARGS+=("$FORCE")

conda run -n "$MAIN_ENV" --no-capture-output \
  python scripts/build_dataset.py "${ARGS[@]}"
