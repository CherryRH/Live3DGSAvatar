#!/usr/bin/env bash
# 基础训练入口。后训练已拆分到独立的 post_train.sh。
#
# 用法：
#   bash train.sh --subject biden --work-name base
#   bash train.sh --subject biden --work-name base -- --dry-run
#   bash train.sh --subject biden --work-name base -- --iterations 8000
#   bash train.sh --subject biden --work-name base -- --force
#
# 产物：models/<对象>/<版本>/；中间产物 output/train/<对象>/<版本>/。
# 后训练：bash post_train.sh --subject <对象> --work-name <版本> [--in-place]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

MAIN_ENV="${MAIN_ENV:-live3dgs}"
PY_RUN=(conda run -n "$MAIN_ENV" --no-capture-output python)

die() { echo "[error] $*" >&2; exit 1; }
info() { echo "[train] $*"; }

usage() {
  sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
}

SUBJECT=""
WORK=""
PASSTHRU=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --subject)   SUBJECT="${2:-}"; shift 2 ;;
    --work-name) WORK="${2:-}"; shift 2 ;;
    -h|--help)   usage ;;
    --)          shift; PASSTHRU+=("$@"); break ;;
    *)           PASSTHRU+=("$1"); shift ;;
  esac
done
[[ -n "$SUBJECT" ]] || die "需要 --subject（例如 --subject biden）"
[[ -n "$WORK" ]] || die "需要 --work-name（例如 --work-name base）"

DRY=0
if [[ ${#PASSTHRU[@]} -gt 0 ]]; then
  printf '%s\n' "${PASSTHRU[@]}" | grep -qx -- '--dry-run' && DRY=1
fi

command -v conda >/dev/null || die "找不到 conda"
[[ -f "datasets/$SUBJECT/render_inputs.npz" ]] || \
  die "找不到数据集 datasets/$SUBJECT/render_inputs.npz"
if (( DRY == 0 )); then
  "${PY_RUN[@]}" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    >/dev/null 2>&1 || die "CUDA 不可用：训练需要 GPU（-- --dry-run 只检查数据）"
fi

ARGS=(--subject "$SUBJECT" --work-name "$WORK")
if [[ ${#PASSTHRU[@]} -gt 0 ]]; then ARGS+=("${PASSTHRU[@]}"); fi
info "环境：$MAIN_ENV"
info "基础训练：python scripts/train.py ${ARGS[*]}"
"${PY_RUN[@]}" scripts/train.py "${ARGS[@]}"

if (( DRY == 0 )); then
  info "已装入 models/$SUBJECT/$WORK/"
  info "后训练：bash post_train.sh --subject $SUBJECT --work-name $WORK"
fi
