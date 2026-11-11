#!/usr/bin/env bash
# 训练一个模型：封装 scripts/train.py。
#
# 用法：
#   bash train.sh --subject crh_1 --work-name run
#   bash train.sh --subject crh_1 --work-name run --iterations 20000
#   bash train.sh --subject crh_1 --work-name run -- --dry-run    # `--` 之后原样透传
#   bash train.sh --subject crh_1 --work-name run --post-train     # 训练后接迭代修复
#
# 环境变量：
#   MAIN_ENV   本项目的 conda 环境（默认 live3dgs）
#
# 数据放在 datasets/<对象>/（`bash build_dataset.sh` 生成）。
# 训练结束**自动**把最佳检查点装进 models/<对象>/<版本>/（同名会覆盖）。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

MAIN_ENV="${MAIN_ENV:-live3dgs}"

die() { echo "[error] $*" >&2; exit 1; }
info() { echo "[train] $*"; }

usage() { sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

SUBJECT=""
WORK=""
PASSTHRU=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subject)    SUBJECT="${2:-}"; shift 2 ;;
    --work-name)  WORK="${2:-}"; shift 2 ;;
    -h|--help)    usage ;;
    --)           shift; PASSTHRU+=("$@"); break ;;
    *)            PASSTHRU+=("$1"); shift ;;
  esac
done

[[ -n "$SUBJECT" ]] || die "需要 --subject（例如 --subject crh_1）"
[[ -n "$WORK" ]] || die "需要 --work-name（例如 --work-name run）"
command -v conda >/dev/null || die "找不到 conda，请先安装并初始化"

ARGS=(--subject "$SUBJECT" --work-name "$WORK")
if [[ ${#PASSTHRU[@]} -gt 0 ]]; then ARGS+=("${PASSTHRU[@]}"); fi

info "环境：$MAIN_ENV"
info "命令：python scripts/train.py ${ARGS[*]}"
conda run -n "$MAIN_ENV" --no-capture-output \
  python scripts/train.py "${ARGS[@]}"

if printf '%s\n' "${PASSTHRU[@]:-}" | grep -qx -- "--dry-run"; then
  info "（dry-run：未落盘）"
else
  info "已装入 models/$SUBJECT/$WORK/"
fi
