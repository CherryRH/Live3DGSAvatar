#!/usr/bin/env bash
# 增强微调：封装 scripts/enhance_model.py（找缺陷 → 外部精修 → 微调 → 落地 → 评测）。
#
# 用法：
#   bash enhance.sh --subject crh_1 --work-name run \
#       --syncfix-dir ~/Projects/SyncFix \
#       --syncfix-model ~/Projects/SyncFix/model/official/ \
#       --syncfix-env syncfix
#
#   bash enhance.sh ... --promote-name run_refined   # 落地到该名字
#   bash enhance.sh ... --skip-find --skip-syncfix   # 复用已有产物
#
# 环境变量：
#   MAIN_ENV     本项目的 conda 环境（默认 live3dgs）
#   SYNCFIX_ENV  精修器所在的 conda 环境（默认 syncfix）
#
# 默认值：缺陷视角上限 16、SyncFix 每批 4 张、微调 600 步。
# 其余参数（--budget / --syncfix-batch / --steps 等）原样透传。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

MAIN_ENV="${MAIN_ENV:-live3dgs}"
SYNCFIX_ENV="${SYNCFIX_ENV:-syncfix}"

die() { echo "[error] $*" >&2; exit 1; }
info() { echo "[enhance] $*"; }

usage() { sed -n '2,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

SUBJECT=""
WORK=""
SYNCFIX_DIR=""
SYNCFIX_MODEL=""
PROMOTE_NAME=""
PASSTHRU=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subject)       SUBJECT="${2:-}"; shift 2 ;;
    --work-name)     WORK="${2:-}"; shift 2 ;;
    --syncfix-dir)   SYNCFIX_DIR="${2:-}"; shift 2 ;;
    --syncfix-model) SYNCFIX_MODEL="${2:-}"; shift 2 ;;
    --promote-name)  PROMOTE_NAME="${2:-}"; shift 2 ;;
    -h|--help)       usage ;;
    --)              shift; PASSTHRU+=("$@"); break ;;
    *)               PASSTHRU+=("$1"); shift ;;
  esac
done

[[ -n "$SUBJECT" ]] || die "需要 --subject"
[[ -n "$WORK" ]] || die "需要 --work-name"

ARGS=(--subject "$SUBJECT" --work-name "$WORK"
      --syncfix-env "$SYNCFIX_ENV")
[[ -n "$SYNCFIX_DIR" ]] && ARGS+=(--syncfix-dir "$SYNCFIX_DIR")
[[ -n "$SYNCFIX_MODEL" ]] && ARGS+=(--syncfix-model "$SYNCFIX_MODEL")
[[ -n "$PROMOTE_NAME" ]] && ARGS+=(--promote-name "$PROMOTE_NAME")
if [[ ${#PASSTHRU[@]} -gt 0 ]]; then ARGS+=("${PASSTHRU[@]}"); fi

info "环境：$MAIN_ENV（精修器：$SYNCFIX_ENV）"
info "命令：python scripts/enhance_model.py ${ARGS[*]}"
conda run -n "$MAIN_ENV" --no-capture-output \
  python scripts/enhance_model.py "${ARGS[@]}"
