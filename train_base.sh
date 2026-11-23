#!/usr/bin/env bash
# 基础模型入口 = **统一管线 + 无修复图 + 无 LPIPS**。
#
# 它同时是：
#   - `avatar` 的单变量对照（两者只差“修复图开/关”）；
#   - 修复图的来源（先训 base，再用它渲染修复图）。
#
# 用法：
#   bash train_base.sh --subject biden                 # work-name 默认 base
#   bash train_base.sh --subject biden --force
#   bash train_base.sh --subject biden -- --iterations 8000
#
# 产物：models/<对象>/base/
# 实现：委托给 train_avatar.sh --no-repairs（LPIPS 默认关闭），两者只差修复图。
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

usage() { sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

SUBJECT=""
WORK="base"
FORCE=0
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --subject)   SUBJECT="${2:-}"; shift 2 ;;
    --work-name) WORK="${2:-}"; shift 2 ;;
    --force)     FORCE=1; shift ;;
    --)          shift; EXTRA=("$@"); break ;;
    -h|--help)   usage ;;
    *)           echo "[error] 未知参数：$1" >&2; exit 2 ;;
  esac
done
[[ -n "$SUBJECT" ]] || { echo "[error] 需要 --subject" >&2; exit 2; }

ARGS=(--subject "$SUBJECT" --work-name "$WORK" --no-repairs)
(( FORCE )) && ARGS+=(--force)
exec bash "$REPO_ROOT/train_avatar.sh" "${ARGS[@]}" -- "${EXTRA[@]}"
