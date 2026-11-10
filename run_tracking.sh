#!/usr/bin/env bash
# 启动实时跟随服务（RDF 追踪），供 GUI 的「实时驱动」使用。
#
# 用法：
#   bash run_tracking.sh                 # 默认端口 8980
#   bash run_tracking.sh --port 9000
#
# 这是**另一个终端**里的进程，和 `bash run_gui.sh` 一起用：
#   终端 A: bash run_tracking.sh
#   终端 B: bash run_gui.sh
# 然后在界面右侧「驱动」里选「RDF 实时追踪」。
#
# 环境变量：
#   RDF_ENV  追踪器的 conda 环境（默认 realdenseface）
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

RDF_ENV="${RDF_ENV:-realdenseface}"

die() { echo "[error] $*" >&2; exit 1; }
info() { echo "[tracking] $*"; }

usage() { sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

PORT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)    PORT="${2:-}"; shift 2 ;;
    -h|--help) usage ;;
    *)         die "未知参数：$1（--help 查看用法）" ;;
  esac
done

command -v conda >/dev/null || die "找不到 conda，请先安装并初始化"

ARGS=()
[[ -n "$PORT" ]] && ARGS+=(--port "$PORT")

info "环境：$RDF_ENV（端口 ${PORT:-默认}）"
conda run -n "$RDF_ENV" --no-capture-output \
  python scripts/rdf_bridge.py "${ARGS[@]}"
