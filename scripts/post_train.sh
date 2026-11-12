#!/usr/bin/env bash
# 兼容入口：只跑后训练，等价于
#   bash train.sh --post-only --subject <subject> --work-name <base-work> --prefix <prefix>
#
# 稳定路线参数只在 train.sh 里维护一份，这里不再重复。
set -euo pipefail
SUBJECT="${1:?usage: post_train.sh <subject> <base-work> [prefix]}"
BASE="${2:?usage: post_train.sh <subject> <base-work> [prefix]}"
PREFIX="${3:-${BASE}_post}"

cd "$(dirname "$0")/.."
exec bash train.sh --post-only \
  --subject "$SUBJECT" --work-name "$BASE" --prefix "$PREFIX"
