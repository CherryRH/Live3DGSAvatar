#!/usr/bin/env bash
# 兼容入口：等价于根目录的 post_train.sh。
set -euo pipefail
SUBJECT="${1:?usage: post_train.sh <subject> <base-work> [prefix]}"
BASE="${2:?usage: post_train.sh <subject> <base-work> [prefix]}"
PREFIX="${3:-${BASE}_post}"
cd "$(dirname "$0")/.."
exec bash post_train.sh --subject "$SUBJECT" --work-name "$BASE" --prefix "$PREFIX"
