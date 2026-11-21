#!/usr/bin/env bash
# 纯 RGBAvatar 基线：只保留"真实帧重建 + 标准加密"，不掺本项目的
# 新视角 spill / alpha 监督 / 修复图 / 盘状初始化。
#
# 用法：
#   bash rgb_train.sh --subject bala          # work_name 默认 rgb
#   bash rgb_train.sh --subject bala -- --iterations 30000
#
# 与 train.sh 的差别（有意为之）：
#   - 初始化：各向同性球（configs/model_rgb.yaml），不是贴面圆片
#   - 损失：L1 + SSIM 0.2，无 LPIPS、无 nv_spill、无 alpha
#   - 数据：只用真实帧，没有修复图
#   - 训练：固定步数（不按组合分早停），**不做加密**（保持 UV 初始化的
#     60,353 个高斯，纯 RGBAvatar 重建口径）
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
MAIN_ENV="${MAIN_ENV:-live3dgs}"
PY_RUN=(conda run -n "$MAIN_ENV" --no-capture-output python)

SUBJECT=""
WORK="rgb"
PASSTHRU=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --subject) SUBJECT="${2:-}"; shift 2 ;;
    --work-name) WORK="${2:-}"; shift 2 ;;
    --) shift; PASSTHRU=("$@"); break ;;
    *) echo "[error] 未知参数：$1" >&2; exit 2 ;;
  esac
done
[[ -n "$SUBJECT" ]] || { echo "[error] 需要 --subject" >&2; exit 2; }

info() { echo "[rgb] $*"; }
info "RGBAvatar 基线：$SUBJECT/$WORK（各向同性球、L1+SSIM、无 spill/alpha/修复图）"

"${PY_RUN[@]}" scripts/train.py \
  --subject "$SUBJECT" --work-name "$WORK" --stage base \
  --model-config configs/model_rgb.yaml \
  --iterations 20000 --batch-size 8 --blend-start 0 \
  --ssim-weight 0.2 --lpips-weight 0 --alpha-weight 0 \
  --nv-spill-weight 0 --nv-hole-weight 0 \
  --patience 9999 --min-delta 0.0 \
  --force "${PASSTHRU[@]}"

info "完成：models/$SUBJECT/$WORK"
info "评测：bash eval.sh --subject $SUBJECT --work-name $WORK"
