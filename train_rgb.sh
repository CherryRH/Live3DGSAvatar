#!/usr/bin/env bash
# RGBAvatar 风格参照：各向同性初始化 + 高 xyz lr + L1/SSIM 0.1、黑底；
# 不掺本项目的 spill/alpha、修复图与加密。
#
# 用法：
#   bash train_rgb.sh --subject bala          # work_name 默认 rgb
#   bash train_rgb.sh --subject bala -- --iterations 30000
#
# 与 train_base.sh 的差别（有意为之）：
#   - 初始化：各向同性球（configs/model_rgb.yaml），不是贴面圆片
#   - 损失：L1 + SSIM 0.2，无 LPIPS、无 nv_spill、无 alpha（黑底固定）
#   - 数据：只用真实帧，没有修复图
#   - 训练：固定步数（不按组合分早停），**不做加密**（保持 UV 初始化的
#     60,353 个高斯，纯 RGBAvatar 重建口径）
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
MAIN_ENV="${MAIN_ENV:-live3dgs}"
PY_RUN=(conda run -n "$MAIN_ENV" --no-capture-output python)

usage() { sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

SUBJECT=""
WORK="rgb"
FORCE=0
RUN_EVAL=1
PASSTHRU=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --subject) SUBJECT="${2:-}"; shift 2 ;;
    -h|--help) usage ;;
    --work-name) WORK="${2:-}"; shift 2 ;;
    --force)     FORCE=1; shift ;;
    --no-eval)   RUN_EVAL=0; shift ;;
    --) shift; PASSTHRU=("$@"); break ;;
    *) echo "[error] 未知参数：$1" >&2; exit 2 ;;
  esac
done
[[ -n "$SUBJECT" ]] || { echo "[error] 需要 --subject" >&2; exit 2; }

info() { echo "[rgb] $*"; }
eval "$(conda run -n "$MAIN_ENV" --no-capture-output python scripts/show_recipe.py --name rgb --shell)"
SSIM_W="${RECIPE_SSIM_WEIGHT:-0.1}"; LPIPS_W="${RECIPE_LPIPS_WEIGHT:-0}"
SPILL_W="${RECIPE_SPILL_WEIGHT:-0}"; ALPHA_W="${RECIPE_ALPHA_WEIGHT:-0}"
MODEL_CFG="${RECIPE_MODEL_CONFIG:-configs/model_rgb.yaml}"
info "RGBAvatar 基线：$SUBJECT/$WORK（各向同性球、L1+SSIM 0.1、黑底、无 spill/alpha/修复图）"

"${PY_RUN[@]}" scripts/train.py \
  --subject "$SUBJECT" --work-name "$WORK" --stage base \
  --model-config "$MODEL_CFG" \
  --iterations 10000 --batch-size 8 --blend-start 0 \
  --ssim-weight "$SSIM_W" --lpips-weight "$LPIPS_W" \
  --alpha-weight "$ALPHA_W" --nv-spill-weight "$SPILL_W" --nv-hole-weight 0 \
  --patience 4 --min-delta 0.05 --eval-every 500 \
  $( (( FORCE )) && echo "--force" ) "${PASSTHRU[@]}"

info "完成：models/$SUBJECT/$WORK"
info "评测：bash eval.sh --subject $SUBJECT --work-name $WORK"

if (( RUN_EVAL )) && [[ -f "$REPO_ROOT/models/$SUBJECT/$WORK/model.ply" ]]; then
  echo "[eval] 评测：bash eval.sh --subject $SUBJECT --work-name $WORK"
  bash "$REPO_ROOT/eval.sh" --subject "$SUBJECT" --work-name "$WORK"
fi
