#!/usr/bin/env bash
# 基础模型入口 = 各向同性初始化 + 高 xyz 学习率 + spill/alpha（“优化版 RGBAvatar”）。
#
# 与 `train_rgb.sh` 的唯一差别是：base 开了 nv_spill 0.05 与 alpha 0.2；
# 与 `avatar` 的差别是整套配方：各向同性 + 高 lr + 无修复图，
# 而 avatar 是盘状 + 低 lr + 修复图。
#
# 它同时是修复图的来源（先训 base，再用它渲染修复图）。
#
# 用法：
#   bash train_base.sh --subject biden                 # work-name 默认 base
#   bash train_base.sh --subject biden --force
#   bash train_base.sh --subject biden -- --iterations 8000
#
# 产物：models/<对象>/base/
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

MAIN_ENV="${MAIN_ENV:-live3dgs}"
PY_RUN=(conda run -n "$MAIN_ENV" --no-capture-output python)

usage() { sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

SUBJECT=""
WORK="base"
FORCE=0
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --subject)   SUBJECT="${2:-}"; shift 2 ;;
    --work-name) WORK="${2:-}"; shift 2 ;;
    --force)     FORCE=1; shift ;;
    -h|--help)   usage ;;
    --)          shift; EXTRA=("$@"); break ;;
    *)           echo "[error] 未知参数：$1" >&2; exit 2 ;;
  esac
done
[[ -n "$SUBJECT" ]] || { echo "[error] 需要 --subject" >&2; exit 2; }

eval "$("${PY_RUN[@]}" scripts/show_recipe.py --name base --shell)"
SSIM_W="${RECIPE_SSIM_WEIGHT:-0.1}"; LPIPS_W="${RECIPE_LPIPS_WEIGHT:-0}"
SPILL_W="${RECIPE_SPILL_WEIGHT:-0.05}"; ALPHA_W="${RECIPE_ALPHA_WEIGHT:-0.2}"
MODEL_CFG="${RECIPE_MODEL_CONFIG:-configs/model_rgb.yaml}"
echo "[base] 优化版 RGBAvatar：$SUBJECT/$WORK（各向同性、xyz lr 默认 8e-4、spill+alpha）"
"${PY_RUN[@]}" scripts/train.py \
  --subject "$SUBJECT" --work-name "$WORK" --stage base \
  --model-config "$MODEL_CFG" \
  --iterations 10000 --batch-size 8 --blend-start 0 \
  --ssim-weight "$SSIM_W" --lpips-weight "$LPIPS_W" \
  --alpha-weight "$ALPHA_W" --nv-spill-weight "$SPILL_W" \
  --patience 4 --min-delta 0.05 --eval-every 500 \
  $( (( FORCE )) && echo "--force" ) "${EXTRA[@]}"

echo "[base] 完成：models/$SUBJECT/$WORK"
echo "[base] 评测：bash eval.sh --subject $SUBJECT --work-name $WORK"
