#!/usr/bin/env bash
# 分层模型训练入口（独立于旧的 train.sh / post_train.sh）。
#
# 当前阶段：
#   base（历史基线，不在这里训练）
#     -> datasets/<subject>/repair/（SyncFix 修复图，正式数据集部分）
#     -> hard_init：从 UV/FLAME 零初始化，冻结 xyz，真实 6 : 修复 2，1000 步
#
# 用法：
#   bash layered_train.sh --subject bala --base-work base
#   bash layered_train.sh --subject bala --base-work base --steps 1000
#   bash layered_train.sh --subject bala --base-work base --name layered_hard_init
#
# 自由层尚未接入；下阶段在此之后继续。
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
MAIN_ENV="${MAIN_ENV:-live3dgs}"
SUBJECT=""; BASE_WORK="base"; NAME="layered_hard_init"; STEPS=1000
while [[ $# -gt 0 ]]; do
  case "$1" in
    --subject) SUBJECT="${2:-}"; shift 2 ;;
    --base-work) BASE_WORK="${2:-}"; shift 2 ;;
    --name) NAME="${2:-}"; shift 2 ;;
    --steps) STEPS="${2:-}"; shift 2 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "[error] 未知参数：$1" >&2; exit 2 ;;
  esac
done
[[ -n "$SUBJECT" ]] || { echo "[error] 需要 --subject" >&2; exit 2; }
[[ -f "datasets/$SUBJECT/repair/manifest.json" ]] || {
  echo "[error] 缺少 datasets/$SUBJECT/repair/manifest.json" >&2; exit 2; }
[[ -d "datasets/$SUBJECT/repair/repairs" ]] || {
  echo "[error] 缺少 datasets/$SUBJECT/repair/repairs/" >&2; exit 2; }
[[ -f "models/$SUBJECT/$BASE_WORK/model.ply" ]] || {
  echo "[warn] 未找到 base 模型 models/$SUBJECT/$BASE_WORK；硬表面初始化不需要它" >&2; }
echo "[layered] $SUBJECT: base=$BASE_WORK -> hard_init ($STEPS 步，冻结 xyz，真实6:修复2)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run -n "$MAIN_ENV" --no-capture-output python scripts/train.py \
  --subject "$SUBJECT" --work-name "$NAME" --stage hard_surface \
  --repair-dir "datasets/$SUBJECT/repair/repairs" \
  --repair-per-batch 2 --batch-size 8 \
  --iterations "$STEPS" --freeze-xyz --ssim-weight 0 --force
echo "[layered] 完成：models/$SUBJECT/$NAME"
