#!/usr/bin/env bash
# 加密训练管线：硬表面只做初始化，之后全部高斯作为统一集合漂移并在高残差处分裂/剪枝。
#
# 用法：
#   bash densify_train.sh --subject bala --hard-model models/bala/layered_hard_soft/model.ply --name densify_soft
#   bash densify_train.sh --subject bala --train-hard --name densify_full
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$REPO_ROOT"
ENV="${MAIN_ENV:-live3dgs}"; SUBJECT="bala"; HARD=""; NAME="densify"; STEPS=2000
TRAIN_HARD=0; HARD_STEPS=5000; TEX=512
while [[ $# -gt 0 ]]; do
 case "$1" in
  --subject) SUBJECT="${2:-}"; shift 2;;
  --hard-model) HARD="${2:-}"; shift 2;;
  --name) NAME="${2:-}"; shift 2;;
  --steps) STEPS="${2:-}"; shift 2;;
  --train-hard) TRAIN_HARD=1; shift;;
  --hard-steps) HARD_STEPS="${2:-}"; shift 2;;
  --tex-size) TEX="${2:-}"; shift 2;;
  -h|--help) sed -n '2,10p' "$0"; exit 0;;
  *) echo "[error] 未知参数：$1" >&2; exit 2;;
 esac
done
if (( TRAIN_HARD == 1 )); then
  HARD="models/$SUBJECT/densify_hard/model.ply"
  echo "[densify] stage1 hard from scratch, early stop mask PSNR"
  conda run -n "$ENV" --no-capture-output python scripts/train.py \
    --subject "$SUBJECT" --work-name densify_hard --stage hard_surface \
    --repair-dir "datasets/$SUBJECT/repair/repairs" --repair-per-batch 2 \
    --batch-size 8 --iterations "$HARD_STEPS" --early-metric psnr \
    --ssim-weight 0 --force
fi
[[ -f "$HARD" ]] || { echo "[error] 缺少 hard model：$HARD" >&2; exit 2; }
base="models/$SUBJECT/$NAME"
mkdir -p "tmp/$SUBJECT"
unified="tmp/$SUBJECT/${NAME}_unified.ply"
cfg="tmp/$SUBJECT/${NAME}_unified.yaml"
python3 - "$HARD" "$unified" "$cfg" "$TEX" <<'PY'
import sys,re
from pathlib import Path
hard,out,cfg,tex=sys.argv[1:]
p=Path(hard); c=Path(cfg)
t=(p.parent/'config.yaml').read_text()
t=re.sub(r'tex_size:\s*\d+',f'tex_size: {tex}',t,count=1)
c.write_text(t)
PY
conda run -n "$ENV" --no-capture-output python scripts/merge_layers.py \
  --hard "$HARD" --out "$unified" --tex-size "$TEX"
echo "[densify] stage2 unified drift + residual split/prune"
conda run -n "$ENV" --no-capture-output python scripts/train.py \
  --subject "$SUBJECT" --work-name "$NAME" --stage base --init-ply "$unified" \
  --model-config "$cfg" --repair-dir "datasets/$SUBJECT/repair/repairs" \
  --repair-per-batch 2 --batch-size 8 --hard-xyz-lr 5e-5 --iterations "$STEPS" \
  --ssim-weight 0 --densify --force
echo "[densify] 完成 models/$SUBJECT/$NAME"
