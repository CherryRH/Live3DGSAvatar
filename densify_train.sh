#!/usr/bin/env bash
# 统一集合加密训练管线（三段式：硬表面初始化 -> 硬表面精修 -> 动态加密）。
#
#   1) init     硬表面从 UV/FLAME 零初始化。只拟合 FLAME 表面内的外观，
#               几何约束最强（贴面/法线/尺度上限），形变基按 init_blend_start 决定。
#   2) refine   以 init 为 warm start。mask 权重恢复 1.0（背景也参与），
#               形变基从第 0 步开启（warm start 的基是 0，晚开只会浪费步数）。
#   3) densify  合并成 tex 512 的统一集合，残差驱动 split，继续训练。
#
# 用法：
#   bash densify_train.sh --subject bala               # 全流程
#   bash densify_train.sh --subject bala --hard-only   # 只到硬表面模型（先看这段）
#   bash densify_train.sh --subject bala --plan        # 只打印将要执行的命令
#   bash densify_train.sh --subject bala --eval        # 每段结束后自动评测
#
# 参数默认值全部来自 configs/train.yaml 的 train.pipeline.densify；
# DENS_* 环境变量可临时覆盖（见脚本中 ${DENS_X:-$CFG_DENSIFY_X} 的写法）。
# 产物：models/<对象>/{<名字>_init,<名字>_hard,<名字>}。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

MAIN_ENV="${MAIN_ENV:-live3dgs}"
PY_RUN=(conda run -n "$MAIN_ENV" --no-capture-output python)

die() { echo "[error] $*" >&2; exit 2; }
info() { echo "[densify] $*"; }

usage() { sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

SUBJECT=""
NAME="densify"
HARD_ONLY=0
PLAN=0
RUN_EVAL=0
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subject)   SUBJECT="${2:-}"; shift 2 ;;
    --name)      NAME="${2:-}"; shift 2 ;;
    --hard-only) HARD_ONLY=1; shift ;;
    --plan|--show-plan) PLAN=1; shift ;;
    --eval)      RUN_EVAL=1; shift ;;
    --force)     FORCE=1; shift ;;
    -h|--help)   usage ;;
    *)           die "未知参数：$1" ;;
  esac
done

[[ -n "$SUBJECT" ]] || die "需要 --subject（例如 --subject bala）"
[[ -f "datasets/$SUBJECT/repair/manifest.json" ]] || \
  die "缺少 datasets/$SUBJECT/repair/manifest.json"
[[ -d "datasets/$SUBJECT/repair/repairs" ]] || \
  die "缺少 datasets/$SUBJECT/repair/repairs/"

cfg_shell="$("${PY_RUN[@]}" scripts/show_config.py --shell-densify)" || \
  die "读取 configs/train.yaml 的 train.pipeline.densify 失败"
eval "$cfg_shell"

# config 默认值 -> DENS_*（同名环境变量优先）
INIT_STEPS="${DENS_INIT_STEPS:-${CFG_DENSIFY_INIT_STEPS:-5000}}"
INIT_BATCH="${DENS_INIT_BATCH:-${CFG_DENSIFY_INIT_BATCH:-8}}"
INIT_REPAIR="${DENS_INIT_REPAIR:-${CFG_DENSIFY_INIT_REPAIR:-2}}"
INIT_BLEND="${DENS_INIT_BLEND_START:-${CFG_DENSIFY_INIT_BLEND_START:-3000}}"

REFINE_STEPS="${DENS_REFINE_STEPS:-${CFG_DENSIFY_REFINE_STEPS:-5000}}"
REFINE_BATCH="${DENS_REFINE_BATCH:-${CFG_DENSIFY_REFINE_BATCH:-8}}"
REFINE_REPAIR="${DENS_REFINE_REPAIR:-${CFG_DENSIFY_REFINE_REPAIR:-1}}"
REFINE_BLEND="${DENS_REFINE_BLEND_START:-${CFG_DENSIFY_REFINE_BLEND_START:-0}}"
REFINE_XYZ_LR="${DENS_REFINE_XYZ_LR:-${CFG_DENSIFY_REFINE_XYZ_LR:-0.00005}}"

STEPS="${DENS_STEPS:-${CFG_DENSIFY_STEPS:-5000}}"
BATCH="${DENS_BATCH:-${CFG_DENSIFY_BATCH:-4}}"
REPAIR="${DENS_REPAIR:-${CFG_DENSIFY_REPAIR:-1}}"
BLEND="${DENS_BLEND_START:-${CFG_DENSIFY_BLEND_START:-0}}"
EVERY="${DENS_EVERY:-${CFG_DENSIFY_EVERY:-300}}"
COUNT="${DENS_COUNT:-${CFG_DENSIFY_COUNT:-300}}"
MAX_GAUSS="${DENS_MAX:-${CFG_DENSIFY_MAX_GAUSSIANS:-120000}}"
PRUNE_OP="${DENS_PRUNE_OPACITY:-${CFG_DENSIFY_PRUNE_OPACITY:-0.01}}"
JITTER="${DENS_SPLIT_JITTER:-${CFG_DENSIFY_SPLIT_JITTER:-0.8}}"
TEX="${DENS_TEX_SIZE:-${CFG_DENSIFY_TEX_SIZE:-512}}"

MODELS="models/$SUBJECT"
INIT_WORK="${NAME}_init"
HARD_WORK="${NAME}_hard"
UNIFIED="tmp/${SUBJECT}_${NAME}_unified.ply"
UNIFIED_CFG="tmp/${SUBJECT}_${NAME}_unified.yaml"

FORCE_FLAG=()
(( FORCE )) && FORCE_FLAG=(--force)

run() {
  echo
  echo "\$ $*"
  (( PLAN )) || "$@"
}

eval_after() {
  (( RUN_EVAL )) || return 0
  bash eval.sh --subject "$SUBJECT" --work-name "$1" --no-lpips
}

echo "=============================================================="
echo "加密训练管线：$SUBJECT / $NAME"
echo "  stage1 init    : $INIT_STEPS 步  batch $INIT_BATCH  repair $INIT_REPAIR" \
     " blend_start $INIT_BLEND"
echo "  stage2 refine  : $REFINE_STEPS 步  batch $REFINE_BATCH  repair $REFINE_REPAIR" \
     " blend_start $REFINE_BLEND  xyz_lr $REFINE_XYZ_LR"
echo "  stage3 densify : $STEPS 步  batch $BATCH  repair $REPAIR" \
     " blend_start $BLEND  every $EVERY count $COUNT max $MAX_GAUSS tex $TEX"
echo "  产物           : $MODELS/{$INIT_WORK,$HARD_WORK,$NAME}"
(( FORCE )) && echo "  --force        : 覆盖同名模型/产物目录"
(( PLAN )) && echo "  --plan         : 只打印命令，不执行"
echo "=============================================================="

# ---------------------------------------------------------------- stage 1 --
info "stage1：硬表面从 UV/FLAME 零初始化（静态几何 + 表面内外观）"
run "${PY_RUN[@]}" scripts/train.py \
  --subject "$SUBJECT" --work-name "$INIT_WORK" --stage hard_surface \
  --repair-dir "datasets/$SUBJECT/repair/repairs" \
  --repair-per-batch "$INIT_REPAIR" \
  --batch-size "$INIT_BATCH" --iterations "$INIT_STEPS" \
  --blend-start "$INIT_BLEND" \
  --no-robust-eval --ssim-weight 0 "${FORCE_FLAG[@]}"
(( PLAN )) || [[ -f "$MODELS/$INIT_WORK/model.ply" ]] || \
  die "stage1 没有产出 $MODELS/$INIT_WORK/model.ply"
info "stage1 完成：$MODELS/$INIT_WORK"
eval_after "$INIT_WORK"

# ---------------------------------------------------------------- stage 2 --
info "stage2：硬表面精修（warm start，形变基从第 $REFINE_BLEND 步开启）"
run "${PY_RUN[@]}" scripts/train.py \
  --subject "$SUBJECT" --work-name "$HARD_WORK" --stage base \
  --init-ply "$MODELS/$INIT_WORK/model.ply" \
  --repair-dir "datasets/$SUBJECT/repair/repairs" \
  --repair-per-batch "$REFINE_REPAIR" \
  --batch-size "$REFINE_BATCH" --iterations "$REFINE_STEPS" \
  --blend-start "$REFINE_BLEND" --hard-xyz-lr "$REFINE_XYZ_LR" \
  --ssim-weight 0 "${FORCE_FLAG[@]}"
(( PLAN )) || [[ -f "$MODELS/$HARD_WORK/model.ply" ]] || \
  die "stage2 没有产出 $MODELS/$HARD_WORK/model.ply"
info "stage2 完成：硬表面模型 $MODELS/$HARD_WORK"
echo "         评测：bash eval.sh --subject $SUBJECT --work-name $HARD_WORK --no-lpips"
eval_after "$HARD_WORK"

if (( HARD_ONLY )); then
  info "按要求停在硬表面模型（--hard-only）"
  exit 0
fi

# ---------------------------------------------------------------- stage 3 --
info "stage3：合并成 tex $TEX 的统一集合并继续训练（残差驱动 split）"
run "${PY_RUN[@]}" - "$MODELS/$HARD_WORK" "$UNIFIED" "$UNIFIED_CFG" "$TEX" <<'PY'
import re
import sys
from pathlib import Path

hard_dir, out_ply, out_cfg, tex = sys.argv[1:]
src = Path(hard_dir) / "config.yaml"
text = src.read_text(encoding="utf-8")
text, n = re.subn(r"tex_size:\s*\d+", f"tex_size: {tex}", text, count=1)
if n != 1:
    raise SystemExit(f"[error] 无法在 {src} 里替换 tex_size")
Path(out_cfg).parent.mkdir(parents=True, exist_ok=True)
Path(out_cfg).write_text(text, encoding="utf-8")
print(f"结构配置：{src} -> {out_cfg}（tex {tex}）")
PY

run "${PY_RUN[@]}" scripts/merge_layers.py \
  --hard "$MODELS/$HARD_WORK/model.ply" --out "$UNIFIED" --tex-size "$TEX"

run "${PY_RUN[@]}" scripts/train.py \
  --subject "$SUBJECT" --work-name "$NAME" --stage base \
  --init-ply "$UNIFIED" --model-config "$UNIFIED_CFG" \
  --repair-dir "datasets/$SUBJECT/repair/repairs" \
  --repair-per-batch "$REPAIR" \
  --batch-size "$BATCH" --iterations "$STEPS" \
  --blend-start "$BLEND" --hard-xyz-lr "$REFINE_XYZ_LR" \
  --densify --densify-every "$EVERY" --densify-count "$COUNT" \
  --densify-max "$MAX_GAUSS" --densify-prune-opacity "$PRUNE_OP" \
  --densify-split-jitter "$JITTER" \
  --ssim-weight 0 "${FORCE_FLAG[@]}"

info "全部完成：$MODELS/$NAME"
echo "         评测：bash eval.sh --subject $SUBJECT --work-name $NAME --no-lpips"
eval_after "$NAME"
