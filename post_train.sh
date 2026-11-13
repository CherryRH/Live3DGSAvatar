#!/usr/bin/env bash
# 后训练独立入口：在已有基础模型上做迭代修复。
#
# 用法：
#   bash post_train.sh --subject biden --work-name base
#   bash post_train.sh --subject biden --work-name best --in-place
#   bash post_train.sh --subject biden --work-name base --repair-mode syncfix
#   bash post_train.sh --subject biden --work-name base --repair-mode none
#   bash post_train.sh --subject biden --work-name base --plan
#
# 修复方式：
#   syncfix（默认）         选角 -> SyncFix 全图多视图 -> 微调
#   mask_diffusion          选角 -> 掩码伪标签（SD1.5/epiC）-> 微调
#   none                    消融：只用真实帧微调，不做任何修复
#
# 参数默认值全部来自 configs/train.yaml 的 train.pipeline.post_train；
# POST_* 环境变量可临时覆盖。产物：output/post/<对象>/<前缀>/。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

MAIN_ENV="${MAIN_ENV:-live3dgs}"
PY_RUN=(conda run -n "$MAIN_ENV" --no-capture-output python)

die() { echo "[error] $*" >&2; exit 1; }
info() { echo "[post] $*"; }

usage() {
  sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
}

SUBJECT=""
WORK=""
PREFIX=""
PREFIX_SET=0
IN_PLACE=0
REPAIR_MODE=""
PLAN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subject)      SUBJECT="${2:-}"; shift 2 ;;
    --work-name)    WORK="${2:-}"; shift 2 ;;
    --prefix)       PREFIX="${2:-}"; PREFIX_SET=1; shift 2 ;;
    --in-place)     IN_PLACE=1; shift ;;
    --repair-mode)  REPAIR_MODE="${2:-}"; shift 2 ;;
    --plan|--show-plan) PLAN=1; shift ;;
    -h|--help)      usage ;;
    *)              die "未知参数：$1" ;;
  esac
done

[[ -n "$SUBJECT" ]] || die "需要 --subject（例如 --subject biden）"
[[ -n "$WORK" ]] || die "需要 --work-name（例如 --work-name base）"

cfg_shell="$("${PY_RUN[@]}" scripts/show_config.py --shell-post)" || \
  die "读取 configs/train.yaml 的 post_train 失败"
eval "$cfg_shell"

# config 默认值 → POST_*（环境变量优先）
POST_BASE_MODEL="${POST_BASE_MODEL:-$CFG_POST_BASE_MODEL}"
POST_IP_ADAPTER="${POST_IP_ADAPTER:-$CFG_POST_IP_ADAPTER}"
POST_TOP_N="${POST_TOP_N:-$CFG_POST_TOP_N}"
POST_RINGS="${POST_RINGS:-$CFG_POST_RINGS}"
POST_STEPS="${POST_STEPS:-$CFG_POST_STEPS}"
POST_REAL_PER_BATCH="${POST_REAL_PER_BATCH:-$CFG_POST_REAL_PER_BATCH}"
POST_REFINED_PER_BATCH="${POST_REFINED_PER_BATCH:-$CFG_POST_REFINED_PER_BATCH}"
POST_LR_XYZ="${POST_LR_XYZ:-$CFG_POST_LR_XYZ}"
POST_MASK_DILATE="${POST_MASK_DILATE:-$CFG_POST_MASK_DILATE}"
POST_CORE_DILATE="${POST_CORE_DILATE:-$CFG_POST_CORE_DILATE}"
POST_STRENGTH="${POST_STRENGTH:-$CFG_POST_STRENGTH}"
POST_MIN_FILL="${POST_MIN_FILL:-$CFG_POST_MIN_FILL}"
POST_TRUSTED_TOL="${POST_TRUSTED_TOL:-$CFG_POST_TRUSTED_TOL}"
POST_SHARPEN="${POST_SHARPEN:-$CFG_POST_SHARPEN}"
POST_MIN_SHARP="${POST_MIN_SHARP:-$CFG_POST_MIN_SHARP}"
POST_MIN_MASK_SHARP="${POST_MIN_MASK_SHARP:-$CFG_POST_MIN_MASK_SHARP}"
POST_QUALITY_TOL="${POST_QUALITY_TOL:-$CFG_POST_QUALITY_TOL}"
POST_PSNR_WEIGHT="${POST_PSNR_WEIGHT:-$CFG_POST_PSNR_WEIGHT}"
POST_FEATHER="${POST_FEATHER:-$CFG_POST_FEATHER}"
POST_COLOR_RING="${POST_COLOR_RING:-$CFG_POST_COLOR_RING}"
POST_COLOR_MATCH="${POST_COLOR_MATCH:-$CFG_POST_COLOR_MATCH}"
POST_HEAD_CUT="${POST_HEAD_CUT:-$CFG_POST_HEAD_CUT}"
POST_HEAD_CUT_MARGIN="${POST_HEAD_CUT_MARGIN:-$CFG_POST_HEAD_CUT_MARGIN}"
POST_PROMPT="${POST_PROMPT:-$CFG_POST_PROMPT}"
POST_NEGATIVE="${POST_NEGATIVE:-$CFG_POST_NEGATIVE}"
POST_REPAIR_MODE="${POST_REPAIR_MODE:-${REPAIR_MODE:-${CFG_POST_REPAIR_MODE:-syncfix}}}"
POST_SYNCFIX_READY="${POST_SYNCFIX_READY:-${CFG_POST_SYNCFIX_READY:-false}}"
POST_SYNCFIX_REPO="${POST_SYNCFIX_REPO:-${CFG_POST_SYNCFIX_REPO:-/home/test/Projects/SyncFix}}"
POST_SYNCFIX_PYTHON="${POST_SYNCFIX_PYTHON:-${CFG_POST_SYNCFIX_PYTHON:-syncfix}}"
POST_SYNCFIX_MODEL="${POST_SYNCFIX_MODEL:-${CFG_POST_SYNCFIX_MODEL:-/home/test/Projects/SyncFix/models/paper}}"
POST_SYNCFIX_N_PER_PASS="${POST_SYNCFIX_N_PER_PASS:-${CFG_POST_SYNCFIX_N_PER_PASS:-4}}"
POST_SYNCFIX_REF_SIZE="${POST_SYNCFIX_REF_SIZE:-${CFG_POST_SYNCFIX_REF_SIZE:-4}}"

case "$POST_REPAIR_MODE" in
  joint_syncfix)
    info "repair-mode 'joint_syncfix' 已更名为 'syncfix'，本次按 syncfix 执行"
    POST_REPAIR_MODE=syncfix ;;
  mask_diffusion|syncfix|none) ;;
  *) die "未知 --repair-mode：$POST_REPAIR_MODE（syncfix / mask_diffusion / none）" ;;
esac

if (( PREFIX_SET == 0 )); then
  if (( IN_PLACE == 1 )); then PREFIX="$WORK"; else PREFIX="${WORK}_post"; fi
fi

print_plan() {
  cat <<EOF
subject=$SUBJECT
work=$WORK
post_prefix=$PREFIX
post_dir=output/post/$SUBJECT/$PREFIX
post_in_place=$IN_PLACE
post_repair_mode=$POST_REPAIR_MODE
post_rings=$POST_RINGS
post_steps=$POST_STEPS
post_quality_tol=$POST_QUALITY_TOL
post_base_model=$POST_BASE_MODEL
post_syncfix_ready=$POST_SYNCFIX_READY
post_syncfix_repo=$POST_SYNCFIX_REPO
post_syncfix_model=$POST_SYNCFIX_MODEL
post_syncfix_n_per_pass=$POST_SYNCFIX_N_PER_PASS
post_syncfix_ref_size=$POST_SYNCFIX_REF_SIZE
post_real_per_batch=$POST_REAL_PER_BATCH
post_refined_per_batch=$POST_REFINED_PER_BATCH
post_train_batch=$((POST_REAL_PER_BATCH + POST_REFINED_PER_BATCH))
EOF
}
if (( PLAN == 1 )); then print_plan; exit 0; fi

preflight() {
  command -v conda >/dev/null || die "找不到 conda"
  [[ -f "models/$SUBJECT/$WORK/model.ply" ]] || \
    die "找不到基础模型 models/$SUBJECT/$WORK/model.ply"
  # 模式相关的配置先查，避免没 GPU 时只报 CUDA、掩盖"SyncFix 未配置"
  if [[ "$POST_REPAIR_MODE" == "mask_diffusion" ]]; then
    "${PY_RUN[@]}" -c "import diffusers" >/dev/null 2>&1 || \
      die "mask_diffusion 需要 diffusers：先跑 bash setup_env.sh"
  elif [[ "$POST_REPAIR_MODE" == "syncfix" ]]; then
    [[ "${POST_SYNCFIX_READY,,}" == "true" ]] || \
      die "SyncFix 环境未配置（syncfix_ready=false）；本次不会调用。配置好后再设 true"
    [[ -f "$POST_SYNCFIX_MODEL/model.safetensors" ]] || \
      die "找不到 SyncFix 权重：$POST_SYNCFIX_MODEL"
  fi
  "${PY_RUN[@]}" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    >/dev/null 2>&1 || die "CUDA 不可用：后训练需要 GPU"
}

# ---- 两种模式共用的训练参数 ----
post_args=(
  --subject "$SUBJECT" --base-work "$WORK" --prefix "$PREFIX"
  --top-n "$POST_TOP_N" --rings "$POST_RINGS" --steps "$POST_STEPS"
  --real-per-batch "$POST_REAL_PER_BATCH"
  --refined-per-batch "$POST_REFINED_PER_BATCH"
  --lr-xyz "$POST_LR_XYZ"
  --quality-tol "$POST_QUALITY_TOL" --psnr-weight "$POST_PSNR_WEIGHT"
  --repair-mode "$POST_REPAIR_MODE"
)
if (( IN_PLACE == 1 )); then post_args+=(--in-place); fi

# ---- 标签生成：各模式独有 ----
if [[ "$POST_REPAIR_MODE" == "mask_diffusion" ]]; then
  post_args+=(
    --mask-dilate "$POST_MASK_DILATE" --core-dilate "$POST_CORE_DILATE"
    --strength "$POST_STRENGTH" --min-fill "$POST_MIN_FILL"
    --trusted-tol "$POST_TRUSTED_TOL"
    --sharpen "$POST_SHARPEN" --min-sharp "$POST_MIN_SHARP"
    --min-mask-sharp "$POST_MIN_MASK_SHARP"
    --feather "$POST_FEATHER" --color-ring "$POST_COLOR_RING"
    --base-model "$POST_BASE_MODEL" --ip-adapter "$POST_IP_ADAPTER"
    --prompt "$POST_PROMPT" --negative "$POST_NEGATIVE"
  )
  if [[ "${POST_COLOR_MATCH,,}" == "true" ]]; then post_args+=(--color-match); else post_args+=(--no-color-match); fi
  if [[ "${POST_HEAD_CUT,,}" == "true" ]]; then post_args+=(--head-cut --head-cut-margin "$POST_HEAD_CUT_MARGIN"); fi
elif [[ "$POST_REPAIR_MODE" == "syncfix" ]]; then
  post_args+=(
    --syncfix-repo "$POST_SYNCFIX_REPO"
    --syncfix-python "$POST_SYNCFIX_PYTHON"
    --syncfix-model "$POST_SYNCFIX_MODEL"
    --syncfix-n-per-pass "$POST_SYNCFIX_N_PER_PASS"
    --syncfix-ref-size "$POST_SYNCFIX_REF_SIZE"
  )
  if [[ "${POST_SYNCFIX_READY,,}" == "true" ]]; then post_args+=(--syncfix-ready); fi
fi
# none 模式只传共用参数（top_n/rings/steps/real/lr/quality），不传标签生成参数

preflight
info "修复方式：$POST_REPAIR_MODE；模型：models/$SUBJECT/$WORK"
"${PY_RUN[@]}" scripts/iterative_repair.py "${post_args[@]}"

report="output/post/$SUBJECT/$PREFIX/report.json"
[[ -f "$report" ]] || die "后训练没有产出报告：$report"
"${PY_RUN[@]}" -c '
import json, sys
from pathlib import Path
report, models = Path(sys.argv[1]), Path(sys.argv[2])
d = json.loads(report.read_text(encoding="utf-8"))
best = d.get("best_model")
if not best:
    print("[post] 没有选出模型（见 {}）".format(report)); raise SystemExit(1)
target = models / best / "model.ply"
if not target.is_file():
    print("[error] 模型不存在：{}".format(target), file=sys.stderr); raise SystemExit(1)
if d.get("accepted") is False:
    print("[post] 没有可接受的轮次，保持 {}".format(target.parent)); raise SystemExit(0)
r = d.get("best_round") or {}; b = r.get("best") or {}
def fmt(v, s): return "nan" if v is None else format(v, s)
print("[post] 最终：{}（PSNR {} / SSIM {} / LPIPS {}）".format(
    best, fmt(b.get("psnr"), ".2f"), fmt(b.get("ssim"), ".4f"), fmt(b.get("lpips"), ".4f")))
print("[post] 模型：{}".format(target)); print("[post] 报告：{}".format(report))
' "$report" "models/$SUBJECT"
