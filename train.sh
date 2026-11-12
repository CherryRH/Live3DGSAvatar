#!/usr/bin/env bash
# 训练入口：基础训练（默认 spill）+ 可选的 epiC 后训练，一条命令跑完。
#
# 用法：
#   bash train.sh --subject biden --work-name base
#   bash train.sh --subject biden --work-name base --post-train   # 基础 + 后训练
#   bash train.sh --subject biden --work-name base --post-only    # 只跑后训练
#   bash train.sh --subject biden --work-name best --post-train --in-place
#                                                   # 每轮写回 best，不生成 *_post_rN
#   bash train.sh --subject biden --work-name base --plan         # 只打印稳定路线
#   bash train.sh --subject biden --work-name base -- --dry-run   # 训练侧 dry-run
#
# 后训练稳定路线（scripts/iterative_repair.py）：
#   生成器  emilianJR/epiCRealism（SD1.5 底）+ normal ControlNet
#           不做 FaceID、不做颜色对齐（这两条实测会换脸/压锐度）
#   参数    mask/core/strength、真实帧:伪标签、步数、选角圈、质量门
#           全部取 configs/train.yaml 的 train.pipeline.post_train；
#           临时消融用 POST_* 环境变量覆盖（不用改脚本）
#   选角    逐圈外扩，每圈按缺陷质量取 top-N
#   选最佳  SSIM − LPIPS + psnr_weight·PSNR；低 quality_tol 即早停
#
# 环境变量（默认即稳定路线，一般不用改）：
#   MAIN_ENV                conda 环境（默认 live3dgs）
#   POST_*                  临时覆盖 configs/train.yaml 的 post_train
#
# 产物：
#   models/<对象>/<版本>/             基础模型（train.py 自动装入）
#   models/<对象>/<版本>_post_r*/     后训练每轮的最佳检查点
#   output/post/<对象>/<版本>_post/    后训练运行目录（report.json + r*/ + cum/）
#   --in-place 时不产生 *_post_r*：每轮更好的结果直接写回 models/<对象>/<版本>/
#
# 注意：train.sh 自己编排后训练，调用 train.py 时固定加 --no-post-train，
# 因此 configs/train.yaml 里的 post_train.enabled 不会在这里重复触发。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

MAIN_ENV="${MAIN_ENV:-live3dgs}"
PY_RUN=(conda run -n "$MAIN_ENV" --no-capture-output python)

# ------------------------------------------------- 后训练参数 ----
# 默认值以 configs/train.yaml 的 train.pipeline.post_train 为准（下面解析后
# 填充）；需要临时消融时用 POST_* 环境变量覆盖，不要在这里再抄一份。
POST_BASE_MODEL="${POST_BASE_MODEL:-}"
POST_IP_ADAPTER="${POST_IP_ADAPTER:-}"
POST_TOP_N="${POST_TOP_N:-}"
POST_RINGS="${POST_RINGS:-}"
POST_STEPS="${POST_STEPS:-}"
POST_REAL_PER_BATCH="${POST_REAL_PER_BATCH:-}"
POST_REFINED_PER_BATCH="${POST_REFINED_PER_BATCH:-}"
POST_LR_XYZ="${POST_LR_XYZ:-}"
POST_MASK_DILATE="${POST_MASK_DILATE:-}"
POST_CORE_DILATE="${POST_CORE_DILATE:-}"
POST_STRENGTH="${POST_STRENGTH:-}"
POST_MIN_FILL="${POST_MIN_FILL:-}"
POST_TRUSTED_TOL="${POST_TRUSTED_TOL:-}"
POST_SHARPEN="${POST_SHARPEN:-}"
POST_MIN_SHARP="${POST_MIN_SHARP:-}"
POST_MIN_MASK_SHARP="${POST_MIN_MASK_SHARP:-}"
POST_QUALITY_TOL="${POST_QUALITY_TOL:-}"
POST_PSNR_WEIGHT="${POST_PSNR_WEIGHT:-}"
POST_FEATHER="${POST_FEATHER:-}"
POST_COLOR_RING="${POST_COLOR_RING:-}"
POST_COLOR_MATCH="${POST_COLOR_MATCH:-}"
POST_HEAD_CUT="${POST_HEAD_CUT:-}"
POST_HEAD_CUT_MARGIN="${POST_HEAD_CUT_MARGIN:-}"
POST_PROMPT="${POST_PROMPT:-}"
POST_NEGATIVE="${POST_NEGATIVE:-}"

die() { echo "[error] $*" >&2; exit 1; }
info() { echo "[train] $*"; }

usage() {
  sed -n '2,33p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
}

SUBJECT=""
WORK=""
PREFIX=""
PREFIX_SET=0
DO_POST=0
POST_ONLY=0
IN_PLACE=0
PLAN=0
PASSTHRU=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subject)    SUBJECT="${2:-}"; shift 2 ;;
    --work-name)  WORK="${2:-}"; shift 2 ;;
    --prefix)     PREFIX="${2:-}"; PREFIX_SET=1; shift 2 ;;
    --post-train) DO_POST=1; shift ;;
    --post-only)  DO_POST=1; POST_ONLY=1; shift ;;
    --in-place)   IN_PLACE=1; shift ;;
    --plan|--show-plan) PLAN=1; shift ;;
    -h|--help)    usage ;;
    --)           shift; PASSTHRU+=("$@"); break ;;
    *)            PASSTHRU+=("$1"); shift ;;
  esac
done

[[ -n "$SUBJECT" ]] || die "需要 --subject（例如 --subject biden）"
[[ -n "$WORK" ]] || die "需要 --work-name（例如 --work-name base）"

# 读 configs/train.yaml（含 configs/local.yaml 覆盖）里的后训练默认值
cfg_shell="$("${PY_RUN[@]}" scripts/show_config.py --shell-post)" || \
  die "读取 configs/train.yaml 的 post_train 失败"
eval "$cfg_shell"
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
if (( PREFIX_SET == 0 )); then
  if (( IN_PLACE == 1 )); then
    PREFIX="$WORK"          # 原地更新：报告/中间目录也用 work 名
  else
    PREFIX="${WORK}_post"
  fi
fi

# `--` 之后的 --dry-run 已经透传给 train.py；后训练阶段不再执行。
DRY=0
if [[ ${#PASSTHRU[@]} -gt 0 ]]; then
  printf '%s\n' "${PASSTHRU[@]}" | grep -qx -- '--dry-run' && DRY=1
fi
if (( POST_ONLY == 1 )); then
  DRY=0   # --post-only 没有训练侧 dry-run 语义；下面会显式报需要 GPU
fi

print_plan() {
  cat <<EOF
subject=$SUBJECT
work=$WORK
post_prefix=$PREFIX
post_dir=output/post/$SUBJECT/$PREFIX
post_enabled=$DO_POST
post_only=$POST_ONLY
post_in_place=$IN_PLACE
post_base_model=$POST_BASE_MODEL
post_ip_adapter=$POST_IP_ADAPTER
post_top_n=$POST_TOP_N
post_rings=$POST_RINGS
post_steps=$POST_STEPS
post_real_per_batch=$POST_REAL_PER_BATCH
post_refined_per_batch=$POST_REFINED_PER_BATCH
post_lr_xyz=$POST_LR_XYZ
post_mask_dilate=$POST_MASK_DILATE
post_core_dilate=$POST_CORE_DILATE
post_strength=$POST_STRENGTH
post_min_fill=$POST_MIN_FILL
post_trusted_tol=$POST_TRUSTED_TOL
post_sharpen=$POST_SHARPEN
post_min_sharp=$POST_MIN_SHARP
post_min_mask_sharp=$POST_MIN_MASK_SHARP
post_quality_tol=$POST_QUALITY_TOL
post_psnr_weight=$POST_PSNR_WEIGHT
post_feather=$POST_FEATHER
post_color_ring=$POST_COLOR_RING
post_color_match=$POST_COLOR_MATCH
post_head_cut=$POST_HEAD_CUT
post_head_cut_margin=$POST_HEAD_CUT_MARGIN
post_controlnet=lllyasviel/control_v11p_sd15_normalbae
EOF
}

if (( PLAN == 1 )); then
  print_plan
  exit 0
fi

preflight() {
  command -v conda >/dev/null || die "找不到 conda，请先安装并初始化"
  [[ -f "datasets/$SUBJECT/render_inputs.npz" ]] || \
    die "找不到数据集 datasets/$SUBJECT/render_inputs.npz（先跑 bash build_dataset.sh）"
  if (( DRY == 1 )); then
    return 0
  fi
  if ! "${PY_RUN[@]}" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
       >/dev/null 2>&1; then
    die "CUDA 不可用：训练需要 GPU（可用 -- --dry-run 只检查数据/配置）"
  fi
  if (( DO_POST == 1 )); then
    # 只有 --post-only 才要求模型已经存在；--post-train 的模型由本次基础训练产出
    if (( POST_ONLY == 1 )); then
      [[ -f "models/$SUBJECT/$WORK/model.ply" ]] || \
        die "找不到基础模型 models/$SUBJECT/$WORK/model.ply（先跑基础训练）"
    fi
    "${PY_RUN[@]}" -c "import diffusers" >/dev/null 2>&1 || \
      die "后训练需要 diffusers：先跑 bash setup_env.sh"
    if [[ ! -d "$HOME/.cache/huggingface/hub/models--emilianJR--epiCRealism" ]]; then
      info "首次后训练会下载 epiCRealism（约 2 GB），请保持网络畅通"
    fi
  fi
}

base_train() {
  local args=(--subject "$SUBJECT" --work-name "$WORK" --no-post-train)
  if [[ ${#PASSTHRU[@]} -gt 0 ]]; then args+=("${PASSTHRU[@]}"); fi
  info "环境：$MAIN_ENV"
  info "基础训练：python scripts/train.py ${args[*]}"
  "${PY_RUN[@]}" scripts/train.py "${args[@]}"
}

post_train() {
  local args=(
    --subject "$SUBJECT" --base-work "$WORK" --prefix "$PREFIX"
    --top-n "$POST_TOP_N" --rings "$POST_RINGS" --steps "$POST_STEPS"
    --real-per-batch "$POST_REAL_PER_BATCH"
    --refined-per-batch "$POST_REFINED_PER_BATCH"
    --lr-xyz "$POST_LR_XYZ"
    --mask-dilate "$POST_MASK_DILATE" --core-dilate "$POST_CORE_DILATE"
    --strength "$POST_STRENGTH" --min-fill "$POST_MIN_FILL"
    --trusted-tol "$POST_TRUSTED_TOL"
    --sharpen "$POST_SHARPEN" --min-sharp "$POST_MIN_SHARP"
    --min-mask-sharp "$POST_MIN_MASK_SHARP"
    --quality-tol "$POST_QUALITY_TOL" --psnr-weight "$POST_PSNR_WEIGHT"
    --feather "$POST_FEATHER" --color-ring "$POST_COLOR_RING"
    --base-model "$POST_BASE_MODEL" --ip-adapter "$POST_IP_ADAPTER"
    --prompt "$POST_PROMPT" --negative "$POST_NEGATIVE"
  )
  if [[ "${POST_COLOR_MATCH,,}" == "true" ]]; then
    args+=(--color-match)
  else
    args+=(--no-color-match)
  fi
  if [[ "${POST_HEAD_CUT,,}" == "true" ]]; then
    args+=(--head-cut --head-cut-margin "$POST_HEAD_CUT_MARGIN")
  fi
  if (( IN_PLACE == 1 )); then args+=(--in-place); fi
  if (( DRY == 1 )); then
    info "后训练（dry-run 不执行）：python scripts/iterative_repair.py ${args[*]}"
    return 0
  fi
  if [[ ! -f "models/$SUBJECT/$WORK/model.ply" ]]; then
    die "找不到基础模型 models/$SUBJECT/$WORK/model.ply"
  fi
  info "后训练：epiC 稳定路线（top${POST_TOP_N}，${POST_RINGS}，${POST_STEPS} 步/轮）"
  "${PY_RUN[@]}" scripts/iterative_repair.py "${args[@]}"
}

post_report() {
  local report="output/post/$SUBJECT/$PREFIX/report.json"
  [[ -f "$report" ]] || die "后训练没有产出报告：$report"
  "${PY_RUN[@]}" -c '
import json
import sys
from pathlib import Path

report, models = Path(sys.argv[1]), Path(sys.argv[2])
d = json.loads(report.read_text(encoding="utf-8"))
best = d.get("best_model")
if not best:
    print("[train] 后训练没有选出最佳轮（见 {}）".format(report))
    raise SystemExit(1)
target = models / best / "model.ply"
if not target.is_file():
    print("[error] 报告里的最佳模型不存在：{}".format(target), file=sys.stderr)
    raise SystemExit(1)


def fmt(value, spec):
    return "nan" if value is None else format(value, spec)


r = d.get("best_round") or {}
b = r.get("best") or {}
print("[train] 后训练最佳：{}（PSNR {} / SSIM {} / LPIPS {}）".format(
    best, fmt(b.get("psnr"), ".2f"), fmt(b.get("ssim"), ".4f"),
    fmt(b.get("lpips"), ".4f")))
print("[train] 模型：{}".format(target))
print("[train] 报告：{}".format(report))
print("[train] 轮次：{}，总耗时 {:.0f}s".format(
    len(d.get("rounds", [])), d.get("total_seconds", 0) or 0))
' "$report" "models/$SUBJECT"
}

preflight

if (( POST_ONLY == 0 )); then
  base_train
else
  info "跳过基础训练（--post-only）"
fi

if (( DO_POST == 1 )); then
  post_train
  if (( DRY == 0 )); then
    post_report
  else
    info "（dry-run：后训练未执行）"
  fi
elif (( DRY == 0 )); then
  info "已装入 models/$SUBJECT/$WORK/（要后训练加 --post-train）"
fi
