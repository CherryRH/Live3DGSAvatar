#!/usr/bin/env bash
# 评测一个模型：封装 scripts/render_test.py（+ 可选的缺陷对比）。
#
# 用法：
#   bash eval.sh --subject <对象> --work-name <版本>                  # 末尾留出集全部
#   bash eval.sh --subject <对象> --work-name <版本> --frames 200     # 留出集前 200 帧
#   bash eval.sh --subject <对象> --work-name <版本> --frame-mode tail --frames 200
#   bash eval.sh --subject <对象> --work-name <版本> --frame-mode head --frames 64
#   bash eval.sh --subject <对象> --work-name <版本> --base-work <基线>
#   bash eval.sh --subject <对象> --work-name <版本> --frames -1      # 留出集全部
#   bash eval.sh --subject <对象> --work-name <版本> --no-lpips   # 跳过 LPIPS
#
# 取帧：--frame-mode heldout（默认，末尾 test_ratio 留出集）/ head（从头）/
#       tail（从尾）/ random（随机，--seed 可重现）。
# 留出集比例默认取 configs/train.yaml 的 train.pipeline.test_ratio（0.3）。
# 视频帧率固定 30（与实时指标口径一致）；渲染耗时与 FPS 仍在报告里。
#
# 产物：output/render_test/<对象>/<版本>/<数据集>/
#   report.json   PSNR / SSIM / LPIPS / 锐度 / 性能
#   compare.mp4   原图 | 渲染的对比视频（每帧叠加指标；缺 imageio 时为 .webp）
#
# 给了 --base-work 时，额外跑一次同视角缺陷对比（缺口 / 云外溢 / 尖刺）。
# 环境变量：MAIN_ENV（默认 live3dgs）
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

MAIN_ENV="${MAIN_ENV:-live3dgs}"

die() { echo "[error] $*" >&2; exit 1; }
info() { echo "[eval] $*"; }

usage() { sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

SUBJECT=""
WORK=""
BASE_WORK=""
PASSTHRU=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subject)    SUBJECT="${2:-}"; shift 2 ;;
    --work-name)  WORK="${2:-}"; shift 2 ;;
    --base-work)  BASE_WORK="${2:-}"; shift 2 ;;
    -h|--help)    usage ;;
    --)           shift; PASSTHRU+=("$@"); break ;;
    *)            PASSTHRU+=("$1"); shift ;;
  esac
done

[[ -n "$SUBJECT" ]] || die "需要 --subject"
[[ -n "$WORK" ]] || die "需要 --work-name"

ARGS=(--subject "$SUBJECT" --work-name "$WORK")
# 默认评测**末尾留出集全部**：与训练/实验报告的口径一致。
# 用户显式给了 --frame-mode / --frames 时不覆盖。
HAS_MODE=0
HAS_FRAMES=0
for arg in "${PASSTHRU[@]:-}"; do
  case "$arg" in
    --frame-mode|--frame-mode=*) HAS_MODE=1 ;;
    --frames|--frames=*)         HAS_FRAMES=1 ;;
  esac
done
if (( HAS_MODE == 0 )); then ARGS+=(--frame-mode heldout); fi
if (( HAS_FRAMES == 0 )); then ARGS+=(--frames -1); fi
if [[ ${#PASSTHRU[@]} -gt 0 ]]; then ARGS+=("${PASSTHRU[@]}"); fi

info "环境：$MAIN_ENV"
info "渲染测试：python scripts/render_test.py ${ARGS[*]}"
conda run -n "$MAIN_ENV" --no-capture-output \
  python scripts/render_test.py "${ARGS[@]}"

if [[ -n "$BASE_WORK" ]]; then
  info "缺陷对比：$BASE_WORK → $WORK"
  conda run -n "$MAIN_ENV" --no-capture-output \
    python scripts/compare_defects.py --subject "$SUBJECT" \
      --work-name "$WORK" --base-work "$BASE_WORK"
fi
