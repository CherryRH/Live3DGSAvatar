#!/usr/bin/env bash
# stage3：在加密模型上做"逐轮角度外扩"的后训练。
#
# 与旧 post_train.sh 的区别：
#   - 修复图由**当前 stage2 模型**渲染后修复，而不是基础模型；
#   - 不选缺陷，按固定 yaw 间隔（默认 5°）从训练段的 p90 一路补到 ±50°；
#   - 锚定帧只用训练段（refine 的 --anchor-scope train），留出集不参与；
#   - 每轮只微调 50 步。
#
# 用法：
#   bash stage3_train.sh --subject bala --work-name densify --name densify_s3
#   bash stage3_train.sh --subject bala --work-name densify --name densify_s3 --real-only
#   bash stage3_train.sh --subject bala --work-name densify --name densify_s3 --plan
#
# 实测（bala）：50 步真实帧微调 +0.40 dB；把 ±40/45/50 的 SyncFix 修复图
# 按 1/8 混进去反而低 0.02 dB、锐度略降 —— 修复图在当前模型上没有额外信息。
# 所以默认（--real-only）只用真实帧；要复现"修复图版"加 --with-repairs。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

MAIN_ENV="${MAIN_ENV:-live3dgs}"
PY_RUN=(conda run -n "$MAIN_ENV" --no-capture-output python)
SYNCFIX_REPO="${SYNCFIX_REPO:-/home/test/Projects/SyncFix}"
SYNCFIX_MODEL="${SYNCFIX_MODEL:-$SYNCFIX_REPO/models/paper}"
SYNCFIX_PYTHON="${SYNCFIX_PYTHON:-syncfix}"
# SyncFix 的输出整体偏软，先做一层轻量 unsharp，再当监督标签用；
# 0 = 不锐化。
SHARPEN="${SHARPEN:-0.35}"

die() { echo "[error] $*" >&2; exit 2; }
info() { echo "[stage3] $*"; }
usage() { sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

SUBJECT=""
WORK=""
NAME=""
START_YAW=""
MAX_YAW=50
STEP=5
STEPS=50
PLAN=0
FORCE=0
WITH_REPAIRS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subject) SUBJECT="${2:-}"; shift 2 ;;
    --work-name) WORK="${2:-}"; shift 2 ;;
    --name) NAME="${2:-}"; shift 2 ;;
    --start-yaw) START_YAW="${2:-}"; shift 2 ;;
    --max-yaw) MAX_YAW="${2:-}"; shift 2 ;;
    --step) STEP="${2:-}"; shift 2 ;;
    --steps) STEPS="${2:-}"; shift 2 ;;
    --with-repairs) WITH_REPAIRS=1; shift ;;
    --real-only) WITH_REPAIRS=0; shift ;;
    --plan) PLAN=1; shift ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage ;;
    *) die "未知参数：$1" ;;
  esac
done

[[ -n "$SUBJECT" ]] || die "需要 --subject"
[[ -n "$WORK" ]] || die "需要 --work-name（起始模型，例如 densify）"
[[ -n "$NAME" ]] || NAME="${WORK}_s3"
[[ -d "$SYNCFIX_REPO" ]] || die "找不到 SyncFix 仓库：$SYNCFIX_REPO"
[[ -e "$SYNCFIX_MODEL" ]] || die "找不到 SyncFix 权重：$SYNCFIX_MODEL"

# 起始角度：训练段 |yaw| 的 p90，向上取整到 step 的整数倍
if [[ -z "$START_YAW" ]]; then
  START_YAW="$("${PY_RUN[@]}" - "$SUBJECT" "$STEP" <<'PY'
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path.cwd() / "src"))
from live3dgsavatar.data.pose import head_yaw
subject, step = sys.argv[1], float(sys.argv[2])
with np.load(Path("datasets") / subject / "render_inputs.npz", allow_pickle=True) as d:
    w = np.asarray(d["blend_weight"], dtype=np.float32)
n = w.shape[0]
held = int(round(n * 0.3))
y = np.abs(head_yaw(w))[: n - held]
p90 = float(np.percentile(y, 90))
print(int(np.ceil(p90 / step) * step))
PY
)"
  START_YAW="${START_YAW##*$'\n'}"
fi

REPAIR_DIR="datasets/$SUBJECT/repair/stage3"
OUT_ROOT="output/stage3/$SUBJECT/$NAME"

info "起始模型：models/$SUBJECT/$WORK"
info "角度圈  ：从 ±${START_YAW}° 到 ±${MAX_YAW}°，步长 ${STEP}°，每轮 ${STEPS} 步"
info "修复图  ：$REPAIR_DIR（累计，重新训练会覆盖）"
mkdir -p "$REPAIR_DIR"

FORCE_FLAG=()
(( FORCE )) && FORCE_FLAG=(--force)
CURRENT="$WORK"
k=0
for ((YAW=START_YAW; YAW<=MAX_YAW; YAW+=STEP)); do
  k=$((k + 1))
  ROUND="$OUT_ROOT/r$k"
  NEXT="${NAME}_r$k"
  info "---- round $k：yaw ±${YAW}°"
  echo "\$ stage3_views --work-name $CURRENT --yaws +$YAW,-$YAW --out $ROUND/defects"
  echo "\$ syncfix_label --defects $ROUND/defects --out $ROUND/syncfix"
  echo "\$ refine_from_images --work-name $CURRENT --steps $STEPS --promote-name $NEXT"
  (( PLAN )) && { CURRENT="$NEXT"; continue; }

  "${PY_RUN[@]}" scripts/stage3_views.py --subject "$SUBJECT" \
    --work-name "$CURRENT" --yaws "+${YAW},-${YAW}" --out "$ROUND/defects"

  if (( ! WITH_REPAIRS )); then
    # 只用真实帧：视图目录仍要生成（refine 需要 selected.json）
    mkdir -p "$ROUND/empty"
    "${PY_RUN[@]}" scripts/refine_from_images.py \
      --subject "$SUBJECT" --work-name "$CURRENT" \
      --defects "$ROUND/defects" --refined-dir "$ROUND/empty" \
      --steps "$STEPS" --real-per-batch 8 --refined-per-batch 0 \
      --anchor-scope train --promote --promote-name "$NEXT" \
      --eval-every "$STEPS" "${FORCE_FLAG[@]}"
    CURRENT="$NEXT"
    continue
  fi

  "${PY_RUN[@]}" scripts/syncfix_label.py \
    --defects "$ROUND/defects" --out "$ROUND/syncfix" \
    --repo "$SYNCFIX_REPO" --python "$SYNCFIX_PYTHON" \
    --model "$SYNCFIX_MODEL" --ref-size 3

  # 修复图并入数据集（同名覆盖）；顺带做轻量 unsharp 抵消 SyncFix 的软化
  "${PY_RUN[@]}" - "$ROUND/syncfix" "$REPAIR_DIR" "$SHARPEN" <<'PYSHARP' || \
    die "syncfix 没有产出任何修复图：$ROUND/syncfix"
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
src, dst, amount = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
dst.mkdir(parents=True, exist_ok=True)
files = sorted(src.glob("*.png"))
if not files:
    raise SystemExit(1)
for p in files:
    im = Image.open(p).convert("RGB")
    if amount > 0:
        blur = np.asarray(im.filter(ImageFilter.GaussianBlur(radius=1.0)),
                          np.float32)
        arr = np.asarray(im, np.float32)
        im = Image.fromarray(np.clip(arr + amount * (arr - blur), 0, 255)
                             .astype(np.uint8))
    im.save(dst / p.name)
print(f"  修复图 {len(files)} 张（unsharp {amount:g}）")
PYSHARP
  ls "$REPAIR_DIR" | wc -l | xargs -I{} echo "  累计修复图：{} 张"

  "${PY_RUN[@]}" scripts/refine_from_images.py \
    --subject "$SUBJECT" --work-name "$CURRENT" \
    --defects "$ROUND/defects" --refined-dir "$REPAIR_DIR" \
    --steps "$STEPS" --real-per-batch 6 --refined-per-batch 2 \
    --anchor-scope train --promote --promote-name "$NEXT" \
    --eval-every "$STEPS" "${FORCE_FLAG[@]}"

  CURRENT="$NEXT"
done

info "完成：最后一轮模型 models/$SUBJECT/$CURRENT"
echo "         评测：bash eval.sh --subject $SUBJECT --work-name $CURRENT --no-lpips"
